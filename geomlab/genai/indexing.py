"""Document indexing for RAG (doc §7.2): corpus → section-aware chunks → embeddings → pgvector.

Corpus layout: ``corpus/<doc_type>/<file>.{pdf,md,txt}`` with an optional sidecar
``<file>.meta.yaml`` giving ``title``, ``city``, ``year``, ``lang``, ``source_url``.
Incremental: a document is re-indexed only when its SHA-256 changed (same change-detection
idea as the ingestion DAGs).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import text

from geomlab.genai.embeddings import get_embedder
from geomlab.logging import get_logger
from geomlab.serving.db import engine
from geomlab.settings import get_settings

log = get_logger(__name__)
SUFFIXES = {".pdf", ".md", ".txt"}
MAX_CHARS = 1500
OVERLAP = 200


@dataclass
class Chunk:
    page: int | None
    section: str | None
    content: str


def _split(text_: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Paragraph-packing splitter; a paragraph longer than max_chars is cut with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text_) if p.strip()]
    out, buf = [], ""
    for p in paras:
        while len(p) > max_chars:
            out.append(p[:max_chars])
            p = p[max_chars - overlap :]
        if len(buf) + len(p) + 2 > max_chars and buf:
            out.append(buf)
            buf = buf[-overlap:] + "\n\n" + p if overlap else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def chunk_markdown(md: str) -> list[Chunk]:
    """Chunk by heading so every chunk knows its section (doc §7.2 'chunking by section')."""
    chunks, section, body = [], None, []
    for line in md.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if body:
                chunks += [Chunk(None, section, c) for c in _split("\n".join(body))]
            section, body = m.group(1).strip(), []
        else:
            body.append(line)
    if body:
        chunks += [Chunk(None, section, c) for c in _split("\n".join(body))]
    return chunks


def chunk_pdf(path: Path) -> tuple[list[Chunk], int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        first = next((ln.strip() for ln in txt.splitlines() if ln.strip()), None)
        chunks += [Chunk(i, first[:120] if first else None, c) for c in _split(txt)]
    return chunks, len(reader.pages)


def load_document(path: Path) -> tuple[dict, list[Chunk]]:
    meta_path = path.with_suffix(path.suffix + ".meta.yaml")
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    n_pages = None
    if path.suffix == ".pdf":
        chunks, n_pages = chunk_pdf(path)
    else:
        chunks = chunk_markdown(path.read_text(encoding="utf-8"))
    rel = path.relative_to(get_settings().corpus_dir)
    doc = {
        "doc_id": str(rel).replace("/", "__"),
        "title": meta.get("title") or path.stem.replace("_", " "),
        "source_path": str(rel),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "doc_type": meta.get("doc_type") or (rel.parts[0] if len(rel.parts) > 1 else None),
        "city": meta.get("city"), "year": meta.get("year"), "lang": meta.get("lang"), "n_pages": n_pages,
    }
    return doc, [c for c in chunks if len(c.content) > 40]


def index_corpus(corpus_dir: Path | None = None, force: bool = False) -> dict:
    corpus_dir = corpus_dir or get_settings().corpus_dir
    emb = get_embedder()
    files = sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in SUFFIXES and p.name != "README.md")
    stats = {"indexed": 0, "unchanged": 0, "removed": 0, "chunks": 0}
    seen = set()
    with engine().begin() as con:
        known = dict(con.execute(text("SELECT doc_id, sha256 FROM rag.documents")).all())
        for path in files:
            doc, chunks = load_document(path)
            seen.add(doc["doc_id"])
            if not force and known.get(doc["doc_id"]) == doc["sha256"]:
                stats["unchanged"] += 1
                continue
            vecs = emb.embed([f"{c.section or ''}\n{c.content}" for c in chunks])
            con.execute(text("DELETE FROM rag.documents WHERE doc_id = :d"), {"d": doc["doc_id"]})
            con.execute(text("""INSERT INTO rag.documents (doc_id, title, source_path, sha256, doc_type, city, year, lang, n_pages)
                VALUES (:doc_id, :title, :source_path, :sha256, :doc_type, :city, :year, :lang, :n_pages)"""), doc)
            if chunks:
                con.execute(text("""INSERT INTO rag.chunks (doc_id, page, section, content, embedding)
                    VALUES (:d, :p, :s, :c, CAST(:e AS vector))"""),
                    [{"d": doc["doc_id"], "p": c.page, "s": c.section, "c": c.content,
                      "e": "[" + ",".join(f"{x:.6f}" for x in v) + "]"} for c, v in zip(chunks, vecs, strict=True)])
            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
            log.info("indexed doc=%s chunks=%d", doc["doc_id"], len(chunks))
        for gone in set(known) - seen:
            con.execute(text("DELETE FROM rag.documents WHERE doc_id = :d"), {"d": gone})
            stats["removed"] += 1
    return stats
