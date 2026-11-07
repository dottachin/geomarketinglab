"""Hybrid retrieval (doc §7.2): PostgreSQL full-text (BM25-like ts_rank) + pgvector cosine,
fused with Reciprocal Rank Fusion, optional cross-encoder reranking (bge-reranker), and
metadata filters (city, document type, year). Low top similarity → ``low_confidence``.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import text

from geomlab.genai.embeddings import get_embedder
from geomlab.serving.db import engine
from geomlab.settings import get_settings

RRF_K = 60
CANDIDATES = 30


def _filters(city, doc_type, year) -> tuple[str, dict]:
    where, params = [], {}
    if city:
        where.append("(d.city IS NULL OR d.city = :city)")
        params["city"] = city
    if doc_type:
        where.append("d.doc_type = :doc_type")
        params["doc_type"] = doc_type
    if year:
        where.append("(d.year IS NULL OR d.year = :year)")
        params["year"] = int(year)
    return (" AND " + " AND ".join(where)) if where else "", params


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos + 1)
    return scores


@lru_cache(maxsize=1)
def _reranker():
    name = get_settings().reranker_model
    if not name:
        return None
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def hybrid_search(query: str, k: int = 5, city: str | None = None, doc_type: str | None = None,
                  year: int | None = None) -> tuple[list[dict], bool]:
    vec = get_embedder().embed([query], is_query=True)[0]
    vec_lit = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
    flt, params = _filters(city, doc_type, year)
    params |= {"q": query, "v": vec_lit, "n": CANDIDATES}
    with engine("ro").connect() as con:
        dense = con.execute(text(f"""
            SELECT c.chunk_id, 1 - (c.embedding <=> CAST(:v AS vector)) AS sim
            FROM rag.chunks c JOIN rag.documents d USING (doc_id)
            WHERE true {flt} ORDER BY c.embedding <=> CAST(:v AS vector) LIMIT :n"""), params).all()
        sparse = con.execute(text(f"""
            SELECT c.chunk_id FROM rag.chunks c JOIN rag.documents d USING (doc_id),
                 websearch_to_tsquery('simple', :q) q
            WHERE c.tsv @@ q {flt} ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT :n"""), params).all()
        sims = {r.chunk_id: float(r.sim) for r in dense}
        fused = rrf([[r.chunk_id for r in dense], [r.chunk_id for r in sparse]])
        ids = sorted(fused, key=fused.get, reverse=True)[: max(k * 3, k)]
        if not ids:
            return [], True
        rows = con.execute(text("""
            SELECT c.chunk_id, c.page, c.section, c.content, d.title, d.source_path, d.doc_type, d.city, d.year,
                   1 - (c.embedding <=> CAST(:v AS vector)) AS sim
            FROM rag.chunks c JOIN rag.documents d USING (doc_id) WHERE c.chunk_id = ANY(:ids)"""),
            {"ids": ids, "v": vec_lit}).mappings().all()
    by_id = {r["chunk_id"]: dict(r) for r in rows}
    ordered = [by_id[i] for i in ids if i in by_id]
    rr = _reranker()
    if rr is not None and ordered:
        scores = rr.predict([(query, r["content"]) for r in ordered])
        for r, s in zip(ordered, scores, strict=True):
            r["rerank"] = float(s)
        ordered.sort(key=lambda r: r["rerank"], reverse=True)
    hits = [{"document": r["title"], "source_path": r["source_path"], "page": r["page"], "section": r["section"],
             "doc_type": r["doc_type"], "city": r["city"], "year": r["year"],
             "similarity": round(float(r["sim"]), 3), "text": r["content"]} for r in ordered[:k]]
    top = max((sims.get(r["chunk_id"], r["sim"]) for r in ordered[:k]), default=0.0)
    return hits, bool(top < get_settings().rag_min_similarity)
