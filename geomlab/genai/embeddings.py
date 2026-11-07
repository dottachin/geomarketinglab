"""Multilingual embeddings (doc §7.2: bge-m3 — French and Arabic support is essential).

Backends (EMBEDDING_BACKEND):
  sentence-transformers  BAAI/bge-m3 locally (1024-d, dense), default
  ollama                 the same model served by Ollama (``ollama pull bge-m3``)
  hash                   deterministic hashing trick, no model download — tests/CI only
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import httpx
import numpy as np

from geomlab.settings import get_settings


class Embedder:
    def __init__(self, backend: str, model: str, dim: int):
        self.backend, self.model_name, self.dim = backend, model, dim

    @lru_cache(maxsize=1)  # noqa: B019 - one model per embedder
    def _st(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:  # noqa: ARG002 - bge-m3 needs no prefix
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "hash":
            vecs = np.stack([_hash_vec(t, self.dim) for t in texts])
        elif self.backend == "ollama":
            r = httpx.post(f"{get_settings().ollama_url}/api/embed", json={"model": self.model_name, "input": texts},
                           timeout=300)
            r.raise_for_status()
            vecs = np.asarray(r.json()["embeddings"], dtype=np.float32)
        else:
            vecs = np.asarray(self._st().encode(texts, batch_size=16, normalize_embeddings=True), dtype=np.float32)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"embedding dim {vecs.shape[1]} != EMBEDDING_DIM {self.dim}")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


def _hash_vec(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 64) % 2 else -1.0
    return v


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    model = s.embedding_model if s.embedding_backend != "ollama" else s.embedding_model.split("/")[-1].lower()
    return Embedder(s.embedding_backend, model, s.embedding_dim)
