"""Embedding and retrieval logic for RAG."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

from src.config import (
    EMBED_MODEL_NAME,
    EMBED_PROVIDER,
    HASH_EMBED_DIM,
    OFFLINE_MODE,
    TOP_K_PER_CLAIM,
)
from src.models import Chunk

log = logging.getLogger(__name__)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _l2_norm(v: list[float]) -> float:
    return (_dot(v, v) + 1e-12) ** 0.5


def _normalize(vectors: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in vectors:
        n = _l2_norm(row)
        normalized.append([value / n for value in row])
    return normalized


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class HashEmbedder:
    """Deterministic offline embedder that requires no network or model downloads."""

    def __init__(self, dim: int = HASH_EMBED_DIM) -> None:
        self._dim = max(64, dim)

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> list[list[float]]:
        del convert_to_numpy
        matrix = [[0.0 for _ in range(self._dim)] for _ in texts]
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:2], "big") % self._dim
                sign = 1.0 if digest[2] % 2 == 0 else -1.0
                matrix[row][idx] += sign
        if normalize_embeddings:
            matrix = _normalize(matrix)
        return matrix


class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        provider = os.getenv("EMBED_PROVIDER", EMBED_PROVIDER).strip().lower()
        offline = os.getenv("OFFLINE_MODE", "1" if OFFLINE_MODE else "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if offline or provider == "hash":
            self._embedder = HashEmbedder()
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(
                    os.getenv("EMBED_MODEL_NAME", EMBED_MODEL_NAME)
                )
            except Exception as exc:
                # Falls back to deterministic hash embeddings when local binary deps are broken.
                log.warning(
                    "Falling back to hash embedder because sentence-transformers failed: %s",
                    exc,
                )
                self._embedder = HashEmbedder()
        texts = [chunk.text for chunk in chunks]
        matrix = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()
        self._matrix = _normalize(matrix)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or TOP_K_PER_CLAIM
        q = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        if hasattr(q, "tolist"):
            q = q.tolist()
        q_vec = _normalize(q)[0]
        scores = [_dot(row, q_vec) for row in self._matrix]
        best_idx = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
        return [
            RetrievedChunk(chunk=self._chunks[idx], score=float(scores[idx]))
            for idx in best_idx
        ]
