"""Embedding and retrieval logic for RAG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_NAME, TOP_K_PER_CLAIM
from src.models import Chunk


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._embedder = SentenceTransformer(EMBED_MODEL_NAME)
        texts = [chunk.text for chunk in chunks]
        matrix = self._embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        self._matrix = _normalize(matrix.astype(np.float32))

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or TOP_K_PER_CLAIM
        q = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        q = _normalize(q.astype(np.float32))
        scores = (self._matrix @ q.T).reshape(-1)
        best_idx = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(chunk=self._chunks[int(idx)], score=float(scores[int(idx)]))
            for idx in best_idx
        ]
