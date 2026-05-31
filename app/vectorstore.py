"""Thin FAISS wrapper.

Because every embedder returns L2-normalised vectors, an inner-product
index (`IndexFlatIP`) returns cosine similarity in [0, 1] directly. We keep
the chunk objects in a parallel list so a FAISS row id maps straight back to
its `document_id` / `chunk_id`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from app.embeddings import Embedder
from app.ingest import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._index: faiss.Index | None = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        self.embedder.fit([c.text for c in chunks])
        matrix = self.embedder.embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._index = index

    def search(self, query: str, top_k: int) -> List[ScoredChunk]:
        if self._index is None:
            raise RuntimeError("VectorStore.build() must be called before search().")
        q = self.embedder.embed([query])
        scores, ids = self._index.search(q, min(top_k, len(self._chunks)))
        results: List[ScoredChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            # clamp tiny negative dot-products from float error to 0
            results.append(ScoredChunk(chunk=self._chunks[idx], score=max(0.0, float(score))))
        return results

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
