"""Retrieval + evidence sufficiency gate.

The gate is what makes this "evidence-controlled". Retrieval always returns
the top_k nearest chunks, but we only allow an answer to be generated if the
retrieved evidence is strong enough. Otherwise the system refuses safely.

Decision rule (all must hold):
  1. top score >= threshold for the active embedding backend, AND
  2. at least `min_supporting_chunks` chunks score >= support_floor,
     where support_floor = threshold * support_floor_ratio.

We expose the numbers we used in `SufficiencyDecision` so the research log
and the API consumer can see exactly why a query was accepted or refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.config import get_settings
from app.embeddings import Embedder
from app.vectorstore import ScoredChunk, VectorStore


@dataclass
class SufficiencyDecision:
    sufficient: bool
    threshold: float
    support_floor: float
    top_score: float
    num_supporting: int
    reason: str


def _threshold_for(embedder: Embedder) -> float:
    s = get_settings()
    return s.sufficiency_threshold_semantic if embedder.is_semantic else s.sufficiency_threshold_tfidf


def evaluate_sufficiency(results: List[ScoredChunk], embedder: Embedder) -> SufficiencyDecision:
    s = get_settings()
    threshold = _threshold_for(embedder)
    support_floor = threshold * s.support_floor_ratio

    if not results:
        return SufficiencyDecision(False, threshold, support_floor, 0.0, 0, "no chunks retrieved")

    top_score = results[0].score
    num_supporting = sum(1 for r in results if r.score >= support_floor)

    if top_score < threshold:
        return SufficiencyDecision(
            False, threshold, support_floor, top_score, num_supporting,
            f"top score {top_score:.3f} < threshold {threshold:.3f}",
        )
    if num_supporting < s.min_supporting_chunks:
        return SufficiencyDecision(
            False, threshold, support_floor, top_score, num_supporting,
            f"only {num_supporting} chunk(s) >= support floor {support_floor:.3f}",
        )
    return SufficiencyDecision(
        True, threshold, support_floor, top_score, num_supporting,
        f"top score {top_score:.3f} >= threshold {threshold:.3f} with "
        f"{num_supporting} supporting chunk(s)",
    )


class Retriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(self, question: str) -> List[ScoredChunk]:
        return self.store.search(question, top_k=get_settings().top_k)

    def assess(self, results: List[ScoredChunk]) -> SufficiencyDecision:
        return evaluate_sufficiency(results, self.store.embedder)
