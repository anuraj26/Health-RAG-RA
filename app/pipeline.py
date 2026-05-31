"""End-to-end orchestration for a single /ask call.

Flow:
  1. Guardrail check (emergency / crisis)  -> if triggered, escalate & stop.
  2. Retrieve top_k chunks from the vector store.
  3. Evidence sufficiency gate              -> if insufficient, refuse safely.
  4. Generate a grounded patient-friendly answer.
  5. Always log the query for research.

The pipeline holds the built index in memory and is created once at startup.
"""
from __future__ import annotations

from typing import List

from app.config import DATA_DIR
from app.embeddings import build_embedder
from app.generator import (
    DISCLAIMER,
    INSUFFICIENT_MESSAGE,
    generate_answer,
)
from app.guardrails import check_guardrails
from app.ingest import load_chunks
from app.logging_store import log_query
from app.retriever import Retriever
from app.schemas import AskResponse, EvidenceItem
from app.vectorstore import ScoredChunk, VectorStore


class RagPipeline:
    def __init__(self) -> None:
        self.embedder = build_embedder()
        self.store = VectorStore(self.embedder)
        self.store.build(load_chunks(DATA_DIR))
        self.retriever = Retriever(self.store)

    @property
    def backend_info(self) -> dict:
        return {
            "embedding_backend": self.embedder.name,
            "is_semantic": self.embedder.is_semantic,
            "num_chunks": self.store.num_chunks,
        }

    @staticmethod
    def _evidence_items(results: List[ScoredChunk]) -> List[EvidenceItem]:
        return [
            EvidenceItem(
                document_id=r.chunk.document_id,
                chunk_id=r.chunk.chunk_id,
                similarity_score=round(r.score, 4),
            )
            for r in results
        ]

    def ask(self, question: str) -> AskResponse:
        # 1) Guardrails first -------------------------------------------------
        guard = check_guardrails(question)
        if guard.triggered:
            log_query(
                {
                    "question": question,
                    "retrieved_evidence": [],
                    "evidence_sufficient": False,
                    "sufficiency_reason": "skipped (guardrail triggered)",
                    "guardrail_triggered": True,
                    "guardrail_category": guard.category,
                    "final_answer": guard.message,
                    "model_name": None,
                    "prompt_summary": "guardrail escalation (no retrieval/generation)",
                }
            )
            return AskResponse(
                answer=guard.message,
                evidence_used=[],
                evidence_sufficient=False,
                guardrail_triggered=True,
                guardrail_category=guard.category,
                disclaimer=None,
            )

        # 2) Retrieve ---------------------------------------------------------
        results = self.retriever.retrieve(question)
        evidence_items = self._evidence_items(results)

        # 3) Sufficiency gate -------------------------------------------------
        decision = self.retriever.assess(results)
        if not decision.sufficient:
            log_query(
                {
                    "question": question,
                    "retrieved_evidence": [e.model_dump() for e in evidence_items],
                    "evidence_sufficient": False,
                    "sufficiency_reason": decision.reason,
                    "guardrail_triggered": False,
                    "guardrail_category": None,
                    "final_answer": INSUFFICIENT_MESSAGE,
                    "model_name": None,
                    "prompt_summary": "refused: evidence below sufficiency threshold",
                }
            )
            return AskResponse(
                answer=INSUFFICIENT_MESSAGE,
                evidence_used=evidence_items,  # transparency: show what we *did* find
                evidence_sufficient=False,
                guardrail_triggered=False,
                disclaimer=DISCLAIMER,
            )

        # 4) Generate ---------------------------------------------------------
        gen = generate_answer(question, results)
        answer_text = f"{gen.answer}\n\n{DISCLAIMER}"

        # 5) Log --------------------------------------------------------------
        log_query(
            {
                "question": question,
                "retrieved_evidence": [e.model_dump() for e in evidence_items],
                "evidence_sufficient": True,
                "sufficiency_reason": decision.reason,
                "guardrail_triggered": False,
                "guardrail_category": None,
                "final_answer": answer_text,
                "model_name": gen.model_name,
                "prompt_summary": gen.prompt_summary,
            }
        )
        return AskResponse(
            answer=answer_text,
            evidence_used=evidence_items,
            evidence_sufficient=True,
            guardrail_triggered=False,
            disclaimer=DISCLAIMER,
        )
