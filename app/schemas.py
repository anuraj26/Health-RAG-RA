"""Pydantic schemas for the public API.

The response shape matches the contract given in the assignment exactly,
plus a couple of optional, additive fields (`guardrail_category`,
`disclaimer`) that never break a client expecting the base shape.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Patient question in natural language.")


class EvidenceItem(BaseModel):
    document_id: str
    chunk_id: str
    similarity_score: float


class AskResponse(BaseModel):
    answer: str
    evidence_used: List[EvidenceItem] = Field(default_factory=list)
    evidence_sufficient: bool
    guardrail_triggered: bool
    # --- additive metadata (safe to ignore for the base contract) -------
    guardrail_category: Optional[str] = None
    disclaimer: Optional[str] = None
