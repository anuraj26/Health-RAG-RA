"""Tests for the core behaviours the assignment cares about.

Run with:  pytest -q
These use the offline TF-IDF backend, so they need no network or API keys.
"""
from __future__ import annotations

import os

# Force the deterministic offline backend so tests are reproducible anywhere.
os.environ.setdefault("embedding_backend", "tfidf")
os.environ.setdefault("llm_backend", "extractive")

import pytest

from app.guardrails import check_guardrails
from app.pipeline import RagPipeline


@pytest.fixture(scope="module")
def pipeline():
    return RagPipeline()


# --- guardrails ------------------------------------------------------------
@pytest.mark.parametrize(
    "question,category",
    [
        ("I have severe chest pain", "cardiac"),
        ("I suddenly can't breathe", "respiratory"),
        ("My mother just fainted", "neurological"),
        ("I feel severely dizzy and my face is drooping", "neurological"),
        ("I am thinking about suicide", "crisis"),
    ],
)
def test_guardrail_triggers(question, category):
    result = check_guardrails(question)
    assert result.triggered is True
    assert result.category == category


def test_guardrail_negation_not_triggered():
    # "no chest pain" should not escalate
    result = check_guardrails("I have no chest pain, what is HFpEF?")
    assert result.triggered is False


def test_guardrail_normal_question_not_triggered():
    result = check_guardrails("What lifestyle changes help with heart failure?")
    assert result.triggered is False


# --- end-to-end pipeline ---------------------------------------------------
def test_emergency_short_circuits_retrieval(pipeline):
    resp = pipeline.ask("I'm having crushing chest pain right now")
    assert resp.guardrail_triggered is True
    assert resp.evidence_sufficient is False
    assert resp.evidence_used == []
    assert "emergency" in resp.answer.lower()


def test_on_topic_question_is_answered_with_citations(pipeline):
    resp = pipeline.ask("What should I ask my doctor about HFpEF treatment options?")
    assert resp.guardrail_triggered is False
    assert resp.evidence_sufficient is True
    assert len(resp.evidence_used) >= 1
    # every cited item carries doc + chunk ids and a numeric score
    for e in resp.evidence_used:
        assert e.document_id.startswith("doc_")
        assert e.chunk_id.startswith("chunk_")
        assert 0.0 <= e.similarity_score <= 1.0


def test_off_topic_question_is_refused(pipeline):
    resp = pipeline.ask("What is the best brand of running shoes for a marathon?")
    assert resp.evidence_sufficient is False
    assert resp.guardrail_triggered is False
    assert "enough" in resp.answer.lower()


def test_vague_question_is_refused(pipeline):
    resp = pipeline.ask("Tell me about it.")
    assert resp.evidence_sufficient is False


def test_logging_writes_a_row(pipeline, tmp_path, monkeypatch):
    # point the log at a temp sqlite and verify a row lands
    import sqlite3

    from app import logging_store
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "log.sqlite"))
    monkeypatch.setattr(settings, "jsonl_path", str(tmp_path / "log.jsonl"))

    logging_store.log_query(
        {
            "question": "test",
            "retrieved_evidence": [],
            "evidence_sufficient": False,
            "sufficiency_reason": "test",
            "guardrail_triggered": False,
            "guardrail_category": None,
            "final_answer": "test answer",
            "model_name": None,
            "prompt_summary": None,
        }
    )
    conn = sqlite3.connect(str(tmp_path / "log.sqlite"))
    count = conn.execute("SELECT COUNT(*) FROM research_log").fetchone()[0]
    conn.close()
    assert count == 1
