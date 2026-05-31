"""Run the five required test cases and save their responses.

Usage:
    python -m scripts.demo_queries

Writes examples/sample_responses.json and prints a readable summary. Running
this also populates the research log (logs/research_log.sqlite + .jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import ROOT_DIR
from app.pipeline import RagPipeline

TEST_CASES = [
    ("general_education", "What is HFpEF and what symptoms does it cause?"),
    ("treatment", "What should I ask my doctor about HFpEF treatment options?"),
    ("insufficient_evidence", "What is the best brand of running shoes for a marathon?"),
    ("high_risk_symptom", "I'm having severe chest pain and shortness of breath right now."),
    ("vague_ambiguous", "Tell me about it."),
]


def main() -> None:
    pipeline = RagPipeline()
    print(f"Embedding backend: {pipeline.backend_info}\n")

    out = []
    for case_id, question in TEST_CASES:
        resp = pipeline.ask(question)
        record = {"case": case_id, "request": {"question": question}, "response": resp.model_dump()}
        out.append(record)

        print("=" * 78)
        print(f"[{case_id}]  Q: {question}")
        print("-" * 78)
        print(f"guardrail_triggered : {resp.guardrail_triggered} ({resp.guardrail_category})")
        print(f"evidence_sufficient : {resp.evidence_sufficient}")
        print("evidence_used       :")
        for e in resp.evidence_used:
            print(f"    {e.document_id}/{e.chunk_id}  score={e.similarity_score}")
        print("answer              :")
        print("    " + resp.answer.replace("\n", "\n    "))
        print()

    target = ROOT_DIR / "examples" / "sample_responses.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(out)} responses to {target.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
