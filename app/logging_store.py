"""Research logging.

Every query is recorded to SQLite (queryable) and mirrored to a JSONL file
(easy to read / diff / share). We log everything the assignment asks for:
timestamp, question, retrieved doc/chunk ids + scores, the sufficiency
decision, the guardrail decision, the final answer, and — when an LLM is
used — the model name and prompt summary.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    retrieved_evidence TEXT NOT NULL,      -- JSON: [{document_id, chunk_id, similarity_score}]
    evidence_sufficient INTEGER NOT NULL,
    sufficiency_reason TEXT,
    guardrail_triggered INTEGER NOT NULL,
    guardrail_category TEXT,
    final_answer TEXT NOT NULL,
    model_name TEXT,
    prompt_summary TEXT
);
"""


def _connect() -> sqlite3.Connection:
    path = Path(get_settings().sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def log_query(record: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a single query record. Returns the record with a timestamp."""
    settings = get_settings()
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}

    # --- SQLite ---
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO research_log
               (timestamp, question, retrieved_evidence, evidence_sufficient,
                sufficiency_reason, guardrail_triggered, guardrail_category,
                final_answer, model_name, prompt_summary)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                record["timestamp"],
                record["question"],
                json.dumps(record.get("retrieved_evidence", [])),
                int(record.get("evidence_sufficient", False)),
                record.get("sufficiency_reason"),
                int(record.get("guardrail_triggered", False)),
                record.get("guardrail_category"),
                record.get("final_answer", ""),
                record.get("model_name"),
                record.get("prompt_summary"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # --- JSONL mirror ---
    jsonl = Path(settings.jsonl_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
