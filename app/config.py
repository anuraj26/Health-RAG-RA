"""Central configuration.

All tunables live here and can be overridden with environment variables
(see .env.example). Keeping them in one place makes the evidence gate and
guardrail behaviour easy to audit and reason about during review.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
INDEX_DIR = ROOT_DIR / ".index"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- Retrieval -------------------------------------------------------
    # Number of chunks pulled from the vector store per query.
    top_k: int = Field(default=4)
    # Target chunk size (characters) and overlap used during ingestion.
    chunk_size: int = Field(default=700)
    chunk_overlap: int = Field(default=120)

    # --- Embedding backend ----------------------------------------------
    # auto | sentence_transformers | tfidf | openai
    # "auto" prefers sentence-transformers if installed/downloadable,
    # otherwise falls back to the dependency-free TF-IDF embedder.
    embedding_backend: str = Field(default="auto")
    st_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # --- Evidence sufficiency gate --------------------------------------
    # Similarity is cosine in [0, 1]. The "right" threshold depends on the
    # embedding backend, so we keep a value per backend and pick at runtime.
    # These are deliberately conservative for a health use case.
    sufficiency_threshold_tfidf: float = Field(default=0.12)
    sufficiency_threshold_semantic: float = Field(default=0.35)
    # A query is "sufficiently supported" only if the top score clears the
    # threshold AND at least `min_supporting_chunks` clear `support_floor`.
    min_supporting_chunks: int = Field(default=1)
    support_floor_ratio: float = Field(default=0.6)  # fraction of threshold

    # --- LLM answer generation ------------------------------------------
    # extractive | openai | anthropic
    # "extractive" (default) builds the answer ONLY from retrieved sentences,
    # so it cannot hallucinate. The LLM paths use a strict grounded prompt.
    llm_backend: str = Field(default="extractive")
    openai_chat_model: str = Field(default="gpt-4o-mini")
    anthropic_model: str = Field(default="claude-3-5-haiku-latest")

    # --- Logging ---------------------------------------------------------
    sqlite_path: str = Field(default=str(LOG_DIR / "research_log.sqlite"))
    jsonl_path: str = Field(default=str(LOG_DIR / "research_log.jsonl"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
