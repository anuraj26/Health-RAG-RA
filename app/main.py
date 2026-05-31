"""FastAPI application exposing the required POST /ask endpoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.pipeline import RagPipeline
from app.schemas import AskRequest, AskResponse

# Built once at startup and reused across requests.
_pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RagPipeline()
    yield
    _pipeline = None


app = FastAPI(
    title="Health AI RAG Backend",
    description="Document-grounded patient-education assistant with evidence control "
    "and safety guardrails. Educational prototype — not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def info() -> dict:
    assert _pipeline is not None
    return {
        "service": "Health AI RAG Backend",
        "status": "ok",
        **_pipeline.backend_info,
        "note": "Educational prototype. Not medical advice.",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    assert _pipeline is not None
    return _pipeline.ask(request.question)
