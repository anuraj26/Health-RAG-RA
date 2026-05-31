"""Answer generation.

Default backend = "extractive": the answer is assembled ONLY from sentences
that appear in the retrieved chunks, so it is grounded by construction and
cannot hallucinate facts that are not in the corpus. This is a deliberate
safety choice for a health prototype.

Optional backends ("openai" / "anthropic") call a hosted model with a strict
grounded prompt template (included below, and logged for every query). The
prompt instructs the model to use ONLY the provided context and to defer to a
clinician otherwise. These paths require the relevant API key + network.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import get_settings
from app.vectorstore import ScoredChunk

DISCLAIMER = (
    "This is general educational information, not medical advice. Always talk "
    "with your own doctor or care team about your specific situation."
)

INSUFFICIENT_MESSAGE = (
    "I don't have enough reliable information in my reference documents to answer "
    "that safely. I'd rather not guess on a health question. Please ask your doctor, "
    "pharmacist, or care team, who can give you advice based on your full medical history."
)

# Strict grounded prompt template used for the LLM backends. Stored so the
# exact instructions can be reproduced and audited from the research log.
GROUNDED_PROMPT_TEMPLATE = """You are a careful patient-education assistant for heart and \
cardio-kidney-metabolic conditions. Answer ONLY using the numbered context passages \
below. Follow these rules strictly:
- Use only facts stated in the context. Do not add outside knowledge.
- If the context does not contain the answer, say you don't have enough information \
and advise the person to speak with their doctor.
- Write in plain, supportive, patient-friendly language (about a grade 8 reading level).
- Do not give a diagnosis, dosage, or personalised treatment instructions.
- End by encouraging the person to discuss it with their care team.
- After the answer, cite the passages you used as [document_id/chunk_id].

Question: {question}

Context:
{context}

Patient-friendly answer:"""


@dataclass
class GeneratedAnswer:
    answer: str
    model_name: str
    prompt_summary: str


# --- sentence helpers ------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    # Drop markdown header lines entirely (they are titles, not prose) and
    # strip bullet/quote markers, then split the remaining prose on sentence
    # boundaries. This stops headers like "## The two big goals" from being
    # glued onto the following paragraph.
    kept_lines = []
    for line in text.splitlines():
        if re.match(r"^\s*#", line):  # markdown header -> skip
            continue
        kept_lines.append(re.sub(r"^\s*[>\-\*]+\s*", "", line))
    cleaned = " ".join(kept_lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def _keywords(question: str) -> set[str]:
    stop = {
        "what", "should", "about", "with", "have", "this", "that", "from", "your",
        "ask", "the", "and", "for", "are", "can", "how", "does", "will", "any",
        "tell", "give", "more", "into", "when", "which", "who", "why", "you", "my",
    }
    words = re.findall(r"[a-z][a-z\-]{2,}", question.lower())
    return {w for w in words if w not in stop}


def _format_citations(chunks: List[ScoredChunk]) -> str:
    seen = []
    for sc in chunks:
        tag = f"{sc.chunk.document_id}/{sc.chunk.chunk_id}"
        if tag not in seen:
            seen.append(tag)
    return ", ".join(f"[{t}]" for t in seen)


# --- extractive backend ----------------------------------------------------
def _generate_extractive(question: str, chunks: List[ScoredChunk]) -> GeneratedAnswer:
    kw = _keywords(question)
    # Track each sentence's original position so we can restore reading order.
    scored: List[Tuple[float, int, str]] = []
    order = 0
    for chunk_rank, sc in enumerate(chunks):
        for sent in _split_sentences(sc.chunk.text):
            overlap = sum(1 for w in kw if w in sent.lower())
            if overlap:
                # relevance favours keyword overlap, tie-broken by chunk score
                scored.append((overlap + sc.score, order, sent))
            order += 1

    # Pick the most relevant sentences (de-duplicated, since overlapping
    # chunks can repeat a sentence), then re-sort by original order so the
    # answer reads naturally (definition before details, etc.).
    scored.sort(key=lambda x: x[0], reverse=True)
    top: List[Tuple[float, int, str]] = []
    seen_sents: set[str] = set()
    for item in scored:
        norm = item[2].strip()
        if norm in seen_sents:
            continue
        seen_sents.add(norm)
        top.append(item)
        if len(top) >= 4:
            break
    top.sort(key=lambda x: x[1])
    picked = [sent for _, _, sent in top]

    if not picked:
        # fall back to the single most relevant chunk's lead sentences
        picked = _split_sentences(chunks[0].chunk.text)[:2]

    body = " ".join(picked)
    citations = _format_citations(chunks)
    answer = (
        f"Here's what the reference material says:\n\n{body}\n\n"
        f"This is a good topic to raise with your doctor, who can tailor it to you.\n\n"
        f"Sources: {citations}"
    )
    return GeneratedAnswer(
        answer=answer,
        model_name="extractive-grounded (no LLM)",
        prompt_summary="Selected sentences from retrieved chunks overlapping the question "
        "keywords; no generative model used (grounded by construction).",
    )


# --- LLM backends ----------------------------------------------------------
def _build_context(chunks: List[ScoredChunk]) -> str:
    lines = []
    for i, sc in enumerate(chunks, 1):
        tag = f"{sc.chunk.document_id}/{sc.chunk.chunk_id}"
        lines.append(f"[{i}] ({tag}) {sc.chunk.text}")
    return "\n\n".join(lines)


def _generate_openai(question: str, chunks: List[ScoredChunk]) -> GeneratedAnswer:
    from openai import OpenAI

    s = get_settings()
    prompt = GROUNDED_PROMPT_TEMPLATE.format(question=question, context=_build_context(chunks))
    client = OpenAI()
    resp = client.chat.completions.create(
        model=s.openai_chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()
    if "Sources:" not in text and "[doc" not in text:
        text += f"\n\nSources: {_format_citations(chunks)}"
    return GeneratedAnswer(text, s.openai_chat_model, "GROUNDED_PROMPT_TEMPLATE (context-only, cite chunks)")


def _generate_anthropic(question: str, chunks: List[ScoredChunk]) -> GeneratedAnswer:
    import anthropic

    s = get_settings()
    prompt = GROUNDED_PROMPT_TEMPLATE.format(question=question, context=_build_context(chunks))
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=s.anthropic_model,
        max_tokens=600,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    if "Sources:" not in text and "[doc" not in text:
        text += f"\n\nSources: {_format_citations(chunks)}"
    return GeneratedAnswer(text, s.anthropic_model, "GROUNDED_PROMPT_TEMPLATE (context-only, cite chunks)")


def generate_answer(question: str, chunks: List[ScoredChunk]) -> GeneratedAnswer:
    backend = get_settings().llm_backend.lower()
    try:
        if backend == "openai" and os.getenv("OPENAI_API_KEY"):
            return _generate_openai(question, chunks)
        if backend == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return _generate_anthropic(question, chunks)
    except Exception:
        # never fail the request because an external LLM is unavailable;
        # fall back to the grounded extractive answer instead.
        pass
    return _generate_extractive(question, chunks)
