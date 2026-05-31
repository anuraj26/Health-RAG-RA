# Health AI RAG Backend (HFpEF / Cardio-Kidney-Metabolic)

A small, document-grounded retrieval-augmented generation (RAG) backend for a
patient-education assistant. It answers questions **only** from a curated set of
reference documents, refuses when the evidence is too weak, escalates urgent
medical situations instead of advising, and logs every query for research.

> **This is an educational prototype, not a medical device.** It does not
> diagnose, prescribe, or replace a clinician. Every answer carries a disclaimer
> and points the user back to their care team.

---

## 1. How to run the system

### Prerequisites
- Python 3.10+

### Setup
```bash
# from the project root
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
The index is built in memory at startup from the files in `data/`. Then:
```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I ask my doctor about HFpEF treatment options?"}'
```
Interactive docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.

### Run the 5 demo questions (also populates the research log)
```bash
python -m scripts.demo_queries
```

### Run the tests
```bash
pytest -q
```

**Zero-config note:** the system runs with **no API keys and no model
downloads** out of the box, because it falls back to a deterministic offline
embedder (see §2). To turn on semantic embeddings or a hosted LLM, copy
`.env.example` to `.env` and follow §2 / §4.

---

## 2. Embedding model and vector database

**Vector store:** [FAISS](https://github.com/facebookresearch/faiss)
(`IndexFlatIP`). All embeddings are L2-normalised, so inner product equals
**cosine similarity in [0, 1]**, which is what the API reports as
`similarity_score`.

**Embeddings are pluggable** (`app/embeddings.py`) behind one interface, chosen
via the `embedding_backend` setting (`auto` by default):

| Backend | Type | When used | Notes |
|---|---|---|---|
| `sentence_transformers` | semantic, local | if installed + model downloadable | `all-MiniLM-L6-v2`; **recommended** |
| `openai` | semantic, hosted | if `OPENAI_API_KEY` set | `text-embedding-3-small` |
| `tfidf` | lexical, offline | always available (fallback) | word + char n-grams, deterministic |

With `auto`, the system prefers OpenAI (if a key is set), then
sentence-transformers (if usable), and finally the offline TF-IDF embedder. This
keeps the prototype runnable anywhere while letting a reviewer upgrade retrieval
quality with a single `pip install sentence-transformers`.

> **About the sample outputs in this README (§9):** they were generated with the
> **offline TF-IDF backend**, since the build environment had no model download
> access. TF-IDF is purely lexical, so similarity numbers are lower and more
> literal than semantic embeddings would produce. The thresholds in §3 are
> calibrated per-backend precisely because of this difference.

---

## 3. How the evidence sufficiency gate works

Retrieval always returns the top-`k` (default 4) nearest chunks, but an answer is
only generated if the evidence clears a gate (`app/retriever.py`). A query is
**sufficient** only if **both** hold:

1. the **top** similarity score ≥ `threshold`, and
2. at least `min_supporting_chunks` (default 1) chunks score ≥ a **support
   floor** = `threshold × support_floor_ratio` (default 0.6 × threshold).

The threshold is **backend-specific** because cosine scores from a lexical
embedder and a semantic embedder are not comparable:

- `sufficiency_threshold_tfidf = 0.12`
- `sufficiency_threshold_semantic = 0.35`

If the gate fails, the system returns a safe refusal *and still reports the weak
evidence it found* (for transparency), with `evidence_sufficient: false`. The
exact numeric reason (e.g. `top score 0.093 < threshold 0.120`) is written to the
research log so every decision is auditable. All thresholds are env-configurable.

---

## 4. How the safety guardrails work

Guardrails run **before** retrieval and generation (`app/guardrails.py`). If a
question describes an urgent or high-risk situation, the system does **not** try
to answer from documents — it returns an escalation message pointing to
emergency care, sets `guardrail_triggered: true`, and skips retrieval entirely.

Detection is **rule-based** (regex over word boundaries) on purpose: it is
transparent, deterministic, testable, and easy to audit and extend — properties
that matter more than cleverness for a safety gate. Categories:

- **cardiac** — chest pain/pressure/tightness, pain radiating to arm/jaw, "heart attack"
- **respiratory** — can't breathe, severe/sudden shortness of breath, choking
- **neurological** — fainting/syncope, severe dizziness, slurred speech, face droop, sudden weakness/numbness, stroke, seizure
- **general_emergency** — coughing/vomiting blood, unconscious/unresponsive
- **crisis** — self-harm / suicidal ideation → routed to a supportive crisis message (988) with highest precedence

**Design choices worth noting:**
- For a patient-facing tool, a **false positive** (escalating a non-emergency) is
  far cheaper than a **false negative**, so matching is intentionally broad.
- A light **negation guard** suppresses obvious non-emergencies like
  *"I have **no** chest pain, what is HFpEF?"* to reduce nuisance escalations.
- Even when the gate passes and an answer is produced, every answer appends a
  medical disclaimer, and the corpus itself contains a dedicated safety document.

---

## 5. What is logged for research

Every query is written to **SQLite** (`logs/research_log.sqlite`, queryable) and
mirrored to **JSONL** (`logs/research_log.jsonl`, easy to read/diff). Each record
contains exactly the fields the assignment asks for:

- `timestamp` (UTC, ISO-8601)
- `question`
- `retrieved_evidence` — list of `{document_id, chunk_id, similarity_score}`
- `evidence_sufficient` + `sufficiency_reason` (the numeric decision)
- `guardrail_triggered` + `guardrail_category`
- `final_answer`
- `model_name` (e.g. `extractive-grounded (no LLM)` or the LLM model used)
- `prompt_summary` (or the full grounded prompt template when an LLM is used)

A real sample is included at `examples/sample_research_log.jsonl`.

---

## 6. Main limitations of this prototype

- **Tiny, hand-written corpus.** Five short documents are enough to demonstrate
  grounding and the gate, but not real coverage. Retrieval quality is bounded by
  what is in `data/`.
- **Lexical fallback in this build.** The shown outputs use TF-IDF, which matches
  words rather than meaning, so paraphrased questions retrieve less well and
  scores are low/literal. Semantic embeddings (a one-line install) fix most of
  this.
- **Extractive answers are stitched sentences.** The default generator is
  hallucination-proof by construction, but reads less fluently than an LLM. The
  LLM backends exist but were not exercised in this build (no keys/network).
- **Rule-based guardrails miss novel phrasings.** Regex is auditable but brittle;
  it won't catch every way a person could describe an emergency.
- **No authentication, rate limiting, persistence of the index, or de-identification.**
  The log stores the raw question, which in production would need PII handling.
- **Thresholds are hand-tuned**, not learned, and were calibrated on this small
  corpus.

---

## 7. What I would improve over a 4-month project

- **Retrieval:** move to strong semantic embeddings (or a clinical model such as
  a biomedical sentence encoder), add **hybrid search** (BM25 + dense) with a
  **re-ranker**, and tune chunking on real documents.
- **Evidence control:** replace the static threshold with a **calibrated**
  decision (e.g. score distributions, an "is-answerable" classifier, or
  cross-encoder relevance), and add **claim-level grounding checks** that verify
  each sentence of the answer is supported by a citation.
- **Guardrails:** layer a learned triage classifier on top of the rules, add
  red-team test suites, clinical review of the escalation logic, and structured
  symptom triage aligned to validated protocols.
- **Content & safety governance:** source documents from vetted guidelines with
  versioning and provenance, add clinician sign-off, reading-level checks, and
  bias/safety evaluations.
- **LLM answer layer:** use a grounded LLM with strict citation enforcement and
  automatic refusal when context is insufficient, plus answer-faithfulness
  evals (e.g. RAGAS-style metrics).
- **Engineering:** persist the FAISS index, add auth + rate limiting, observability
  and evaluation dashboards, de-identified logging with consent, CI, and
  containerised deployment.
- **Research logging → research value:** capture feedback labels, enable A/B of
  retrieval/threshold settings, and build an offline eval harness over a labelled
  question set.

---

## 8. AI Tool Use Disclosure


- **Tools used:** Claude (Anthropic).



---

## 9. Required test cases (actual outputs)

Generated with `python -m scripts.demo_queries` using the **offline TF-IDF**
backend. Full structured JSON is in `examples/sample_responses.json`.

### 1. General HFpEF education question — *answered*
**Q:** "What is HFpEF and what symptoms does it cause?"
`evidence_sufficient: true`, `guardrail_triggered: false`
Evidence: `doc_1/chunk_0 (0.135)`, `doc_1/chunk_1 (0.129)`, `doc_1/chunk_2 (0.129)`, `doc_1/chunk_5 (0.100)`
> Heart failure with preserved ejection fraction, often shortened to HFpEF, is a
> type of heart failure in which the heart's main pumping chamber (the left
> ventricle) squeezes out a normal amount of blood with each beat, but the
> chamber has become stiff and does not relax and fill properly between beats.
> In HFpEF the ejection fraction is normal or near normal, usually 50 percent or
> higher... *(Sources: doc_1/chunk_0, chunk_1, chunk_2, chunk_5; + disclaimer)*

### 2. Treatment-related question — *answered*
**Q:** "What should I ask my doctor about HFpEF treatment options?"
`evidence_sufficient: true`, `guardrail_triggered: false`
Evidence: `doc_2/chunk_0 (0.330)`, `doc_2/chunk_5 (0.154)`, `doc_2/chunk_3 (0.100)`, `doc_2/chunk_4 (0.068)`
> ...Treatment for HFpEF generally aims to do two things: relieve symptoms
> (mostly by removing extra fluid) and treat the underlying conditions that make
> HFpEF worse, such as high blood pressure, diabetes, obesity, atrial
> fibrillation, and kidney disease... *(Sources: doc_2/chunk_0, chunk_5, chunk_3, chunk_4; + disclaimer)*

### 3. Insufficient evidence — *refused safely*
**Q:** "What is the best brand of running shoes for a marathon?"
`evidence_sufficient: false`, `guardrail_triggered: false`
Top score `0.091 < threshold 0.120`.
> I don't have enough reliable information in my reference documents to answer
> that safely. I'd rather not guess on a health question. Please ask your doctor,
> pharmacist, or care team...

### 4. High-risk symptom — *guardrail escalation*
**Q:** "I'm having severe chest pain and shortness of breath right now."
`evidence_sufficient: false`, `guardrail_triggered: true`, `guardrail_category: cardiac`
(retrieval skipped entirely)
> This looks like it could be a medical emergency. I can't give medical advice
> for urgent symptoms... please call your local emergency number (911 in
> Canada/US) or go to the nearest emergency department right now...

### 5. Vague / ambiguous question — *refused safely*
**Q:** "Tell me about it."
`evidence_sufficient: false`, `guardrail_triggered: false`
Top score `0.094 < threshold 0.120`.
> I don't have enough reliable information in my reference documents to answer
> that safely...

---

## Project structure

```
health-rag/
├── app/
│   ├── config.py          # all tunables (thresholds, backends, paths)
│   ├── schemas.py         # request/response models (API contract)
│   ├── ingest.py          # load .txt/.md, chunk, assign doc_id/chunk_id
│   ├── embeddings.py      # sentence-transformers | openai | tfidf
│   ├── vectorstore.py     # FAISS cosine index
│   ├── retriever.py       # retrieval + evidence sufficiency gate
│   ├── guardrails.py      # emergency / crisis detection (pre-retrieval)
│   ├── generator.py       # grounded extractive answer (+ optional LLM)
│   ├── logging_store.py   # SQLite + JSONL research log
│   ├── pipeline.py        # orchestration
│   └── main.py            # FastAPI app, POST /ask
├── data/                  # 5 reference documents
├── scripts/demo_queries.py
├── tests/test_pipeline.py
├── examples/              # curl/.http, sample_responses.json, sample_research_log.jsonl
├── logs/                  # generated research log
├── requirements.txt
└── .env.example
```
```
POST /ask
  → guardrail check ──(triggered)──> escalate + log, stop
  → retrieve top-k from FAISS
  → sufficiency gate ──(insufficient)──> safe refusal + log
  → generate grounded answer (extractive or LLM)
  → append disclaimer, log, return
```
