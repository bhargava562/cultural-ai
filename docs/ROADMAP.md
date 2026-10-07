# CAL-LLM Technical Implementation Roadmap

```text
Phase 1 (W1-W2)              Phase 2 (W3-W4)              Phase 3 (W5-W6)              Phase 4 (W7-W8)
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│ Dataset Ingestion     │───>│ Pipeline & Clients    │───>│ Calibration & Eval    │───>│ Benchmarking & Demo   │
│ - Chunking            │    │ - Unified Free APIs   │    │ - Scoring Rubrics     │    │ - INDICA Test Suite   │
│ - Local Vector Store  │    │ - Anthropological RAG │    │ - Self-Refine Loop    │    │ - Streamlit UI / Demo │
└───────────────────────┘    └───────────────────────┘    └───────────────────────┘    └───────────────────────┘
```

---

## Phase 1: Knowledge Base Ingestion & Vector Indexing (Weeks 1–2)

- [x] Create local storage directories (`data/datasets/`, `data/raw/`, `vector_store/chroma_db/`).
- [x] Implement local dataset ingestion engine (`src/retriever/ingest.py`) supporting Parquet, JSON, JSONL, and CSV.
- [x] Configure 512-token chunking with 64-token overlap.
- [x] Set up local dense vector embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` / `bge-m3`).
- [ ] Ingest full chunks from `wickkiey/tamil-wikipedia-chunked` and `mozhi-ai/tamil-corpus`.
- [x] Verify vector search retrieval accuracy for local entities (Pongal rituals, Joint family decision structures, regional health concepts).

---

## Phase 2: Pipeline Interceptor & Free API Routing (Weeks 3–4)

- [x] Configure `src/utils/llm_clients.py` with LiteLLM supporting Groq and Google Gemini Flash free tiers.
- [x] Implement the generator prompt builder (`src/pipeline/generator.py`) utilizing Anthropological Persona conditioning and domain classification.
- [x] Implement `src/pipeline/evaluator.py` enforcing structured JSON evaluation output matching CAS formula.
- [x] Add automated retries with exponential backoff for HTTP 429 rate limits.

---

## Phase 3: Feedback Resampling & Sentiment Checking (Weeks 5–6)

- [x] Wire the Self-Refine resampling loop in `src/pipeline/interceptor.py`.
- [x] Set hyperparameter threshold defaults ($\tau = 0.75$, max retry count = 2).
- [ ] Integrate `tamilmixsentiment` heuristics to guard against hostile or unnatural code-mixed Tanglish tone.
- [x] Validate feedback-guided iterative improvement via unit and mock tests.

---

## Phase 4: Benchmarking, Validation & Demo Interface (Weeks 7–8)

- [ ] Create an evaluation benchmark set of 150 test queries across 3 target verticals:
  - **Healthcare:** Dietary practices, elder care, pregnancy norms.
  - **Banking / Finance:** Chit funds (seetu), self-help groups (SHG), family asset ownership.
  - **Social Commonsense:** Career guidance, festivals, kinship naming.
- [ ] Measure Cultural Alignment Score (CAS) uplift comparing raw foundation LLMs vs. CAL-LLM.
- [ ] Package a Streamlit / FastAPI comparison playground demonstrating raw vs. aligned outputs side-by-side.
