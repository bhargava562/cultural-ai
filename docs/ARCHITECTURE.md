# CAL-LLM System Architecture Specification

The **Cultural Alignment Layer for Regional LLMs (CAL-LLM)** functions as an operational, non-invasive middleware interposed between regional user queries and LLM generation. It resolves systematic cultural dissonance in multilingual models where linguistic fluency defaults to Western or non-regional cultural frames.

```text
Standard Flow: User Query (Tamil/Tanglish) ────> Base LLM ────> Fluent but Misaligned Answer
Proposed Layer: User Query ─> Context RAG ─> Base LLM ─> Evaluator Judge ─> Resampled / Grounded Output
```

---

## 1. Five-Layer Architecture Overview

```text
                                 ┌─────────────────────────┐
                                 │   User Query (Prompt)   │
                                 └────────────┬────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │    Layer 1: Language & Linguistic Routing     │
                      │  - Language Detection (Tamil / Tanglish / EN) │
                      │  - Code-Mixing Analysis (TamilMixSentiment)   │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │      Layer 2: Regional Cultural RAG Engine    │
                      │  - Dense Retrieval via Local ChromaDB Index   │
                      │  - Sources: Tamil Wiki + Mozhi Corpus Chunks  │
                      │  - Injects Anthropological / Cultural Profile │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │    Layer 3: Candidate Generator (LLM 1)       │
                      │  - Free-Tier Model (Groq LPU / Mistral Saba)  │
                      │  - Produces Initial Candidate Response R_0    │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │     Layer 4: Alignment & Safety Judge         │
                      │  - Rubric Evaluation (Gemini 1.5 Flash Free)  │
                      │  - Computes Cultural Alignment Score (CAS)    │
                      └───────────────┬───────────────────────────────┘
                                      │
                         [Threshold Score >= 0.75 & Safe]
                                     / \
                                    /   \
                             PASS  /     \  FAIL
                                  /       \
                                 ▼         ▼
             ┌────────────────────────┐   ┌───────────────────────────┐
             │ Final Aligned Response │   │ Layer 5: Feedback-Guided  │
             └────────────────────────┘   │ Resampling Loop (Max = 2) │
                                          └─────────────┬─────────────┘
                                                        │
                                                        └───> Rewrites R_1
```

---

## 2. Component Breakdown & Dataset Roles

| Pipeline Layer | Primary Function | Dataset / Asset Source | Engine / Model Used |
|---|---|---|---|
| **Linguistic Routing** | Detect script, dialect markers, domain category, and code-mixed sentiment | `community-datasets/tamilmixsentiment` | FastText / regex token & sentiment analyzer |
| **Cultural Retriever** | Fetch sub-regional context (customs, dietary norms, social practices) | `wickkiey/tamil-wikipedia-chunked` & `mozhi-ai/tamil-corpus` | Local ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2` / `bge-m3` |
| **Candidate Generator** | Produce candidate response under cultural persona framing | Anthropological Prompting System Prompt | Free-Tier API (`groq/openai/gpt-oss-120b`, `groq/mistral-saba-24b`) |
| **Cultural Evaluator** | Score response quality across 4 rubric dimensions | Hofstede CAT + UNESCO UFCS Taxonomy | Free-Tier API (`gemini/gemini-1.5-flash`) |
| **Feedback Resampler** | Critique and regenerate failed candidate responses | Alignment critique JSON string | Self-Refine loop execution (max 2 retries) |

---

## 3. Detailed Component Flows

### 3.1 Ingestion Flow
1. **Source Ingestion:** Stream or batch load dataset files (`tamil-wikipedia-chunked`, `tamil-corpus`, or user-uploaded files in `data/datasets/`).
2. **Preprocessing & Chunking:** Strip metadata artifacts, clean text, and split into **512-token chunks** with **64-token overlap**.
3. **Dense Vector Embeddings:** Compute dense vector embeddings locally using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or `BAAI/bge-m3`.
4. **Persistent Vector Indexing:** Upsert vectors and chunks into a local persistent ChromaDB collection (`./vector_store/chroma_db`).

### 3.2 Inference Flow
1. **Query Inspection:**
   - Detect language: Tamil, Tanglish, or Regional English.
   - Classify domain category: **Healthcare**, **Banking / Finance**, or **Everyday Commonsense**.
2. **Context Retrieval:**
   - Dense semantic query against local ChromaDB for top-$k$ ($k=3$) knowledge chunks.
3. **First-Pass Generation (Anthropological Prompting):**
   ```text
   Region: Tamil Nadu, India
   Tone: Respectful, Culturally Grounded
   Domain: {domain}
   Language Mode: {language_mode}
   Context: {retrieved_knowledge_chunks}
   Query: {user_query}
   ```
4. **Judge Scoring & Cultural Alignment Score (CAS):**
   - The judge parses the query, context, and candidate into structured JSON:
   ```json
   {
     "context_understanding": 0.90,
     "social_appropriateness": 0.85,
     "regional_relevance": 0.95,
     "domain_risk_detected": false,
     "alignment_score": 0.90,
     "verdict": "PASS",
     "critique": "None"
   }
   ```
5. **Conditional Branching:**
   - If $\text{CAS} \ge 0.75$ and $\text{domain\_risk\_detected} == \text{false}$: deliver directly to client.
   - If $\text{CAS} < 0.75$ or risk flagged: invoke feedback resampling with critique instructions (maximum 2 retries).

---

## 4. Target Performance Benchmarks

- **Win Rate:** $\ge 85\%$ win rate on regional commonsense prompts compared to raw foundation models.
- **Latency:** Sub-2-second end-to-end latency for first-pass accepted answers.
- **Cost:** Zero external cloud database or paid compute dependencies.
