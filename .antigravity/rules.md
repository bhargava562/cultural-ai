# Antigravity Agent Directives: Cultural Alignment Middleware

## Project Goals
- Build a non-invasive cultural alignment middleware layer for regional LLM queries (Tamil/English/Tanglish).
- Implement a two-stage pipeline: Cultural RAG Context Injection -> Candidate Generation -> Alignment Judge -> Feedback-Guided Regeneration.

## Development Constraints
1. **Free-Tier API Economy:**
   - Use free providers (Groq Cloud API, Google Gemini Flash API via AI Studio, OpenRouter free endpoints).
   - Use LiteLLM or native async clients with rate-limit retries and backoff.
   - Gracefully fallback across free providers when encountering HTTP 429 / TPM limits.
2. **Local Vector Storage:**
   - Use local ChromaDB with open-source multilingual sentence embeddings (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or `bge-m3`). Zero cloud vector DB costs.
3. **Data Handling:**
   - Ingest from Hugging Face (`wickkiey/tamil-wikipedia-chunked`, `community-datasets/tamilmixsentiment`, `mozhi-ai/tamil-corpus`).
   - Treat datasets as streaming/batched to preserve local memory and disk space.
4. **Scoring Contract:**
   - Evaluator MUST return structured JSON conforming strictly to `docs/EVALUATION_RUBRIC.md` and validated by Pydantic models.
5. **Code Quality & Testing:**
   - All modules must include type annotations, docstrings, and robust error handling.
   - Keep mock tests executable offline without requiring active API keys.
