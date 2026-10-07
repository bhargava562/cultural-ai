# Agent Memory Checkpoint & Operating Guide (`AGENTS.md`)

This repository hosts the **Cultural Alignment Layer**, an open-source, non-invasive middleware designed to evaluate, guide, and adapt LLM responses for regional nuance, socio-cultural appropriateness, and linguistic authenticity across **Tamil, Tanglish, and English**.

---

## 1. Quick Orientation

- **Language Support**: Tamil (தமிழ்), Tanglish (Romanized Tamil), Indian English with regional idioms.
- **Architecture**: Two-stage loop:
  1. `Retriever`: Queries local ChromaDB index populated with Tamil cultural and linguistic corpora.
  2. `Generator`: Generates candidate answers with cultural context injection and anthropological persona.
  3. `Evaluator (Judge)`: Scores candidates against 4 cultural dimensions (Context Understanding, Social Appropriateness, Regional Relevance, Stereotype/Risk).
  4. `Interceptor`: Orchestrates feedback-guided rewriting if the score falls below threshold (`0.75`) or flags risk (max 2 iterations).
- **Core Technology Stack**:
  - Python 3.10+
  - LiteLLM (multi-provider routing over free tiers: Groq, Gemini Flash, OpenRouter)
  - ChromaDB (local persistent vector store)
  - Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
  - Pydantic v2 (strict JSON contract enforcement)
  - Rich & Pytest

---

## 2. Directory Map

```text
cultural-alignment-layer/
├── .antigravity/                   # Agent operational directives
│   └── rules.md
├── docs/                           # Domain knowledge base
│   ├── ARCHITECTURE.md             # Pipeline specifications & flow diagrams
│   ├── EVALUATION_RUBRIC.md        # Alignment dimensions & JSON schemas
│   └── TAXONOMY.md                 # Tamil socio-cultural dimensions & Hofstede/UNESCO mapping
├── data/
│   ├── raw/                        # Hugging Face downloaded chunks
│   └── processed/                  # Filtered JSONL / CSV benchmarks
├── vector_store/                   # Local ChromaDB persistent index
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central settings, environment variables, thresholds
│   ├── retriever/                  # Cultural RAG
│   │   ├── __init__.py
│   │   ├── ingest.py               # Batch/streaming ingestion from Hugging Face
│   │   └── search.py               # ChromaDB query & similarity search
│   ├── pipeline/                   # Feedback-guided alignment loop
│   │   ├── __init__.py
│   │   ├── generator.py            # Cultural candidate generation
│   │   ├── evaluator.py            # Multi-aspect cultural judge
│   │   └── interceptor.py          # Decision engine & rewriting loop
│   └── utils/
│       ├── __init__.py
│       └── llm_clients.py          # Unified LiteLLM wrapper with rate-limit retries
├── tests/
│   ├── __init__.py
│   └── test_alignment_loop.py      # Unit & integration tests
├── requirements.txt
├── .env.example
├── README.md
├── AGENTS.md                       # (This file) Memory checkpoint for AI agents
└── CLAUDE.md                       # Claude developer guide
```

---

## 3. Critical Constraints & Operating Rules

1. **Free-Tier API Economy**:
   - Primary Generator: `groq/openai/gpt-oss-120b` or `groq/mistral-saba-24b` (Indic-specialized).
   - Primary Judge: `gemini/gemini-1.5-flash` via Google AI Studio or OpenRouter free tiers.
   - Never assume unbounded token limits or unmetered rates. Implement exponential backoff for HTTP 429.
2. **Deterministic Evaluation Contract**:
   - The Judge output MUST always parse into `CulturalEvaluationResult` (Pydantic model in `src/pipeline/evaluator.py`).
   - Fields:
     - `context_understanding`: Float [0.0, 1.0]
     - `social_appropriateness`: Float [0.0, 1.0]
     - `regional_relevance`: Float [0.0, 1.0]
     - `stereotype_risk_flag`: Boolean
     - `overall_score`: Float [0.0, 1.0]
     - `critique`: String (actionable guidance for rewrite)
     - `strengths`: List[str]
     - `weaknesses`: List[str]
3. **Local Vector Storage Only**:
   - No paid cloud vector DBs (Pinecone, Weaviate Cloud, etc.).
   - All embeddings must run locally through `sentence-transformers`.
4. **Cultural Grounding**:
   - Refer to `docs/TAXONOMY.md` for honorifics (e.g., *vaanga/ponga*, *thambi*, *akka*, *aiyya*), kinship relations, festival etiquette, and anti-bias guidelines.

---

## 4. Key Commands

```bash
# Activate Virtual Environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt

# Run Unit & Offline Mock Tests
pytest -v tests/

# Ingest Sample Cultural Data into Local ChromaDB
python -m src.retriever.ingest --sample

# Run the Cultural Alignment Interceptor CLI
python -m src.pipeline.interceptor --query "How should I invite my elder relatives for Pongal festival?"
```

---

## 5. Development Conventions

- **Formatting & Typing**: Modern Python 3.10+ typing (`list[str]`, `dict[str, Any]`, `Optional[T]`).
- **Imports**: Absolute imports from `src.` (e.g., `from src.config import settings`).
- **Environment Management**: Load from `.env` via `pydantic-settings` or `python-dotenv`.
- **Logs & CLI Output**: Use `rich.console.Console` for readable, formatted debug prints.
