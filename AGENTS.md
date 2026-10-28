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
├── backend/                        # Python FastAPI backend & Cultural Alignment Engine
│   ├── .env                        # Backend environment configuration & API keys
│   ├── .env.example                # Backend environment template
│   ├── requirements.txt            # Python dependencies
│   ├── data/
│   │   ├── datasets/               # Structured Tamil tradition datasets
│   │   ├── raw/                    # Raw corpora
│   │   └── processed/              # Filtered benchmarks
│   ├── vector_store/               # Local ChromaDB persistent index
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI mediation API server
│   │   ├── config.py               # Settings, environment paths, thresholds
│   │   ├── retriever/              # Cultural RAG & vector store search
│   │   ├── pipeline/               # Generator, evaluator, interceptor, guardrails
│   │   └── utils/                  # LiteLLM client wrapper with key rotation
│   └── tests/
│       ├── __init__.py
│       └── test_alignment_loop.py  # Unit & integration tests
├── frontend/                       # Vite + React + TypeScript web app & Kiosk mode
│   ├── .env                        # Frontend environment variables
│   ├── .env.example                # Frontend environment template
│   ├── package.json
│   └── src/                        # React UI, KioskAvatar, Web Speech STT/TTS
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

# Install Backend Dependencies
pip install -r backend/requirements.txt

# Run Unit & Offline Mock Tests (from root or backend directory)
pytest -v backend/tests/
# or: cd backend && pytest -v tests/

# Ingest Sample Cultural Data into Local ChromaDB
cd backend && python -m src.retriever.ingest --sample

# Run FastAPI Backend Server
cd backend && python -m uvicorn src.main:app --host 127.0.0.1 --port 8000

# Run Frontend Vite Server
cd frontend && npm run dev
```

---

## 5. Development Conventions

- **Formatting & Typing**: Modern Python 3.10+ typing (`list[str]`, `dict[str, Any]`, `Optional[T]`).
- **Imports**: Absolute imports from `src.` (e.g., `from src.config import settings`).
- **Environment Management**: Load from `.env` via `pydantic-settings` or `python-dotenv`.
- **Logs & CLI Output**: Use `rich.console.Console` for readable, formatted debug prints.
