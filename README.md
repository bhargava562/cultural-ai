<div align="center">

# 🏛️ Cultural Alignment Layer
### *கலாச்சார ஒருங்கிணைப்பு அடுக்கு*

**An open-source, non-invasive middleware evaluating and adapting LLMs for regional nuance, socio-cultural etiquette, and linguistic authenticity across Tamil, Tanglish, and English.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/LiteLLM-Multi--Provider-000000?style=for-the-badge&logo=fastapi&logoColor=white" alt="LiteLLM" />
  <img src="https://img.shields.io/badge/ChromaDB-Local%20Vector%20Store-FF6F61?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Groq-LPU%20Inference-F05A28?style=for-the-badge&logo=speedtest&logoColor=white" alt="Groq" />
  <img src="https://img.shields.io/badge/Gemini%20Flash-Judge-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License" />
</p>

[Architecture](docs/ARCHITECTURE.md) • [Evaluation Rubric](docs/EVALUATION_RUBRIC.md) • [Taxonomy](docs/TAXONOMY.md) • [Agent Directives](AGENTS.md)

</div>

---

## ⚡ Overview

Foundational frontier models frequently misjudge South Asian cultural hierarchies, honorific registers (*Maryadhai*), kinship terminology, and colloquial code-switching (Tanglish).

The **Cultural Alignment Layer** sits as an intelligent middleware between user queries and LLM outputs, executing a **two-stage feedback loop**:
1. **Context Augmentation:** Retrieves local cultural and linguistic archives via ChromaDB.
2. **Anthropological Steering:** Injects cultural personas and dialect awareness into high-speed generation models.
3. **Multi-Aspect Cultural Judge:** Evaluates candidates across 4 rubric dimensions with strict Pydantic JSON contracts.
4. **Iterative Rewriting:** Intercepts low scores or cultural risks, executing actionable feedback loops.

---

## 🔄 Two-Stage Pipeline Flow

```mermaid
flowchart LR
    Q([User Query]) --> DET[1. Language & Register Detector]
    DET --> RAG[(2. Cultural RAG\nChromaDB)]
    RAG --> GEN[3. Candidate Generator\nAnthropological Persona]
    GEN --> CAND[Candidate Output]
    CAND --> JUDGE[4. Cultural Judge\n4-Aspect Rubric]
    JUDGE --> DEC{Score >= 0.75\n& No Risk?}
    DEC -- Yes --> OUT([Approved Response])
    DEC -- No --> FB[Feedback Critique Loop]
    FB --> GEN
```

---

## 🎯 The 4 Alignment Dimensions

| Dimension | Weight | Description |
|---|---|---|
| **Context Understanding** | 25% | Grasp of implicit social subtext and situational expectations. |
| **Social Appropriateness** | 35% | Strict honorific alignment (*vaanga/ponga*) and kinship address (*Mama/Chithi*). |
| **Regional Relevance** | 25% | Idiomatic authenticity, local customs, proverbs (*Pazhamozhi*), and culinary traditions. |
| **Stereotype & Risk Flag** | 15% | Hard blocker preventing casteist tropes, gender stereotypes, or linguistic snobbery. |

*Detailed criteria, scoring levels, and validation schemas: [`docs/EVALUATION_RUBRIC.md`](docs/EVALUATION_RUBRIC.md).*

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/bhargava562/cultural-ai.git
cd cultural-ai

python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Free API Keys in `.env`

```bash
GROQ_API_KEY="gsk_..."
GEMINI_API_KEY="AIza..."
OPENROUTER_API_KEY="sk-or-..."
```

### 3. Ingest Cultural Seed Data

```bash
python -m src.retriever.ingest --sample
```

### 4. Run the Alignment Interceptor

```bash
python -m src.pipeline.interceptor --query "How should I invite my elder maternal uncle (Mama) for lunch politely?"
```

### 5. Run Offline Tests

```bash
pytest -v tests/
```

---

## 📂 Repository Map

```text
cultural-ai/
├── docs/                           # Deep domain documentation
│   ├── ARCHITECTURE.md             # Pipeline specifications & flow diagrams
│   ├── EVALUATION_RUBRIC.md        # Alignment dimensions & JSON schemas
│   └── TAXONOMY.md                 # Tamil socio-cultural taxonomy & Hofstede/UNESCO mapping
├── src/
│   ├── retriever/                  # Local ChromaDB cultural index & search
│   ├── pipeline/                   # Generator, Judge, and Interceptor loops
│   ├── utils/                      # LiteLLM resilient free-tier client wrapper
│   └── config.py                   # Central settings & threshold management
├── tests/                          # Offline test suite (mock LLM loops)
├── AGENTS.md                       # Memory checkpoint for AI agent pair-programming
├── CLAUDE.md                       # Fast commands cheatsheet
└── requirements.txt
```

---

## 📚 Deep Documentation

For full architectural blueprints, taxonomies, and agent prompt guides, refer to:
- 📐 [System Architecture Blueprint](docs/ARCHITECTURE.md)
- 📊 [Evaluation Rubric & Data Contract](docs/EVALUATION_RUBRIC.md)
- 🌺 [Tamil Socio-Cultural Taxonomy](docs/TAXONOMY.md)
- 🤖 [Agent Directives & Vibecoding Rules](AGENTS.md)

---

## 📄 License

Distributed under the [MIT License](LICENSE).
