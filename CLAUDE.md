# CLAUDE.md - Developer & Agent Guide

This document contains key instructions, architecture details, and workflow commands for developing and maintaining the **Cultural Alignment Layer** project.

---

## 1. Project Overview

The **Cultural Alignment Layer** is a lightweight, non-invasive middleware designed for regional LLM queries (focusing on Tamil, Tanglish, and regional English). It uses a two-stage feedback-guided pipeline:
- **Cultural Context Retrieval**: ChromaDB with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Candidate Generation**: Prompt-steered anthropological persona via free LLM APIs (Groq, Gemini Flash, OpenRouter).
- **Cultural Alignment Judge**: Evaluates 4 dimensions and outputs structured JSON conforming to `docs/EVALUATION_RUBRIC.md`.
- **Feedback-Guided Interceptor**: Automatically rewrites responses failing threshold (`overall_score < 0.75` or `stereotype_risk_flag == True`).

---

## 2. Common Commands

### Environment Setup
```bash
# Create venv
python -m venv .venv

# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate venv (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration
cp .env.example .env
```

### Running Tests
```bash
# Run all tests
pytest

# Run tests with verbose output and coverage
pytest -v -s tests/test_alignment_loop.py
```

### Ingestion & Search
```bash
# Ingest curated sample data into ChromaDB
python -m src.retriever.ingest --sample

# Ingest from Hugging Face dataset (batched/streaming)
python -m src.retriever.ingest --dataset wickkiey/tamil-wikipedia-chunked --limit 1000

# Test retrieval search
python -m src.retriever.search --query "பொங்கல் திருவிழா மரபுகள்"
```

### Pipeline Execution
```bash
# Test the full interceptor loop via CLI
python -m src.pipeline.interceptor --query "Suggest a polite greeting for an elder uncle in Chennai"
```

---

## 3. Architecture & Code Style Guidelines

- **Zero-Cost API Economy**:
  - Always use `src.utils.llm_clients.generate_completion` to ensure automatic retries, backoff, and JSON mode enforcement.
  - Rely on Groq (`groq/llama-3.3-70b-versatile`) for high-speed generation and Gemini Flash (`gemini/gemini-1.5-flash`) for multi-faceted evaluation.
- **Contract Adherence**:
  - Pydantic models in `src.pipeline.evaluator.py` must remain synchronized with `docs/EVALUATION_RUBRIC.md`.
  - Avoid modifying evaluation score bounds (0.0 to 1.0) or threshold (`0.75`).
- **Data Grounding**:
  - Reference `docs/TAXONOMY.md` when refining system prompts to preserve cultural authenticity, honorific systems, and kinship terminology.
- **Python Conventions**:
  - Strict typing everywhere.
  - Error handling: Never raise unhandled HTTP exceptions; catch rate limits and fallback smoothly.
  - Use `rich` for CLI output formatting.
