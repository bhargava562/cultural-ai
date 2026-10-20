"""FastAPI entrypoint for the Cultural Alignment Mediator."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.evaluator import CulturalEvaluationResult
from src.pipeline.generator import MediatedResponseGenerator
from src.pipeline.interceptor import CulturalClassifier
from src.retriever.search import CulturalRetriever

app = FastAPI(title="Cultural Alignment Mediator", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
classifier = CulturalClassifier()
retriever = CulturalRetriever()
generator = MediatedResponseGenerator()


def normalize_provider_model(model: Optional[str]) -> Optional[str]:
    """Route the Groq-hosted OpenAI-compatible model through Groq in LiteLLM."""
    if model == "openai/gpt-oss-120b":
        return "groq/openai/gpt-oss-120b"
    if model and model.startswith("groq/"):
        return model
    return model


class MediateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    model: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)


class MediateResponse(BaseModel):
    cultural_profile: dict[str, Any]
    retrieved_practices: list[dict[str, Any]]
    mediated_response: str
    evaluation_scores: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/mediate", response_model=MediateResponse)
def mediate(request: MediateRequest) -> MediateResponse:
    profile = classifier.classify(request.query)
    practices = retriever.search(request.query, profile=profile, top_k=request.top_k)
    response = generator.generate(request.query, profile, practices, model=normalize_provider_model(request.model))
    evaluation = CulturalEvaluationResult(
        context_understanding=0.75 if practices else 0.55,
        social_appropriateness=0.85 if profile.detected_culture != "Neutral / Unresolved" else 0.65,
        regional_relevance=0.85 if practices else 0.55,
        domain_risk_detected=False,
        alignment_score=CulturalEvaluationResult.compute_cas(0.75 if practices else 0.55, 0.85 if profile.detected_culture != "Neutral / Unresolved" else 0.65, 0.85 if practices else 0.55, False),
        verdict="PASS" if practices else "REVIEW",
        critique="Retrieved practice grounding is present." if practices else "No local practice was retrieved; verify regional variation.",
    )
    return MediateResponse(cultural_profile=profile.model_dump(), retrieved_practices=practices, mediated_response=response, evaluation_scores={"CAS": evaluation.alignment_score, **evaluation.model_dump()})
