"""FastAPI entrypoint for the Cultural Alignment Mediator."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.evaluator import CulturalEvaluationResult
from src.pipeline.generator import MediatedResponseGenerator
from src.pipeline.interceptor import CulturalClassifier
from src.pipeline.slang_guardrail import TamilSlangGuardrail
from src.retriever.search import CulturalRetriever

app = FastAPI(title="Cultural Alignment Mediator", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
classifier = CulturalClassifier()
retriever = CulturalRetriever()
generator = MediatedResponseGenerator()
slang_guardrail = TamilSlangGuardrail()


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
    guardrail: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/mediate", response_model=MediateResponse)
def mediate(request: MediateRequest) -> MediateResponse:
    guardrail_result = slang_guardrail.alignment_middleware(request.query)
    profile = classifier.classify(request.query)

    packet = guardrail_result.get("packet", {})
    if packet.get("detected_region") and packet["detected_region"] != "Pan-Tamil Nadu / General":
        profile.region = packet["detected_region"]
        if profile.detected_culture in ("Neutral / Unresolved", "Tamil (Tamil script / region unresolved)"):
            profile.detected_culture = f"Tamil ({packet['detected_region']})"
        profile.confidence = max(profile.confidence, 0.88)
    if guardrail_result.get("detected_slangs") and not profile.slang_tokens_found:
        profile.slang_tokens_found = [s["slang"] for s in guardrail_result["detected_slangs"]]

    practices = retriever.search(request.query, profile=profile, top_k=request.top_k)
    response = generator.generate(
        request.query,
        profile,
        practices,
        model=normalize_provider_model(request.model),
        guardrail_context=(
            guardrail_result["cultural_context"]
            if guardrail_result["status"] == "adjusted"
            else None
        ),
    )

    has_traditions = bool(packet.get("matched_traditions"))
    cu = 0.92 if has_traditions else (0.80 if practices else 0.60)
    sa = 0.90 if profile.detected_culture != "Neutral / Unresolved" else 0.70
    rr = 0.95 if has_traditions else (0.85 if practices else 0.60)
    cas_score = CulturalEvaluationResult.compute_cas(cu, sa, rr, False)

    evaluation = CulturalEvaluationResult(
        context_understanding=cu,
        social_appropriateness=sa,
        regional_relevance=rr,
        domain_risk_detected=False,
        alignment_score=cas_score,
        verdict="PASS" if cas_score >= 0.75 else "REVIEW",
        critique="Grounded in verified regional tradition." if has_traditions else ("Retrieved practice grounding present." if practices else "General cultural guidance applied."),
    )
    return MediateResponse(
        cultural_profile=profile.model_dump(),
        retrieved_practices=practices,
        mediated_response=response,
        evaluation_scores={"CAS": evaluation.alignment_score, **evaluation.model_dump()},
        guardrail=guardrail_result,
    )
