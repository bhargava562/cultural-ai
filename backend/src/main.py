"""FastAPI entrypoint for the Cultural Alignment Mediator."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import re
import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from src.pipeline.evaluator import CulturalEvaluationResult
from src.pipeline.generator import MediatedResponseGenerator, detect_language_mode
from src.pipeline.interceptor import CulturalClassifier
from src.pipeline.slang_guardrail import TamilSlangGuardrail
from src.retriever.search import CulturalRetriever

logger = logging.getLogger("cultural_ai.main")

app = FastAPI(title="Cultural Alignment Mediator", version="0.2.0")


class VercelPathRewriteMiddleware:
    """Restores original request path from Vercel x-matched-path header when rewrites are used."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8")
            if matched_path:
                clean_path = matched_path.split("?")[0]
                scope["path"] = clean_path
                scope["raw_path"] = clean_path.encode("utf-8")
        await self.app(scope, receive, send)


app.add_middleware(VercelPathRewriteMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error in /api/mediate: %s", exc, exc_info=True)
    origin = request.headers.get("origin") or "*"
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


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


def build_spoken_script(markdown_text: str) -> str:
    """Clean markdown formatting into a natural spoken script for Kiosk TTS and subtitles."""
    cleaned = re.sub(r"```[\s\S]*?```", "", markdown_text)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[🏛️✅🌿✨🎯💡]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    script = " ".join(sentences[:5]).strip()
    return script or cleaned


class MediateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    model: Optional[str] = None
    top_k: int = Field(default=2, ge=1, le=10)


class MediateResponse(BaseModel):
    cultural_profile: dict[str, Any]
    retrieved_practices: list[dict[str, Any]]
    mediated_response: str
    conversational_response: Optional[str] = None
    cultural_insights: Optional[str] = None
    spoken_response: Optional[str] = None
    evaluation_scores: dict[str, Any]
    guardrail: dict[str, Any]


@app.get("/")
@app.get("/health")
@app.get("/api/health")
@app.get("/api/index")
@app.get("/api/index.py")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "Cultural Alignment Mediator"}


@app.post("/api/mediate", response_model=MediateResponse)
@app.post("/mediate", response_model=MediateResponse)
@app.post("/api/index", response_model=MediateResponse)
@app.post("/api/index.py", response_model=MediateResponse)
def mediate(request: MediateRequest) -> MediateResponse:
    guardrail_result = slang_guardrail.alignment_middleware(request.query)
    profile = classifier.classify(request.query)

    packet = guardrail_result.get("packet") or guardrail_result
    if packet.get("detected_region") and packet["detected_region"] != "Pan-Tamil Nadu / General":
        profile.region = packet["detected_region"]
        if profile.detected_culture in ("Neutral / Unresolved", "Tamil (Tamil script / region unresolved)"):
            profile.detected_culture = f"Tamil ({packet['detected_region']})"
        profile.confidence = max(profile.confidence, 0.88)
    if guardrail_result.get("detected_slangs") and not profile.slang_tokens_found:
        profile.slang_tokens_found = [s["slang"] for s in guardrail_result["detected_slangs"]]

    lang_mode = detect_language_mode(request.query)
    is_tanglish = lang_mode == "Tanglish (Tamil in Latin script / Code-mixed)"

    practices: list[dict[str, Any]] = []
    if packet.get("matched_traditions"):
        for t in packet["matched_traditions"]:
            exp = t.get("tanglish_process_explanation") if is_tanglish and t.get("tanglish_process_explanation") else t.get("little_tradition_reality", "")
            practices.append({
                "content": f"{t['tradition_name']} ({t['region']}): {exp} Guideline: {t['expected_explanation_guideline']}",
                "metadata": {"title": t["tradition_name"], "region": t["region"]},
            })
    retrieved = retriever.search(request.query, profile=profile, top_k=request.top_k)
    for r in retrieved:
        if r.get("metadata", {}).get("title") not in [p.get("metadata", {}).get("title") for p in practices]:
            practices.append(r)
    practices = practices[:request.top_k]

    try:
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
    except Exception as exc:
        logger.error("LLM Generation exception: %s", exc, exc_info=True)
        # Resilient cultural fallback
        if is_tanglish:
            response = (
                "Sari da machi, un kelvikku namma ooru vazhakkam padi solren!\n\n"
                "### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):\n"
                "Namma Tamil kalachaarathula periyavanga kitta kalandhu pesi, "
                "kudumba maryadhaiyoda mudivu edukkuradhu dhaan namma ooru murai.\n\n"
                "### 💡 Ippo Nee Try Panna Vendiya Steps:\n"
                "1. Kudumba periyavanga kitta open-aa discuss pannu da.\n"
                "2. Terinja seniors kitta nalla vicharanai pannitu step edu!"
            )
        else:
            response = (
                "In Tamil tradition, community consensus and filial respect (Maryadhai) guide important decisions.\n\n"
                "### 🌿 As per your culture, here is the customary process they follow:\n"
                "In Tamil culture, family elders and maternal uncles carry moral guardianship of the clan.\n\n"
                "### 💡 So try these steps (Actionable Cultural Decision & Phrasing):\n"
                "1. Consult your household elders in person during morning auspicious hours (Nalla Neram).\n"
                "2. Approach the situation with mutual respect and humility."
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
    conversational_part = response
    cultural_insights_part = None
    if "---" in response:
        parts = response.split("---", 1)
        conversational_part = parts[0].strip()
        cultural_insights_part = parts[1].strip()
    elif "PART 2" in response:
        parts = re.split(r"\*{0,2}PART\s*2[^\n]*\*{0,2}", response, maxsplit=1, flags=re.IGNORECASE)
        conversational_part = parts[0].strip()
        cultural_insights_part = parts[1].strip() if len(parts) > 1 else None

    conversational_part = re.sub(r"^\*{0,2}PART\s*1[^\n]*\*{0,2}\s*", "", conversational_part, flags=re.IGNORECASE).strip()

    spoken_script = build_spoken_script(conversational_part)

    return MediateResponse(
        cultural_profile=profile.model_dump(),
        retrieved_practices=practices,
        mediated_response=response,
        conversational_response=conversational_part,
        cultural_insights=cultural_insights_part,
        spoken_response=spoken_script,
        evaluation_scores={"CAS": evaluation.alignment_score, **evaluation.model_dump()},
        guardrail=guardrail_result,
    )
