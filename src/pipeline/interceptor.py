"""Feedback-guided Interceptor Loop for Cultural Alignment."""

import argparse
import sys
import logging
import re
from typing import Any, Optional
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Force UTF-8 on Windows consoles to render Tamil scripts properly
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import settings
from src.retriever.search import query_cultural_context, format_cultural_context
from src.pipeline.generator import generate_candidate
from src.pipeline.evaluator import evaluate_candidate, CulturalEvaluationResult

console = Console(highlight=False)
logger = logging.getLogger("cultural_ai.interceptor")


from src.pipeline.slang_guardrail import TamilSlangGuardrail, CulturalContextPacket

class CulturalProfile(BaseModel):
    """Typed sociolinguistic profile produced before cultural retrieval."""

    detected_culture: str = "Neutral / Unresolved"
    region: str = "South Asia"
    sub_region: str = "Pan-Tamil Nadu / General"
    slang_tokens_found: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    meaning_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dialect_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    communication_style: str = "Neutral, context-seeking"
    inferred_social_norms: list[str] = Field(default_factory=list)
    matched_tradition_name: Optional[str] = None
    little_tradition_reality: Optional[str] = None
    expected_explanation_guideline: Optional[str] = None


class StyleProfiler:
    """Fast, explainable lexical profiler for Tamil, Tanglish, and Hinglish."""

    def __init__(self) -> None:
        self.guardrail = TamilSlangGuardrail()

    def profile(self, text: str) -> CulturalProfile:
        packet = self.guardrail.analyze_query(text)
        lowered = text.lower()
        tamil_script = bool(re.search(r"[\u0B80-\u0BFF]", text))

        # Check for Hinglish
        hinglish_words = [
            "arey", "arre", "bhai", "behen", "aap", "aapka", "aapki", "aapko", "kaise", "kaisa",
            "kaisi", "hai", "hain", "haan", "kya", "kyun", "kyon", "main", "mujhe", "mera", "meri",
            "mere", "tum", "tumhara", "tumhari", "tujhe", "tera", "teri", "nahi", "nahin", "bahut",
            "bohot", "achha", "accha", "theek", "thik", "yaar", "dost", "batao", "bolo", "chalo",
            "phir", "abhi", "ghar", "wala", "wali", "namaste", "shukriya", "ji",
        ]
        matched_hindi = [w for w in hinglish_words if re.search(rf"\b{re.escape(w)}\b", lowered)]

        if matched_hindi and len(matched_hindi) > len(packet.detected_slang_markers):
            meaning_conf = min(0.95, 0.70 + (0.05 * len(matched_hindi)))
            dialect_conf = min(0.90, 0.50 + (0.08 * len(matched_hindi)))
            return CulturalProfile(
                detected_culture="Hindi (Hinglish)",
                region="North India / South Asia",
                sub_region="North India",
                slang_tokens_found=matched_hindi,
                confidence=dialect_conf,
                meaning_confidence=meaning_conf,
                dialect_confidence=dialect_conf,
                communication_style="Colloquial, code-mixed, high-context",
                inferred_social_norms=[
                    "Preserve the user's code-mixed script and register",
                    "Use respectful aap/ji forms when the relationship is unclear",
                ],
            )

        if not packet.detected_slang_markers and not tamil_script:
            return CulturalProfile()

        # Build Tamil/Tanglish profile
        detected_culture = (
            "Tamil (Tamil script)"
            if tamil_script
            else f"Tamil ({packet.detected_region} Tanglish)"
        )

        norms = [
            "Preserve the user's code-mixed script and register",
            "Always explain the cultural tradition and validate why it is normative",
            "Use neenga/vaanga/ponga for elders or unfamiliar people",
            "Keep Tamil kinship terms relationship-specific",
        ]

        trad_name = packet.matched_traditions[0].tradition_name if packet.matched_traditions else None
        reality = packet.matched_traditions[0].little_tradition_reality if packet.matched_traditions else None
        guideline = packet.matched_traditions[0].expected_explanation_guideline if packet.matched_traditions else None

        meaning_conf = min(0.98, 0.75 + (0.04 * len(packet.detected_slang_markers)))
        dialect_conf = (
            min(0.95, 0.60 + (0.08 * len(packet.detected_slang_markers)))
            if packet.detected_region != "Pan-Tamil Nadu / General"
            else 0.50
        )

        return CulturalProfile(
            detected_culture=detected_culture,
            region=f"{packet.detected_region} / Tamil Nadu",
            sub_region=packet.detected_region,
            slang_tokens_found=packet.detected_slang_markers,
            confidence=dialect_conf,
            meaning_confidence=meaning_conf,
            dialect_confidence=dialect_conf,
            communication_style="Colloquial, code-mixed, culturally grounded",
            inferred_social_norms=norms,
            matched_tradition_name=trad_name,
            little_tradition_reality=reality,
            expected_explanation_guideline=guideline,
        )


class CulturalClassifier(StyleProfiler):
    """Compatibility name for the mediator's lightweight classifier."""

    def classify(self, text: str) -> CulturalProfile:
        return self.profile(text)


style_profiler = CulturalClassifier()


class InterceptorOutput(BaseModel):
    """Complete diagnostic output of the cultural alignment interceptor loop."""

    query: str
    cultural_profile: CulturalProfile = Field(default_factory=CulturalProfile)
    final_response: str
    initial_candidate: str
    passed_threshold: bool
    iterations_used: int
    final_evaluation: CulturalEvaluationResult
    trajectory: list[dict[str, Any]]


def run_cultural_alignment_pipeline(
    query: str,
    top_k_context: int = 3,
    generator_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    max_loops: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> InterceptorOutput:
    """Execute the full two-stage Cultural RAG -> Generation -> Judge -> Interceptor loop."""
    threshold = score_threshold if score_threshold is not None else settings.score_threshold
    loops_allowed = max_loops if max_loops is not None else settings.max_feedback_loops
    profile = style_profiler.profile(query)

    console.print(f"[bold cyan]1. Retrieving cultural context for query:[/bold cyan] '{query}'")
    try:
        snippets = query_cultural_context(query, n_results=top_k_context)
        context_str = format_cultural_context(snippets)
    except Exception as exc:
        logger.warning("Retriever encountered error or empty index: %s", exc)
        context_str = "No local cultural archive retrieved."

    console.print("[bold cyan]2. Generating initial candidate...[/bold cyan]")
    current_candidate = generate_candidate(
        query=query,
        cultural_context=context_str,
        critique_feedback=None,
        model=generator_model,
    )
    initial_candidate = current_candidate

    trajectory: list[dict[str, Any]] = []
    current_eval: Optional[CulturalEvaluationResult] = None
    iteration = 0

    while True:
        iteration += 1
        console.print(f"[bold cyan]3. Evaluating candidate (Iteration {iteration})...[/bold cyan]")
        current_eval = evaluate_candidate(
            query=query,
            candidate=current_candidate,
            cultural_context=context_str,
            model=judge_model,
        )

        trajectory.append({
            "iteration": iteration,
            "candidate": current_candidate,
            "evaluation": current_eval.model_dump(),
        })

        is_acceptable = (
            current_eval.overall_score >= threshold
            and not current_eval.stereotype_risk_flag
        )

        if is_acceptable:
            console.print(
                f"[bold green]Candidate PASSED cultural alignment threshold ({current_eval.overall_score:.2f} >= {threshold:.2f}).[/bold green]"
            )
            break

        if iteration > loops_allowed:
            console.print(
                f"[bold yellow]Max feedback loops ({loops_allowed}) reached. Returning current candidate.[/bold yellow]"
            )
            break

        console.print(
            f"[bold magenta]Score {current_eval.overall_score:.2f} < {threshold:.2f} (or risk flagged). Triggering Feedback Loop {iteration}...[/bold magenta]"
        )
        console.print(f"[dim]Critique: {current_eval.critique}[/dim]")

        # Regenerate candidate guided by critique
        current_candidate = generate_candidate(
            query=query,
            cultural_context=context_str,
            critique_feedback=current_eval.critique,
            model=generator_model,
        )

    passed = (
        current_eval.overall_score >= threshold
        and not current_eval.stereotype_risk_flag
    )

    return InterceptorOutput(
        query=query,
        cultural_profile=profile,
        final_response=current_candidate,
        initial_candidate=initial_candidate,
        passed_threshold=passed,
        iterations_used=iteration,
        final_evaluation=current_eval,
        trajectory=trajectory,
    )


def print_diagnostic_report(output: InterceptorOutput) -> None:
    """Print a visually rich diagnostic report."""
    table = Table(title="Cultural Alignment Score Card", show_header=True, header_style="bold magenta")
    table.add_column("Dimension", style="dim", width=26)
    table.add_column("Score / Status", justify="right")

    fe = output.final_evaluation
    table.add_row("Context Understanding (30%)", f"{fe.context_understanding:.2f}")
    table.add_row("Social Appropriateness (35%)", f"{fe.social_appropriateness:.2f}")
    table.add_row("Regional Relevance (35%)", f"{fe.regional_relevance:.2f}")
    risk_style = "bold red" if fe.domain_risk_detected else "bold green"
    table.add_row("Domain Risk Detected", f"[{risk_style}]{fe.domain_risk_detected}[/{risk_style}]")
    score_style = "bold green" if output.passed_threshold else "bold yellow"
    table.add_row("Cultural Alignment Score (CAS)", f"[{score_style}]{fe.alignment_score:.2f}[/{score_style}]")
    table.add_row("Verdict", f"[{score_style}]{fe.verdict}[/{score_style}]")
    table.add_row("Iterations Used", str(output.iterations_used))
    table.add_row("Status", "[bold green]PASSED[/bold green]" if output.passed_threshold else "[bold red]FAILED[/bold red]")

    console.print(table)
    console.print(Panel(output.final_response, title="Final Response Output", border_style="green"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cultural Alignment Interceptor on a query.")
    parser.add_argument("--query", type=str, required=True, help="User query to process")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k context chunks")
    args = parser.parse_args()

    result = run_cultural_alignment_pipeline(query=args.query, top_k_context=args.top_k)
    print_diagnostic_report(result)


if __name__ == "__main__":
    main()
