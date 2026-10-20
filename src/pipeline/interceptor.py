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


class CulturalProfile(BaseModel):
    """Typed sociolinguistic profile produced before cultural retrieval."""

    detected_culture: str = "Neutral / Unresolved"
    region: str = "South Asia"
    slang_tokens_found: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    meaning_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dialect_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    communication_style: str = "Neutral, context-seeking"
    inferred_social_norms: list[str] = Field(default_factory=list)


class StyleProfiler:
    """Fast, explainable lexical profiler for Tamil, Tanglish, and Hinglish."""

    LEXICONS = {
        "Tamil (Tamil Nadu / Chennai Slang)": (
            "machi", "macha", "mapla", "naina", "thambi", "akka", "anna", "pa", "da", "di",
            "vanakkam", "vaanga", "ponga", "sollunga", "eppadi", "irukinga", "romba", "nalla",
            "konjam", "ippo", "enna", "yaaru", "saaptiya", "nanba", "saptiya", "sowkiyama", "la", "enga", "yenga",
            "oorla", "kulir", "kool", "uthuranga", "adhuku", "yen", "epdi", "kupudradhu", "koopidradhu",
            "pesura", "solra", "irukku", "maadhiri", "paathirukkinga", "seyyavum", "ungalukku",
        ),
        "Hindi (Hinglish)": (
            "arey", "arre", "bhai", "behen", "aap", "aapka", "aapki", "aapko", "kaise", "kaisa",
            "kaisi", "hai", "hain", "haan", "kya", "kyun", "kyon", "main", "mujhe", "mera", "meri",
            "mere", "tum", "tumhara", "tumhari", "tujhe", "tera", "teri", "nahi", "nahin", "bahut",
            "bohot", "achha", "accha", "theek", "thik", "yaar", "dost", "batao", "bolo", "chalo",
            "phir", "abhi", "ghar", "wala", "wali", "namaste", "shukriya", "ji",
        ),
    }

    def profile(self, text: str) -> CulturalProfile:
        lowered = text.lower()
        scores = {culture: [token for token in words if re.search(rf"\b{re.escape(token)}\b", lowered)] for culture, words in self.LEXICONS.items()}
        culture, tokens = max(scores.items(), key=lambda item: len(item[1]))
        tamil_script = bool(re.search(r"[\u0B80-\u0BFF]", text))
        if tamil_script:
            culture, tokens = "Tamil (Tamil script / region unresolved)", []
        if not tokens and not tamil_script:
            return CulturalProfile()
        common_tamil_markers = {"thambi", "sowkiyama", "nalla", "enna", "epdi", "enga", "yenga"}
        strong_markers = set(tokens) - common_tamil_markers
        meaning_confidence = min(0.98, 0.72 + (0.05 * len(set(tokens))))
        dialect_confidence = min(0.92, 0.35 + (0.12 * len(strong_markers))) if culture.startswith("Tamil") else min(0.9, 0.5 + (0.08 * len(set(tokens))))
        is_hindi = culture.startswith("Hindi")
        norms = ["Preserve the user's code-mixed script and register", "Ask before assuming a practice is universal"]
        if is_hindi:
            norms.append("Use respectful aap/ji forms when the relationship is unclear")
        else:
            norms.extend(["Use neenga/vaanga/ponga for elders or unfamiliar people", "Keep Tamil kinship terms relationship-specific"])
        strong_tamil_region = {"chennai", "madurai", "coimbatore", "kongu", "tirunelveli", "nagercoil", "kanyakumari", "jaffna", "batticaloa"}
        requested_region = next((word.title() for word in strong_tamil_region if word in lowered), None)
        if culture.startswith("Tamil"):
            culture = f"Tamil ({requested_region} / requested style)" if requested_region else "Tamil (Romanized / region unresolved)"
        return CulturalProfile(
            detected_culture=culture,
            region=(f"{requested_region} / Tamil-speaking region" if requested_region else "Tamil-speaking region") if not is_hindi else "North India / South Asia",
            slang_tokens_found=sorted(set(tokens)),
            confidence=dialect_confidence,
            meaning_confidence=meaning_confidence,
            dialect_confidence=dialect_confidence,
            communication_style="Colloquial, code-mixed, high-context" if tokens else "Neutral, context-seeking",
            inferred_social_norms=norms,
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
