"""Feedback-guided Interceptor Loop for Cultural Alignment."""

import argparse
import sys
import logging
from typing import Any, Optional
from pydantic import BaseModel
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


class InterceptorOutput(BaseModel):
    """Complete diagnostic output of the cultural alignment interceptor loop."""

    query: str
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
    table.add_row("Context Understanding", f"{fe.context_understanding:.2f}")
    table.add_row("Social Appropriateness", f"{fe.social_appropriateness:.2f}")
    table.add_row("Regional Relevance", f"{fe.regional_relevance:.2f}")
    risk_style = "bold red" if fe.stereotype_risk_flag else "bold green"
    table.add_row("Stereotype / Risk Flag", f"[{risk_style}]{fe.stereotype_risk_flag}[/{risk_style}]")
    score_style = "bold green" if output.passed_threshold else "bold yellow"
    table.add_row("Overall Score", f"[{score_style}]{fe.overall_score:.2f}[/{score_style}]")
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
