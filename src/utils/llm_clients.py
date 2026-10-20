"""Unified LiteLLM and Free-Tier Model Client Wrapper."""

import os
import time
import logging
from typing import Any, Optional
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
logger = logging.getLogger("cultural_ai.llm")
console = Console()


def generate_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.6,
    json_mode: bool = False,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> str:
    """Unified wrapper handling free-tier models via LiteLLM with rate-limit retries."""
    response_format = {"type": "json_object"} if json_mode else None

    # Inject keys into litellm environment if present
    for env_var in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
        val = os.getenv(env_var)
        if val and env_var not in os.environ:
            os.environ[env_var] = val

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            # Import on first request so API startup never performs LiteLLM's
            # optional remote model-cost-map lookup or network initialization.
            from litellm import completion

            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            err_msg = str(exc).lower()
            is_rate_limit = (
                "rate limit" in err_msg
                or "429" in err_msg
                or "quota" in err_msg
                or "resource_exhausted" in err_msg
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = backoff_factor ** (attempt + 1)
                console.print(
                    f"[yellow]Rate limit reached on {model}. Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...[/yellow]"
                )
                time.sleep(wait_time)
                continue
            logger.error("LLM completion failed for model %s: %s", model, exc)
            raise

    raise RuntimeError(f"Exhausted retries ({max_retries}) for {model}: {last_error}")
