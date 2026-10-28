"""Unified LiteLLM and Free-Tier Model Client Wrapper with Multi-Key Rotation."""

import os
import re
import time
import logging
from typing import Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

_backend_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _backend_env.exists():
    load_dotenv(_backend_env)
else:
    load_dotenv()
logger = logging.getLogger("cultural_ai.llm")
console = Console()


class KeyRotator:
    """Manages multi-key pools and automatic rotation upon rate limit / quota exhaustion."""

    def __init__(self, provider_prefix: str, env_vars: list[str]) -> None:
        self.provider_prefix = provider_prefix
        self.env_vars = env_vars
        self.keys = self._discover_keys()
        self.current_idx = 0

    def _discover_keys(self) -> list[str]:
        keys: list[str] = []
        for var in self.env_vars:
            val = os.getenv(var, "")
            if val:
                for k in re.split(r"[,;\n\s]+", val):
                    cleaned = k.strip().strip('"').strip("'")
                    if cleaned and cleaned not in keys and not cleaned.startswith("gsk_...") and not cleaned.startswith("AIza..."):
                        keys.append(cleaned)
        # Check numbered variants: e.g. GROQ_API_KEY_1, GROQ_API_KEY_2
        for i in range(1, 10):
            val = os.getenv(f"{self.provider_prefix}_{i}", "")
            if val:
                cleaned = val.strip().strip('"').strip("'")
                if cleaned and cleaned not in keys and not cleaned.startswith("gsk_..."):
                    keys.append(cleaned)
        return keys

    def get_current_key(self) -> Optional[str]:
        if not self.keys:
            return None
        return self.keys[self.current_idx % len(self.keys)]

    def rotate(self) -> Optional[str]:
        if len(self.keys) <= 1:
            return self.get_current_key()
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        next_key = self.get_current_key()
        masked = f"{next_key[:6]}...{next_key[-4:]}" if next_key and len(next_key) > 10 else "***"
        console.print(f"[cyan]Rotating {self.provider_prefix} to key [{self.current_idx + 1}/{len(self.keys)}]: {masked}[/cyan]")
        return next_key


groq_rotator = KeyRotator("GROQ_API_KEY", ["GROQ_API_KEY", "GROQ_API_KEYS"])
gemini_rotator = KeyRotator("GEMINI_API_KEY", ["GEMINI_API_KEY", "GEMINI_API_KEYS"])
openrouter_rotator = KeyRotator("OPENROUTER_API_KEY", ["OPENROUTER_API_KEY", "OPENROUTER_API_KEYS"])


def _apply_current_key(model: str) -> None:
    model_lower = model.lower()
    if "groq" in model_lower:
        k = groq_rotator.get_current_key()
        if k:
            os.environ["GROQ_API_KEY"] = k
    elif "gemini" in model_lower:
        k = gemini_rotator.get_current_key()
        if k:
            os.environ["GEMINI_API_KEY"] = k
    elif "openrouter" in model_lower:
        k = openrouter_rotator.get_current_key()
        if k:
            os.environ["OPENROUTER_API_KEY"] = k


def generate_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.6,
    json_mode: bool = False,
    max_retries: int = 4,
    backoff_factor: float = 2.0,
) -> str:
    """Unified wrapper handling free-tier models via LiteLLM with rate-limit retries, key rotation, and model fallback."""
    response_format = {"type": "json_object"} if json_mode else None

    # Fallback model candidate map if TPM bucket is exhausted on current model
    fallbacks = {
        "groq/openai/gpt-oss-120b": "groq/mistral-saba-24b",
        "openai/gpt-oss-120b": "groq/mistral-saba-24b",
    }

    current_model = model
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            _apply_current_key(current_model)
            from litellm import completion

            response = completion(
                model=current_model,
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
                or "tokens per minute" in err_msg
            )
            if is_rate_limit:
                # 1. Attempt instant key rotation first if multiple keys available
                model_lower = current_model.lower()
                rotated = False
                if "groq" in model_lower and len(groq_rotator.keys) > 1:
                    new_key = groq_rotator.rotate()
                    os.environ["GROQ_API_KEY"] = new_key or ""
                    rotated = True
                elif "gemini" in model_lower and len(gemini_rotator.keys) > 1:
                    new_key = gemini_rotator.rotate()
                    os.environ["GEMINI_API_KEY"] = new_key or ""
                    rotated = True

                if rotated:
                    time.sleep(0.5)
                    continue

                # 2. If single key or all keys exhausted for this model, try fallback model with separate TPM quota
                if attempt >= 1 and current_model in fallbacks:
                    alt_model = fallbacks[current_model]
                    console.print(
                        f"[yellow]TPM limit on {current_model}. Switching to alternative Indic model {alt_model}...[/yellow]"
                    )
                    current_model = alt_model
                    time.sleep(1.0)
                    continue

                # 3. Check if server specified exact wait time
                wait_match = re.search(r"try again in ([\d\.]+)s", str(exc), re.IGNORECASE)
                if wait_match:
                    wait_time = min(float(wait_match.group(1)) + 1.0, 15.0)
                else:
                    wait_time = backoff_factor ** (attempt + 1)

                if attempt < max_retries - 1:
                    console.print(
                        f"[yellow]Rate limit reached on {current_model}. Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...[/yellow]"
                    )
                    time.sleep(wait_time)
                    continue

            logger.error("LLM completion failed for model %s: %s", current_model, exc)
            raise

    raise RuntimeError(f"Exhausted retries ({max_retries}) for {current_model}: {last_error}")
