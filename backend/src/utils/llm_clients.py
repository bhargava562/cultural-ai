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

# Auto-fallback to GROQ_API_KEY_1 if GROQ_API_KEY is not defined
if not os.getenv("GROQ_API_KEY"):
    _fallback_groq = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY_2")
    if _fallback_groq:
        os.environ["GROQ_API_KEY"] = _fallback_groq.strip().strip('"').strip("'")

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
        # 1. Numbered variants take explicit priority order: e.g. GROQ_API_KEY_1, GROQ_API_KEY_2
        for i in range(1, 10):
            val = os.getenv(f"{self.provider_prefix}_{i}", "")
            if val:
                cleaned = val.strip().strip('"').strip("'")
                if cleaned and cleaned not in keys and not cleaned.startswith("gsk_...") and not cleaned.startswith("your_"):
                    keys.append(cleaned)

        # 2. Base env vars (e.g. GROQ_API_KEY, GROQ_API_KEYS) if not already added
        for var in self.env_vars:
            val = os.getenv(var, "")
            if val:
                for k in re.split(r"[,;\n\s]+", val):
                    cleaned = k.strip().strip('"').strip("'")
                    if cleaned and cleaned not in keys and not cleaned.startswith("gsk_...") and not cleaned.startswith("AIza...") and not cleaned.startswith("your_"):
                        keys.append(cleaned)
        return keys

    def reload_keys(self) -> list[str]:
        """Refresh key list and reset pointer (useful for test isolation or runtime updates)."""
        self.keys = self._discover_keys()
        self.current_idx = 0
        return self.keys

    def get_current_key(self) -> Optional[str]:
        if not self.keys:
            return None
        return self.keys[self.current_idx % len(self.keys)]

    def rotate(self) -> Optional[str]:
        if len(self.keys) <= 1:
            return self.get_current_key()
        old_idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        next_key = self.get_current_key()
        masked = f"{next_key[:6]}...{next_key[-4:]}" if next_key and len(next_key) > 10 else "***"
        console.print(
            f"[yellow]🔄 Rate limit or exhaustion on {self.provider_prefix} key #{old_idx + 1}. "
            f"Automatically rotating to key #{self.current_idx + 1}/{len(self.keys)}: {masked}[/yellow]"
        )
        return next_key


groq_rotator = KeyRotator("GROQ_API_KEY", ["GROQ_API_KEY", "GROQ_API_KEYS"])
gemini_rotator = KeyRotator("GEMINI_API_KEY", ["GEMINI_API_KEY", "GEMINI_API_KEYS"])
openrouter_rotator = KeyRotator("OPENROUTER_API_KEY", ["OPENROUTER_API_KEY", "OPENROUTER_API_KEYS"])


def _apply_current_key(model: str) -> Optional[str]:
    model_lower = model.lower()
    current_key = None
    if "groq" in model_lower:
        current_key = groq_rotator.get_current_key()
        if current_key:
            os.environ["GROQ_API_KEY"] = current_key
    elif "gemini" in model_lower:
        current_key = gemini_rotator.get_current_key()
        if current_key:
            os.environ["GEMINI_API_KEY"] = current_key
    elif "openrouter" in model_lower:
        current_key = openrouter_rotator.get_current_key()
        if current_key:
            os.environ["OPENROUTER_API_KEY"] = current_key
    return current_key


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
            active_key = _apply_current_key(current_model)
            from litellm import completion

            kwargs: dict[str, Any] = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = response_format
            if active_key:
                kwargs["api_key"] = active_key

            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            err_msg = str(exc).lower()
            is_rate_limit = (
                "rate limit" in err_msg
                or "rate_limit" in err_msg
                or "429" in err_msg
                or "quota" in err_msg
                or "resource_exhausted" in err_msg
                or "tokens per minute" in err_msg
                or "requests per minute" in err_msg
                or "tpm" in err_msg
                or "rpm" in err_msg
                or "too many requests" in err_msg
            )
            is_auth_error = (
                "invalid api key" in err_msg
                or "invalid_api_key" in err_msg
                or "unauthorized" in err_msg
                or "401" in err_msg
                or "authentication" in err_msg
            )

            if is_rate_limit or is_auth_error:
                model_lower = current_model.lower()
                rotated = False
                if "groq" in model_lower and len(groq_rotator.keys) > 1:
                    new_key = groq_rotator.rotate()
                    if new_key:
                        os.environ["GROQ_API_KEY"] = new_key
                    rotated = True
                elif "gemini" in model_lower and len(gemini_rotator.keys) > 1:
                    new_key = gemini_rotator.rotate()
                    if new_key:
                        os.environ["GEMINI_API_KEY"] = new_key
                    rotated = True

                if rotated:
                    time.sleep(0.3)
                    continue

                # 2. If single key or all keys exhausted for this model, try fallback model with separate TPM quota
                if attempt >= 1 and current_model in fallbacks:
                    alt_model = fallbacks[current_model]
                    console.print(
                        f"[yellow]TPM limit on {current_model}. Switching to alternative Indic model {alt_model}...[/yellow]"
                    )
                    current_model = alt_model
                    time.sleep(0.5)
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
