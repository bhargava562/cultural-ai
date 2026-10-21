"""Lightweight Tamil/Tanglish slang detection for response-style alignment."""

from __future__ import annotations

import re
from typing import Any


class TamilSlangGuardrail:
    """Detect registered colloquialisms without making identity or location claims."""

    slang_lexicon: dict[str, dict[str, str]] = {
        "machi": {"type": "casual", "intent": "friendly", "response_tone": "warm_peer"},
        "machan": {"type": "casual", "intent": "friendly", "response_tone": "warm_peer"},
        "gethu": {"type": "expression", "intent": "pride/style", "response_tone": "appreciative"},
        "lusu": {"type": "informal", "intent": "teasing", "response_tone": "lighthearted"},
        "vera level": {"type": "tanglish", "intent": "high_praise", "response_tone": "enthusiastic"},
        "kadrai": {"type": "colloquial", "intent": "nonsense/talk", "response_tone": "humorous"},
    }

    def __init__(self) -> None:
        # Longest-first prevents a shorter entry from winning inside a phrase.
        alternatives = sorted(self.slang_lexicon, key=len, reverse=True)
        self._patterns = {
            slang: re.compile(rf"(?<!\w){re.escape(slang)}(?!\w)", re.IGNORECASE)
            for slang in alternatives
        }

    def detect_slang(self, text: str) -> list[dict[str, Any]]:
        """Return registered slang markers found as whole words or phrases."""
        detected: list[dict[str, Any]] = []
        for slang in self.slang_lexicon:
            if self._patterns[slang].search(text):
                detected.append({"slang": slang, **self.slang_lexicon[slang]})
        return detected

    def alignment_middleware(self, user_input: str) -> dict[str, Any]:
        """Build a generation instruction while preserving the user's original query."""
        detected_slangs = self.detect_slang(user_input)
        if not detected_slangs:
            return {
                "status": "pass",
                "adjusted_prompt": user_input,
                "cultural_context": "Standard / Neutral Tamil context",
                "detected_slangs": [],
            }

        primary_tone = detected_slangs[0]["response_tone"]
        names = ", ".join(item["slang"] for item in detected_slangs)
        instruction = (
            "[Cultural Guardrail Active]: Detected Tamil/Tanglish colloquialisms "
            f"({names}). Match the user's {primary_tone} conversational register "
            "while remaining helpful, respectful, and socially appropriate. "
            "Do not infer the user's identity, ethnicity, or exact locality from slang."
        )
        return {
            "status": "adjusted",
            "detected_slangs": detected_slangs,
            "cultural_context": instruction,
            "adjusted_prompt": f"{instruction}\nUser Query: {user_input}",
        }
