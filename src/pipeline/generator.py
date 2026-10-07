"""Candidate response generation with cultural context injection and language adaptation."""

import re
from typing import Optional
from src.config import settings
from src.utils.llm_clients import generate_completion


def detect_language_mode(text: str) -> str:
    """Detect language and code-mixing mode (Tamil, Tanglish, English)."""
    # Tamil Unicode block: U+0B80 to U+0BFF
    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
    total_chars = max(len(re.findall(r"\w", text)), 1)
    tamil_ratio = tamil_chars / total_chars

    if tamil_ratio > 0.3:
        return "Tamil (தமிழ்)"

    # Tanglish indicator patterns
    tanglish_patterns = [
        r"\b(machi|macha|mapla|naina|thambi|akka|anna|da|di|pa|la|baama)\b",
        r"\b(vanakkam|vaanga|ponga|sollunga|theriyum|puriyala|eppadi|irukenga|saapdengala)\b",
        r"\b(romba|nalla|konjam|ippo|enga|enna|yaaru)\b",
    ]
    text_lower = text.lower()
    tanglish_matches = sum(1 for p in tanglish_patterns if re.search(p, text_lower))

    if tanglish_matches >= 1:
        return "Tanglish (Tamil in Latin script / Code-mixed)"

    return "Regional English"


ANTHROPOLOGICAL_SYSTEM_PROMPT = """You are an expert Cultural Anthropologist and Native Linguistic Guide specialized in Tamil Nadu, Dravidian culture, and South Asian regional dynamics.

Your Mission:
Respond to the user with deep cultural grounding, authentic socio-linguistic markers, and respectful situational etiquette.

Operating Principles:
1. Grounded Honorifics: Always align with age, hierarchy, and relationship registers (e.g., 'vaanga/ponga' vs 'vaa/po', 'neenga' vs 'nee').
2. Accurate Kinship: Distinctly respect family networks (Periyamma vs Chithi, Periyappa vs Chithappa, Mama, Athai).
3. Authentic Tone: When responding in Tamil or Tanglish, use natural colloquial cadences without awkward literal translations or robotic syntax.
4. Dignity & Anti-Bias: Never perpetuate caste stereotypes, regressive caricatures, or patronizing tropes.
5. Inclusivity: Foster warm hospitality (Virundhombal) and moral righteousness (Aram).

Language Mode: {language_mode}

Cultural Reference Knowledge:
{cultural_context}
"""


def generate_candidate(
    query: str,
    cultural_context: str = "",
    critique_feedback: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.6,
) -> str:
    """Generate candidate response incorporating retrieved cultural context and optional feedback."""
    lang_mode = detect_language_mode(query)
    sys_prompt = ANTHROPOLOGICAL_SYSTEM_PROMPT.format(
        language_mode=lang_mode,
        cultural_context=cultural_context or "No specific external context provided.",
    )

    messages = [{"role": "system", "content": sys_prompt}]

    if critique_feedback:
        messages.append({
            "role": "user",
            "content": query,
        })
        messages.append({
            "role": "assistant",
            "content": "Previous draft was evaluated as needing cultural refinement.",
        })
        messages.append({
            "role": "user",
            "content": (
                f"Please regenerate your response addressing this cultural critique:\n"
                f"{critique_feedback}\n\nOriginal Query: {query}"
            ),
        })
    else:
        messages.append({"role": "user", "content": query})

    target_model = model or settings.generator_model
    return generate_completion(
        model=target_model,
        messages=messages,
        temperature=temperature,
        json_mode=False,
    )
