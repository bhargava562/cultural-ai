"""Candidate response generation with cultural context injection and language adaptation."""

import re
from typing import Any, Optional
from pydantic import BaseModel
from src.config import settings
from src.utils.llm_clients import generate_completion


class MediatedResponse(BaseModel):
    """Response contract exposed by the mediator generator."""

    cultural_context_observation: str
    customary_practice: str
    response: str

    def as_markdown(self) -> str:
        return f"**Cultural Context Observation:**\n{self.cultural_context_observation}\n\n**Customary Practice & Regional Norm:**\n{self.customary_practice}\n\n---\n**Response:**\n{self.response}"


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


def detect_domain_category(text: str) -> str:
    """Detect domain category: Healthcare, Banking/Finance, or Everyday Commonsense."""
    text_lower = text.lower()

    healthcare_keywords = [
        "health", "doctor", "medicine", "diet", "food", "fever", "illness", "pregnant",
        "pregnancy", "baby", "elder", "care", "hospital", "maruthuvam", "vaithiyam",
        "kashayam", "rasam", "treatment", "pain", "sick"
    ]
    finance_keywords = [
        "bank", "loan", "money", "chit fund", "seetu", "shg", "self-help", "interest",
        "vaddi", "savings", "gold", "jewel", "property", "asset", "finance", "cash",
        "rupee", "paisa", "invest"
    ]

    for kw in healthcare_keywords:
        if kw in text_lower:
            return "Healthcare"
    for kw in finance_keywords:
        if kw in text_lower:
            return "Banking / Finance"

    return "Everyday Commonsense"


ANTHROPOLOGICAL_SYSTEM_PROMPT = """You are an expert Cultural Anthropologist and Native Linguistic Guide specialized in Tamil Nadu, Dravidian culture, and South Asian regional dynamics.

Your Mission:
Produce a culturally grounded, socio-linguistically authentic response to the user query according to Anthropological Prompting principles.

Operating Principles:
1. Grounded Honorifics: Always align with age, hierarchy, and relationship registers (e.g., 'vaanga/ponga' vs 'vaa/po', 'neenga' vs 'nee').
2. Accurate Kinship: Distinctly respect family networks (Periyamma vs Chithi, Periyappa vs Chithappa, Mama, Athai).
3. Authentic Tone: When responding in Tamil or Tanglish, use natural colloquial cadences without awkward literal translations or robotic syntax.
4. Dignity & Anti-Bias: Never perpetuate caste stereotypes, regressive caricatures, or unscientific harms.
5. Inclusivity: Foster warm hospitality (Virundhombal) and moral righteousness (Aram).

Region: Tamil Nadu, India
Tone: Respectful, Culturally Grounded
Domain: {domain}
Language Mode: {language_mode}

Context:
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
    domain = detect_domain_category(query)
    sys_prompt = ANTHROPOLOGICAL_SYSTEM_PROMPT.format(
        domain=domain,
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


class MediatedResponseGenerator:
    """Compose a grounded mediation prompt using profile and retrieved facts."""

    def generate(
        self,
        query: str,
        profile: Any,
        retrieved_practices: list[dict[str, Any]],
        model: Optional[str] = None,
        guardrail_context: Optional[str] = None,
    ) -> str:
        practices = "\n".join(
            f"- {item.get('content', item.get('cultural_practice_fact', ''))}" for item in retrieved_practices
        ) or "- No verified local practice was retrieved; state uncertainty instead of inventing one."
        prompt = f"""Answer the user directly and naturally. Do not begin with a cultural report, do not say "You seem to be from...", and do not infer the user's location, ethnicity, or identity from slang.

LANGUAGE AND DIALECT RULES
- The user's input is likely Romanized Tamil/Tanglish when Tamil markers are present, even if most words are English-looking Latin text.
- First understand the meaning and conversational tone. Preserve the user's politeness level, familiar words, particles, and regional expressions; do not normalize the reply into textbook or generic Tamil.
- Mirror the input's warmth and familiarity. For example, "Ennanga thambi, sowkiyama irukkingala?" can naturally receive "Nalla irukken nga, neenga sowkiyama irukkingala?".
- Register examples: "mapla", "machi", "da", and "dei" usually signal an informal peer/friend register, so prefer "nee/irukka" or "da" in the reply; reserve "neenga/irukkinga" for elders, unfamiliar people, or explicitly polite contexts.
- Treat any named place or dialect in the request as the TARGET STYLE, not proof of who the user is.
- Never label common words such as thambi, sowkiyama, nalla, enna, epdi, or enga as uniquely Chennai, Kongu, Madurai, Kanyakumari, or Sri Lankan Tamil.
- Treat a greeting using these broad markers as possibly Kongu-associated or broadly regional, not definitive proof of Kongu origin.
- Keep meaning confidence separate from dialect confidence. Meaning can be high while dialect confidence stays low. Never report high dialect confidence when the exact locality is uncertain.
- If no specific region is requested, use natural, broadly understandable Romanized Tamil with light English code-switching. Do not invent exaggerated slang.
- Preserve the meaning and answer in a warm conversational register. Use respectful forms when the user is addressing an elder; do not turn a simple request into medical, cultural, or lifestyle advice.

SIMPLE REQUEST CONTRACT
For a slang, dialect, wording, or "what should I reply?" request, return only:
**Locality / Dialect:** [likely language community or locality; say "Tamil dialect — exact locality unclear" when the markers are broad]
\n**Slang detected:** [the relevant slang words, or "No strong locality-specific slang"]
\n**Suggested response:** [the actual natural Romanized Tamil/Tanglish reply the user can say or send]
\n**Meaning confidence:** [High, Medium, or Low]
\n**Dialect confidence:** [High, Medium, or Low; never High when exact locality is unclear]
Do not add a separate English meaning section. Do not claim the user's identity. Do not add unrelated questions, suggestions, lists, or lengthy explanations.

OTHER QUESTIONS
For non-translation questions, answer in the user's language mode and tone. Use retrieved facts only when relevant, mention regional variation briefly, and keep the answer practical.

Internal profile (do not expose as an identity claim): {profile.model_dump_json()}
Retrieved practices (use only when relevant):
{practices}

{f"Style guardrail for this request:\n{guardrail_context}" if guardrail_context else ""}

User query: {query}"""
        return generate_completion(
            model=model or settings.generator_model,
            messages=[{"role": "system", "content": ANTHROPOLOGICAL_SYSTEM_PROMPT + "\nPrioritize dialect accuracy, naturalness, and calibrated uncertainty."}, {"role": "user", "content": prompt}],
            temperature=0.45,
            json_mode=False,
        )
