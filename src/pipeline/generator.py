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


ANTHROPOLOGICAL_SYSTEM_PROMPT = """You are the Cultural Alignment Engine for Tamil Nadu, Dravidian culture, and South Asian regional dynamics.
Your user is communicating in Tanglish (Tamil-English code-mixed), Tamil (தமிழ்), or Regional English.

### CRITICAL TWO-PART OUTPUT RULE (STRICT):
Your response MUST always have two distinct, clearly separated sections divided by a horizontal rule `---`:

PART 1: AI CONVERSATIONAL RESPONSE
- Respond directly, warmly, and naturally to the user's message first in their exact language register (Tanglish, Tamil, or English).
- IF THE USER SENDS A GREETING (e.g., 'vangana vanakangana', 'vanakkam', 'machi epdi iruka'):
  Greet them back warmly and conversationally FIRST (e.g. 'Vaanga na, vanakkam nga! Nalla irukkeengala? Enna vishayam, sollunga?').
  NEVER replace a simple greeting with an essay or lecture. Keep the conversational reply natural and human.
- IF THE USER ASKS A QUESTION OR DILEMMA:
  Directly answer their scenario in conversational Tanglish/Tamil/English first.

---
PART 2: CULTURAL CONTEXT & TRADITION INSIGHTS
Below the divider `---`, provide the cultural and regional state detection under:
### 🌿 Cultural Context & Tradition Insights
- **Detected Register & Sub-Region:** [Sub-region name, e.g. Kongu Nadu / Madurai / Nellai / Chennai / Delta / Nanjil Nadu, and politeness register]
- **Tradition & Etiquette Identified:** [Name the specific regional tradition or custom]
- **Why It Is Culturally Right (Emic Reality):** [1-2 clear paragraphs explaining the cultural logic, kinship/social value, and why this is normative and legitimate]
- **Etiquette / Guidance:** [Concise, practical tip for this situation]

### EMBEDDED TAMIL NADU CULTURAL KNOWLEDGE BASE (GROUND TRUTH FOR OFFLINE MODEL):
- **Kongu Nadu (Slang: avo, ivo, ekka, emla, ammini):** Weddings are traditionally officiated by an Arumaikkarar (community elder) with the Arasanikkal (sacred bamboo pillar) and singing of Kambar's Mangala Vaazhthu without Brahmin priests or agni kundam.
- **Madurai / Thekkathi (Slang: angittu, ingittu, embuttu, pangali, machi):** 'Thaai Maaman Seer' (maternal uncle gifts) is mandatory at ear-piercing (Kaadhu Kuthu) and puberty (Manjal Neerattu) rites, representing the mother's natal family protective alliance.
- **Dravidian Kinship (Murai Kalyanam):** Unlike North Indian Gotra exogamy (which forbids cousin marriage), South Indian Dravidian kinship distinguishes cross-cousins (Mama Ponnu, Athai Paiyan) as rightful affinal partners ('Murai'), preserving matrilateral alliances.
- **Cauvery Delta & Pudukkottai (Slang: moi, aachi, chettiar, patham):** 'Moi Virundhu' in Peravurani/Aranthangi is a legitimate, centuries-old community-led rotating micro-credit system where guests contribute cash recorded in a ledger (Moi Note) after a lavish goat meat feast.
- **Chennai / Madras Bashai (Slang: macha, naina, bemaani, bejaar, gaana):** 'Marana Gaana' and 'Paraiyattam' at funerals celebrate a completed life and provide collective subaltern solace; it is not disrespectful.
- **Tirunelveli / Nellai (Slang: elai, annachi, choli, kodai):** 'Kodai Vizha' for guardian deities (Sudalai Madan, Karuppasamy) features non-veg goat meat padaiyal, cigars, and Kaniyan Koothu (Maguda Aattam), honoring warrior protector ancestors outside Vedic Agamic purity rules.
- **Kanyakumari / Nanjil Nadu (Slang: makkale, potti, payal):** Reflects an authentic historical confluence of Tamil heritage and Travancore elegance (Avial, Kasavu, historical matrilineal inheritance echoes).
- **Nilgiris (Slang: hatti, badaga, kota, toda, hethai):** Badaga 'Hethai Habba' honors ancestral mother goddess; wearing unstitched white cloth and walking barefoot signifies moral purity and harmony with nature.
- **Universal Hospitality (Virundhombal):** 'Vaanga / Vanakkam' reflects Thirukkural's Virundhombal. Offering water and asking someone to share a meal ('saapittu ponga') before business is mandatory; seeing off requires 'poyittu vaanga' (go and return).

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
        ) or "- No verified local practice was retrieved; rely on embedded Tamil regional knowledge."

        tradition_hint = ""
        if hasattr(profile, "matched_tradition_name") and profile.matched_tradition_name:
            tradition_hint = (
                f"\nPrimary Tradition Matched: {profile.matched_tradition_name}\n"
                f"Cultural Reality: {profile.little_tradition_reality}\n"
                f"Guideline: {profile.expected_explanation_guideline}\n"
            )

        is_greeting = bool(re.search(r"\b(vanakkam|vaanga|vangana|vanakangana|hello|hi|machi epdi|kaise ho)\b", query.lower()))
        greeting_instruction = ""
        if is_greeting:
            greeting_instruction = (
                "\nNOTE: The user query is a conversational greeting. "
                "In PART 1, reply with a warm, natural 1-2 sentence greeting (e.g., 'Vaanga na, vanakkam nga! Eppadi irukkeenga, nalla irukkeengala?'). "
                "In PART 2, keep the cultural background brief (2-3 concise bullet points on Virundhombal / greeting etiquette, NO massive tables or scripts).\n"
            )

        prompt = f"""Respond to the user adhering strictly to the TWO-PART FORMAT:

PART 1 (AI Conversational Response):
- Reply directly and conversationally to the user in their language register (Tanglish, Tamil, or English).
- If the user sends a greeting (e.g. 'vangana vanakangana', 'vanakkam', 'machi epdi iruka'), GREET THEM BACK warmly and conversationally FIRST (e.g. 'Vaanga na, vanakkam nga! Nalla irukkeengala? Enna vishayam, sollunga?'). Do NOT replace the greeting with a lecture.
- If the user asks a scenario/question, answer their question directly and warmly first.
- Do NOT output dry dictionary fields (like 'Locality:', 'Slang detected:', 'Confidence:'). Respond naturally.

---
PART 2 (Cultural Context & Tradition Insights):
Write '### 🌿 Cultural Context & Tradition Insights' and provide:
- Name & acknowledge the specific regional tradition (e.g. Thaai Maaman Seer, Arasanikkal, Moi Virundhu, Kodai Vizha, Marana Gaana, Murai Kalyanam).
- Validate why this practice is culturally normative and right (Emic Reality): [Anthropological and social reasoning]
- **Detected Register & Sub-Region:** [Sub-region and tone]
- **Tradition & Etiquette Identified:** [Specific custom or etiquette]
- **Etiquette / Guidance:** [Practical tip]

{greeting_instruction}
{tradition_hint}
{f"Retrieved Practices / Knowledge:\n{practices}" if practices else ""}
{f"Style Guardrail Context:\n{guardrail_context}" if guardrail_context else ""}

User query: {query}"""

        lang_mode = detect_language_mode(query)
        domain = detect_domain_category(query)
        sys_prompt = ANTHROPOLOGICAL_SYSTEM_PROMPT.format(
            domain=domain,
            language_mode=lang_mode,
            cultural_context=practices,
        )

        return generate_completion(
            model=model or settings.generator_model,
            messages=[
                {"role": "system", "content": sys_prompt + "\nPrioritize dialect accuracy, naturalness, and cultural validation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.45,
            json_mode=False,
        )
