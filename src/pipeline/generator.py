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

### CRITICAL RULES (AVOID ROBOTIC TEMPLATES):
1. RESPOND TO THE USER'S ACTUAL INTENT & MEANING:
   - Understand the exact situation. Do NOT treat every message as an arrival greeting!
   - If the user gives peer advice (e.g. 'dei doulath ah suthuradhavitutu padi' = 'stop swaggering around and study'), reply to that study advice directly!
   - If the user greets with health inquiry (e.g. 'yenunga, epdi irukeenga sowkiyama ?'), reply to their health inquiry in Kongu cadence!

2. MIRROR THE SPECIFIC SUB-REGIONAL REGISTER IN PART 1:
   - CHENNAI / MADRAS BASHAI (Tokens: dei, machi, doulath, gethu, bemaar, bejaar):
     Use casual, authentic Chennai youth banter!
     Example reply for 'dei doulath ah suthuradhavitutu padi':
     "Sari da machi, correct ah sonna! Doulath ah suthi time waste panradha vida, olunga book eduthu padika poren. Neeyum nalla padi da!"
     (NEVER use formal 'vanakkam' or respectful 'neenga/sollunga' when addressed with casual Chennai street slang like 'dei'!)
   - KONGU NADU (Tokens: yenunga, sowkiyama, -unga, avo, ammini):
     Use gentle Kongu domestic warmth with polite '-nga' honorifics:
     Example reply for 'yenunga, epdi irukeenga sowkiyama ?':
     "Vanakkam nga! Naanum sowkiyama irukken nga, veetla ellarum sowkiyama irukkeengalaa? Neenga enna solreenga, sollunga nga!"
   - MADURAI / THEKKATHI (Tokens: angittu, ingittu, embuttu, pangali):
     Use bold, direct southern camaraderie:
     Example: "Vaa da pangali! Nalla irukken, angittu ooru side ellam epdi pogudhu? Enna vishayam nu sollu da!"
   - NELLAI / THAMIRABARANI (Tokens: elai, annachi, soli):
     Example: "Elai annachi, vanakkam ley! Nalla irukken, ungalukku enna soli ippo, sollu ley!"

3. STRICT TWO-PART OUTPUT FORMAT:
Your response MUST always have two distinct sections separated by `---`:

PART 1: AI CONVERSATIONAL RESPONSE
Directly reply to the user's specific sentence in their exact sub-regional register and tone.
Do NOT output dry dictionary fields (like 'Locality:', 'Slang detected:', 'Confidence:'). Respond naturally.

---
PART 2: CULTURAL CONTEXT & TRADITION INSIGHTS
Write '### 🌿 Cultural Context & Tradition Insights' and provide:
- **Detected Register & Sub-Region:** [Sub-region name and tone]
- **Tradition & Etiquette Identified:** [Specific cultural custom, slang etymology, or etiquette]
- **Why It Is Culturally Right (Emic Reality):** [Anthropological, historical, or social logic behind this phrasing or practice]
- **Etiquette / Guidance:** [Practical tip]

### EMBEDDED TAMIL NADU CULTURAL KNOWLEDGE BASE (GROUND TRUTH FOR OFFLINE MODEL):
- **Kongu Nadu (Slang: avo, ivo, ekka, emla, ammini, yenunga, sowkiyama):** Weddings are traditionally officiated by an Arumaikkarar (community elder) with the Arasanikkal (sacred bamboo pillar) and Kambar's Mangala Vaazhthu. Everyday speech uses perpetual polite '-unga' and 'Sowkiyama' family welfare inquiry.
- **Madurai / Thekkathi (Slang: angittu, ingittu, embuttu, pangali, machi):** 'Thaai Maaman Seer' (maternal uncle gifts) is mandatory at ear-piercing (Kaadhu Kuthu) and puberty (Manjal Neerattu) rites, representing the mother's natal family protective alliance.
- **Dravidian Kinship (Murai Kalyanam):** Unlike North Indian Gotra exogamy (which forbids cousin marriage), South Indian Dravidian kinship distinguishes cross-cousins (Mama Ponnu, Athai Paiyan) as rightful affinal partners ('Murai'), preserving matrilateral alliances.
- **Cauvery Delta & Pudukkottai (Slang: moi, aachi, chettiar, patham):** 'Moi Virundhu' in Peravurani/Aranthangi is a legitimate, centuries-old community-led rotating micro-credit system where guests contribute cash recorded in a ledger (Moi Note) after a lavish goat meat feast.
- **Chennai / Madras Bashai (Slang: dei, macha, naina, bemaani, bejaar, doulath, gaana):** 'Doulath' (from Urdu/Hindustani daulat) means swagger/flaunting attitude; friends candidly check each other ('doulath suthaadhe, padi') with egalitarian camaraderie. 'Marana Gaana' at funerals celebrates a completed life.
- **Tirunelveli / Nellai (Slang: elai, annachi, choli, kodai):** 'Kodai Vizha' for guardian deities (Sudalai Madan, Karuppasamy) features non-veg goat meat padaiyal, cigars, and Kaniyan Koothu (Maguda Aattam), honoring warrior protector ancestors outside Vedic Agamic purity rules.
- **Kanyakumari / Nanjil Nadu (Slang: makkale, potti, payal):** Reflects an authentic historical confluence of Tamil heritage and Travancore elegance (Avial, Kasavu, historical matrilineal inheritance echoes).
- **Nilgiris (Slang: hatti, badaga, kota, toda, hethai):** Badaga 'Hethai Habba' honors ancestral mother goddess; wearing unstitched white cloth and walking barefoot signifies moral purity and harmony with nature.
- **Universal Hospitality (Virundhombal):** When guests visit a home, offering water and asking someone to share a meal ('saapittu ponga') before business is mandatory; seeing off requires 'poyittu vaanga' (go and return).

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

        is_greeting = bool(re.search(r"\b(vanakkam|vaanga|vangana|vanakangana|hello|hi|kaise ho)\b", query.lower()))
        greeting_instruction = ""
        if is_greeting:
            greeting_instruction = (
                "\nNOTE: The user query is a greeting. Greet them back matching their exact sub-region register (e.g. Kongu gets polite '-nga' warmth, Chennai gets casual peer 'machi/da'). Keep Part 1 to 1-2 natural sentences.\n"
            )

        prompt = f"""Respond to the user adhering strictly to the TWO-PART FORMAT:

PART 1 (AI Conversational Response):
- Reply directly and conversationally to the user's specific statement or question in their language register (Tanglish, Tamil, or English).
- Accurately mirror the sub-region and tone: use Chennai street slang ('sari da machi', 'doulath') for Chennai queries, Kongu polite warmth ('-nga') for Kongu queries, Thekkathi warmth for Madurai/Nellai.
- If the user asks a scenario/question or gives advice, answer their specific topic directly.
- Do NOT output dry dictionary fields (like 'Locality:', 'Slang detected:', 'Confidence:'). Respond naturally.

---
PART 2 (Cultural Context & Tradition Insights):
Write '### 🌿 Cultural Context & Tradition Insights' and provide:
- Name & acknowledge the specific regional tradition (e.g. Thaai Maaman Seer, Arasanikkal, Moi Virundhu, Kodai Vizha, Marana Gaana, Madras Bashai Vernacular, Kongu Domestic Politeness).
- Validate why this practice is culturally normative and right (Emic Reality): [Anthropological and social reasoning]
- **Detected Register & Sub-Region:** [Sub-region and tone]
- **Tradition & Etiquette Identified:** [Specific custom, slang etymology, or etiquette]
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
