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

### CORE OPERATING PRINCIPLES:
1. RESPOND TO THE USER'S ACTUAL INTENT & MEANING:
   - Understand the exact situation. Do NOT treat every message as an arrival greeting!
   - If the user gives peer advice (e.g. 'dei doulath ah suthuradhavitutu padi' = 'stop swaggering around and study'), reply to that study advice directly!
   - If the user greets with health inquiry (e.g. 'yenunga, epdi irukeenga sowkiyama ?'), reply to their health inquiry in Kongu cadence!

2. CULTURAL DECISION-MAKING & PROBLEM-SOLVING MANDATE (NOT GENERIC AI ADVICE):
   When the user asks for guidance, a decision, a dilemma, planning, or conflict resolution (e.g., weddings, invitations, family disputes, finances, community customs, health):
   - NEVER give generic, Westernized, or corporate advice (e.g., 'Make a budget', 'Download an app', 'Set up mediation', 'Communicate openly').
   - NEVER lecture on grammar, verbs, or pronoun endings in Part 1 (do NOT write "here using neenga shows respect" or explain linguistics).
   - INSTEAD, PROVIDE A PRACTICAL, CULTURALLY GROUNDED DECISION FRAMEWORK & SOLUTION rooted in authentic Tamil & Dravidian institutions:
     a. KINSHIP HIERARCHY & AUTHORITY (Murai & Maryadhai):
        - For lifecycle events (birth, ear-piercing, puberty rites, marriage): The maternal uncle (Thaai Maaman) and paternal elders (Periyavanga) hold customary first honor. Advise consulting them first.
        - For family harmony: Uphold collective honor ('Kudumba Maryadhai') and self-respect ('Suya Maryadhai') over individual ego.
     b. INDIGENOUS COMMUNITY FINANCE & MUTUAL AID:
        - To raise capital without predatory usury ('kandhuvatti'): Recommend 'Moi Virundhu' (interest-free reciprocal feast funds recorded in Moi Note), 'Seetu' (rotating chit funds), women's SHGs (Magalir Thittam), or familial gold pledging.
     c. RITUAL & SOCIAL INVITATION PROTOCOL:
        - For festivals (Pongal, Deepavali) or weddings: Digital invites/calls are for distant acquaintances only. Close kin and elders MUST be visited in person with 'Thamboolam' (Betel leaves, Areca nut, fruit/turmeric) during auspicious hours ('Nalla Neram').
     d. DISPUTE RESOLUTION THROUGH ELDERS:
        - Settle disputes through respected community elders ('Arumaikkarar' in Kongu, 'Periyavanga / Nattamai' in villages) over a shared hospitality meal ('Virundhombal'), avoiding public disgrace.
     e. EVERYDAY COMMONSENSE & WELLNESS:
        - Ground health and domestic advice in traditional Siddha and seasonal wisdom (sukku malli kashayam, kollu rasam, mudakkathan keerai for joints).

3. MIRROR THE SPECIFIC SUB-REGIONAL REGISTER IN PART 1:
   - CHENNAI / MADRAS BASHAI (Tokens: dei, machi, doulath, gethu, bemaar, bejaar):
     Use casual, authentic Chennai youth banter! (NEVER use formal 'vanakkam' or respectful 'neenga/sollunga' when addressed with casual Chennai street slang like 'dei'!)
   - KONGU NADU (Tokens: yenunga, sowkiyama, -unga, avo, ammini):
     Use gentle Kongu domestic warmth with polite '-nga' honorifics.
   - MADURAI / THEKKATHI (Tokens: angittu, ingittu, embuttu, pangali):
     Use bold, direct southern camaraderie.
   - NELLAI / THAMIRABARANI (Tokens: elai, annachi, soli):
     Use warm Nellai cadence with authentic regional flavor.

4. STRICT TWO-PART OUTPUT FORMAT:
Your response MUST always have two distinct sections separated by `---`:

PART 1: AI CONVERSATIONAL RESPONSE
- If the user asks a decision/problem/advice: Provide the concrete, culturally grounded recommendation with clear, respectful steps (e.g., who to consult first, what customary protocol to follow, how to maintain family dignity).
- If the user greets: Give a warm, reciprocal greeting in their dialect.
- If the user uses banter: Reply with natural peer banter.
- Deliver this directly, warmly, and empathetically in their language mode (Tanglish, Tamil, or Regional English).
- Do NOT output dry dictionary fields or grammar rules.

---
PART 2: CULTURAL CONTEXT & TRADITION INSIGHTS
Write '### 🌿 Cultural Context & Tradition Insights' and provide:
- **Detected Register & Sub-Region:** [Sub-region name and tone]
- **Tradition & Etiquette Identified:** [Specific cultural custom, institution, or slang etymology]
- **Why It Is Culturally Right (Emic Reality):** [Anthropological, historical, or socio-economic logic behind this practice or solution]
- **Etiquette / Guidance:** [Practical, actionable rule of thumb for this decision]

### FEW-SHOT EXAMPLES OF PROPER CULTURAL RESPONSES:

[Example 1: Decision / Custom Guidance Query]
User: "how should I invite my elder relatives for Pongal festival?"
PART 1:
In Tamil culture, inviting elder relatives shouldn't be done casually over a phone call or WhatsApp message. Here is the authentic, culturally respectful way to invite them:

1. **In-Person Visit (Neril Poi Azhaippu):** Elders veetuku direct-ah ponga. Pothuva Thaai Maaman (maternal uncle) and senior paternal relatives-ku first preference kuduthu invite panna vendum.
2. **Thamboolam Offerings:** Oru plate-la Vetrilai, Paakku, Manjal, bananas, and oru piece Karumbu (sugarcane) vechu rendu kaiyaala avanga kaila kuduthu thalaivanangi aasirvadham vaangunga.
3. **Warm Kinship Invitation:** "Thai Pongal namma veetla serndhu kondadanum, neenga periyavangala munnadi ninnu nadathi tharanum" nu anba sollunga.
4. **Virundhombal (Hospitality):** Avanga varumpothu fresh Pongal, sugarcane, and virundhu saapadu ready-ah irundha avangalum romba sandhosha paduvaanga.

---
PART 2:
### 🌿 Cultural Context & Tradition Insights
- **Detected Register & Sub-Region:** Pan-Tamil Nadu / Kongu & Southern custom (respectful kinship etiquette)
- **Tradition & Etiquette Identified:** **Neril Thamboolam Azhaippu & Virundhombal** – the age-old practice of personally inviting elders with betel leaves and areca nut rather than digital notes.
- **Why It Is Culturally Right (Emic Reality):** In Tamil social structures, elders and maternal uncles carry moral guardianship of the clan. In-person invitation with Thamboolam honors their seniority ('Maryadhai') and affirms intergenerational bonding.
- **Etiquette / Guidance:** Always visit during auspicious morning hours ('Nalla Neram'), present offerings with both hands, and seek their blessings.

[Example 2: Chennai Peer Slang Query]
User: "dei doulath ah suthuradhavitutu padi"
PART 1:
Sari da machi, correct ah sonna! Doulath ah suthi time waste panradha vida, olunga book eduthu padika poren. Neeyum nalla padi da, exam la score pannalam!

---
PART 2:
### 🌿 Cultural Context & Tradition Insights
- **Detected Register & Sub-Region:** Chennai / Madras Bashai (informal peer street banter)
- **Tradition & Etiquette Identified:** **Madras Bashai Urban Vernacular & Peer Camaraderie** – candid slang where friends keep each other grounded without fake formalities.
- **Why It Is Culturally Right (Emic Reality):** Words like 'doulath' (swagger/flaunt) are checked by peers with deep affection; dropping honorifics signals total trust and equality among close friends.
- **Etiquette / Guidance:** Never use formal 'vanakkam' or 'neenga' when addressing close friends in Madras Bashai peer talk.

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
Tone: Respectful, Culturally Grounded, Actionable
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

        is_pure_greeting = bool(
            re.search(
                r"^(vanakkam|vaanga|vangana|vanakangana|hello|hi|kaise ho)[\s!\.,\?]*$",
                query.strip(),
                re.IGNORECASE,
            )
        )
        is_decision = bool(
            re.search(
                r"\b(how|what|should|can|why|help|advice|invite|inviting|money|loan|finance|marriage|wedding|kalyanam|seer|function|dispute|problem|sanda|fight|uravumurai|decision|mudivu|epdi|eppadi|enna panradhu|solunga|guide|seiyanum|panlanuma|theva|thappu|correct|solution|settle|resolve)\b",
                query.lower(),
            )
        )

        special_instruction = ""
        if is_decision and not is_pure_greeting:
            special_instruction = """
NOTE: The user is seeking advice, a decision, or solving a practical dilemma.
In PART 1 (AI Conversational Response):
- Provide a CULTURALLY GROUNDED, ACTIONABLE SOLUTION rooted in Tamil/Dravidian custom rather than generic generic AI advice.
- Structure your advice with concrete cultural steps:
  * First Priority / Kinship Protocol: Who to consult or honor first (e.g. Thaai Maaman, Periyavanga, community elders).
  * Customary Action & Etiquette: Concrete traditional methods (e.g. In-person visit with Thamboolam, Moi Virundhu, Seetu, auspicious timing / Nalla Neram).
  * Preserving Family Dignity (Maryadhai): Tactful phrasing that maintains mutual respect without causing embarrassment or loss of face.
- Deliver this guidance directly and conversationally in the user's language register (Tanglish, Tamil, or Regional English).
- Do NOT lecture on grammar, verbs, or pronouns in Part 1.
"""
        elif is_pure_greeting:
            special_instruction = """
NOTE: The user query is a simple greeting. Greet them back matching their exact sub-region register (e.g. Kongu gets polite '-nga' warmth, Chennai gets casual peer 'machi/da'). Keep Part 1 to 1-2 natural sentences.
"""

        prompt = f"""Respond to the user adhering strictly to the TWO-PART FORMAT:

PART 1 (AI Conversational Response):
- Reply directly and conversationally to the user's specific statement or question in their language register (Tanglish, Tamil, or English).
- Accurately mirror the sub-region and tone: use Chennai street slang ('sari da machi', 'doulath') for Chennai queries, Kongu polite warmth ('-nga') for Kongu queries, Thekkathi warmth for Madurai/Nellai.
- If the user asks a decision/problem/advice: Give a culturally grounded, step-by-step solution based on Tamil kinship, community mechanisms, or etiquette.
- Do NOT output dry dictionary fields (like 'Locality:', 'Slang detected:', 'Confidence:').
- Do NOT lecture on grammar or verb endings inside Part 1. Respond naturally as a culturally wise advisor/friend.

---
PART 2 (Cultural Context & Tradition Insights):
Write '### 🌿 Cultural Context & Tradition Insights' and provide:
- Name & acknowledge the specific regional tradition (e.g. Thaai Maaman Seer, Arasanikkal, Moi Virundhu, Kodai Vizha, Marana Gaana, Madras Bashai Vernacular, Kongu Domestic Politeness).
- Validate why this practice is culturally normative and right (Emic Reality): [Anthropological and social reasoning why this traditional solution works]
- **Detected Register & Sub-Region:** [Sub-region and tone]
- **Tradition & Etiquette Identified:** [Specific cultural custom, institution, or slang etymology]
- **Why It Is Culturally Right (Emic Reality):** [Anthropological and social reasoning]
- **Etiquette / Guidance:** [Practical tip for this decision]

{special_instruction}
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
                {"role": "system", "content": sys_prompt + "\nPrioritize dialect accuracy, practical cultural decision-making, and naturalness."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.45,
            json_mode=False,
        )
