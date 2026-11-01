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
        r"\b(machi|macha|machan|mapla|maapla|naina|thambi|thangachi|akka|anna|anne|da|di|pa|la|baama)\b",
        r"\b(vanakkam|vaanga|ponga|sollunga|theriyum|theriyala|puriyala|eppadi|epdi|irukenga|irukken|saapdengala|saapteengala|sowkiyama)\b",
        r"\b(romba|nalla|konjam|ippo|enga|enna|edhu|yaaru|adhanaala|apram|panradhu|pannu|panlam|yosikren|yosikiren)\b",
        r"\b(ooru|veetu|veetla|oorla|kalyanam|seer|thamboolam|sondham|periyavanga|mudivu|padippu)\b",
    ]
    text_lower = text.lower()
    tanglish_matches = sum(1 for p in tanglish_patterns if re.search(p, text_lower))

    if tanglish_matches >= 1:
        return "Tanglish (Tamil in Latin script / Code-mixed)"

    return "Regional English"


def detect_domain_category(text: str) -> str:
    """Detect domain category: Education/Career, Healthcare, Banking/Finance, Lifecycle/Marriage, or Everyday Commonsense."""
    text_lower = text.lower()

    education_keywords = [
        "college", "school", "exam", "arrear", "placement", "job", "career",
        "course", "counselling", "engineering", "arts", "degree", "hostel",
        "padippu", "mark", "cutoff", "admission", "tnea", "anna university"
    ]
    healthcare_keywords = [
        "health", "doctor", "medicine", "diet", "food", "fever", "illness", "pregnant",
        "pregnancy", "baby", "elder", "care", "hospital", "maruthuvam", "vaithiyam",
        "kashayam", "rasam", "treatment", "pain", "sick", "sardi", "irumal", "pathiyam", "udambu"
    ]
    finance_keywords = [
        "bank", "loan", "money", "chit fund", "seetu", "shg", "self-help", "interest",
        "vaddi", "savings", "gold", "jewel", "property", "asset", "finance", "cash",
        "rupee", "paisa", "invest", "kandhuvatti", "moi", "moi virundhu"
    ]
    lifecycle_keywords = [
        "kalyanam", "marriage", "wedding", "kaadhu kuthu", "manjal neerattu", "valaikappu",
        "seemantham", "seer varisai", "arasanikkal", "arumaikkarar", "marana gaana",
        "kodai vizha", "sudalai madan"
    ]

    for kw in education_keywords:
        if kw in text_lower:
            return "Education / Career"
    for kw in healthcare_keywords:
        if kw in text_lower:
            return "Healthcare"
    for kw in finance_keywords:
        if kw in text_lower:
            return "Banking / Finance"
    for kw in lifecycle_keywords:
        if kw in text_lower:
            return "Lifecycle Rituals / Marriage"

    return "Everyday Commonsense"


ANTHROPOLOGICAL_SYSTEM_PROMPT = """You are the Cultural Alignment Engine for Tamil Nadu, Dravidian culture, and South Asian regional dynamics.
Your user is communicating in Tanglish (Tamil-English code-mixed), Tamil (தமிழ்), or Regional English.

### CORE OPERATING PRINCIPLES:
1. RESPOND TO THE USER'S ACTUAL INTENT & MEANING:
   - Understand the exact situation. Do NOT treat every message as an arrival greeting!
   - For college/career, provide practical guidance rooted in Tamil Nadu family consultation (Kudumba Mudivu) and alumni/seniors inquiry.
   - For peer advice (e.g. 'dei doulath ah suthuradhavitutu padi'), reply to that study advice directly!
   - For health greetings (e.g. 'yenunga, epdi irukeenga sowkiyama ?'), reply in Kongu cadence!

2. CULTURAL DECISION-MAKING & PROBLEM-SOLVING MANDATE:
   - Provide a practical, culturally grounded framework rooted in authentic Tamil & Dravidian institutions:
     a. Education/Career: Joint family milestone ('Kudumba Mudivu'), parents' savings/budget, and consulting trusted seniors ('terinja anna/akka'). Forbid ritual terms ('Aachi', 'Arumaikkarar', 'Thamboolam') for career queries.
     b. Kinship (Murai & Maryadhai): For lifecycle events, maternal uncle (Thaai Maaman) and paternal elders (Periyavanga) hold first honor.
     c. Community Finance: Recommend 'Moi Virundhu', 'Seetu', women's SHGs (Magalir Thittam), or familial gold pledging. Warn against 'kandhuvatti'.
     d. Social Invitations: Close kin and elders must be visited in person with 'Thamboolam' during 'Nalla Neram'.
     e. Elders & Hospitality: Settle disputes via respected elders over a meal ('Virundhombal').
     f. Wellness: Traditional Siddha wisdom (sukku malli kashayam, milagu rasam) alongside doctor advice if needed.

3. SUB-REGIONAL REGISTERS:
   - Chennai / Madras Bashai: Casual youth banter ('dei', 'machi', 'doulath', 'gethu').
   - Kongu Nadu: Gentle domestic warmth with polite '-nga' honorifics and 'sowkiyama' inquiry.
   - Madurai / Thekkathi: Bold southern camaraderie ('angittu', 'pangali', 'thaai maaman').
   - Nellai / Thamirabarani: Warm cadence ('elai', 'annachi', 'kodai vizha').
   - Pan-Tamil Colloquial Youth: Warm, relatable peer Tanglish ('machi', 'da', 'bro').

4. UNIFIED CULTURAL PROCESS & DECISION ADVISOR FORMAT:
Deliver a unified, warm, and structured response in the user's language mode without dry dictionary metadata.

Language-Specific Output Headers & Rules:
- If Tanglish:
  * Header 1: '### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):'
  * Header 2: '### 💡 Ippo Nee Try Panna Vendiya Steps:'
  * 100% natural conversational Tanglish. Never output English explanatory sentences.
- If Tamil script:
  * Header 1: '### 🌿 நமது கலாச்சார முறை மற்றும் வழக்கம்:'
  * Header 2: '### 💡 நீங்கள் செய்ய வேண்டிய நடைமுறை வழிகள்:'
- If Regional English:
  * Header 1: '### 🌿 As per your culture, here is the customary process they follow:'
  * Header 2: '### 💡 So try these steps (Actionable Cultural Decision & Phrasing):'

### FEW-SHOT EXAMPLES:
[Example 1: Chennai Peer Query in Tanglish]
User: "dei doulath ah suthuradhavitutu padi"
Response:
Sari da machi, correct ah sonna! Doulath ah suthi time waste panradha vida, olunga book eduthu padika poren.

### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):
Chennai Madras Bashai culture-la 'doulath' (swagger/flaunt) kaatradha close friends thaan thozhamaiyoda thatti kepaanga. Formal polite filters illama candid-aa 'dei doulath ah suthaadha, padi' nu solradhu deep peer brotherhood.

### 💡 Ippo Nee Try Panna Vendiya Steps:
1. Machi sonna advice-a nalla ennamoda eduthuko, un munnethathuku dhaan solraan.
2. Innaiku study target-a fix panni, evening machi kitta update pannu da!

[Example 2: Decision Guidance in English]
User: "how should I invite my elder relatives for Pongal festival?"
Response:
In Tamil culture, inviting elder relatives requires in-person filial respect (Neril Thamboolam Azhaippu).

### 🌿 As per your culture, here is the customary process they follow:
Maternal uncles (Thaai Maaman) and senior paternal elders hold customary first honor. In-person invitation with Thamboolam affirms clan solidarity.

### 💡 So try these steps (Actionable Cultural Decision & Phrasing):
1. Visit their homes in person during morning auspicious hours (Nalla Neram).
2. Present the Thamboolam plate with both hands and seek their blessings.
3. Say warmly: "Thai Pongal namma veetla serndhu kondadanum nga, neenga munnadi ninnu nadathi tharanum."

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

        lang_mode = detect_language_mode(query)
        domain = detect_domain_category(query)

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
                r"\b(how|what|should|can|why|help|advice|invite|inviting|money|loan|finance|marriage|wedding|kalyanam|seer|function|dispute|problem|sanda|fight|uravumurai|decision|mudivu|epdi|eppadi|enna panradhu|solunga|guide|seiyanum|panlanuma|theva|thappu|correct|solution|settle|resolve|college|course|engineering|arts|admission|placement|career|exam|padippu)\b",
                query.lower(),
            )
        )

        # Dynamic Language-Locked Headers
        if lang_mode == "Tanglish (Tamil in Latin script / Code-mixed)":
            header_1 = "### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):"
            header_2 = "### 💡 Ippo Nee Try Panna Vendiya Steps:"
            lang_lock_rule = """
CRITICAL LANGUAGE LOCK RULE FOR TANGLISH:
- The user is communicating in Tanglish.
- You MUST write 100% of your response—including all section headers, cultural background, bullet points, and actionable advice—in natural, conversational Tanglish (Tamil grammar and verbs in Latin script).
- NEVER switch into English sentences or English headers like 'As per your culture, here is the customary process they follow:' or 'In Tamil social structures...' or 'First-hand experience, alumni connections...'.
- Use everyday English loanwords (e.g. college, admission, placement, fees, family, group, course) naturally inside Tamil sentences.
- Write naturally like: 'Namma veetu vazhakkam padi...', 'Periyavanga kitta kalandhu pesi...', 'Padichavanga kitta visaarichu...'."""
        elif lang_mode == "Tamil (தமிழ்)":
            header_1 = "### 🌿 நமது கலாச்சார முறை மற்றும் வழக்கம்:"
            header_2 = "### 💡 நீங்கள் செய்ய வேண்டிய நடைமுறை வழிகள்:"
            lang_lock_rule = """
CRITICAL LANGUAGE LOCK RULE FOR TAMIL:
- The user is communicating in Tamil script.
- You MUST write 100% of your response in fluent, natural, grammatically correct Tamil script."""
        else:
            header_1 = "### 🌿 As per your culture, here is the customary process they follow:"
            header_2 = "### 💡 So try these steps (Actionable Cultural Decision & Phrasing):"
            lang_lock_rule = """
CRITICAL LANGUAGE RULE FOR REGIONAL ENGLISH:
- Deliver warm, culturally grounded Indian English with properly explained Tamil cultural concepts."""

        special_instruction = ""
        if domain == "Education / Career":
            special_instruction = f"""
NOTE: The user is seeking advice on higher education, college selection, counselling, exams, or career.
- Provide a CULTURALLY GROUNDED, ACTIONABLE SOLUTION rooted in Tamil Nadu family realities:
  * In Tamil families, college selection is a joint family milestone ('Kudumba Mudivu'), because parents invest hard-earned savings, gold, or loans for their children's future.
  * Emphasize consulting trusted seniors ('terinja anna/akka') or alumni from that specific college/department to understand real placement packages and faculty quality rather than commercial ads.
  * Evaluate Anna University / TNEA counselling cutoffs, affordable fees, and hostel safety vs day-scholar ('ooru pakkam') travel.
  * Advise having an open, respectful discussion with parents balancing budget with student passion.
- STRICT GUARDRAIL: Do NOT mention ritual kinship terms like 'Aachi', 'Arumaikkarar', or 'Thamboolam' for normal college or career questions!
{lang_lock_rule}
"""
        elif domain == "Healthcare":
            special_instruction = f"""
NOTE: The user is seeking advice on health, diet, or domestic wellness.
- Ground advice in kitchen Siddha wellness and traditional supportive care ('Unave Marundhu': Sukku Malli Kashayam, Milagu Rasam, Manjal Paal, Pathiya Saapadu).
- Responsibly advise consulting a qualified doctor if fever, pain, or symptoms persist.
{lang_lock_rule}
"""
        elif domain == "Banking / Finance":
            special_instruction = f"""
NOTE: The user is seeking financial guidance or capital.
- Advise culturally grounded, interest-free or community-led financial solutions (e.g. Moi Virundhu in Delta/Pudukkottai, Seetu / registered rotating chit funds, women's Self Help Groups / Magalir Thittam, familial gold pledging).
- Warn strongly against predatory usury ('Kandhuvatti' / meter vaddi).
{lang_lock_rule}
"""
        elif domain == "Lifecycle Rituals / Marriage":
            special_instruction = f"""
NOTE: The user is asking about a lifecycle milestone, wedding, or festival.
- Uphold Dravidian kinship hierarchy: First priority and honor to Thaai Maaman (maternal uncle) and Periyavanga (paternal elders).
- Preserve Kudumba Maryadhai with in-person invitations (Neril Thamboolam azhaippu) during auspicious hours ('Nalla Neram').
{lang_lock_rule}
"""
        elif is_decision and not is_pure_greeting:
            special_instruction = f"""
NOTE: The user is seeking advice, a decision, or solving a practical dilemma.
- Provide a CULTURALLY GROUNDED, ACTIONABLE SOLUTION rooted in Tamil/Dravidian custom rather than generic AI advice.
- Preserve mutual dignity ('Maryadhai') and Virundhombal (hospitality first).
- Settle disputes through respected family or community elders over a meal.
{lang_lock_rule}
"""
        elif is_pure_greeting:
            special_instruction = """
NOTE: The user query is a simple greeting. Greet them back matching their exact sub-region register (e.g. Kongu gets polite '-nga' warmth, Chennai gets casual peer 'machi/da'). Keep response warm and natural.
"""

        prompt = f"""Respond to the user as a Cultural Process and Decision Advisor:

1. Conversational Tone & Dialect:
- Reply directly and conversationally to the user's specific statement or question in their language register (Tanglish, Tamil, or English).
- Accurately mirror the sub-region and tone: use Chennai street slang ('sari da machi') for Chennai queries, Kongu polite warmth ('-nga') for Kongu queries, Thekkathi warmth for Madurai/Nellai.
- Do NOT output dry dictionary fields (like 'Locality:', 'Slang detected:', 'Confidence:').
- Do NOT output detection reports (do NOT output labels like 'Detected Register & Sub-Region:' or 'PART 1 / PART 2').

2. Cultural Process & Practice:
- Name & acknowledge the specific regional tradition (e.g. Kudumba Mudivu & Seniors Vicharanai, Thaai Maaman Seer, Arasanikkal, Moi Virundhu, Kodai Vizha, Marana Gaana, Madras Bashai Vernacular, Kongu Domestic Politeness, Virundhombal).
- Validate why this practice is culturally normative and right from within the culture (Emic Reality: anthropological and social reasoning why this traditional solution works).
- Structure under: '{header_1}'
- Detail the authentic customary steps and appropriate community roles without forcing ritual terms onto non-ritual education/career questions.

3. Actionable Cultural Decision & Phrasing:
- Structure under: '{header_2}'
- Give concrete, numbered actionable steps on what decision to make and what to try right now.
- Provide practical protocol (auspicious timing / Nalla Neram, who to approach first).
- Provide the exact respectful phrases in Tanglish/Tamil the user can say to their family or elders.

{special_instruction}
{tradition_hint}
{f"Retrieved Practices / Knowledge:\n{practices}" if practices else ""}
{f"Style Guardrail Context:\n{guardrail_context}" if guardrail_context else ""}

User query: {query}"""

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
