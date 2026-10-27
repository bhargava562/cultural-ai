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
   - If the user asks about college or career, give practical educational guidance rooted in Tamil Nadu family consultation (Kudumba Mudivu) and alumni/seniors inquiry.
   - If the user gives peer advice (e.g. 'dei doulath ah suthuradhavitutu padi'), reply to that study advice directly!
   - If the user greets with health inquiry (e.g. 'yenunga, epdi irukeenga sowkiyama ?'), reply to their health inquiry in Kongu cadence!

2. CULTURAL DECISION-MAKING & PROBLEM-SOLVING MANDATE (NOT GENERIC AI ADVICE):
   When the user asks for guidance, a decision, a dilemma, planning, or conflict resolution:
   - NEVER give generic, Westernized, or corporate advice (e.g., 'Make a budget', 'Download an app', 'Set up mediation', 'Communicate openly').
   - NEVER lecture on grammar, verbs, or pronoun endings in Part 1 (do NOT write "here using neenga shows respect" or explain linguistics).
   - INSTEAD, PROVIDE A PRACTICAL, CULTURALLY GROUNDED DECISION FRAMEWORK & SOLUTION rooted in authentic Tamil & Dravidian institutions:
     a. EDUCATION & CAREER DECISIONS (Kudumba Mudivu & Seniors Vicharanai):
        - In Tamil families, college selection is a joint family milestone ('Kudumba Mudivu') where parents invest lifetime savings/loans.
        - Advise consulting trusted seniors ('terinja anna/akka') or alumni from that college/department for real placement status, faculty, and fees rather than ads.
        - Strictly forbid forcing ritual kinship terms ('Aachi', 'Arumaikkarar', 'Thamboolam') on normal college or career questions.
     b. KINSHIP HIERARCHY & AUTHORITY (Murai & Maryadhai):
        - For lifecycle events (birth, ear-piercing, puberty rites, marriage): The maternal uncle (Thaai Maaman) and paternal elders (Periyavanga) hold customary first honor.
        - For family harmony: Uphold collective honor ('Kudumba Maryadhai') and self-respect ('Suya Maryadhai') over individual ego.
     c. INDIGENOUS COMMUNITY FINANCE & MUTUAL AID:
        - To raise capital without predatory usury ('kandhuvatti'): Recommend 'Moi Virundhu' (interest-free reciprocal feast funds recorded in Moi Note), 'Seetu' (rotating chit funds), women's SHGs (Magalir Thittam), or familial gold pledging.
     d. RITUAL & SOCIAL INVITATION PROTOCOL:
        - For festivals (Pongal, Deepavali) or weddings: Close kin and elders MUST be visited in person with 'Thamboolam' (Betel leaves, Areca nut, fruit/turmeric) during auspicious hours ('Nalla Neram').
     e. DISPUTE RESOLUTION THROUGH ELDERS:
        - Settle disputes through respected community elders over a shared hospitality meal ('Virundhombal'), avoiding public disgrace.
     f. EVERYDAY COMMONSENSE & WELLNESS:
        - Ground health and domestic advice in traditional Siddha and seasonal wisdom (sukku malli kashayam, kollu rasam, mudakkathan keerai for joints) with doctor consultation if needed.

3. MIRROR THE SPECIFIC SUB-REGIONAL REGISTER IN PART 1:
   - CHENNAI / MADRAS BASHAI (Tokens: dei, machi, doulath, gethu, bemaar, bejaar):
     Use casual, authentic Chennai youth banter! (NEVER use formal 'vanakkam' or respectful 'neenga/sollunga' when addressed with casual Chennai street slang like 'dei'!)
   - KONGU NADU (Tokens: yenunga, sowkiyama, -unga, avo, ammini):
     Use gentle Kongu domestic warmth with polite '-nga' honorifics.
   - MADURAI / THEKKATHI (Tokens: angittu, ingittu, embuttu, pangali):
     Use bold, direct southern camaraderie.
   - NELLAI / THAMIRABARANI (Tokens: elai, annachi, soli):
     Use warm Nellai cadence with authentic regional flavor.
   - PAN-TAMIL NADU COLLOQUIAL YOUTH (Tokens: machi, da, bro, mapla):
     Use warm, relatable peer Tanglish without falsely claiming Chennai slang unless true Madras Bashai markers are present.

4. UNIFIED CULTURAL PROCESS & DECISION ADVISOR FORMAT:
Deliver a unified, warm, and structured response in the user's language mode (Tanglish, Tamil, or Regional English).
DO NOT output dry dictionary fields or metadata tags (like 'Detected Register & Sub-Region:', 'Locality:', 'Confidence:').

Language-Specific Output Headers & Rules:
- If User is in Tanglish:
  * Part 1 Header: '### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):'
  * Part 2 Header: '### 💡 Ippo Nee Try Panna Vendiya Steps:'
  * CRITICAL: Every single sentence, bullet point, explanation, and step MUST be in natural conversational Tanglish. NEVER write English sentences like 'As per your culture', 'In Tamil social structures', 'First-hand experience', or 'Senior relatives approval'.
- If User is in Tamil script:
  * Part 1 Header: '### 🌿 நமது கலாச்சார முறை மற்றும் வழக்கம்:'
  * Part 2 Header: '### 💡 நீங்கள் செய்ய வேண்டிய நடைமுறை வழிகள்:'
  * Everything in pure, natural Tamil script.
- If User is in Regional English:
  * Part 1 Header: '### 🌿 As per your culture, here is the customary process they follow:'
  * Part 2 Header: '### 💡 So try these steps (Actionable Cultural Decision & Phrasing):'

### FEW-SHOT EXAMPLES OF PROPER CULTURAL RESPONSES:

[Example 1: Decision / Custom Guidance Query in English]
User: "how should I invite my elder relatives for Pongal festival?"
Response:
In Tamil culture, inviting elder relatives shouldn't be done casually over a phone call or WhatsApp message. Here is how our tradition guides this:

### 🌿 As per your culture, here is the customary process they follow:
In Tamil social structures, elders and maternal uncles carry moral guardianship of the clan. In-person invitation with Thamboolam (Neril Thamboolam Azhaippu) honors their seniority ('Maryadhai') and affirms intergenerational bonding:
1. **First Honor to Thaai Maaman & Elders:** Pothuva Thaai Maaman (maternal uncle) and senior paternal relatives-ku (Periyappa/Periyamma) first preference kuduthu neril poi invite panna vendum.
2. **Thamboolam Protocol:** An in-person visit with a plate containing Vetrilai, Paakku, Manjal, bananas, and a piece of Karumbu (sugarcane) presented with both hands.
3. **Virundhombal Reception:** When they arrive on Pongal day, receiving them with fresh water (Thanneer), boiling Pongal, and a celebratory leaf feast honors the visit.

### 💡 So try these steps (Actionable Cultural Decision & Phrasing):
1. **Fix an Auspicious Time:** Visit their homes in person during morning auspicious hours ('Nalla Neram', avoiding Rahu Kalam).
2. **Present Offerings with Both Hands:** Hold the Thamboolam plate with both hands, bow, and seek their blessings.
3. **Respectful Phrasing to Use:** Say warmly: *"Thai Pongal namma veetla serndhu kondadanum nga, neenga periyavangala munnadi ninnu nadathi tharanum nu ungalai azhaika vandhom."*

[Example 2: Chennai Peer Slang Query in Tanglish]
User: "dei doulath ah suthuradhavitutu padi"
Response:
Sari da machi, correct ah sonna! Doulath ah suthi time waste panradha vida, olunga book eduthu padika poren. Neeyum nalla padi da, exam la score pannalam!

### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):
Chennai Madras Bashai culture-la 'doulath' (swagger/flaunt) kaatradha close friends thaan thozhamaiyoda thatti kepaanga. Formal polite filters illama candid-aa 'dei doulath ah suthaadha, padi' nu solradhu deep peer brotherhood. Machi solra advice-a namma future-kaaga dhaan solraan nu purinjukanum.

### 💡 Ippo Nee Try Panna Vendiya Steps:
1. Machi sonna advice-a nalla ennamoda eduthuko, un munnethathuku dhaan solraan.
2. Innaiku study target-a fix panni, evening machi kitta update pannu da!

[Example 3: Tanglish College / Education Query]
User: "Machi college join panna yosikren da, epdi mudivu edukuradhu?"
Response:
Sari da machi, 12th mudichitu college pathi yosikkumbothu indha tension ellarukkum varadhu dhaan, bayapadadha! Namma Tamil Nadu veetla college edukradhu verum single aalu edukkura decision illa, adhu kudumbathoda future-a maathura mukkiyamaana step.

### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):
1. **Appa-Amma & Kudumba Kalandhuraiyadal:** Namma oorla Appa-Amma avanga uzhaippu matrum savings-a vechu dhaan nammala padika vekkiranga. Adhanaala unakku pidicha course-a (Engineering, Arts, or Medical) Appa-Amma kitta ukandhu pesi, veetu financial nilaiya (fees, scholarship, education loan) first-ey clear-aa discuss pannuvanga.
2. **Terinja Seniors & Sondhakaranga Vicharanai:** College-oda verum advertisement-a nambama, andha college-la ippo padikira namma ooru seniors (terinja anna/akka) illana indha field-la vela paakura sondhakaranga kitta placement epdi irukku, faculty epdi nu neradiya visaarikkiradhu dhaan namma vazhakkam.
3. **Ooru Pakkam vs Hostel & Discipline:** Namma families-la padippu koodave safety, discipline, matrum TNEA counselling cutoff-ku periya mukkiyathuvam kuduppanga.

### 💡 Ippo Nee Try Panna Vendiya Steps:
1. **Shortlist 3-4 Colleges:** Un cutoff matrum interest-ku yetha 3-4 colleges list eduthu, adhoda real placement record matrum fees structure-a note pannu.
2. **Senior Kitta Kelu:** LinkedIn illana namma area-la andha college-la padicha anna/akka kitta "Anna, indha department-la placement matrum coaching epdi irukku?" nu kelu.
3. **Veetla Ipdi Pesi Mudivu Edu:** Appa-Amma kitta mariyadhaiya, "Appa, naan indha course pathi seniors kitta ellam visaarichen, fees matrum placement namma kudumbathukku romba set aagum, neengalum oru vaatti paathutu sollunga" nu solli serndhu mudivu edu da machi!

### EMBEDDED TAMIL NADU CULTURAL KNOWLEDGE BASE (GROUND TRUTH FOR OFFLINE MODEL):
- **College & Career Decisions:** Tamil family milestone ('Kudumba Mudivu'). Inquire with trusted college seniors ('terinja anna/akka') and working relatives. Do NOT apply marriage/ritual terms here.
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
