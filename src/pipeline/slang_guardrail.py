"""Tamil & Tanglish Regional Slang & Cultural Tradition Guardrail.

Detects sub-regional dialects and cultural tradition triggers across Tamil Nadu
to guide LLM generation with authentic ethnographic context rather than dry slang reports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.config import BASE_DIR

DATASET_PATH = BASE_DIR / "data" / "datasets" / "tn_regional_traditions_tanglish.json"


class MatchedTradition(BaseModel):
    """Contextual metadata regarding a detected Tamil cultural practice."""

    tradition_id: str
    tradition_name: str
    region: str
    domain: str
    little_tradition_reality: str
    expected_explanation_guideline: str
    great_tradition_misconception: Optional[str] = None
    tanglish_process_explanation: Optional[str] = None
    tanglish_action_steps: list[str] = Field(default_factory=list)


class CulturalContextPacket(BaseModel):
    """Internal middleware packet injected into the LLM generation context."""

    detected_region: str = "Pan-Tamil Nadu / General"
    dialect_name: str = "Colloquial Tamil / Tanglish"
    detected_slang_markers: list[str] = Field(default_factory=list)
    matched_traditions: list[MatchedTradition] = Field(default_factory=list)
    politeness_register: str = "warm_peer"
    status: str = "pass"
    cultural_context_instruction: str = ""


# Comprehensive Two-Tier Sub-Regional Tanglish Lexicon
# Tier 1: Pan-Tamil youth / colloquial markers (Weight = 1, Region = "Pan-Tamil Nadu")
# Tier 2: High-Precision Regional Dialect Markers (Weight = 3, locks specific sub-region)
SUB_REGIONAL_LEXICON: dict[str, dict[str, Any]] = {
    # =========================================================================
    # TIER 1: PAN-TAMIL NADU COLLOQUIAL & YOUTH MARKERS (Weight = 1)
    # =========================================================================
    "machi": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "macha": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "machan": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "mapla": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "maapla": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "da": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "di": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "pa": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "ma": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "affectionate_peer"},
    "bro": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "nanba": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "warm_peer"},
    "thala": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "affectionate_peer"},
    "thalapathy": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "affectionate_peer"},
    "thambi": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "affectionate_junior"},
    "thangachi": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "affectionate_junior"},
    "anna": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_brother_respect"},
    "anne": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_brother_respect"},
    "akka": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_sister_respect"},
    "periyappa": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "periyamma": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "chithappa": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "chithi": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "mama": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "athai": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "paatti": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "thatha": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "kudumbam": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "familial_respect"},
    "sondham": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "familial_respect"},
    "uravumurai": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "familial_respect"},
    "periyavanga": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "elder_respect"},
    "sollu": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "casual_peer"},
    "sollunga": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "formal_respect"},
    "vaanga": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "formal_respect"},
    "ponga": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "formal_respect"},
    "irukku": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "neutral"},
    "irukanga": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "polite_respect"},
    "pannu": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "casual"},
    "panradhu": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "casual"},
    "panlam": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "collaborative"},
    "mudiyala": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "expressive"},
    "theriyala": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "expressive"},
    "puriyala": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "expressive"},
    "yosikiren": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "reflective"},
    "yosikren": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "reflective"},
    "paakalam": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "casual"},
    "kelu": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "direct"},
    "kekkalam": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "collaborative"},
    "veetla": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "domestic"},
    "oorla": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "regional"},
    "romba": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "expressive"},
    "konjam": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "moderate"},
    "nalla": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "positive"},
    "ketta": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "cautionary"},
    "sema": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "enthusiastic"},
    "vera level": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "enthusiastic_praise"},
    "mass": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "enthusiastic"},
    "pakka": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "approving"},
    "summa": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "casual"},
    "vetti": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "casual_slang"},
    "scene": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "playful"},
    "kavalai": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "empathetic"},
    "sandhosham": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "celebratory"},
    "bayam": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "empathetic"},
    "kandippa": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "affirmative"},
    "adhanaala": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "explanatory"},
    "apram": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "connective"},
    "ippo": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "temporal"},
    "eppo": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "temporal"},
    "enga": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "spatial"},
    "epdi": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "inquiry"},
    "yenna": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "inquiry"},
    "enna": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "inquiry"},
    "edhu": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "inquiry"},
    "yaaru": {"region": "Pan-Tamil Nadu", "tier": 1, "tone": "inquiry"},

    # Pan-Tamil Pan-Traditions (Cross-Cousin Kinship, Virundhombal, Education, Health)
    "mama ponnu": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "athai paiyan": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "muraiponnu": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "muraipaiyan": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "murai kalyanam": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "moraponnu": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "athai ponnu": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "mama paiyan": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "virundhombal": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "moral_duty"},
    "saapittu ponga": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "hospitality_etiquette"},
    "poyittu vaanga": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "protective_etiquette"},
    "thamboolam": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "respectful_traditional"},
    "vetrilai paaku": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "respectful_traditional"},
    "pen paathal": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "lifecycle_ritual"},
    "ponnu paaka": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "lifecycle_ritual"},
    "valaikappu": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "lifecycle_ritual"},
    "seemantham": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "lifecycle_ritual"},
    "paal kaachuradhu": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "domestic_ritual"},
    "grahapravesam": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HOSPITALITY-001", "tone": "domestic_ritual"},

    # Education / Career Pan-Tradition
    "college": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "academic_guidance"},
    "counselling": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "academic_guidance"},
    "tnea counselling": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "academic_guidance"},
    "tnea": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "academic_guidance"},
    "anna university": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "academic_guidance"},
    "placement": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "career_guidance"},
    "cutoff": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "academic_guidance"},
    "arrear": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-EDU-001", "tone": "academic_guidance"},

    # Healthcare / Wellness Pan-Tradition
    "sukku malli kashayam": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HEALTH-001", "tone": "home_wellness"},
    "sukku malli": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HEALTH-001", "tone": "home_wellness"},
    "paatti vaithiyam": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HEALTH-001", "tone": "home_wellness"},
    "milagu rasam": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HEALTH-001", "tone": "home_wellness"},
    "pathiyam": {"region": "Pan-Tamil Nadu", "tier": 1, "tradition": "TN-HEALTH-001", "tone": "home_wellness"},

    # =========================================================================
    # TIER 2: HIGH-PRECISION REGIONAL DIALECT MARKERS (Weight = 3)
    # =========================================================================

    # 1. KONGU NADU (Coimbatore, Erode, Tiruppur, Pollachi, Salem, Karur)
    "arasanikkal": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-001", "tone": "respectful_traditional"},
    "arumaikkarar": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-001", "tone": "respectful_traditional"},
    "mangala vaazhai": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-001", "tone": "traditional_festive"},
    "mangala vaazhthu": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-001", "tone": "traditional_festive"},
    "coimbatore kalyanam": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-001", "tone": "regional_festive"},
    "kongu kalyanam": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-001", "tone": "regional_festive"},
    "yenunga": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-002", "tone": "polite_respectful"},
    "enanga": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-002", "tone": "polite_respectful"},
    "ennanga": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-002", "tone": "polite_respectful"},
    "sowkiyama": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-002", "tone": "familial_warmth"},
    "sowkkiyama": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-002", "tone": "familial_warmth"},
    "arisi paruppu sadham": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-002", "tone": "hospitality_warmth"},
    "arisi paruppu": {"region": "Kongu Nadu", "tier": 2, "tradition": "TN-KONGU-002", "tone": "hospitality_warmth"},
    "pachapuli rasam": {"region": "Kongu Nadu", "tier": 2, "tone": "culinary_heritage"},
    "saapteengala": {"region": "Kongu Nadu", "tier": 2, "tone": "domestic_politeness"},
    "saapteenghala": {"region": "Kongu Nadu", "tier": 2, "tone": "domestic_politeness"},
    "solreenga": {"region": "Kongu Nadu", "tier": 2, "tone": "polite_verbal"},
    "vaanganna": {"region": "Kongu Nadu", "tier": 2, "tone": "polite_familial"},
    "ponganna": {"region": "Kongu Nadu", "tier": 2, "tone": "polite_familial"},
    "sollunganna": {"region": "Kongu Nadu", "tier": 2, "tone": "polite_familial"},
    "ammini": {"region": "Kongu Nadu", "tier": 2, "tone": "affectionate_respect"},
    "ammani": {"region": "Kongu Nadu", "tier": 2, "tone": "affectionate_respect"},
    "enna kannu": {"region": "Kongu Nadu", "tier": 2, "tone": "affectionate_warmth"},
    "kuttachi": {"region": "Kongu Nadu", "tier": 2, "tone": "affectionate_kin"},
    "appuchi": {"region": "Kongu Nadu", "tier": 2, "tone": "kinship_respect"},
    "apparu": {"region": "Kongu Nadu", "tier": 2, "tone": "kinship_respect"},
    "appatha": {"region": "Kongu Nadu", "tier": 2, "tone": "kinship_respect"},
    "ammasai": {"region": "Kongu Nadu", "tier": 2, "tone": "kinship_respect"},
    "ammachi": {"region": "Kongu Nadu", "tier": 2, "tone": "kinship_respect"},
    "avo": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "ivo": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "ekka": {"region": "Kongu Nadu", "tier": 2, "tone": "familial_warmth"},
    "emla": {"region": "Kongu Nadu", "tier": 2, "tone": "friendly_peer"},
    "poravu": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "ikkatta": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "akkatta": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "avathala": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "ivathala": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "orukka": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "marukka": {"region": "Kongu Nadu", "tier": 2, "tone": "colloquial_kongu"},
    "vellamai": {"region": "Kongu Nadu", "tier": 2, "tone": "agrarian_heritage"},
    "ootukku": {"region": "Kongu Nadu", "tier": 2, "tone": "domestic_colloquial"},
    "saapadu aakku": {"region": "Kongu Nadu", "tier": 2, "tone": "domestic_colloquial"},
    "vilakku podu": {"region": "Kongu Nadu", "tier": 2, "tone": "domestic_traditional"},

    # 2. CHENNAI / MADRAS BASHAI (North Chennai & Metro)
    "doulath ah suthuradhavitutu": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "doulath": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-002", "tone": "swagger_pride"},
    "dowlath": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-002", "tone": "swagger_pride"},
    "suthuradhavitutu": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "suthuradha": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "suthuradhu": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "suthaadhe": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "naina": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "elder_respect_colloquial"},
    "baama": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "peer_colloquial"},
    "beema": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "peer_colloquial"},
    "annathe": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "elder_respect_colloquial"},
    "maamu": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "friendly_peer"},
    "bemaani": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_slang"},
    "kasmal": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_slang"},
    "kasmalam": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_slang"},
    "bejaar": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "colloquial_distress"},
    "bejaaru": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "colloquial_distress"},
    "galatta": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_chaos"},
    "galaata": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_chaos"},
    "jalsa": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "celebratory"},
    "dhool": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "enthusiastic"},
    "dabaikaradhu": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "dabaai": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "aattaya podu": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "allakai": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "somberi": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_slang"},
    "jaadai": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "kabodhi": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_slang"},
    "kailaam": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "marana gaana": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-001", "tone": "subaltern_solace"},
    "saavu gaana": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-001", "tone": "subaltern_solace"},
    "parai adichu": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-001", "tone": "subaltern_solace"},
    "paraiyattam": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-001", "tone": "subaltern_solace"},
    "gaana": {"region": "Chennai / Madras Bashai", "tier": 2, "tradition": "TN-CHENNAI-001", "tone": "subaltern_art"},
    "vadaacurry": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "culinary_heritage"},
    "naastha": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "culinary_colloquial"},
    "kasimedu": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "regional_landmark"},
    "royapuram": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "regional_landmark"},
    "vyasarpadi": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "regional_landmark"},
    "mylapore": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "regional_landmark"},
    "merina": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "regional_landmark"},
    "kalachify": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "teasing"},
    "bluff": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "playful_slang"},
    "jujube": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "oc": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "benaa": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "figuru": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "casual_slang"},
    "kettavan": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "street_slang"},
    "attagasam": {"region": "Chennai / Madras Bashai", "tier": 2, "tone": "enthusiastic"},

    # 3. MADURAI & THEKKATHI BELT (Madurai, Dindigul, Theni, Sivaganga, Ramnad, Virudhunagar)
    "thaai maaman": {"region": "Madurai & Thekkathi", "tier": 2, "tradition": "TN-MADURAI-001", "tone": "familial_reverent"},
    "maaman seer": {"region": "Madurai & Thekkathi", "tier": 2, "tradition": "TN-MADURAI-001", "tone": "familial_reverent"},
    "kaadhu kuthu": {"region": "Madurai & Thekkathi", "tier": 2, "tradition": "TN-MADURAI-001", "tone": "lifecycle_ritual"},
    "manjal neerattu": {"region": "Madurai & Thekkathi", "tier": 2, "tradition": "TN-MADURAI-001", "tone": "lifecycle_ritual"},
    "pachai koodaram": {"region": "Madurai & Thekkathi", "tier": 2, "tradition": "TN-MADURAI-001", "tone": "lifecycle_ritual"},
    "parivattam": {"region": "Madurai & Thekkathi", "tier": 2, "tradition": "TN-MADURAI-001", "tone": "ritual_honor"},
    "angittu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "ingittu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "engittu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "anguttu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "inguttu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "embuttu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "imbuttu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "ambuttu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "yela": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "direct_peer"},
    "yele": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "direct_peer"},
    "yov": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "direct_peer"},
    "yenney": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "direct_peer"},
    "yekka": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "familial_warmth"},
    "verasa": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "vellaney": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "adhukaandi": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "idhukaandi": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "ennathukku": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "varainga": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "porainga": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "solrainga": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "seirainga": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "patraya podu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_thekkathi"},
    "soodhanama": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "cautious_care"},
    "kokka makka": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "exclamation"},
    "pangali": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "kinship_solidarity"},
    "thekkathi": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "regional_pride"},
    "kida vettu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "festive_feast"},
    "kari virundhu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "festive_feast"},
    "saami kida": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "sacred_offering"},
    "marudhu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "heritage_pride"},
    "veechu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "colloquial_style"},
    "jallikattu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "martial_heritage"},
    "manjuvirattu": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "martial_heritage"},
    "chithirai thiruvizha": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "sacred_festival"},
    "kallazhagar": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "sacred_festival"},
    "jigarthanda": {"region": "Madurai & Thekkathi", "tier": 2, "tone": "culinary_heritage"},

    # 4. TIRUNELVELI & THOOTHUKUDI (Nellai / Thamirabarani)
    "kodai vizha": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tradition": "TN-NELLAI-001", "tone": "folk_devotional"},
    "sudalai madan": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tradition": "TN-NELLAI-001", "tone": "ancestral_guardian"},
    "saamiattam": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tradition": "TN-NELLAI-001", "tone": "sacred_trance"},
    "saamiyadi": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tradition": "TN-NELLAI-001", "tone": "sacred_trance"},
    "kaniyan koothu": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tradition": "TN-NELLAI-001", "tone": "sacred_ritual_art"},
    "magudam": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tradition": "TN-NELLAI-001", "tone": "sacred_ritual_art"},
    "villu paattu": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tradition": "TN-NELLAI-001", "tone": "sacred_ritual_art"},
    "elai": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "warm_nellai_peer"},
    "ela": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "warm_nellai_peer"},
    "annachi": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "elder_brother_respect"},
    "yetti": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "affectionate_female_peer"},
    "etti": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "affectionate_female_peer"},
    "setha neram": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "colloquial_nellai"},
    "anthani": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "colloquial_nellai"},
    "choli": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "colloquial_nellai"},
    "soli": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "colloquial_nellai"},
    "saram": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "colloquial_nellai"},
    "vaariyal": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "domestic_colloquial"},
    "kurukka saanju": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "colloquial_nellai"},
    "isakki amman": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "sacred_guardian"},
    "pechi amman": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "sacred_guardian"},
    "karuppasamy": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "sacred_guardian"},
    "padaiyal": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "sacred_offering"},
    "suruttu": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "sacred_offering"},
    "aadu vettu": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "sacred_offering"},
    "sodhi": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "culinary_heritage"},
    "mapillai sodhi": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "culinary_heritage"},
    "iruttukadai": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "culinary_landmark"},
    "macron": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "culinary_landmark"},
    "parotta salna": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "culinary_heritage"},
    "thamirabarani": {"region": "Tirunelveli & Thoothukudi", "tier": 2, "tone": "river_heritage"},

    # 5. KANYAKUMARI (Nanjil Nadu)
    "makkale": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "affectionate_collective"},
    "potti": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "payal": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "payale": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "edey": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "enthiru": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "veenditu vaa": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "chakkai": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "thengu": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "thenga": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "poochai": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "ormai": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "etharai": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "appam": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "uchai": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "pazhanji": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "culinary_heritage"},
    "puriyam": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "thalla": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "familial_reverent"},
    "kootukaran": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "warm_peer"},
    "ulundhanchoru": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "culinary_heritage"},
    "pazham nurukku": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "culinary_heritage"},
    "avial": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "culinary_heritage"},
    "theeyal": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "culinary_heritage"},
    "kasavu pudavai": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "cultural_synthesis"},
    "kasavu": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "cultural_synthesis"},
    "marumakkathayam": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "heritage_lineage"},
    "nanjil": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tradition": "TN-NANJIL-001", "tone": "regional_pride"},
    "ayyavazhi": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "sacred_heritage"},
    "nizhal thangal": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "sacred_heritage"},
    "nagercoil": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "regional_landmark"},
    "marthandam": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "regional_landmark"},
    "choodu": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},
    "nalla chooda": {"region": "Kanyakumari / Nanjil Nadu", "tier": 2, "tone": "colloquial_nanjil"},

    # 6. CAUVERY DELTA & PUDUKKOTTAI (Thanjavur, Tiruvarur, Nagapattinam, Pudukkottai, Aranthangi)
    "moi virundhu": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "community_solidarity"},
    "moi note": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "indigenous_finance"},
    "moi ezhudhura": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "indigenous_finance"},
    "moi ezhudhudhal": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "indigenous_finance"},
    "rettippu moi": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "indigenous_finance"},
    "peravurani": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "regional_landmark"},
    "aranthangi": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "regional_landmark"},
    "pattukkottai": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tradition": "TN-DELTA-001", "tone": "regional_landmark"},
    "oi": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "colloquial_delta"},
    "oay": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "colloquial_delta"},
    "appachi": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "kinship_respect"},
    "vaangonna": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "polite_respect"},
    "uppukandam": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "culinary_heritage"},
    "aadi perukku": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "agrarian_festive"},
    "padinettam perukku": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "agrarian_festive"},
    "kaaveri": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "river_heritage"},
    "kollidam": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "river_heritage"},
    "samba": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "agrarian_heritage"},
    "kuruvai": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "agrarian_heritage"},
    "thalai vaazhai": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "hospitality_etiquette"},
    "degree kaapi": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "culinary_heritage"},
    "ashoka halwa": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "culinary_heritage"},
    "thiruvaiyaru": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "regional_landmark"},
    "patham": {"region": "Cauvery Delta & Pudukkottai", "tier": 2, "tone": "colloquial_delta"},

    # 7. CHETTINAD (Karaikudi, Devakottai, Kanadukathan, Sivaganga)
    "aachi": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "matriarchal_respect"},
    "chettiar": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "community_respect"},
    "seer varisai": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "stridhana_tradition"},
    "kavuni arisi": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "culinary_heritage"},
    "nagarathar": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "community_identity"},
    "pulli": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "community_lineage"},
    "kovil pirivu": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "kinship_exogamy"},
    "padaippu": {"region": "Chettinad", "tier": 2, "tradition": "TN-CHETTINAD-001", "tone": "ancestral_worship"},
    "vattakai": {"region": "Chettinad", "tier": 2, "tone": "traditional_vessel"},
    "paal paniyaram": {"region": "Chettinad", "tier": 2, "tone": "culinary_heritage"},
    "kandarappam": {"region": "Chettinad", "tier": 2, "tone": "culinary_heritage"},
    "seepu seedai": {"region": "Chettinad", "tier": 2, "tone": "culinary_heritage"},
    "kai murukku": {"region": "Chettinad", "tier": 2, "tone": "culinary_heritage"},
    "kandangi": {"region": "Chettinad", "tier": 2, "tone": "textile_heritage"},
    "valavu veedu": {"region": "Chettinad", "tier": 2, "tone": "architectural_heritage"},
    "magamai": {"region": "Chettinad", "tier": 2, "tone": "community_charity"},
    "athangudi": {"region": "Chettinad", "tier": 2, "tone": "craft_heritage"},

    # 8. VELLORE & NORTHERN ARCOT BELT (Thondaimandalam)
    "baasha": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "colloquial_northern"},
    "kaja": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "colloquial_northern"},
    "nikker": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "colloquial_northern"},
    "jasti": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "colloquial_northern"},
    "kammi": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "colloquial_northern"},
    "ishtam": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "colloquial_northern"},
    "annamalaiyar": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "sacred_heritage"},
    "girivalam": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "sacred_heritage"},
    "ambur biriyani": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "culinary_heritage"},
    "makkan peda": {"region": "Northern Belt / Vellore", "tier": 2, "tone": "culinary_heritage"},

    # 9. NILGIRIS HILL COMMUNITIES (Badaga, Kota, Toda)
    "hatti": {"region": "Nilgiris", "tier": 2, "tradition": "TN-NILGIRIS-001", "tone": "indigenous_village"},
    "hethai habba": {"region": "Nilgiris", "tier": 2, "tradition": "TN-NILGIRIS-001", "tone": "ancestral_mother_devotion"},
    "hethai amman": {"region": "Nilgiris", "tier": 2, "tradition": "TN-NILGIRIS-001", "tone": "ancestral_mother_devotion"},
    "hethai": {"region": "Nilgiris", "tier": 2, "tradition": "TN-NILGIRIS-001", "tone": "ancestral_mother_devotion"},
    "badaga": {"region": "Nilgiris", "tier": 2, "tradition": "TN-NILGIRIS-001", "tone": "indigenous_identity"},
    "toda": {"region": "Nilgiris", "tier": 2, "tone": "indigenous_identity"},
    "kota": {"region": "Nilgiris", "tier": 2, "tone": "indigenous_identity"},
    "seeme": {"region": "Nilgiris", "tier": 2, "tone": "territory_division"},
    "mund": {"region": "Nilgiris", "tier": 2, "tone": "toda_settlement"},
    "dhodda habba": {"region": "Nilgiris", "tier": 2, "tone": "community_feast"},
    "dhodda": {"region": "Nilgiris", "tier": 2, "tone": "indigenous_identity"},
}


class TamilSlangGuardrail:
    """Advanced dialect & cultural tradition guardrail for regional South Asian interactions."""

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.traditions_cache: dict[str, dict[str, Any]] = {}
        path = dataset_path or DATASET_PATH
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        self.traditions_cache[item["id"]] = item
            except Exception:
                pass

        # Sort multi-word phrases first to prevent sub-string clash
        sorted_keys = sorted(SUB_REGIONAL_LEXICON.keys(), key=len, reverse=True)
        self._patterns = {
            token: re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.IGNORECASE)
            for token in sorted_keys
        }

    def detect_slang(self, text: str) -> list[dict[str, Any]]:
        """Return registered slang and tradition markers found as whole words or phrases."""
        detected = []
        lowered = text.lower()
        for token, pattern in self._patterns.items():
            if pattern.search(lowered):
                meta = SUB_REGIONAL_LEXICON[token]
                detected.append({
                    "slang": token,
                    "response_tone": meta.get("tone", "warm_peer"),
                    **meta,
                })
        return detected

    def analyze_query(self, user_input: str) -> CulturalContextPacket:
        """Analyze query, extract sub-region, slang markers, and matched cultural traditions."""
        detected_slangs = self.detect_slang(user_input)
        if not detected_slangs:
            return CulturalContextPacket(
                detected_region="Pan-Tamil Nadu / General",
                dialect_name="Colloquial Tamil / Tanglish",
                detected_slang_markers=[],
                matched_traditions=[],
                politeness_register="respectful_friendly",
                status="pass",
                cultural_context_instruction="Address the user warmly in natural Tamil/Tanglish respecting customary decorum.",
            )

        # Separate Tier 1 (Pan-Tamil, weight 1) vs Tier 2 (Regional dialect, weight 3)
        region_scores: dict[str, int] = {}
        matched_traditions_list: list[MatchedTradition] = []
        seen_traditions: set[str] = set()
        tier2_detected: bool = False

        for item in detected_slangs:
            reg = item.get("region", "Pan-Tamil Nadu")
            tier = item.get("tier", 2)
            weight = 1 if tier == 1 or reg == "Pan-Tamil Nadu" else 3
            if tier == 2 and reg != "Pan-Tamil Nadu":
                tier2_detected = True

            region_scores[reg] = region_scores.get(reg, 0) + weight

            trad_id = item.get("tradition")
            if trad_id and trad_id not in seen_traditions and trad_id in self.traditions_cache:
                t = self.traditions_cache[trad_id]
                matched_traditions_list.append(
                    MatchedTradition(
                        tradition_id=trad_id,
                        tradition_name=t["tradition_name"],
                        region=t["region"],
                        domain=t["domain"],
                        little_tradition_reality=t["little_tradition_reality"],
                        expected_explanation_guideline=t["expected_explanation_guideline"],
                        great_tradition_misconception=t.get("great_tradition_misconception"),
                        tanglish_process_explanation=t.get("tanglish_process_explanation"),
                        tanglish_action_steps=t.get("tanglish_action_steps", []),
                    )
                )
                seen_traditions.add(trad_id)

        # Sub-region resolution: only assign specific regional dialect if Tier 2 marker present
        if tier2_detected:
            # Regional candidates excluding Pan-Tamil Nadu
            sub_region_candidates = {r: s for r, s in region_scores.items() if r != "Pan-Tamil Nadu"}
            if sub_region_candidates:
                primary_region = max(sub_region_candidates.items(), key=lambda x: x[1])[0]
            else:
                primary_region = "Pan-Tamil Nadu / General"
        else:
            primary_region = "Pan-Tamil Nadu / General"

        marker_names = [item["slang"] for item in detected_slangs]
        all_tones = [item.get("tone") or item.get("response_tone") or "warm_peer" for item in detected_slangs]
        tones_str = ", ".join(dict.fromkeys(all_tones))
        primary_tone = all_tones[0]

        # Instruction for the Generator to EXPLAIN AND VALIDATE, NOT produce a dictionary
        instruction_lines = [
            f"[Cultural Alignment Active]: Inferred Sub-Region: {primary_region}.",
            f"Conversational Tone: Match the user's natural {tones_str} register.",
            "CRITICAL: Do NOT output a slang dictionary or translation analysis.",
        ]

        if matched_traditions_list:
            t = matched_traditions_list[0]
            instruction_lines.append(f"Primary Tradition Identified: {t.tradition_name}")
            instruction_lines.append(f"Cultural Reality: {t.little_tradition_reality}")
            instruction_lines.append(f"Validation Guideline: {t.expected_explanation_guideline}")
            if t.tanglish_process_explanation:
                instruction_lines.append(f"Tanglish Customary Process: {t.tanglish_process_explanation}")

        instruction = "\n".join(instruction_lines)

        dialect_label = (
            f"{primary_region} Dialect"
            if primary_region != "Pan-Tamil Nadu / General"
            else "Pan-Tamil Colloquial / Youth Tanglish"
        )

        return CulturalContextPacket(
            detected_region=primary_region,
            dialect_name=dialect_label,
            detected_slang_markers=marker_names,
            matched_traditions=matched_traditions_list,
            politeness_register=primary_tone,
            status="adjusted",
            cultural_context_instruction=instruction,
        )

    def alignment_middleware(self, user_input: str) -> dict[str, Any]:
        """Backward-compatible middleware interface returning dict representation."""
        packet = self.analyze_query(user_input)
        packet_dict = packet.model_dump()
        return {
            "status": packet.status,
            "packet": packet_dict,
            "detected_slangs": [{"slang": m} for m in packet.detected_slang_markers],
            "cultural_context": packet.cultural_context_instruction,
            "detected_region": packet.detected_region,
            "matched_traditions": [t.model_dump() for t in packet.matched_traditions],
            "adjusted_prompt": (
                f"{packet.cultural_context_instruction}\n\nUser Query: {user_input}"
                if packet.status == "adjusted"
                else user_input
            ),
        }
