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


class CulturalContextPacket(BaseModel):
    """Internal middleware packet injected into the LLM generation context."""

    detected_region: str = "Pan-Tamil Nadu / General"
    dialect_name: str = "Colloquial Tamil / Tanglish"
    detected_slang_markers: list[str] = Field(default_factory=list)
    matched_traditions: list[MatchedTradition] = Field(default_factory=list)
    politeness_register: str = "warm_peer"
    status: str = "pass"
    cultural_context_instruction: str = ""


# Comprehensive Sub-Regional Tanglish Lexicon & Tradition Triggers
SUB_REGIONAL_LEXICON: dict[str, dict[str, Any]] = {
    # 1. Kongu Nadu
    "arasanikkal": {"region": "Kongu Nadu", "tradition": "TN-KONGU-001", "tone": "respectful_traditional"},
    "arumaikkarar": {"region": "Kongu Nadu", "tradition": "TN-KONGU-001", "tone": "respectful_traditional"},
    "mangala vaazhai": {"region": "Kongu Nadu", "tradition": "TN-KONGU-001", "tone": "traditional_festive"},
    "yenunga": {"region": "Kongu Nadu", "tradition": "TN-KONGU-002", "tone": "polite_respectful"},
    "enanga": {"region": "Kongu Nadu", "tradition": "TN-KONGU-002", "tone": "polite_respectful"},
    "sowkiyama": {"region": "Kongu Nadu", "tradition": "TN-KONGU-002", "tone": "familial_warmth"},
    "sowkkiyama": {"region": "Kongu Nadu", "tradition": "TN-KONGU-002", "tone": "familial_warmth"},
    "saapteengala": {"region": "Kongu Nadu", "tone": "domestic_politeness"},
    "saapteenghala": {"region": "Kongu Nadu", "tone": "domestic_politeness"},
    "solreenga": {"region": "Kongu Nadu", "tone": "polite_verbal"},
    "avo": {"region": "Kongu Nadu", "tone": "colloquial_kongu"},
    "ivo": {"region": "Kongu Nadu", "tone": "colloquial_kongu"},
    "ekka": {"region": "Kongu Nadu", "tone": "familial_warmth"},
    "emla": {"region": "Kongu Nadu", "tone": "friendly_peer"},
    "ammini": {"region": "Kongu Nadu", "tone": "affectionate_respect"},
    "enna kannu": {"region": "Kongu Nadu", "tone": "affectionate_warmth"},
    "kannu": {"region": "Kongu Nadu", "tone": "affectionate_warmth"},
    "saapteenghala": {"region": "Kongu Nadu", "tone": "polite_respectful"},

    # 2. Madurai & Thekkathi Belt
    "thaai maaman": {"region": "Madurai & Thekkathi", "tradition": "TN-MADURAI-001", "tone": "familial_reverent"},
    "maaman seer": {"region": "Madurai & Thekkathi", "tradition": "TN-MADURAI-001", "tone": "familial_reverent"},
    "kaadhu kuthu": {"region": "Madurai & Thekkathi", "tradition": "TN-MADURAI-001", "tone": "lifecycle_ritual"},
    "manjal neerattu": {"region": "Madurai & Thekkathi", "tradition": "TN-MADURAI-001", "tone": "lifecycle_ritual"},
    "angittu": {"region": "Madurai & Thekkathi", "tone": "colloquial_thekkathi"},
    "ingittu": {"region": "Madurai & Thekkathi", "tone": "colloquial_thekkathi"},
    "embuttu": {"region": "Madurai & Thekkathi", "tone": "colloquial_thekkathi"},
    "pangali": {"region": "Madurai & Thekkathi", "tone": "kinship_solidarity"},
    "thekkathi": {"region": "Madurai & Thekkathi", "tone": "regional_pride"},

    # 3. Dravidian Kinship & Cross-Cousin Marriage
    "mama ponnu": {"region": "Pan-Tamil Nadu", "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "athai paiyan": {"region": "Pan-Tamil Nadu", "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "muraiponnu": {"region": "Pan-Tamil Nadu", "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "muraipaiyan": {"region": "Pan-Tamil Nadu", "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "murai kalyanam": {"region": "Pan-Tamil Nadu", "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},
    "moraponnu": {"region": "Pan-Tamil Nadu", "tradition": "TN-SOUTH-002", "tone": "kinship_normative"},

    # 4. Cauvery Delta & Pudukkottai / Chettinad
    "moi virundhu": {"region": "Cauvery Delta & Pudukkottai", "tradition": "TN-DELTA-001", "tone": "community_solidarity"},
    "moi note": {"region": "Cauvery Delta & Pudukkottai", "tradition": "TN-DELTA-001", "tone": "indigenous_finance"},
    "moi ezhudhura": {"region": "Cauvery Delta & Pudukkottai", "tradition": "TN-DELTA-001", "tone": "indigenous_finance"},
    "peravurani": {"region": "Cauvery Delta & Pudukkottai", "tradition": "TN-DELTA-001", "tone": "regional_landmark"},
    "aranthangi": {"region": "Cauvery Delta & Pudukkottai", "tradition": "TN-DELTA-001", "tone": "regional_landmark"},
    "aachi": {"region": "Chettinad", "tradition": "TN-CHETTINAD-001", "tone": "matriarchal_respect"},
    "chettiar": {"region": "Chettinad", "tradition": "TN-CHETTINAD-001", "tone": "community_respect"},
    "seer varisai": {"region": "Chettinad", "tradition": "TN-CHETTINAD-001", "tone": "stridhana_tradition"},
    "kavuni arisi": {"region": "Chettinad", "tradition": "TN-CHETTINAD-001", "tone": "culinary_heritage"},

    # 5. Chennai / Madras Bashai
    "marana gaana": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-001", "tone": "subaltern_solace"},
    "saavu gaana": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-001", "tone": "subaltern_solace"},
    "parai adichu": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-001", "tone": "subaltern_solace"},
    "doulath ah suthuradhavitutu": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "doulath": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-002", "tone": "swagger_pride"},
    "dowlath": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-002", "tone": "swagger_pride"},
    "suthuradhavitutu": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "suthuradha": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "suthuradhu": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "suthaadhe": {"region": "Chennai / Madras Bashai", "tradition": "TN-CHENNAI-002", "tone": "street_camaraderie"},
    "alaparai": {"region": "Chennai / Madras Bashai", "tone": "overaction"},
    "kalachify": {"region": "Chennai / Madras Bashai", "tone": "teasing"},
    "vetti": {"region": "Chennai / Madras Bashai", "tone": "casual_slang"},
    "macha": {"region": "Chennai / Madras Bashai", "tone": "warm_peer"},
    "machi": {"region": "Chennai / Madras Bashai", "tone": "warm_peer"},
    "machan": {"region": "Chennai / Madras Bashai", "tone": "warm_peer"},
    "maamu": {"region": "Chennai / Madras Bashai", "tone": "friendly_peer"},
    "naina": {"region": "Chennai / Madras Bashai", "tone": "elder_respect_colloquial"},
    "gethu": {"region": "Chennai / Madras Bashai", "tone": "pride_style"},
    "bemaani": {"region": "Chennai / Madras Bashai", "tone": "playful_slang"},
    "bejaar": {"region": "Chennai / Madras Bashai", "tone": "colloquial_distress"},
    "kasmal": {"region": "Chennai / Madras Bashai", "tone": "playful_slang"},
    "dei": {"region": "Chennai / Madras Bashai", "tone": "informal_peer"},
    "vera level": {"region": "Pan-Tamil Nadu", "tone": "enthusiastic_praise"},

    # 6. Tirunelveli & Thoothukudi (Nellai / Thamirabarani)
    "kodai vizha": {"region": "Tirunelveli & Thoothukudi", "tradition": "TN-NELLAI-001", "tone": "folk_devotional"},
    "sudalai madan": {"region": "Tirunelveli & Thoothukudi", "tradition": "TN-NELLAI-001", "tone": "ancestral_guardian"},
    "saamiattam": {"region": "Tirunelveli & Thoothukudi", "tradition": "TN-NELLAI-001", "tone": "sacred_trance"},
    "kaniyan koothu": {"region": "Tirunelveli & Thoothukudi", "tradition": "TN-NELLAI-001", "tone": "sacred_ritual_art"},
    "elai": {"region": "Tirunelveli & Thoothukudi", "tone": "warm_nellai_peer"},
    "annachi": {"region": "Tirunelveli & Thoothukudi", "tone": "elder_brother_respect"},
    "choli": {"region": "Tirunelveli & Thoothukudi", "tone": "colloquial_nellai"},
    "soli": {"region": "Tirunelveli & Thoothukudi", "tone": "colloquial_nellai"},

    # 7. Kanyakumari (Nanjil Nadu)
    "makkale": {"region": "Kanyakumari / Nanjil Nadu", "tone": "affectionate_collective"},
    "potti": {"region": "Kanyakumari / Nanjil Nadu", "tone": "colloquial_nanjil"},
    "payal": {"region": "Kanyakumari / Nanjil Nadu", "tone": "colloquial_nanjil"},
    "enthiru": {"region": "Kanyakumari / Nanjil Nadu", "tone": "colloquial_nanjil"},
    "nanjil": {"region": "Kanyakumari / Nanjil Nadu", "tradition": "TN-NANJIL-001", "tone": "regional_pride"},
    "kasavu pudavai": {"region": "Kanyakumari / Nanjil Nadu", "tradition": "TN-NANJIL-001", "tone": "cultural_synthesis"},

    # 8. Nilgiris Hill Communities
    "hatti": {"region": "Nilgiris", "tradition": "TN-NILGIRIS-001", "tone": "indigenous_village"},
    "hethai habba": {"region": "Nilgiris", "tradition": "TN-NILGIRIS-001", "tone": "ancestral_mother_devotion"},
    "badaga": {"region": "Nilgiris", "tradition": "TN-NILGIRIS-001", "tone": "indigenous_identity"},
    "toda": {"region": "Nilgiris", "tone": "indigenous_identity"},
    "kota": {"region": "Nilgiris", "tone": "indigenous_identity"},

    # 9. Universal Hospitality & Honorifics
    "virundhombal": {"region": "Pan-Tamil Nadu", "tradition": "TN-HOSPITALITY-001", "tone": "moral_duty"},
    "saapittu ponga": {"region": "Pan-Tamil Nadu", "tradition": "TN-HOSPITALITY-001", "tone": "hospitality_etiquette"},
    "poyittu vaanga": {"region": "Pan-Tamil Nadu", "tradition": "TN-HOSPITALITY-001", "tone": "protective_etiquette"},
    "vaanga": {"region": "Pan-Tamil Nadu", "tone": "formal_respect"},
    "ponga": {"region": "Pan-Tamil Nadu", "tone": "formal_respect"},
    "sollunga": {"region": "Pan-Tamil Nadu", "tone": "formal_respect"},
    "thambi": {"region": "Pan-Tamil Nadu", "tone": "affectionate_junior"},
    "akka": {"region": "Pan-Tamil Nadu", "tone": "elder_sister_respect"},
    "anna": {"region": "Pan-Tamil Nadu", "tone": "elder_brother_respect"},
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
                detected.append({"slang": token, "response_tone": meta.get("tone", "warm_peer"), **meta})
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

        # Region vote from detected markers
        region_counts: dict[str, int] = {}
        matched_traditions_list: list[MatchedTradition] = []
        seen_traditions: set[str] = set()

        for item in detected_slangs:
            reg = item.get("region", "Pan-Tamil Nadu")
            region_counts[reg] = region_counts.get(reg, 0) + 1

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
                    )
                )
                seen_traditions.add(trad_id)

        primary_region = max(region_counts.items(), key=lambda x: x[1])[0]
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

        instruction = "\n".join(instruction_lines)

        return CulturalContextPacket(
            detected_region=primary_region,
            dialect_name=f"{primary_region} Dialect",
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
