import sys
from pathlib import Path

# Add backend directory to sys.path so 'src' imports resolve from root or backend directory
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import json
from unittest.mock import patch, MagicMock
import pytest

from src.pipeline.generator import detect_language_mode, detect_domain_category, generate_candidate
from src.pipeline.evaluator import (
    CulturalEvaluationResult,
    evaluate_candidate,
    _clean_json_string,
)
from src.pipeline.interceptor import run_cultural_alignment_pipeline
from src.pipeline.slang_guardrail import TamilSlangGuardrail
from src.retriever.search import format_cultural_context
from src.retriever.ingest import chunk_text


def test_detect_language_mode():
    """Test language and code-mixing detection."""
    tamil_query = "பொங்கல் திருவிழாவை எவ்வாறு கொண்டாடுவது?"
    assert detect_language_mode(tamil_query) == "Tamil (தமிழ்)"

    tanglish_query = "Machi, how to greet my elder uncle properly da?"
    assert detect_language_mode(tanglish_query) == "Tanglish (Tamil in Latin script / Code-mixed)"

    english_query = "What are the common social etiquettes when visiting a family in South India?"
    assert detect_language_mode(english_query) == "Regional English"


def test_tamil_slang_guardrail_detects_phrases_and_respects_word_boundaries():
    guardrail = TamilSlangGuardrail()

    result = guardrail.alignment_middleware("Machi, this output is vera level!")

    assert result["status"] == "adjusted"
    assert set(item["slang"] for item in result["detected_slangs"]) == {"machi", "vera level"}
    assert "warm_peer" in result["cultural_context"]
    assert guardrail.detect_slang("machinery and level") == []


def test_tamil_slang_guardrail_passes_neutral_input():
    result = TamilSlangGuardrail().alignment_middleware("How does sorting work?")

    assert result["status"] == "pass"
    assert result["detected_slangs"] == []
    assert result["adjusted_prompt"] == "How does sorting work?"


def test_detect_domain_category():
    """Test domain categorization into Healthcare, Banking/Finance, or Commonsense."""
    health_q = "What diet is advised for a pregnant woman in Tamil tradition?"
    assert detect_domain_category(health_q) == "Healthcare"

    finance_q = "How do women participate in local chit fund (seetu) groups?"
    assert detect_domain_category(finance_q) == "Banking / Finance"

    social_q = "How to invite relatives for Pongal festival?"
    assert detect_domain_category(social_q) == "Everyday Commonsense"


def test_cas_formula_computation():
    """Test Cultural Alignment Score (CAS) formula computation."""
    # CAS = 0.30*0.90 + 0.35*0.85 + 0.35*0.95 = 0.27 + 0.2975 + 0.3325 = 0.90
    cas = CulturalEvaluationResult.compute_cas(0.90, 0.85, 0.95, domain_risk_detected=False)
    assert cas == 0.90

    # Risk detected -> penalty = 1.0 -> clamped to 0.0
    cas_risk = CulturalEvaluationResult.compute_cas(0.90, 0.85, 0.95, domain_risk_detected=True)
    assert cas_risk == 0.0


def test_chunk_text():
    """Test text chunking with overlap."""
    words = "word " * 600
    chunks = chunk_text(words, chunk_size=512, chunk_overlap=64)
    assert len(chunks) >= 2


def test_cultural_evaluation_model_validation():
    """Test CulturalEvaluationResult validation and bounds."""
    data = {
        "context_understanding": 0.85,
        "social_appropriateness": 0.90,
        "regional_relevance": 0.80,
        "domain_risk_detected": False,
        "alignment_score": 0.85,
        "verdict": "PASS",
        "critique": "Solid cultural grounding.",
        "strengths": ["Polite honorifics"],
        "weaknesses": [],
    }
    result = CulturalEvaluationResult(**data)
    assert result.alignment_score == 0.85
    assert result.overall_score == 0.85
    assert not result.domain_risk_detected
    assert not result.stereotype_risk_flag
    assert result.verdict == "PASS"

    # Test out of bounds validation error
    with pytest.raises(Exception):
        CulturalEvaluationResult(
            context_understanding=1.5,
            social_appropriateness=0.8,
            regional_relevance=0.8,
            domain_risk_detected=False,
            alignment_score=0.8,
            critique="Test",
        )


def test_clean_json_string():
    """Test markdown extraction from LLM response."""
    raw_md = "```json\n{\"test\": 123}\n```"
    assert _clean_json_string(raw_md) == '{"test": 123}'

    raw_plain = '{"test": 123}'
    assert _clean_json_string(raw_plain) == '{"test": 123}'


@patch("src.pipeline.generator.generate_completion")
def test_generate_candidate_with_context(mock_llm):
    """Test candidate generator calling wrapper with proper messages."""
    mock_llm.return_value = "வணக்கம்! நீங்கள் நலம் தானா?"

    response = generate_candidate(
        query="Greeting for elder",
        cultural_context="Use respectful honorifics.",
    )

    assert response == "வணக்கம்! நீங்கள் நலம் தானா?"
    mock_llm.assert_called_once()
    call_args = mock_llm.call_args[1]
    assert "Context:" in call_args["messages"][0]["content"]
    assert "Region: Tamil Nadu, India" in call_args["messages"][0]["content"]


@patch("src.pipeline.evaluator.generate_completion")
def test_evaluate_candidate(mock_llm):
    """Test evaluator parsing structured JSON from Judge LLM."""
    mock_llm.return_value = json.dumps({
        "context_understanding": 0.88,
        "social_appropriateness": 0.92,
        "regional_relevance": 0.85,
        "stereotype_risk_flag": False,
        "overall_score": 0.88,
        "critique": "Exemplary honorific address.",
        "strengths": ["Used vaanga/ponga"],
        "weaknesses": [],
    })

    eval_result = evaluate_candidate(
        query="Invite uncle for lunch",
        candidate="மாமா, வீட்டிற்கு சாப்பிட வாருங்கள்.",
    )

    assert eval_result.overall_score == 0.88
    assert eval_result.social_appropriateness == 0.92
    assert not eval_result.stereotype_risk_flag


@patch("src.pipeline.interceptor.query_cultural_context")
@patch("src.pipeline.interceptor.generate_candidate")
@patch("src.pipeline.interceptor.evaluate_candidate")
def test_alignment_loop_first_pass_success(mock_eval, mock_gen, mock_rag):
    """Test scenario where initial candidate immediately passes threshold."""
    mock_rag.return_value = [{"content": "Context snippet", "metadata": {"title": "Doc1"}}]
    mock_gen.return_value = "Respectful candidate 1"
    mock_eval.return_value = CulturalEvaluationResult(
        context_understanding=0.9,
        social_appropriateness=0.9,
        regional_relevance=0.85,
        stereotype_risk_flag=False,
        overall_score=0.88,
        critique="Looks great.",
        strengths=["Polite"],
        weaknesses=[],
    )

    output = run_cultural_alignment_pipeline(
        query="Invite uncle",
        score_threshold=0.75,
        max_loops=2,
    )

    assert output.passed_threshold is True
    assert output.iterations_used == 1
    assert output.final_response == "Respectful candidate 1"
    assert mock_gen.call_count == 1
    assert mock_eval.call_count == 1


@patch("src.pipeline.interceptor.query_cultural_context")
@patch("src.pipeline.interceptor.generate_candidate")
@patch("src.pipeline.interceptor.evaluate_candidate")
def test_alignment_loop_feedback_regeneration(mock_eval, mock_gen, mock_rag):
    """Test scenario where initial candidate fails threshold, feedback loop triggers, second passes."""
    mock_rag.return_value = []
    # 1st draft fails, 2nd draft generated passes
    mock_gen.side_effect = ["Initial casual response", "Refined respectful response"]

    eval_fail = CulturalEvaluationResult(
        context_understanding=0.5,
        social_appropriateness=0.5,
        regional_relevance=0.5,
        stereotype_risk_flag=False,
        overall_score=0.5,
        critique="Too casual. Use vaanga/ponga.",
        strengths=[],
        weaknesses=["Used nee instead of neenga"],
    )
    eval_pass = CulturalEvaluationResult(
        context_understanding=0.85,
        social_appropriateness=0.90,
        regional_relevance=0.80,
        stereotype_risk_flag=False,
        overall_score=0.85,
        critique="Addressed critique well.",
        strengths=["Polite plural verbs"],
        weaknesses=[],
    )
    mock_eval.side_effect = [eval_fail, eval_pass]

    output = run_cultural_alignment_pipeline(
        query="Invite elder relative",
        score_threshold=0.75,
        max_loops=2,
    )

    assert output.passed_threshold is True
    assert output.iterations_used == 2
    assert output.final_response == "Refined respectful response"
    assert mock_gen.call_count == 2
    assert mock_eval.call_count == 2


def test_format_cultural_context():
    """Test formatting retrieved snippets."""
    snippets = [
        {"content": "Snippet 1 text", "metadata": {"title": "Doc A"}},
        {"content": "Snippet 2 text", "metadata": {"title": "Doc B"}},
    ]
    formatted = format_cultural_context(snippets)
    assert "[1] Doc A:" in formatted
    assert "Snippet 1 text" in formatted
    assert "[2] Doc B:" in formatted


def test_sub_regional_tradition_detection_kongu():
    """Test Tanglish guardrail detecting Kongu Arasanikkal tradition."""
    guardrail = TamilSlangGuardrail()
    query = "Enoda ekka kalyanam Coimbatore la nadakudhu, iyer illama arumaikkarar vechu arasanikkal nattu panranga"
    packet = guardrail.analyze_query(query)
    assert packet.detected_region == "Kongu Nadu"
    assert "arasanikkal" in packet.detected_slang_markers
    assert "arumaikkarar" in packet.detected_slang_markers
    assert len(packet.matched_traditions) > 0
    assert packet.matched_traditions[0].tradition_name == "Arasanikkal & Arumaikkarar Wedding Tradition (Kongu Kalyanam)"


def test_sub_regional_tradition_detection_delta_moi_virundhu():
    """Test Tanglish guardrail detecting Delta / Pudukkottai Moi Virundhu tradition."""
    guardrail = TamilSlangGuardrail()
    query = "Moi virundhu la cash collect panradhu scam ah illa culture ah machi?"
    packet = guardrail.analyze_query(query)
    assert packet.detected_region == "Cauvery Delta & Pudukkottai"
    assert "moi virundhu" in packet.detected_slang_markers
    assert len(packet.matched_traditions) > 0
    assert "Moi Virundhu" in packet.matched_traditions[0].tradition_name


def test_sub_regional_tradition_detection_nellai_kodai():
    """Test Tanglish guardrail detecting Nellai Kodai Vizha Sudalai Madan tradition."""
    guardrail = TamilSlangGuardrail()
    query = "Elai, enga Tirunelveli ooru Sudalai Madan kovil Kodai vizha la aadu vetti suruttu vekkuranga"
    packet = guardrail.analyze_query(query)
    assert packet.detected_region == "Tirunelveli & Thoothukudi"
    assert "elai" in packet.detected_slang_markers
    assert "sudalai madan" in packet.detected_slang_markers
    assert "kodai vizha" in packet.detected_slang_markers


@patch("src.pipeline.generator.generate_completion")
def test_mediated_generator_prompt_no_dry_dictionary(mock_llm):
    """Test MediatedResponseGenerator instructs model to explain tradition, not produce dry dictionary."""
    from src.pipeline.generator import MediatedResponseGenerator
    from src.pipeline.interceptor import CulturalClassifier

    mock_llm.return_value = "Detailed cultural explanation..."
    gen = MediatedResponseGenerator()
    classifier = CulturalClassifier()

    query = "Moi virundhu la cash collect panradhu scam ah?"
    profile = classifier.classify(query)
    res = gen.generate(query, profile, [{"content": "Moi virundhu is microfinance"}])

    assert res == "Detailed cultural explanation..."
    call_args = mock_llm.call_args[1]
    prompt_sent = call_args["messages"][1]["content"]
    assert "Do NOT output dry dictionary fields" in prompt_sent
    assert "Name & acknowledge the specific regional tradition" in prompt_sent
    assert "Validate why this practice is culturally normative" in prompt_sent


@patch("src.pipeline.generator.generate_completion")
def test_tanglish_college_decision_classification_and_headers(mock_llm):
    """Test Tanglish college query classifies as Pan-Tamil Nadu and uses Tanglish headers."""
    from src.pipeline.generator import MediatedResponseGenerator, detect_domain_category, detect_language_mode
    from src.pipeline.interceptor import CulturalClassifier

    mock_llm.return_value = "Detailed Tanglish response..."
    gen = MediatedResponseGenerator()
    classifier = CulturalClassifier()

    query = "Machi college join panna yosikren da, epdi mudivu edukuradhu?"
    assert detect_language_mode(query) == "Tanglish (Tamil in Latin script / Code-mixed)"
    assert detect_domain_category(query) == "Education / Career"

    profile = classifier.classify(query)
    assert profile.sub_region == "Pan-Tamil Nadu / General"
    assert "machi" in profile.slang_tokens_found
    assert "da" in profile.slang_tokens_found
    assert profile.matched_tradition_name == "Tamil Nadu College & Career Family Decision Process (Kudumba Mudivu & Seniors Vicharanai)"

    res = gen.generate(query, profile, [{"content": "College selection is Kudumba Mudivu"}])
    assert res == "Detailed Tanglish response..."

    call_args = mock_llm.call_args[1]
    prompt_sent = call_args["messages"][1]["content"]
    assert "### 🌿 Namma Ooru Culture & Vazhakkam (Epdi Nadakkum-na):" in prompt_sent
    assert "### 💡 Ippo Nee Try Panna Vendiya Steps:" in prompt_sent
    assert "CRITICAL LANGUAGE LOCK RULE FOR TANGLISH" in prompt_sent


def test_regional_slang_tier2_locking():
    """Test that high-precision Tier 2 slang markers correctly lock sub-regions."""
    guardrail = TamilSlangGuardrail()

    # Kongu Nadu
    p_kongu = guardrail.analyze_query("Yenunga, epdi irukeenga sowkiyama?")
    assert p_kongu.detected_region == "Kongu Nadu"

    # Madurai & Thekkathi
    p_madurai = guardrail.analyze_query("Angittu poi parivattam katti thaai maaman seer seiyanum")
    assert p_madurai.detected_region == "Madurai & Thekkathi"

    # Kanyakumari / Nanjil Nadu
    p_nanjil = guardrail.analyze_query("Makkale, kasavu pudavai katti avial virundhu saapidalam")
    assert p_nanjil.detected_region == "Kanyakumari / Nanjil Nadu"

    # Chennai / Madras Bashai
    p_chennai = guardrail.analyze_query("Dei doulath ah suthuradhavitutu padi da")
    assert p_chennai.detected_region == "Chennai / Madras Bashai"

    # Cauvery Delta & Pudukkottai
    p_delta = guardrail.analyze_query("Enga peravurani side la moi note ezhudhi kida virundhu poduvanga")
    assert p_delta.detected_region == "Cauvery Delta & Pudukkottai"


def test_key_rotator_discovers_numbered_keys_and_rotates(monkeypatch):
    """Test KeyRotator discovers GROQ_API_KEY_1 and GROQ_API_KEY_2 and rotates smoothly."""
    from src.utils.llm_clients import KeyRotator

    monkeypatch.setenv("GROQ_API_KEY_1", "gsk_test_key_one_123456789")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk_test_key_two_987654321")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    rotator = KeyRotator("GROQ_API_KEY", ["GROQ_API_KEY", "GROQ_API_KEYS"])
    assert len(rotator.keys) >= 2
    assert rotator.keys[0] == "gsk_test_key_one_123456789"
    assert rotator.keys[1] == "gsk_test_key_two_987654321"

    # Initial key is key 1
    assert rotator.get_current_key() == "gsk_test_key_one_123456789"

    # Rotating advances to key 2
    next_key = rotator.rotate()
    assert next_key == "gsk_test_key_two_987654321"
    assert rotator.get_current_key() == "gsk_test_key_two_987654321"


@patch("litellm.completion")
def test_generate_completion_rotates_on_rate_limit(mock_litellm, monkeypatch):
    """Test generate_completion catches HTTP 429 rate limit, rotates from key 1 to key 2, and succeeds."""
    import os
    from src.utils.llm_clients import generate_completion, groq_rotator

    monkeypatch.setenv("GROQ_API_KEY_1", "gsk_test_key_one_123456789")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk_test_key_two_987654321")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    groq_rotator.reload_keys()

    # First attempt fails with 429 Rate Limit; second attempt succeeds with rotated key
    mock_success_response = MagicMock()
    mock_success_response.choices = [MagicMock(message=MagicMock(content="Culturally aligned answer"))]
    mock_litellm.side_effect = [
        Exception("Rate limit reached: 429 Too Many Requests. Rate limit: 6000 TPM."),
        mock_success_response,
    ]

    answer = generate_completion("groq/openai/gpt-oss-120b", [{"role": "user", "content": "Vanakkam"}])
    assert answer == "Culturally aligned answer"
    assert mock_litellm.call_count == 2
    # Verify second call received the rotated key (key 2)
    assert mock_litellm.call_args_list[1][1]["api_key"] == "gsk_test_key_two_987654321"



