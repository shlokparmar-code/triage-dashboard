"""Unit tests for Module C: Symptom Triage NLP pipeline."""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from modules.symptom_triage.model import SymptomTriageModel, predict_urgency
from modules.symptom_triage.red_flags import check_red_flags
from modules.symptom_triage.preprocessor import preprocess_text, validate_input


@pytest.fixture(scope="module")
def triage_engine():
    return SymptomTriageModel()


def test_standard_interface_keys(triage_engine):
    """Verify output dictionary adheres strictly to standard schema."""
    result = triage_engine.predict("Mild headache and feeling tired after work")
    assert isinstance(result, dict)
    assert result["module"] == "symptom_triage"
    assert result["urgency"] in [config.TIER_LOW, config.TIER_MEDIUM, config.TIER_HIGH]
    assert isinstance(result["score"], (float, int))
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["explanation"], str)
    assert len(result["explanation"]) > 0
    assert "details" in result
    assert "probabilities" in result["details"]


def test_red_flag_overrides():
    """Verify red flags force HIGH urgency immediately."""
    emergency_inputs = [
        "Crushing chest pain radiating to left arm and cold sweats",
        "Cannot breathe and struggling to inhale air",
        "Face is drooping on one side and slurred speech",
        "Coughing up blood and feeling faint",
        "Severe allergic reaction with throat closing",
    ]
    for text in emergency_inputs:
        is_flagged, reasons = check_red_flags(text)
        assert is_flagged, f"Failed to detect red flag in: {text}"
        res = predict_urgency(text)
        assert res["urgency"] == config.TIER_HIGH
        assert res["score"] == 1.0
        assert res["details"]["red_flag_triggered"] is True


def test_negation_awareness():
    """Ensure negated emergency terms do NOT trigger false alarms."""
    negated_inputs = [
        "Mild headache but no chest pain and no trouble breathing",
        "Slight cough for two days, denies shortness of breath, no fever",
        "Feeling a bit tired today, without chest pressure and no palpitations",
    ]
    for text in negated_inputs:
        is_flagged, reasons = check_red_flags(text)
        assert not is_flagged, f"False positive red flag triggered for negated text: {text} -> {reasons}"
        res = predict_urgency(text)
        assert res["details"]["red_flag_triggered"] is False
        assert res["urgency"] in [config.TIER_LOW, config.TIER_MEDIUM]


def test_typo_correction():
    """Verify common typos are normalized."""
    text = "chesst hurt and breathin fast with hedache"
    cleaned = preprocess_text(text)
    assert "chest" in cleaned
    assert "breathing" in cleaned
    assert "headache" in cleaned


def test_edge_case_empty_input(triage_engine):
    """Handle empty or whitespace text without crashing."""
    res = triage_engine.predict("   ")
    assert res["details"]["validation_status"] == "EMPTY"
    assert res["score"] == 0.0
    assert "No symptoms entered" in res["explanation"]


def test_edge_case_gibberish(triage_engine):
    """Handle gibberish / repeated characters gracefully."""
    res = triage_engine.predict("zzzzzzzzzz aaaaaaa bbbbbbb")
    assert res["details"]["validation_status"] == "GIBBERISH"
    assert "Unable to assess" in res["explanation"] or "characters" in res["explanation"]


def test_edge_case_non_medical(triage_engine):
    """Handle non-medical greetings / conversational text."""
    res = triage_engine.predict("hello how are you today?")
    assert res["details"]["validation_status"] == "NON_MEDICAL"
    assert "Unable to assess" in res["explanation"]


def test_very_long_input(triage_engine):
    """Handle inputs exceeding normal length without crash."""
    long_text = "I have a mild runny nose and cough. " * 100
    res = triage_engine.predict(long_text)
    assert isinstance(res, dict)
    assert res["urgency"] in [config.TIER_LOW, config.TIER_MEDIUM, config.TIER_HIGH]
