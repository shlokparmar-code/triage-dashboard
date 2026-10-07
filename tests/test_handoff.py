"""Unit tests for SBAR Doctor Handoff Summary generation."""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from modules.fusion.handoff import generate_handoff, load_referral_levels


def test_referral_levels_loading():
    """Verify referral levels JSON load."""
    levels = load_referral_levels()
    assert "LOW" in levels
    assert "MEDIUM" in levels
    assert "HIGH" in levels
    assert "District Hospital" in levels["HIGH"]["facility"]


def test_handoff_sbar_sections():
    """Verify SBAR structure and disclaimer presence."""
    patient = {
        "name": "Ramesh Kumar",
        "age": 58,
        "sex": "Male",
        "patient_id": "PAT-DEMO-001",
        "complaint": "Chest tightness and sweating"
    }
    module_results = {
        "symptom_triage": {"urgency": "HIGH", "score": 0.90, "explanation": "Crushing chest pain"},
        "vitals_anomaly": {"urgency": "HIGH", "score": 0.88, "explanation": "SpO2 88%"},
        "risk_screening": {"urgency": "HIGH", "score": 0.78, "explanation": "High BP and glucose"}
    }
    final_result = {
        "final_urgency": "HIGH",
        "final_score": 0.92,
        "safety_override": True,
        "clinical_synthesis": "Critical emergency patient requiring priority resuscitation."
    }
    history = [
        {"hospital_name": "Alwar CHC", "final_urgency": "LOW", "timestamp": "2026-08-01 10:00:00"},
        {"hospital_name": "St. Jude Clinic", "final_urgency": "MEDIUM", "timestamp": "2026-09-15 14:30:00"}
    ]

    handoff = generate_handoff(patient, module_results, final_result, history=history)

    assert "[S] SITUATION" in handoff
    assert "[B] BACKGROUND" in handoff
    assert "[A] CLINICAL ASSESSMENT" in handoff
    assert "[R] RECOMMENDATION & REFERRAL DIRECTIVE" in handoff
    assert "Ramesh Kumar" in handoff
    assert "PAT-DEMO-001" in handoff
    assert "Alwar CHC" in handoff
    assert "St. Jude Clinic" in handoff
    assert "AI decision-support output. Not a medical diagnosis. Requires review by a qualified clinician." in handoff


def test_handoff_handles_missing_data():
    """Verify handoff does not crash on completely empty inputs."""
    handoff = generate_handoff({}, {}, {})
    assert "[S] SITUATION" in handoff
    assert "[B] BACKGROUND" in handoff
    assert "[A] CLINICAL ASSESSMENT" in handoff
    assert "[R] RECOMMENDATION" in handoff
    assert "Unknown Patient" in handoff
