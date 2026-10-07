"""Unit tests for Fusion and Recommendation Engine."""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from modules.fusion.engine import fuse_triage_modalities
from modules.fusion.recommendations import get_recommendations, load_knowledge_base


def test_knowledge_base_loading():
    """Verify editable recommendations knowledge base loads and has required keys."""
    kb = load_knowledge_base()
    assert "emergency_high" in kb
    assert "yoga_asanas" in kb
    assert "dietary_guidelines" in kb
    assert "disclaimer" in kb
    assert len(kb["yoga_asanas"]) >= 3


def test_all_low_fusion():
    """Verify all low modules combine into LOW overall triage."""
    mod_a = {"module": "risk_screening", "urgency": "LOW", "score": 0.15, "explanation": "Low chronic risk"}
    mod_b = {"module": "vitals_anomaly", "urgency": "LOW", "score": 0.12, "explanation": "Stable vitals"}
    mod_c = {"module": "symptom_triage", "urgency": "LOW", "score": 0.10, "explanation": "Mild cold"}

    res = fuse_triage_modalities(mod_a, mod_b, mod_c)
    assert res["final_urgency"] == config.TIER_LOW
    assert res["final_score"] < config.RISK_LOW_MAX
    assert res["safety_override"] is False
    assert not res["recommendations"]["is_emergency"]
    assert len(res["recommendations"]["yoga_asanas"]) > 0


def test_safety_override_single_high_symptom():
    """Verify that a single HIGH in Module C forces final result to HIGH."""
    mod_a = {"module": "risk_screening", "urgency": "LOW", "score": 0.10, "explanation": "Low risk"}
    mod_b = {"module": "vitals_anomaly", "urgency": "LOW", "score": 0.10, "explanation": "Normal vitals"}
    mod_c = {"module": "symptom_triage", "urgency": "HIGH", "score": 1.0, "explanation": "Crushing chest pain"}

    res = fuse_triage_modalities(mod_a, mod_b, mod_c)
    assert res["final_urgency"] == config.TIER_HIGH
    assert res["safety_override"] is True
    assert res["final_score"] >= 0.85
    assert res["recommendations"]["is_emergency"] is True
    assert "while_waiting_steps" in res["recommendations"]


def test_safety_override_single_high_vitals():
    """Verify that a single HIGH in Module B forces final result to HIGH."""
    mod_a = {"module": "risk_screening", "urgency": "LOW", "score": 0.20, "explanation": "Low risk"}
    mod_b = {"module": "vitals_anomaly", "urgency": "HIGH", "score": 1.0, "explanation": "Critical hypoxia SpO2 86%"}
    mod_c = {"module": "symptom_triage", "urgency": "LOW", "score": 0.15, "explanation": "Mild fatigue"}

    res = fuse_triage_modalities(mod_a, mod_b, mod_c)
    assert res["final_urgency"] == config.TIER_HIGH
    assert res["safety_override"] is True
    assert res["recommendations"]["is_emergency"] is True


def test_personalized_recommendations():
    """Verify recommendations personalize asanas and diet to clinical inputs."""
    # Patient with high BP
    recs_bp = get_recommendations("LOW", {"resting_bp": 155, "glucose": 95})
    assert any("Anulom Vilom" in a["name"] for a in recs_bp["yoga_asanas"])
    assert any("Sodium" in d.get("focus", "") or "DASH" in d.get("focus", "") for d in recs_bp["dietary_guidelines"])

    # Patient with high glucose
    recs_glucose = get_recommendations("LOW", {"resting_bp": 115, "glucose": 180})
    assert any("Mandukasana" in a["name"] for a in recs_glucose["yoga_asanas"])
    assert any("Glycemic" in d.get("focus", "") or "Fiber" in d.get("focus", "") for d in recs_glucose["dietary_guidelines"])


def test_country_emergency_number_config():
    """Verify emergency number adapts to country config."""
    recs_india = get_recommendations("HIGH", emergency_country="India")
    assert "112" in recs_india["emergency_number"] or "108" in recs_india["emergency_number"]

    recs_us = get_recommendations("HIGH", emergency_country="US")
    assert "911" in recs_us["emergency_number"]
