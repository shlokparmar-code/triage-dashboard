"""Unit tests for Maternal and Child Danger Signs Special Population layer."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from modules.special_populations.rules import (
    detect_patient_category,
    get_pediatric_bracket,
    evaluate_pediatric_vitals,
    evaluate_special_population,
)
from modules.fusion.engine import fuse_triage_modalities
import config


def test_detect_patient_category():
    assert detect_patient_category(35, "Male") == "Adult"
    assert detect_patient_category(28, "Female", is_pregnant=True) == "Pregnant / Postpartum"
    assert detect_patient_category(28, "Female", is_postpartum=True) == "Pregnant / Postpartum"
    assert detect_patient_category(3, "Female") == "Pediatric (<5)"
    assert detect_patient_category(10, "Male") == "Pediatric (5-18)"


def test_pediatric_bracket_lookup():
    b_neonate = get_pediatric_bracket(1.0)
    assert b_neonate["name"].startswith("Neonate")

    b_infant = get_pediatric_bracket(6.0)
    assert b_infant["name"].startswith("Infant")

    b_young = get_pediatric_bracket(24.0)
    assert b_young["name"].startswith("Young Child")

    b_adolescent = get_pediatric_bracket(180.0)
    assert b_adolescent["name"].startswith("Adolescent")


def test_under5_danger_sign_forces_high():
    # Child unable to drink or breastfeed
    patient = {
        "age": 2,
        "age_months": 24,
        "sex": "Female",
        "child_danger_signs": ["unable_to_drink"]
    }
    result = evaluate_special_population(patient)
    assert result["urgency"] == config.TIER_HIGH
    assert result["score"] >= 0.90
    assert "unable_to_drink" in str(result["details"]["danger_signs_detected"]).lower() or "drink" in str(result["details"]["danger_signs_detected"]).lower()
    assert result["details"]["module_a_applicable"] is False
    assert len(result["details"]["emergency_instructions"]) > 0


def test_fast_breathing_forces_high():
    # Infant with respiratory rate 65 (fast breathing threshold is 50 for infant)
    patient = {
        "age": 0.5,
        "age_months": 6,
        "sex": "Male",
        "vitals": {"rr": 65, "hr": 130, "spo2": 96}
    }
    result = evaluate_special_population(patient)
    assert result["urgency"] == config.TIER_HIGH
    assert result["details"]["vitals_evaluation"]["fast_breathing"] is True


def test_pregnancy_red_flag_forces_high():
    # Pregnant woman with heavy bleeding
    patient = {
        "age": 26,
        "sex": "Female",
        "is_pregnant": True,
        "gestational_weeks": 32,
        "pregnancy_red_flags": ["preg_heavy_bleeding"]
    }
    result = evaluate_special_population(patient)
    assert result["urgency"] == config.TIER_HIGH
    assert result["score"] >= 0.90
    assert len(result["details"]["red_flags_detected"]) > 0


def test_preeclampsia_detection():
    # Pregnant woman with BP 150/95 and severe headache
    patient = {
        "age": 30,
        "sex": "Female",
        "is_pregnant": True,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "pregnancy_red_flags": ["preg_severe_headache_bp"]
    }
    result = evaluate_special_population(patient)
    assert result["urgency"] == config.TIER_HIGH
    assert result["details"]["preeclampsia_risk"] is True
    assert "preeclampsia" in str(result["details"]["red_flags_detected"]).lower()


def test_fusion_excludes_module_a_for_child():
    child_patient = {
        "age": 4,
        "age_months": 48,
        "sex": "Male"
    }
    spec_result = evaluate_special_population(child_patient)

    # Dummy module results
    risk_res = {"module": "Risk Screening", "urgency": config.TIER_HIGH, "score": 0.88, "explanation": "Adult risk"}
    vit_res = {"module": "Vitals Anomaly", "urgency": config.TIER_LOW, "score": 0.10, "explanation": "Vitals normal"}
    sym_res = {"module": "Symptom Triage", "urgency": config.TIER_LOW, "score": 0.15, "explanation": "Mild cold"}

    fused = fuse_triage_modalities(
        risk_result=risk_res,
        vitals_result=vit_res,
        symptom_result=sym_res,
        patient_features=child_patient,
        special_population_result=spec_result
    )

    # Since patient is child, Module A is NOT applicable and must NOT force high
    assert fused["module_contributions"]["risk_screening"]["applicable"] is False
    assert fused["module_contributions"]["risk_screening"]["status"] == "Not applicable (adult model)"
    assert fused["final_urgency"] == config.TIER_LOW


def test_fusion_override_with_special_population():
    child_patient = {
        "age": 2,
        "age_months": 24,
        "sex": "Male",
        "child_danger_signs": ["convulsions"]
    }
    spec_result = evaluate_special_population(child_patient)

    risk_res = {"module": "Risk Screening", "urgency": config.TIER_LOW, "score": 0.05, "explanation": "N/A"}
    vit_res = {"module": "Vitals Anomaly", "urgency": config.TIER_LOW, "score": 0.10, "explanation": "Normal"}
    sym_res = {"module": "Symptom Triage", "urgency": config.TIER_LOW, "score": 0.10, "explanation": "Fever"}

    fused = fuse_triage_modalities(
        risk_result=risk_res,
        vitals_result=vit_res,
        symptom_result=sym_res,
        patient_features=child_patient,
        special_population_result=spec_result
    )

    assert fused["final_urgency"] == config.TIER_HIGH
    assert fused["safety_override"] is True
    assert any("Special Population" in mod for mod in fused["overriding_modules"])
