"""Unit tests for Mode configuration and mode invariance."""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from modules.fusion.engine import fuse_triage_modalities


def test_mode_config_keys():
    """Verify MODE_CONFIG has required keys and mode-specific feature toggles."""
    assert config.MODE_HOSPITAL in config.MODE_CONFIG
    assert config.MODE_HOME in config.MODE_CONFIG

    h_cfg = config.MODE_CONFIG[config.MODE_HOSPITAL]
    home_cfg = config.MODE_CONFIG[config.MODE_HOME]

    # Hospital has clinical language, queue, handoff, registry search
    assert h_cfg["clinical_language"] is True
    assert h_cfg["show_queue"] is True
    assert h_cfg["show_handoff"] is True
    assert h_cfg["show_registry_search"] is True

    # Home has plain language, no queue, no handoff, no registry search
    assert home_cfg["clinical_language"] is False
    assert home_cfg["show_queue"] is False
    assert home_cfg["show_handoff"] is False
    assert home_cfg["show_registry_search"] is False


def test_mode_invariance_of_triage_results():
    """Verify that identical inputs yield identical final urgency tier across modes."""
    risk_res = {"module": "risk_screening", "urgency": "MEDIUM", "score": 0.50, "explanation": "Risk moderate"}
    vitals_res = {"module": "vitals_anomaly", "urgency": "MEDIUM", "score": 0.48, "explanation": "Vitals borderline"}
    symptom_res = {"module": "symptom_triage", "urgency": "MEDIUM", "score": 0.52, "explanation": "Symptoms moderate"}

    res_1 = fuse_triage_modalities(risk_res, vitals_res, symptom_res)
    res_2 = fuse_triage_modalities(risk_res, vitals_res, symptom_res)

    assert res_1["final_urgency"] == res_2["final_urgency"]
    assert res_1["final_score"] == res_2["final_score"]
    assert res_1["final_urgency"] == "MEDIUM"


def test_dual_mode_pdf_generation():
    """Verify both clinical hospital PDF and patient-friendly home PDF generate valid bytes."""
    from dashboard.pdf_export import generate_triage_pdf, generate_home_pdf

    risk_res = {"module": "Risk Screening", "urgency": "LOW", "score": 0.15, "explanation": "Low risk"}
    vit_res = {"module": "Vitals Anomaly", "urgency": "LOW", "score": 0.10, "explanation": "Vitals normal"}
    sym_res = {"module": "Symptom Triage", "urgency": "LOW", "score": 0.12, "explanation": "Mild fatigue"}

    fusion_res = fuse_triage_modalities(risk_res, vit_res, sym_res)
    patient_info = {
        "name": "Sunita Devi",
        "age": 28,
        "sex": "Female",
        "mobile_number": "+919876543210"
    }

    # Hospital PDF
    hosp_pdf = generate_triage_pdf(fusion_res, risk_res, vit_res, sym_res, patient_info)
    assert len(hosp_pdf) > 1000
    assert isinstance(hosp_pdf, bytes)

    # Home PDF
    home_pdf = generate_home_pdf(fusion_res, patient_info)
    assert len(home_pdf) > 1000
    assert isinstance(home_pdf, bytes)

