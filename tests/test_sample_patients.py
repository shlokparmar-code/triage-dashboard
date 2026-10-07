"""End-to-end verification of the 3 canonical benchmark patients."""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from modules.risk_screening.model import RiskScreeningModel
from modules.vitals_anomaly.model import VitalsAnomalyDetector
from modules.vitals_anomaly.generator import generate_synthetic_vitals
from modules.symptom_triage.model import SymptomTriageModel
from modules.fusion.engine import fuse_triage_modalities
from dashboard.pdf_export import generate_triage_pdf


@pytest.fixture(scope="module")
def pipeline():
    return {
        "risk": RiskScreeningModel(),
        "vitals": VitalsAnomalyDetector(),
        "symptom": SymptomTriageModel()
    }


def test_patient_1_low_flow(pipeline):
    """Verify Patient 1: Mild Cold -> LOW urgency triage."""
    # 1. Symptoms
    sym_res = pipeline["symptom"].predict("Mild runny nose, sneezing for 2 days, no fever, no chest pain")
    assert sym_res["urgency"] == config.TIER_LOW

    # 2. Vitals
    vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none", seed=42)
    vit_res = pipeline["vitals"].detect(vitals_df)
    assert vit_res["urgency"] == config.TIER_LOW

    # 3. Tabular Risk
    risk_res = pipeline["risk"].predict({
        "age": 28, "sex": 0, "resting_bp": 115, "cholesterol": 175, "glucose": 88, "bmi": 21.8
    })
    assert risk_res["urgency"] == config.TIER_LOW

    # 4. Fusion
    fusion_res = fuse_triage_modalities(risk_res, vit_res, sym_res, {"resting_bp": 115, "glucose": 88})
    assert fusion_res["final_urgency"] == config.TIER_LOW
    assert fusion_res["final_score"] < config.RISK_LOW_MAX
    assert fusion_res["safety_override"] is False
    assert not fusion_res["recommendations"]["is_emergency"]

    # 5. PDF generation
    pdf_bytes = generate_triage_pdf(fusion_res, risk_res, vit_res, sym_res, {"age": 28, "sex": "Female"})
    assert len(pdf_bytes) > 1000


def test_patient_2_medium_flow(pipeline):
    """Verify Patient 2: Uncontrolled Glycemia & Migraine / Infection -> MEDIUM urgency triage."""
    # 1. Symptoms
    sym_res = pipeline["symptom"].predict("Fever of 102F for three days with persistent shivering and nausea")
    assert sym_res["urgency"] in [config.TIER_MEDIUM, config.TIER_HIGH]

    # 2. Vitals
    vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="septic_fever", seed=42)
    vit_res = pipeline["vitals"].detect(vitals_df)

    # 3. Tabular Risk
    risk_res = pipeline["risk"].predict({
        "age": 54, "sex": 1, "resting_bp": 142, "cholesterol": 245, "glucose": 175, "bmi": 29.5
    })

    # 4. Fusion
    fusion_res = fuse_triage_modalities(risk_res, vit_res, sym_res, {"resting_bp": 142, "glucose": 175})
    assert fusion_res["final_urgency"] in [config.TIER_MEDIUM, config.TIER_HIGH]
    assert len(fusion_res["clinical_synthesis"]) > 20

    # 5. PDF generation
    pdf_bytes = generate_triage_pdf(fusion_res, risk_res, vit_res, sym_res, {"age": 54, "sex": "Male"})
    assert len(pdf_bytes) > 1000


def test_patient_3_high_emergency_flow(pipeline):
    """Verify Patient 3: Acute Chest Pain & Severe Hypoxia -> HIGH emergency triage."""
    # 1. Symptoms (Red flag)
    sym_res = pipeline["symptom"].predict("Crushing chest pain radiating down left arm, gasping for air, sweating cold")
    assert sym_res["urgency"] == config.TIER_HIGH
    assert sym_res["details"]["red_flag_triggered"] is True

    # 2. Vitals (Critical desaturation)
    vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="hypoxia", seed=42)
    vit_res = pipeline["vitals"].detect(vitals_df)
    assert vit_res["urgency"] == config.TIER_HIGH

    # 3. Tabular Risk
    risk_res = pipeline["risk"].predict({
        "age": 66, "sex": 1, "resting_bp": 178, "cholesterol": 295, "glucose": 210, "bmi": 33.0
    })

    # 4. Fusion
    fusion_res = fuse_triage_modalities(risk_res, vit_res, sym_res, {"resting_bp": 178, "glucose": 210})
    assert fusion_res["final_urgency"] == config.TIER_HIGH
    assert fusion_res["safety_override"] is True
    assert fusion_res["final_score"] >= 0.85
    assert fusion_res["recommendations"]["is_emergency"] is True
    assert "112" in fusion_res["recommendations"]["emergency_number"]

    # 5. PDF generation
    pdf_bytes = generate_triage_pdf(fusion_res, risk_res, vit_res, sym_res, {"age": 66, "sex": "Male"})
    assert len(pdf_bytes) > 1000
