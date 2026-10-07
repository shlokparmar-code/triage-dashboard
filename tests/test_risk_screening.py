"""Unit tests for Module A: Tabular Chronic Risk Screening."""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from modules.risk_screening.model import RiskScreeningModel, predict_risk
from modules.risk_screening.data_loader import load_heart_disease_data, load_diabetes_data


@pytest.fixture(scope="module")
def risk_model():
    return RiskScreeningModel()


def test_data_loaders():
    """Verify dataset loading and fallback resilience."""
    h_df, h_src = load_heart_disease_data()
    assert len(h_df) > 50
    assert "target" in h_df.columns

    d_df, d_src = load_diabetes_data()
    assert len(d_df) > 50
    assert "Outcome" in d_df.columns


def test_standard_interface_keys(risk_model):
    """Verify output dictionary adheres strictly to standard schema."""
    features = {
        "age": 52,
        "sex": 1,
        "resting_bp": 130,
        "cholesterol": 220,
        "glucose": 110,
        "bmi": 27.0
    }
    res = risk_model.predict(features)
    assert isinstance(res, dict)
    assert res["module"] == "risk_screening"
    assert res["urgency"] in config.TIERS
    assert isinstance(res["score"], (float, int))
    assert 0.0 <= res["score"] <= 1.0
    assert isinstance(res["explanation"], str)
    assert len(res["explanation"]) > 0
    assert "details" in res
    assert "heart_disease_risk" in res["details"]
    assert "diabetes_risk" in res["details"]


def test_low_risk_patient(risk_model):
    """Verify young, healthy profile yields LOW risk."""
    healthy_patient = {
        "age": 25,
        "sex": 0,
        "resting_bp": 110,
        "cholesterol": 160,
        "glucose": 85,
        "bmi": 21.5,
        "thalach": 170,
        "cp": 0,
        "exang": 0,
        "oldpeak": 0.0
    }
    res = risk_model.predict(healthy_patient)
    assert res["urgency"] == config.TIER_LOW
    assert res["score"] < config.RISK_MEDIUM_MAX


def test_high_risk_patient(risk_model):
    """Verify elderly, multimorbid profile yields HIGH risk."""
    high_risk_patient = {
        "age": 74,
        "sex": 1,
        "resting_bp": 185,
        "cholesterol": 340,
        "glucose": 260,
        "bmi": 38.5,
        "thalach": 105,
        "cp": 2,
        "exang": 1,
        "oldpeak": 3.8
    }
    res = risk_model.predict(high_risk_patient)
    assert res["urgency"] == config.TIER_HIGH
    assert res["score"] >= config.RISK_MEDIUM_MAX


def test_missing_features_resilience(risk_model):
    """Verify empty/partial feature dictionary defaults gracefully without exceptions."""
    res = risk_model.predict({})
    assert isinstance(res, dict)
    assert res["urgency"] in config.TIERS
    assert 0.0 <= res["score"] <= 1.0
