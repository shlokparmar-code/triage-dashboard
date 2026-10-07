"""Unit tests for Module B: Time-Series Vitals Anomaly Detection."""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from modules.vitals_anomaly.generator import generate_synthetic_vitals, stream_vitals_generator
from modules.vitals_anomaly.model import VitalsAnomalyDetector, detect_anomalies, check_clinical_safety_rules


@pytest.fixture(scope="module")
def vitals_detector():
    return VitalsAnomalyDetector()


def test_generator_structure():
    """Verify synthetic vitals dataframe conforms to expected schema."""
    df = generate_synthetic_vitals(duration_minutes=15, sampling_interval_sec=10, anomaly_type="none")
    assert len(df) == 90
    for col in ["timestamp", "heart_rate", "spo2", "respiratory_rate", "temperature", "ground_truth_anomaly"]:
        assert col in df.columns


def test_standard_interface_keys(vitals_detector):
    """Verify standard dictionary interface output."""
    df = generate_synthetic_vitals(duration_minutes=20, sampling_interval_sec=10, anomaly_type="none")
    res = vitals_detector.detect(df)
    assert isinstance(res, dict)
    assert res["module"] == "vitals_anomaly"
    assert res["urgency"] in config.TIERS
    assert isinstance(res["score"], (float, int))
    assert 0.0 <= res["score"] <= 1.0
    assert isinstance(res["explanation"], str)
    assert len(res["explanation"]) > 0
    assert "details" in res
    assert "model_used" in res["details"]


def test_clinical_safety_override_hypoxia(vitals_detector):
    """Verify SpO2 < 90% strictly triggers HIGH urgency override."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="10s"),
        "heart_rate": [75] * 10,
        "spo2": [97, 96, 95, 88.5, 87.0, 89.0, 93, 95, 96, 97],  # Drops to 87.0
        "respiratory_rate": [16] * 10,
        "temperature": [37.0] * 10
    })
    is_critical, violations, extremes = check_clinical_safety_rules(df)
    assert is_critical
    assert any("Hypoxia" in v for v in violations)

    res = vitals_detector.detect(df)
    assert res["urgency"] == config.TIER_HIGH
    assert res["score"] == 1.0
    assert res["details"]["safety_override"] is True


def test_clinical_safety_override_tachycardia(vitals_detector):
    """Verify HR > 140 bpm triggers HIGH urgency override."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="10s"),
        "heart_rate": [75, 80, 110, 145, 152, 148, 120, 95, 80, 75],  # Peaks at 152
        "spo2": [98] * 10,
        "respiratory_rate": [16] * 10,
        "temperature": [37.0] * 10
    })
    res = vitals_detector.detect(df)
    assert res["urgency"] == config.TIER_HIGH
    assert res["score"] == 1.0
    assert res["details"]["safety_override"] is True


def test_clinical_safety_override_bradycardia(vitals_detector):
    """Verify HR < 40 bpm triggers HIGH urgency override."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="10s"),
        "heart_rate": [60, 50, 42, 35, 34, 38, 45, 52, 58, 62],  # Drops to 34
        "spo2": [98] * 10,
        "respiratory_rate": [16] * 10,
        "temperature": [37.0] * 10
    })
    res = vitals_detector.detect(df)
    assert res["urgency"] == config.TIER_HIGH
    assert res["score"] == 1.0
    assert res["details"]["safety_override"] is True


def test_normal_vitals_yield_low_urgency(vitals_detector):
    """Verify normal resting baseline vitals produce LOW urgency."""
    df = generate_synthetic_vitals(duration_minutes=30, sampling_interval_sec=10, anomaly_type="none", seed=42)
    res = vitals_detector.detect(df)
    assert res["urgency"] == config.TIER_LOW
    assert res["score"] < config.RISK_LOW_MAX


def test_configurable_model_selection(vitals_detector):
    """Verify both Isolation Forest and Autoencoder models can be selected."""
    df = generate_synthetic_vitals(duration_minutes=20, sampling_interval_sec=10, anomaly_type="none")
    res_iso = vitals_detector.detect(df, model_type="isolation_forest")
    assert res_iso["details"]["model_used"] in ["Isolation Forest", "Heuristic Baseline"]

    res_ae = vitals_detector.detect(df, model_type="autoencoder")
    assert res_ae["details"]["model_used"] in ["Reconstruction Autoencoder", "Heuristic Baseline"]


def test_streaming_simulation():
    """Verify streaming chunks generator yields contiguous sequential frames."""
    df = generate_synthetic_vitals(duration_minutes=5, sampling_interval_sec=10, anomaly_type="none")
    chunks = list(stream_vitals_generator(df, chunk_size=5))
    assert len(chunks) == 6
    assert len(chunks[0]) == 5
