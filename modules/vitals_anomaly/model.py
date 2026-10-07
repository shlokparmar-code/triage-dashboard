"""Vitals Anomaly Detection inference engine with hard clinical safety overrides."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.vitals_anomaly.features import extract_rolling_features
from modules.vitals_anomaly.lstm_model import VitalsAutoencoder

logger = logging.getLogger(__name__)


def check_clinical_safety_rules(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Evaluate absolute clinical safety boundaries that override ML predictions.

    Rules:
    - SpO2 < 90% => Force HIGH (Severe Hypoxemia)
    - Heart Rate > 140 bpm => Force HIGH (Severe Tachycardia)
    - Heart Rate < 40 bpm => Force HIGH (Severe Bradycardia)
    - Respiratory Rate < 8 or > 32 => Force HIGH (Acute Respiratory Distress)
    - Temperature > 39.5 C => Force HIGH (Severe Hyperpyrexia)
    """
    violations = []
    extreme_values = {}

    if "spo2" in df.columns:
        min_spo2 = float(df["spo2"].min())
        extreme_values["min_spo2"] = min_spo2
        if min_spo2 < config.VITALS_SPO2_CRITICAL:
            violations.append(f"Critical Hypoxia: SpO2 dropped to {min_spo2:.1f}% (< {config.VITALS_SPO2_CRITICAL}%)")

    if "heart_rate" in df.columns:
        max_hr = float(df["heart_rate"].max())
        min_hr = float(df["heart_rate"].min())
        extreme_values["max_hr"] = max_hr
        extreme_values["min_hr"] = min_hr

        if max_hr > config.VITALS_HR_HIGH_CRITICAL:
            violations.append(f"Severe Tachycardia: Heart rate reached {max_hr:.0f} bpm (> {config.VITALS_HR_HIGH_CRITICAL} bpm)")
        if min_hr < config.VITALS_HR_LOW_CRITICAL:
            violations.append(f"Severe Bradycardia: Heart rate dropped to {min_hr:.0f} bpm (< {config.VITALS_HR_LOW_CRITICAL} bpm)")

    if "respiratory_rate" in df.columns:
        max_rr = float(df["respiratory_rate"].max())
        min_rr = float(df["respiratory_rate"].min())
        extreme_values["max_rr"] = max_rr
        if max_rr > config.VITALS_RR_HIGH_CRITICAL:
            violations.append(f"Severe Tachypnea: Respiratory rate reached {max_rr:.0f} breaths/min")
        elif min_rr < config.VITALS_RR_LOW_CRITICAL and min_rr > 0:
            violations.append(f"Bradypnea / Hypoventilation: Respiratory rate fell to {min_rr:.0f} breaths/min")

    if "temperature" in df.columns:
        max_temp = float(df["temperature"].max())
        extreme_values["max_temp"] = max_temp
        if max_temp > config.VITALS_TEMP_HIGH_CRITICAL:
            violations.append(f"Hyperpyrexia: Core temperature reached {max_temp:.1f} C")

    return len(violations) > 0, violations, extreme_values


class VitalsAnomalyDetector:
    """Configurable multi-variate vitals anomaly detection engine (Isolation Forest or Autoencoder)."""

    def __init__(self, model_type: str = "isolation_forest"):
        self.model_type = model_type  # 'isolation_forest' or 'autoencoder'
        self.isoforest = None
        self.scaler = None
        self.autoencoder = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained models from models/ directory."""
        iso_path = config.MODELS_DIR / "vitals_isoforest.joblib"
        scaler_path = config.MODELS_DIR / "vitals_scaler.joblib"
        ae_path = config.MODELS_DIR / "vitals_autoencoder.joblib"

        try:
            if iso_path.exists():
                self.isoforest = joblib.load(iso_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            if ae_path.exists():
                self.autoencoder = joblib.load(ae_path)
            logger.info("Loaded Module B vitals anomaly artifacts.")
        except Exception as e:
            logger.warning(f"Error loading vitals anomaly models: {e}")

    def detect(self, df: pd.DataFrame, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Process time-series vitals dataframe and return standard urgency output.

        Args:
            df: DataFrame containing at least 'heart_rate' and 'spo2'.
            model_type: Optional override ('isolation_forest' or 'autoencoder').

        Returns:
            Dict conforming to standard module interface.
        """
        chosen_model = model_type or self.model_type

        # Ensure required columns exist or map aliases
        df_clean = df.copy()
        if "hr" in df_clean.columns and "heart_rate" not in df_clean.columns:
            df_clean["heart_rate"] = df_clean["hr"]
        if "o2" in df_clean.columns and "spo2" not in df_clean.columns:
            df_clean["spo2"] = df_clean["o2"]
        if "rr" in df_clean.columns and "respiratory_rate" not in df_clean.columns:
            df_clean["respiratory_rate"] = df_clean["rr"]
        if "temp" in df_clean.columns and "temperature" not in df_clean.columns:
            df_clean["temperature"] = df_clean["temp"]

        # 1. HARD CLINICAL SAFETY RULES OVERRIDE
        is_critical, safety_violations, extremes = check_clinical_safety_rules(df_clean)
        if is_critical:
            flagged_indices = []
            if "timestamp" in df_clean.columns:
                flagged_indices = df_clean["timestamp"].astype(str).tolist()

            explanation = (
                f"CRITICAL SAFETY OVERRIDE: {safety_violations[0]}. "
                "Immediate medical attention required."
            )
            return {
                "module": "vitals_anomaly",
                "urgency": config.TIER_HIGH,
                "score": 1.0,
                "explanation": explanation,
                "details": {
                    "model_used": "CLINICAL_SAFETY_RULE_OVERRIDE",
                    "safety_override": True,
                    "violations": safety_violations,
                    "extremes": extremes,
                    "mean_hr": round(float(df_clean["heart_rate"].mean()), 1) if "heart_rate" in df_clean.columns else None,
                    "min_spo2": round(float(df_clean["spo2"].min()), 1) if "spo2" in df_clean.columns else None,
                    "flagged_windows_count": len(df_clean),
                    "anomaly_indices": list(range(len(df_clean)))[:25]
                }
            }

        # 2. FEATURE EXTRACTION
        feat_df, feat_cols = extract_rolling_features(df_clean)

        # 3. MODEL INFERENCE
        anomaly_scores = np.zeros(len(df_clean))

        if chosen_model == "autoencoder" and self.autoencoder is not None:
            model_name = "Reconstruction Autoencoder"
            anomaly_scores = self.autoencoder.compute_anomaly_scores(feat_df.values)
        elif self.isoforest is not None and self.scaler is not None:
            model_name = "Isolation Forest"
            X_scaled = self.scaler.transform(feat_df.values)
            # IsolationForest decision_function: positive is normal, negative is anomalous
            raw_scores = self.isoforest.decision_function(X_scaled)
            # Centered sigmoid with offset: normal values (+0.1) map to ~0.15-0.25, anomalies (-0.1) map to >0.75
            anomaly_scores = 1.0 / (1.0 + np.exp(12.0 * (raw_scores - 0.02)))
        else:
            model_name = "Heuristic Baseline"
            # Fallback heuristic if models not yet serialized
            hr_dist = np.abs(df_clean["heart_rate"] - 75.0) / 40.0 if "heart_rate" in df_clean.columns else 0
            spo2_dist = np.maximum(0, 96.0 - df_clean["spo2"]) / 10.0 if "spo2" in df_clean.columns else 0
            anomaly_scores = np.clip(hr_dist * 0.4 + spo2_dist * 0.6, 0.0, 1.0)

        # Peak and average anomaly scores
        peak_anomaly_score = float(np.max(anomaly_scores))
        avg_anomaly_score = float(np.mean(anomaly_scores))
        anomalous_mask = anomaly_scores > config.VITALS_ANOMALY_PERCENTILE_THRESHOLD
        flagged_count = int(np.sum(anomalous_mask))

        # Flagged window timestamps
        flagged_windows = []
        if "timestamp" in df_clean.columns:
            flagged_windows = df_clean.loc[anomalous_mask, "timestamp"].astype(str).tolist()
        else:
            flagged_windows = [f"Step_{i}" for i in np.where(anomalous_mask)[0]]

        # Determine Urgency Tier
        # HIGH requires either extreme anomaly score or substantial fraction of anomalous windows
        if peak_anomaly_score >= 0.75 or flagged_count >= max(5, int(0.15 * len(df_clean))):
            urgency = config.TIER_HIGH
            score = max(peak_anomaly_score, 0.75)
        elif peak_anomaly_score >= 0.55 and flagged_count >= 3:
            urgency = config.TIER_MEDIUM
            score = max(peak_anomaly_score, 0.45)
        else:
            urgency = config.TIER_LOW
            score = min(avg_anomaly_score, config.RISK_LOW_MAX - 0.05)

        # Clinical Explanation
        mean_hr = float(df_clean["heart_rate"].mean()) if "heart_rate" in df_clean.columns else 72.0
        min_spo2 = float(df_clean["spo2"].min()) if "spo2" in df_clean.columns else 98.0

        if urgency == config.TIER_HIGH:
            explanation = (
                f"Physiological instability detected by {model_name} (Peak score: {peak_anomaly_score:.1%}). "
                f"{flagged_count} anomalous time windows flagged (Min SpO2: {min_spo2:.1f}%, Mean HR: {mean_hr:.0f} bpm)."
            )
        elif urgency == config.TIER_MEDIUM:
            explanation = (
                f"Borderline vital fluctuations observed by {model_name} (Score: {peak_anomaly_score:.1%}). "
                f"Occasional erratic readings flagged across {flagged_count} windows. Monitor closely."
            )
        else:
            explanation = (
                f"Vital signs are stable and within normal physiological limits "
                f"(Mean HR: {mean_hr:.0f} bpm, Min SpO2: {min_spo2:.1f}%)."
            )

        return {
            "module": "vitals_anomaly",
            "urgency": urgency,
            "score": round(score, 4),
            "explanation": explanation,
            "details": {
                "model_used": model_name,
                "safety_override": False,
                "peak_anomaly_score": round(peak_anomaly_score, 4),
                "avg_anomaly_score": round(avg_anomaly_score, 4),
                "flagged_windows_count": flagged_count,
                "flagged_windows": flagged_windows[:15],
                "mean_hr": round(mean_hr, 1),
                "min_spo2": round(min_spo2, 1),
                "anomaly_series": np.round(anomaly_scores, 3).tolist()
            }
        }


_cached_vitals_detector: Optional[VitalsAnomalyDetector] = None


def detect_anomalies(df: pd.DataFrame, model_type: str = "isolation_forest") -> Dict[str, Any]:
    """Exposed functional interface for Module B."""
    global _cached_vitals_detector
    if _cached_vitals_detector is None or _cached_vitals_detector.isoforest is None:
        _cached_vitals_detector = VitalsAnomalyDetector(model_type=model_type)
    return _cached_vitals_detector.detect(df, model_type=model_type)
