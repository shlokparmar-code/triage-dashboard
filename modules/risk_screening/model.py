"""Tabular Risk Screening inference engine with SHAP explainability."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.risk_screening.explain import explain_prediction_shap

logger = logging.getLogger(__name__)


class RiskScreeningModel:
    """Chronic disease risk assessment engine (Cardiovascular & Diabetes)."""

    def __init__(self):
        self.heart_model = None
        self.diabetes_model = None
        self.heart_bg = None
        self.diabetes_bg = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained models and background baseline distributions."""
        heart_path = config.MODELS_DIR / "heart_disease_model.joblib"
        diab_path = config.MODELS_DIR / "diabetes_model.joblib"
        heart_bg_path = config.MODELS_DIR / "heart_background.joblib"
        diab_bg_path = config.MODELS_DIR / "diabetes_background.joblib"

        try:
            if heart_path.exists():
                self.heart_model = joblib.load(heart_path)
            if diab_path.exists():
                self.diabetes_model = joblib.load(diab_path)
            if heart_bg_path.exists():
                self.heart_bg = joblib.load(heart_bg_path)
            if diab_bg_path.exists():
                self.diabetes_bg = joblib.load(diab_bg_path)
            logger.info("Loaded Module A risk screening model artifacts.")
        except Exception as e:
            logger.warning(f"Error loading RiskScreening artifacts: {e}")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compute heart disease & diabetes risk with SHAP explainability.

        Args:
            features: Dictionary containing patient clinical metrics.

        Returns:
            Dict conforming to standard module interface.
        """
        # Default fallback values for missing features
        age = float(features.get("age", 50))
        sex = int(features.get("sex", 1))
        trestbps = float(features.get("trestbps", features.get("resting_bp", 125)))
        chol = float(features.get("chol", features.get("cholesterol", 210)))
        fbs = int(features.get("fbs", 1 if features.get("glucose", 100) > 120 else 0))
        thalach = float(features.get("thalach", features.get("max_hr", 145)))
        cp = int(features.get("cp", 0))
        exang = int(features.get("exang", 0))
        oldpeak = float(features.get("oldpeak", 0.8))
        restecg = int(features.get("restecg", 0))

        bmi = float(features.get("bmi", 26.5))
        glucose = float(features.get("glucose", 105))
        bp_diastolic = float(features.get("blood_pressure", features.get("diastolic_bp", 75)))
        pregnancies = float(features.get("pregnancies", 1 if sex == 0 else 0))
        skin_thickness = float(features.get("skin_thickness", 20.0))
        insulin = float(features.get("insulin", 85.0))
        pedigree = float(features.get("pedigree", 0.45))

        # 1. Heart Risk Inference
        heart_prob = 0.30
        heart_shap_factors = []
        heart_shap_summary = ""
        heart_patient_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak
        }])

        if self.heart_model is not None:
            try:
                # If pipeline, model expects dataframe matching column names
                heart_prob = float(self.heart_model.predict_proba(heart_patient_df)[0][1])
                if self.heart_bg is not None:
                    heart_shap_factors, heart_shap_summary = explain_prediction_shap(
                        self.heart_model, heart_patient_df, self.heart_bg
                    )
            except Exception as e:
                logger.warning(f"Heart model inference error: {e}")

        # 2. Diabetes Risk Inference
        diab_prob = 0.25
        diab_shap_factors = []
        diab_shap_summary = ""
        diab_patient_df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp_diastolic,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }])

        if self.diabetes_model is not None:
            try:
                diab_prob = float(self.diabetes_model.predict_proba(diab_patient_df)[0][1])
                if self.diabetes_bg is not None:
                    diab_shap_factors, diab_shap_summary = explain_prediction_shap(
                        self.diabetes_model, diab_patient_df, self.diabetes_bg
                    )
            except Exception as e:
                logger.warning(f"Diabetes model inference error: {e}")

        # Composite risk score is dominated by highest clinical liability
        composite_score = max(heart_prob, diab_prob)

        # Urgency Tier Mapping
        if composite_score >= config.RISK_MEDIUM_MAX:
            urgency = config.TIER_HIGH
        elif composite_score >= config.RISK_LOW_MAX:
            urgency = config.TIER_MEDIUM
        else:
            urgency = config.TIER_LOW

        # Generate combined clinical explanation
        primary_disease = "Cardiovascular Disease" if heart_prob >= diab_prob else "Type 2 Diabetes"
        lead_summary = heart_shap_summary if heart_prob >= diab_prob else diab_shap_summary
        lead_factors = heart_shap_factors if heart_prob >= diab_prob else diab_shap_factors

        explanation = (
            f"Baseline chronic risk is {urgency} (Score: {composite_score:.1%}). "
            f"Primary risk driver is {primary_disease} (Heart Risk: {heart_prob:.1%}, Diabetes Risk: {diab_prob:.1%}). "
            f"{lead_summary}"
        )

        return {
            "module": "risk_screening",
            "urgency": urgency,
            "score": round(composite_score, 4),
            "explanation": explanation,
            "details": {
                "heart_disease_risk": round(heart_prob, 4),
                "diabetes_risk": round(diab_prob, 4),
                "primary_driver": primary_disease,
                "lead_shap_factors": lead_factors,
                "heart_shap_factors": heart_shap_factors,
                "diabetes_shap_factors": diab_shap_factors,
                "patient_metrics": {
                    "age": age,
                    "sex": "Male" if sex == 1 else "Female",
                    "resting_bp": trestbps,
                    "cholesterol": chol,
                    "bmi": bmi,
                    "glucose": glucose
                }
            }
        }


_cached_risk_model: Optional[RiskScreeningModel] = None


def predict_risk(features: Dict[str, Any]) -> Dict[str, Any]:
    """Exposed functional interface for Module A."""
    global _cached_risk_model
    if _cached_risk_model is None or _cached_risk_model.heart_model is None:
        _cached_risk_model = RiskScreeningModel()
    return _cached_risk_model.predict(features)
