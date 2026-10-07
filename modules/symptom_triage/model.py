"""Symptom Triage Model inference and lifecycle management."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.symptom_triage.red_flags import check_red_flags
from modules.symptom_triage.preprocessor import preprocess_text, validate_input
from modules.symptom_triage.explain import extract_token_attributions, generate_symptom_explanation

logger = logging.getLogger(__name__)


class SymptomTriageModel:
    """End-to-end NLP Symptom Triage inference engine with clinical safety overrides."""

    def __init__(self, model_path: Optional[Path] = None, vectorizer_path: Optional[Path] = None):
        self.model_path = model_path or (config.MODELS_DIR / "symptom_model.joblib")
        self.vectorizer_path = vectorizer_path or (config.MODELS_DIR / "symptom_vectorizer.joblib")
        self.classifier = None
        self.vectorizer = None
        self.classes_ = [config.TIER_LOW, config.TIER_MEDIUM, config.TIER_HIGH]
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized ML artifacts if present."""
        if self.model_path.exists() and self.vectorizer_path.exists():
            try:
                self.classifier = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                if hasattr(self.classifier, "classes_"):
                    self.classes_ = list(self.classifier.classes_)
                logger.info("Successfully loaded Symptom Triage model artifacts.")
            except Exception as e:
                logger.warning(f"Could not load symptom triage artifacts: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        """Triage free-text symptoms into standard urgency output dict.

        Args:
            text: Raw symptom description.

        Returns:
            Dict conforming to standard module interface.
        """
        # 1. Rule-Based Red-Flag Clinical Override (Evaluated first to guarantee patient safety)
        is_red_flag, red_flag_reasons = check_red_flags(text)
        if is_red_flag:
            return {
                "module": "symptom_triage",
                "urgency": config.TIER_HIGH,
                "score": 1.0,
                "explanation": f"CRITICAL OVERRIDE: {red_flag_reasons[0]}. Seek immediate emergency care.",
                "details": {
                    "validation_status": "VALID",
                    "probabilities": {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 1.0},
                    "red_flag_triggered": True,
                    "red_flags": red_flag_reasons,
                    "top_tokens": [{"token": r.split("'")[1] if "'" in r else "critical_symptom", "weight": 1.0, "influence": "escalating"} for r in red_flag_reasons]
                }
            }

        # 2. Edge Case Validation
        is_valid, status_code, validation_msg = validate_input(text)
        if not is_valid:
            return {
                "module": "symptom_triage",
                "urgency": config.TIER_LOW,
                "score": 0.0,
                "explanation": validation_msg,
                "details": {
                    "validation_status": status_code,
                    "probabilities": {c: 0.0 for c in config.TIERS},
                    "red_flag_triggered": False,
                    "red_flags": [],
                    "top_tokens": []
                }
            }

        # 3. Model Inference (Fallback to rule/keyword if un-trained)
        if self.classifier is None or self.vectorizer is None:
            return {
                "module": "symptom_triage",
                "urgency": config.TIER_MEDIUM,
                "score": 0.50,
                "explanation": "NLP model not yet loaded. Defaulting to cautionary MEDIUM triage.",
                "details": {
                    "validation_status": "MODEL_UNINITIALIZED",
                    "probabilities": {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.25},
                    "red_flag_triggered": False,
                    "red_flags": [],
                    "top_tokens": []
                }
            }

        # Preprocess text
        preprocessed = preprocess_text(text)
        X_vec = self.vectorizer.transform([preprocessed])
        
        # Probabilities
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X_vec)[0]
        else:
            # Fallback for models without predict_proba
            pred = self.classifier.predict(X_vec)[0]
            probs = [1.0 if c == pred else 0.0 for c in self.classes_]

        prob_dict = {cls_name: float(probs[idx]) for idx, cls_name in enumerate(self.classes_)}
        p_high = prob_dict.get(config.TIER_HIGH, 0.0)
        p_med = prob_dict.get(config.TIER_MEDIUM, 0.0)
        p_low = prob_dict.get(config.TIER_LOW, 0.0)

        # 4. Clinical Threshold Tuning: Prioritize HIGH recall to prevent missing emergencies
        # Urgency score measures clinical severity (0.0 = minimal risk, 1.0 = emergency)
        if p_high >= 0.28:
            urgency = config.TIER_HIGH
            score = max(p_high, 0.75)
        elif p_med >= 0.35:
            urgency = config.TIER_MEDIUM
            score = max(p_med, 0.45)
        else:
            urgency = config.TIER_LOW
            # Severity score for low tier is residual non-benign probability
            score = min(round(1.0 - p_low, 4), config.RISK_LOW_MAX - 0.05)

        # 5. Low-Confidence Fallback: escalate to MEDIUM if confidence is below threshold
        max_prob = max(p_high, p_med, p_low)
        low_confidence_escalated = False
        if max_prob < config.NLP_LOW_CONFIDENCE_THRESHOLD and urgency == config.TIER_LOW:
            urgency = config.TIER_MEDIUM
            score = 0.50
            low_confidence_escalated = True

        # 6. Explainability: Token Attributions
        target_idx = self.classes_.index(urgency) if urgency in self.classes_ else 0
        top_tokens = extract_token_attributions(
            text=text,
            preprocessed_text=preprocessed,
            vectorizer=self.vectorizer,
            classifier=self.classifier,
            target_class_idx=target_idx,
            top_k=6
        )

        explanation = generate_symptom_explanation(
            urgency=urgency,
            red_flag_triggered=False,
            red_flags=[],
            top_tokens=top_tokens,
            confidence=max_prob
        )
        if low_confidence_escalated:
            explanation += " (Low confidence in primary assessment; escalated to MEDIUM for safety caution)."

        return {
            "module": "symptom_triage",
            "urgency": urgency,
            "score": round(float(score), 4),
            "explanation": explanation,
            "details": {
                "validation_status": "VALID",
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
                "red_flag_triggered": False,
                "red_flags": [],
                "top_tokens": top_tokens,
                "low_confidence_fallback": low_confidence_escalated,
                "preprocessed_text": preprocessed
            }
        }


# Singleton model cache
_cached_symptom_model: Optional[SymptomTriageModel] = None


def predict_urgency(text: str) -> Dict[str, Any]:
    """Exposed standard function for symptom triage."""
    global _cached_symptom_model
    if _cached_symptom_model is None or _cached_symptom_model.classifier is None:
        _cached_symptom_model = SymptomTriageModel()
    return _cached_symptom_model.predict(text)
