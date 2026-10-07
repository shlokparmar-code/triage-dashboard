"""Explainability module for Symptom Triage NLP.

Extracts token-level feature weights to highlight influential clinical words
and produces human-interpretable clinical explanations.
"""

from typing import Dict, List, Tuple
import numpy as np


def extract_token_attributions(
    text: str,
    preprocessed_text: str,
    vectorizer,
    classifier,
    target_class_idx: int,
    top_k: int = 6
) -> List[Dict]:
    """Calculate token-level attributions for the predicted urgency class.

    Returns a list of token dictionaries with token, weight, and clinical polarity.
    """
    if vectorizer is None or classifier is None:
        return []

    try:
        feature_names = vectorizer.get_feature_names_out()
        tfidf_vec = vectorizer.transform([preprocessed_text])
        non_zero_indices = tfidf_vec.indices

        if len(non_zero_indices) == 0:
            return []

        # Extract model coefficients for target class
        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_[target_class_idx]
        elif hasattr(classifier, "calibrated_classifiers_"):
            # Average coefficients across calibrated folds if linear base estimator
            base_estimator = classifier.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, "coef_"):
                coefs = np.mean([clf.estimator.coef_[target_class_idx] for clf in classifier.calibrated_classifiers_], axis=0)
            else:
                return []
        else:
            return []

        token_weights = []
        for idx in non_zero_indices:
            feat_name = feature_names[idx]
            tfidf_val = tfidf_vec[0, idx]
            weight = float(coefs[idx] * tfidf_val)
            token_weights.append({
                "token": feat_name,
                "weight": round(weight, 4),
                "influence": "escalating" if weight > 0 else "de-escalating"
            })

        # Sort by magnitude of contribution
        token_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return token_weights[:top_k]

    except Exception:
        return []


def generate_symptom_explanation(
    urgency: str,
    red_flag_triggered: bool,
    red_flags: List[str],
    top_tokens: List[Dict],
    confidence: float
) -> str:
    """Compose a plain-English clinical rationale based on predictions and attributions."""
    if red_flag_triggered:
        triggers_str = ", ".join(red_flags)
        return f"CRITICAL SAFETY OVERRIDE: {triggers_str}. Immediate medical evaluation required."

    if not top_tokens:
        return f"Symptoms triaged as {urgency} (confidence: {confidence:.0%}) based on overall clinical presentation."

    positive_tokens = [t["token"] for t in top_tokens if t["weight"] > 0]
    if positive_tokens:
        influential = ", ".join([f"'{t}'" for t in positive_tokens[:3]])
        if urgency == "HIGH":
            return f"High-risk clinical indicators identified: {influential}. Priority evaluation recommended (confidence: {confidence:.0%})."
        elif urgency == "MEDIUM":
            return f"Key urgent features observed: {influential}. Timely physician assessment advised (confidence: {confidence:.0%})."
        else:
            return f"Features consistent with mild self-limiting condition: {influential}. Supportive self-care indicated (confidence: {confidence:.0%})."
    
    return f"Assessed as {urgency} urgency (confidence: {confidence:.0%})."
