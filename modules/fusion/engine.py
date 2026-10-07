"""Master Urgency Fusion and Recommendation Orchestration Engine."""

import logging
from typing import Any, Dict, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.fusion.recommendations import get_recommendations

logger = logging.getLogger(__name__)


def fuse_triage_modalities(
    risk_result: Dict[str, Any],
    vitals_result: Dict[str, Any],
    symptom_result: Dict[str, Any],
    patient_features: Optional[Dict[str, Any]] = None,
    emergency_country: str = "India"
) -> Dict[str, Any]:
    """Combine Risk Screening, Vitals Anomaly, and Symptom Triage into ONE explained urgency score.

    Transparent method:
    - Base: Weighted average of individual module scores.
    - Safety Override: If ANY module returns HIGH urgency, final urgency is FORCED to HIGH.

    Args:
        risk_result: Standard output dict from Module A.
        vitals_result: Standard output dict from Module B.
        symptom_result: Standard output dict from Module C.
        patient_features: Raw clinical attributes for personalization.
        emergency_country: Country for emergency dispatcher contact info.

    Returns:
        Consolidated multi-modal triage dictionary.
    """
    weights = config.FUSION_WEIGHTS
    w_sym = weights.get("symptom_triage", 0.45)
    w_vit = weights.get("vitals_anomaly", 0.35)
    w_rsk = weights.get("risk_screening", 0.20)

    score_sym = float(symptom_result.get("score", 0.0))
    score_vit = float(vitals_result.get("score", 0.0))
    score_rsk = float(risk_result.get("score", 0.0))

    urgency_sym = symptom_result.get("urgency", config.TIER_LOW)
    urgency_vit = vitals_result.get("urgency", config.TIER_LOW)
    urgency_rsk = risk_result.get("urgency", config.TIER_LOW)

    # 1. Base Weighted Score Calculation
    weighted_score = (w_sym * score_sym) + (w_vit * score_vit) + (w_rsk * score_rsk)
    weighted_score = round(float(weighted_score), 4)

    # 2. Hard Clinical Safety Override: If ANY module is HIGH, final urgency is strictly HIGH
    has_high_module = (
        urgency_sym == config.TIER_HIGH
        or urgency_vit == config.TIER_HIGH
        or urgency_rsk == config.TIER_HIGH
    )

    overriding_modules = []
    if urgency_sym == config.TIER_HIGH:
        overriding_modules.append("Symptom Triage (NLP Red-Flag / Acute)")
    if urgency_vit == config.TIER_HIGH:
        overriding_modules.append("Vitals Anomaly (Severe Physiological Instability)")
    if urgency_rsk == config.TIER_HIGH:
        overriding_modules.append("Risk Screening (High Chronic Liability)")

    if has_high_module:
        final_urgency = config.TIER_HIGH
        final_score = max(weighted_score, 0.85, score_sym, score_vit, score_rsk)
        safety_override = True
    elif weighted_score >= config.RISK_MEDIUM_MAX:
        final_urgency = config.TIER_HIGH
        final_score = weighted_score
        safety_override = False
    elif weighted_score >= config.RISK_LOW_MAX or urgency_sym == config.TIER_MEDIUM or urgency_vit == config.TIER_MEDIUM:
        final_urgency = config.TIER_MEDIUM
        final_score = max(weighted_score, 0.45)
        safety_override = False
    else:
        final_urgency = config.TIER_LOW
        final_score = min(weighted_score, 0.30)
        safety_override = False

    # 3. 2-3 Sentence Plain-English Clinical Synthesis
    synthesis_sentences = []
    if final_urgency == config.TIER_HIGH:
        synthesis_sentences.append(
            f"URGENT CLINICAL ALERT: Patient triaged as HIGH urgency ({final_score:.0%}) due to acute indicators in {', '.join(overriding_modules)}."
        )
        v_exp = vitals_result.get("explanation", "").strip().rstrip(".")
        s_exp = symptom_result.get("explanation", "").strip().rstrip(".")
        synthesis_sentences.append(
            f"Vitals presentation reveals {v_exp} while symptoms indicate {s_exp}."
        )
        synthesis_sentences.append(
            "Immediate physician evaluation or emergency care transport is strongly indicated."
        )
    elif final_urgency == config.TIER_MEDIUM:
        synthesis_sentences.append(
            f"Patient is triaged as MEDIUM urgency ({final_score:.0%}), warranting formal physician consultation within 24 to 48 hours."
        )
        v_exp = vitals_result.get("explanation", "").strip()
        synthesis_sentences.append(
            f"Primary findings show moderate symptom burden. {v_exp}"
        )
        synthesis_sentences.append(
            "Begin outpatient monitoring and implement targeted supportive self-care in the interim."
        )
    else:
        synthesis_sentences.append(
            f"Patient presentation is triaged as LOW urgency ({final_score:.0%}) with stable vital parameters and no acute emergency flags."
        )
        r_exp = risk_result.get("explanation", "").strip()
        synthesis_sentences.append(
            f"{r_exp} Symptoms reflect mild self-limiting features."
        )
        synthesis_sentences.append(
            "Recommended for primary supportive home care, hydration, and healthy lifestyle practices."
        )

    clinical_synthesis = " ".join(synthesis_sentences)

    # 4. Personalized Recommendations
    recommendations = get_recommendations(
        urgency=final_urgency,
        patient_features=patient_features,
        emergency_country=emergency_country
    )

    # 5. Output
    return {
        "final_urgency": final_urgency,
        "final_score": round(final_score, 4),
        "safety_override": safety_override,
        "overriding_modules": overriding_modules,
        "clinical_synthesis": clinical_synthesis,
        "module_contributions": {
            "symptom_triage": {
                "urgency": urgency_sym,
                "score": score_sym,
                "weight": w_sym,
                "weighted_contribution": round(w_sym * score_sym, 4)
            },
            "vitals_anomaly": {
                "urgency": urgency_vit,
                "score": score_vit,
                "weight": w_vit,
                "weighted_contribution": round(w_vit * score_vit, 4)
            },
            "risk_screening": {
                "urgency": urgency_rsk,
                "score": score_rsk,
                "weight": w_rsk,
                "weighted_contribution": round(w_rsk * score_rsk, 4)
            }
        },
        "recommendations": recommendations,
        "disclaimer": recommendations.get("disclaimer", "")
    }
