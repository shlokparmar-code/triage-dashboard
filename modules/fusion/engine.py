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
    emergency_country: str = "India",
    special_population_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Combine Risk Screening, Vitals Anomaly, and Symptom Triage into ONE explained urgency score.

    Transparent method:
    - Base: Weighted average of individual module scores.
    - Safety Override: If ANY module returns HIGH urgency, final urgency is FORCED to HIGH.
    - Special Populations: Any pediatric danger sign or pregnancy red flag forces HIGH urgency.
      If patient is under 18, Module A is marked "Not applicable (adult model)" and excluded from fusion.

    Args:
        risk_result: Standard output dict from Module A.
        vitals_result: Standard output dict from Module B.
        symptom_result: Standard output dict from Module C.
        patient_features: Raw clinical attributes for personalization.
        emergency_country: Country for emergency dispatcher contact info.
        special_population_result: Optional output dict from Special Population safety layer.

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

    # Check if Module A applies (adult only)
    is_pediatric = False
    if special_population_result:
        is_pediatric = not special_population_result.get("details", {}).get("module_a_applicable", True)
    elif patient_features and float(patient_features.get("age", 30)) < 18.0:
        is_pediatric = True

    # 1. Base Weighted Score Calculation
    if is_pediatric:
        # Exclude Module A, renormalize weights between Module B & C
        denom = w_sym + w_vit if (w_sym + w_vit) > 0 else 1.0
        w_sym_adj = round(w_sym / denom, 4)
        w_vit_adj = round(w_vit / denom, 4)
        w_rsk_adj = 0.0
        weighted_score = (w_sym_adj * score_sym) + (w_vit_adj * score_vit)
    else:
        w_sym_adj = w_sym
        w_vit_adj = w_vit
        w_rsk_adj = w_rsk
        weighted_score = (w_sym * score_sym) + (w_vit * score_vit) + (w_rsk * score_rsk)

    weighted_score = round(float(weighted_score), 4)

    # 2. Hard Clinical Safety Override: If ANY module is HIGH, final urgency is strictly HIGH
    urgency_spec = config.TIER_LOW
    score_spec = 0.0
    if special_population_result:
        urgency_spec = special_population_result.get("urgency", config.TIER_LOW)
        score_spec = float(special_population_result.get("score", 0.0))

    has_high_module = (
        urgency_sym == config.TIER_HIGH
        or urgency_vit == config.TIER_HIGH
        or (urgency_rsk == config.TIER_HIGH and not is_pediatric)
        or urgency_spec == config.TIER_HIGH
    )

    overriding_modules = []
    if urgency_sym == config.TIER_HIGH:
        overriding_modules.append("Symptom Triage (NLP Red-Flag / Acute)")
    if urgency_vit == config.TIER_HIGH:
        overriding_modules.append("Vitals Anomaly (Severe Physiological Instability)")
    if urgency_rsk == config.TIER_HIGH and not is_pediatric:
        overriding_modules.append("Risk Screening (High Chronic Liability)")
    if urgency_spec == config.TIER_HIGH:
        overriding_modules.append("Special Population (Maternal / Pediatric Danger Signs)")

    if has_high_module:
        final_urgency = config.TIER_HIGH
        candidates = [weighted_score, 0.85, score_sym, score_vit]
        if not is_pediatric:
            candidates.append(score_rsk)
        if special_population_result:
            candidates.append(score_spec)
        final_score = max(candidates)
        safety_override = True
    elif weighted_score >= config.RISK_MEDIUM_MAX:
        final_urgency = config.TIER_HIGH
        final_score = weighted_score
        safety_override = False
    elif (
        weighted_score >= config.RISK_LOW_MAX
        or urgency_sym == config.TIER_MEDIUM
        or urgency_vit == config.TIER_MEDIUM
        or urgency_spec == config.TIER_MEDIUM
    ):
        final_urgency = config.TIER_MEDIUM
        final_score = max(weighted_score, 0.45)
        safety_override = False
    else:
        final_urgency = config.TIER_LOW
        final_score = min(weighted_score, 0.30)
        safety_override = False

    # 3. 2-3 Sentence Plain-English Clinical Synthesis
    synthesis_sentences = []

    # Check for special population alerts to include first
    if special_population_result:
        spec_details = special_population_result.get("details", {})
        if spec_details.get("preeclampsia_risk"):
            synthesis_sentences.append(
                "CRITICAL OBSTETRIC ALERT: Suspected preeclampsia/eclampsia risk identified with elevated blood pressure and neurological signs."
            )
        elif spec_details.get("danger_signs_detected"):
            ds_text = ", ".join(spec_details["danger_signs_detected"][:2])
            synthesis_sentences.append(
                f"CRITICAL PEDIATRIC ALERT: Severe danger signs detected ({ds_text})."
            )

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
        if not is_pediatric:
            r_exp = risk_result.get("explanation", "").strip()
            synthesis_sentences.append(
                f"{r_exp} Symptoms reflect mild self-limiting features."
            )
        else:
            synthesis_sentences.append(
                "Pediatric screening confirms normal physiological ranges with mild self-limiting features."
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

    # Append emergency instructions from special populations if present
    if special_population_result and special_population_result.get("details", {}).get("emergency_instructions"):
        spec_inst = special_population_result["details"]["emergency_instructions"]
        existing_actions = recommendations.get("action_plan", [])
        recommendations["special_population_instructions"] = spec_inst
        # Prioritize special population instructions in action plan for HIGH
        if final_urgency == config.TIER_HIGH:
            recommendations["action_plan"] = spec_inst + existing_actions

    # 5. Output
    module_contrib = {
        "symptom_triage": {
            "urgency": urgency_sym,
            "score": score_sym,
            "weight": w_sym_adj,
            "weighted_contribution": round(w_sym_adj * score_sym, 4)
        },
        "vitals_anomaly": {
            "urgency": urgency_vit,
            "score": score_vit,
            "weight": w_vit_adj,
            "weighted_contribution": round(w_vit_adj * score_vit, 4)
        },
        "risk_screening": {
            "urgency": "N/A" if is_pediatric else urgency_rsk,
            "score": 0.0 if is_pediatric else score_rsk,
            "weight": 0.0 if is_pediatric else w_rsk_adj,
            "weighted_contribution": 0.0 if is_pediatric else round(w_rsk_adj * score_rsk, 4),
            "applicable": not is_pediatric,
            "status": "Not applicable (adult model)" if is_pediatric else "Active"
        }
    }

    if special_population_result:
        module_contrib["special_population"] = {
            "urgency": urgency_spec,
            "score": score_spec,
            "weight": "Safety Gate",
            "category": special_population_result.get("details", {}).get("category", "General")
        }

    return {
        "final_urgency": final_urgency,
        "final_score": round(final_score, 4),
        "safety_override": safety_override,
        "overriding_modules": overriding_modules,
        "clinical_synthesis": clinical_synthesis,
        "module_contributions": module_contrib,
        "special_population": special_population_result,
        "recommendations": recommendations,
        "disclaimer": recommendations.get("disclaimer", "")
    }
