"""Special Populations Rule-Based Safety Layer.

Covers:
- WHO IMNCI danger signs for children under 5.
- Age-adjusted pediatric vital sign thresholds.
- Maternal red flags (preeclampsia, eclampsia, hemorrhage, sepsis).
- Module A exclusion for age < 18.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

logger = logging.getLogger(__name__)

_PEDIATRIC_DATA = None


def load_pediatric_data() -> Dict[str, Any]:
    """Load pediatric ranges and danger signs definition from JSON file."""
    global _PEDIATRIC_DATA
    if _PEDIATRIC_DATA is not None:
        return _PEDIATRIC_DATA

    filepath = Path(config.PEDIATRIC_RANGES_PATH)
    if not filepath.exists():
        # Fallback to default path relative to project
        filepath = Path(__file__).resolve().parents[2] / "data" / "pediatric_ranges.json"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            _PEDIATRIC_DATA = json.load(f)
            return _PEDIATRIC_DATA
    except Exception as e:
        logger.error(f"Failed to load pediatric ranges from {filepath}: {e}")
        return {
            "age_brackets": [],
            "who_imnci_danger_signs_under5": [],
            "pregnancy_red_flags": []
        }


def detect_patient_category(
    age_years: float,
    sex: str,
    is_pregnant: bool = False,
    is_postpartum: bool = False
) -> str:
    """Classify patient into broad triage clinical population."""
    sex_norm = str(sex).strip().capitalize()
    if (is_pregnant or is_postpartum) and sex_norm in ["Female", "F"]:
        return "Pregnant / Postpartum"
    if age_years < 5.0:
        return "Pediatric (<5)"
    if age_years < 18.0:
        return "Pediatric (5-18)"
    return "Adult"


def get_pediatric_bracket(age_months: float) -> Optional[Dict[str, Any]]:
    """Return matching age bracket dictionary based on age in months."""
    data = load_pediatric_data()
    brackets = data.get("age_brackets", [])
    for b in brackets:
        min_m = b.get("min_age_months", 0)
        max_m = b.get("max_age_months", 9999)
        if min_m <= age_months < max_m:
            return b
    if brackets and age_months >= 216:
        return brackets[-1]
    return brackets[0] if brackets else None


def evaluate_pediatric_vitals(
    age_months: float,
    vitals: Dict[str, Any]
) -> Dict[str, Any]:
    """Check pediatric vitals against age-bracket norms.

    Returns dict with warnings, fast_breathing flag, and severity.
    """
    bracket = get_pediatric_bracket(age_months)
    if not bracket:
        return {"abnormal": False, "warnings": [], "fast_breathing": False, "bracket_name": "Unknown"}

    warnings = []
    is_severe = False
    is_moderate = False
    fast_breathing = False

    hr = vitals.get("heart_rate") or vitals.get("hr")
    rr = vitals.get("respiratory_rate") or vitals.get("rr")
    spo2 = vitals.get("spo2") or vitals.get("o2_sat")
    temp = vitals.get("temperature") or vitals.get("temp")

    # Heart Rate
    hr_norm = bracket.get("hr_normal", [60, 100])
    if hr is not None:
        try:
            hr_val = float(hr)
            if hr_val < hr_norm[0] * 0.7 or hr_val > hr_norm[1] * 1.3:
                warnings.append(f"Critical Heart Rate: {hr_val:.0f} bpm (Age norm: {hr_norm[0]}-{hr_norm[1]})")
                is_severe = True
            elif hr_val < hr_norm[0] or hr_val > hr_norm[1]:
                warnings.append(f"Abnormal Heart Rate: {hr_val:.0f} bpm (Age norm: {hr_norm[0]}-{hr_norm[1]})")
                is_moderate = True
        except (ValueError, TypeError):
            pass

    # Respiratory Rate & Fast Breathing check
    rr_norm = bracket.get("rr_normal", [12, 24])
    rr_fast_thresh = bracket.get("rr_fast_threshold", 40)
    if rr is not None:
        try:
            rr_val = float(rr)
            if rr_val >= rr_fast_thresh:
                fast_breathing = True
                warnings.append(f"Fast Breathing for age (WHO criteria): {rr_val:.0f} breaths/min (Threshold: >={rr_fast_thresh})")
                is_severe = True
            elif rr_val > rr_norm[1] or rr_val < rr_norm[0]:
                warnings.append(f"Abnormal Respiratory Rate: {rr_val:.0f} breaths/min (Age norm: {rr_norm[0]}-{rr_norm[1]})")
                is_moderate = True
        except (ValueError, TypeError):
            pass

    # SpO2
    spo2_min = bracket.get("spo2_normal_min", 95.0)
    if spo2 is not None:
        try:
            spo2_val = float(spo2)
            if spo2_val < 90.0:
                warnings.append(f"Severe Hypoxemia: SpO2 {spo2_val:.1f}% (< 90%)")
                is_severe = True
            elif spo2_val < spo2_min:
                warnings.append(f"Low Oxygen Saturation: SpO2 {spo2_val:.1f}% (Norm: >={spo2_min:.0f}%)")
                is_moderate = True
        except (ValueError, TypeError):
            pass

    # Temperature
    temp_norm = bracket.get("temp_normal", [36.5, 37.5])
    if temp is not None:
        try:
            temp_val = float(temp)
            if temp_val >= 39.0 or temp_val < 35.5:
                warnings.append(f"High Fever / Hypothermia: {temp_val:.1f}°C (Norm: {temp_norm[0]}-{temp_norm[1]}°C)")
                is_severe = True
            elif temp_val > temp_norm[1] or temp_val < temp_norm[0]:
                warnings.append(f"Mild Temperature Deviation: {temp_val:.1f}°C")
                is_moderate = True
        except (ValueError, TypeError):
            pass

    return {
        "bracket_name": bracket.get("name", "Unknown"),
        "warnings": warnings,
        "is_severe": is_severe,
        "is_moderate": is_moderate,
        "fast_breathing": fast_breathing,
        "hr_normal": hr_norm,
        "rr_normal": rr_norm,
        "spo2_normal_min": spo2_min
    }


def evaluate_special_population(
    patient_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate patient for pediatric danger signs or maternal red flags.

    Follows standard module interface:
    {
        "module": "Special Population",
        "urgency": "LOW" | "MEDIUM" | "HIGH",
        "score": float (0-1),
        "explanation": str,
        "details": dict
    }
    """
    age_years = float(patient_data.get("age", 30))
    age_months = float(patient_data.get("age_months", age_years * 12.0))
    sex = str(patient_data.get("sex", "Male")).strip()
    is_pregnant = bool(patient_data.get("is_pregnant", False))
    is_postpartum = bool(patient_data.get("is_postpartum", False))
    gestational_weeks = patient_data.get("gestational_weeks")

    bp_sys = patient_data.get("systolic_bp") or patient_data.get("resting_bp") or patient_data.get("trestbps")
    bp_dia = patient_data.get("diastolic_bp") or patient_data.get("resting_bp_dia")
    try:
        bp_sys = float(bp_sys) if bp_sys is not None else None
    except (ValueError, TypeError):
        bp_sys = None
    try:
        bp_dia = float(bp_dia) if bp_dia is not None else None
    except (ValueError, TypeError):
        bp_dia = None

    vitals = patient_data.get("vitals", {})
    if not isinstance(vitals, dict):
        vitals = {}

    category = detect_patient_category(age_years, sex, is_pregnant, is_postpartum)
    module_a_applicable = (age_years >= 18.0)

    danger_signs_found: List[str] = []
    red_flags_found: List[str] = []
    preeclampsia_risk = False
    emergency_instructions: List[str] = []

    # 1. Pediatric Under-5 Evaluation (WHO IMNCI)
    vitals_eval = {}
    if age_years < 18.0:
        vitals_eval = evaluate_pediatric_vitals(age_months, vitals)

    child_signs = patient_data.get("child_danger_signs") or []
    if isinstance(child_signs, str):
        child_signs = [child_signs]

    if age_years < 5.0 or category == "Pediatric (<5)":
        data = load_pediatric_data()
        imnci_defs = {s["id"]: s["label"] for s in data.get("who_imnci_danger_signs_under5", [])}

        for sign in child_signs:
            label = imnci_defs.get(sign, sign)
            danger_signs_found.append(label)

        if vitals_eval.get("fast_breathing"):
            if "Fast breathing for age" not in danger_signs_found:
                danger_signs_found.append("Fast breathing for age (WHO criteria)")

        if vitals_eval.get("is_severe"):
            for w in vitals_eval.get("warnings", []):
                if w not in danger_signs_found:
                    danger_signs_found.append(w)

        if danger_signs_found:
            emergency_instructions = [
                "Keep the child warm and dry; wrap in a blanket.",
                "Do NOT force solid foods or oral fluids if the child is lethargic, convulsing, or vomiting everything.",
                "If breastfeeding and child can swallow, offer small, frequent breast milk feeds.",
                "If convulsing, place child gently on their side in recovery position, clear airway, do NOT place anything in mouth.",
                "Transport immediately to the nearest hospital or emergency pediatric facility."
            ]
    elif age_years < 18.0:
        # Older pediatric
        if vitals_eval.get("is_severe"):
            for w in vitals_eval.get("warnings", []):
                danger_signs_found.append(w)
            emergency_instructions = [
                "Keep patient at rest and monitor breathing.",
                "Transport immediately for pediatric medical evaluation."
            ]

    # 2. Maternal / Pregnancy Red Flag Evaluation
    preg_flags = patient_data.get("pregnancy_red_flags") or []
    if isinstance(preg_flags, str):
        preg_flags = [preg_flags]

    preg_concerns = patient_data.get("pregnancy_concerns") or []
    if isinstance(preg_concerns, str):
        preg_concerns = [preg_concerns]

    if is_pregnant or is_postpartum or category == "Pregnant / Postpartum":
        data = load_pediatric_data()
        preg_defs = {s["id"]: s for s in data.get("pregnancy_red_flags", [])}

        has_severe_headache_or_vision = False

        for flag in preg_flags:
            item = preg_defs.get(flag)
            label = item["label"] if item else flag
            red_flags_found.append(label)
            if item and item.get("preeclampsia"):
                has_severe_headache_or_vision = True
            elif "headache" in str(flag).lower() or "vision" in str(flag).lower() or "convulsion" in str(flag).lower() or "swelling" in str(flag).lower():
                has_severe_headache_or_vision = True

        # Check blood pressure for preeclampsia/hypertensive disorders of pregnancy
        is_hypertensive = (bp_sys is not None and bp_sys >= 140) or (bp_dia is not None and bp_dia >= 90)
        is_severe_hypertensive = (bp_sys is not None and bp_sys >= 160) or (bp_dia is not None and bp_dia >= 110)

        if is_severe_hypertensive:
            red_flags_found.append(f"Severe Gestational Hypertension (BP: {bp_sys:.0f}/{bp_dia if bp_dia else '?'})")
            preeclampsia_risk = True

        if is_hypertensive and has_severe_headache_or_vision:
            preeclampsia_risk = True
            if "Suspected Preeclampsia / Eclampsia" not in red_flags_found:
                red_flags_found.append(f"Suspected Preeclampsia with Neurological Signs (BP >= 140/90 with headache/visual changes)")

        if red_flags_found:
            emergency_instructions = [
                "Position patient on her left side (lateral tilt) to optimize placental perfusion.",
                "Do NOT give unprescribed medications or traditional remedies.",
                "If convulsing (eclampsia), keep patient on left side, maintain patent airway, do not restrain.",
                "In case of vaginal bleeding, note flow and keep warm; do NOT insert anything vaginally.",
                "Seek immediate emergency obstetric transfer to a Comprehensive Emergency Obstetric and Newborn Care (CEmONC) facility."
            ]

    # 3. Urgency and Score Determination
    if danger_signs_found or red_flags_found:
        urgency = config.TIER_HIGH
        score = 0.95
        all_reasons = danger_signs_found + red_flags_found
        explanation = (
            f"CRITICAL SAFETY ALERT: Detected {len(all_reasons)} special-population emergency sign(s): "
            + "; ".join(all_reasons[:3])
            + ("..." if len(all_reasons) > 3 else ".")
        )
    elif preg_concerns or (vitals_eval.get("is_moderate") and age_years < 18.0):
        urgency = config.TIER_MEDIUM
        score = 0.55
        reasons = preg_concerns + vitals_eval.get("warnings", [])
        explanation = (
            f"Moderate concern in special population: "
            + "; ".join(reasons[:2])
            + ". Requires clinical evaluation within 24-48 hours."
        )
    else:
        urgency = config.TIER_LOW
        score = 0.15
        if not module_a_applicable:
            explanation = f"Patient category: {category} (age {age_years:.1f}y). No maternal/pediatric acute danger signs detected."
        else:
            explanation = "Special population safety screening: Normal adult parameters, no maternal flags."

    details = {
        "category": category,
        "age_years": age_years,
        "age_months": age_months,
        "module_a_applicable": module_a_applicable,
        "module_a_status": "Applicable" if module_a_applicable else "Not applicable (adult model)",
        "danger_signs_detected": danger_signs_found,
        "red_flags_detected": red_flags_found,
        "preeclampsia_risk": preeclampsia_risk,
        "vitals_evaluation": vitals_eval,
        "emergency_instructions": emergency_instructions
    }

    return {
        "module": "Special Population",
        "urgency": urgency,
        "score": score,
        "explanation": explanation,
        "details": details
    }
