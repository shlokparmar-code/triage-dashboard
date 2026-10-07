"""Personalized recommendation generator reading from editable knowledge base."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

logger = logging.getLogger(__name__)


def load_knowledge_base() -> Dict[str, Any]:
    """Load JSON knowledge base from models/ directory."""
    kb_path = config.MODELS_DIR / "recommendations_kb.json"
    if kb_path.exists():
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {kb_path}: {e}")
    return {}


def get_recommendations(
    urgency: str,
    patient_features: Optional[Dict[str, Any]] = None,
    emergency_country: str = "India"
) -> Dict[str, Any]:
    """Generate personalized recommendations tailored to patient urgency and clinical risk profile.

    Args:
        urgency: Overall urgency tier ('LOW', 'MEDIUM', 'HIGH').
        patient_features: Dict of measured patient metrics (BP, glucose, age, symptoms, etc.).
        emergency_country: Country for local emergency contact info.

    Returns:
        Structured recommendation dictionary.
    """
    kb = load_knowledge_base()
    patient_features = patient_features or {}

    bp = float(patient_features.get("resting_bp", patient_features.get("trestbps", 120)))
    glucose = float(patient_features.get("glucose", 100))
    age = float(patient_features.get("age", 45))
    symptom_text = str(patient_features.get("symptom_text", "")).lower()

    disclaimer = kb.get(
        "disclaimer",
        "DISCLAIMER: This multi-modal triage tool is an AI-powered clinical decision-support prototype. It does NOT constitute a medical diagnosis."
    )

    if urgency == config.TIER_HIGH:
        emergency_data = kb.get("emergency_high", {})
        emergency_numbers = emergency_data.get("emergency_numbers", {})
        chosen_number = emergency_numbers.get(emergency_country, config.DEFAULT_EMERGENCY_NUMBER)

        return {
            "tier": config.TIER_HIGH,
            "is_emergency": True,
            "title": emergency_data.get("alert_title", "EMERGENCY CARE REQUIRED"),
            "action": emergency_data.get("primary_action", "Seek immediate emergency physician care."),
            "emergency_number": chosen_number,
            "while_waiting_steps": emergency_data.get("while_waiting_steps", [
                "Remain seated upright and still.",
                "Loosen restrictive clothing.",
                "Call emergency medical transport immediately."
            ]),
            "disclaimer": disclaimer
        }

    # LOW or MEDIUM: Lifestyle, Yoga, Diet, and Exercise Guidance
    all_asanas = kb.get("yoga_asanas", [])
    selected_asanas = []

    # Personalize Yoga Asanas
    if bp >= 135:
        # Match hypertension / cardiac risk asana
        for a in all_asanas:
            if "high_bp" in a.get("target_conditions", []):
                selected_asanas.append(a)
                break

    if glucose >= 125:
        # Match glycemic control asana
        for a in all_asanas:
            if "high_glucose" in a.get("target_conditions", []):
                selected_asanas.append(a)
                break

    if any(k in symptom_text for k in ["cough", "throat", "breath", "cold"]):
        for a in all_asanas:
            if "respiratory_mild" in a.get("target_conditions", []):
                selected_asanas.append(a)
                break

    # If no specific condition met, provide calming / general asana
    if not selected_asanas:
        for a in all_asanas:
            if "general_wellness" in a.get("target_conditions", []) or "stress" in a.get("target_conditions", []):
                selected_asanas.append(a)
                break

    # Personalize Dietary Guidance
    diet_guidelines = kb.get("dietary_guidelines", {})
    selected_diet = []

    if bp >= 135:
        selected_diet.append(diet_guidelines.get("hypertension", {}))
    if glucose >= 125:
        selected_diet.append(diet_guidelines.get("hyperglycemia", {}))
    if any(k in symptom_text for k in ["cough", "throat", "cold"]):
        selected_diet.append(diet_guidelines.get("respiratory_cold", {}))
    
    if not selected_diet:
        selected_diet.append(diet_guidelines.get("general_wellness", {}))

    # Physical Activity Guidance
    phys_act = kb.get("physical_activity", {})
    if bp >= 145 or glucose >= 160 or age >= 65:
        exercise_plan = phys_act.get("sedentary_or_high_risk", {}).get(
            "plan", "Gentle 20-30 min daily walking and joint mobility. Avoid strenuous sudden exertion."
        )
    else:
        exercise_plan = phys_act.get("moderate_fit", {}).get(
            "plan", "30-40 min moderate aerobic activity (brisk walking/cycling) 5 days per week."
        )

    return {
        "tier": urgency,
        "is_emergency": False,
        "title": "Preventive Lifestyle & Supportive Care Plan" if urgency == config.TIER_LOW else "Clinical Monitoring & Supportive Care Plan",
        "action": "Maintain routine self-care and preventive habits." if urgency == config.TIER_LOW else "Schedule an outpatient physician visit within 24-48 hours for clinical evaluation.",
        "yoga_asanas": selected_asanas,
        "dietary_guidelines": selected_diet,
        "exercise_plan": exercise_plan,
        "disclaimer": disclaimer
    }
