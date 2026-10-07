"""Doctor Handoff Summary generator using SBAR (Situation-Background-Assessment-Recommendation) protocol."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


def load_referral_levels() -> Dict[str, Any]:
    """Load referral level configuration from data/referral_levels.json."""
    ref_path = config.REFERRAL_LEVELS_PATH
    if ref_path.exists():
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "LOW": {
            "level": "Primary Self-Care & Community Health",
            "facility": "Home / Community Health Post",
            "timeline": "Routine / Elective",
            "action": "Supportive self-care, oral hydration, rest."
        },
        "MEDIUM": {
            "level": "Primary Care Physician Consultation",
            "facility": "Primary Health Centre (PHC) / Community Health Centre (CHC)",
            "timeline": "Within 24 to 48 hours",
            "action": "Outpatient clinic evaluation and diagnostic labs."
        },
        "HIGH": {
            "level": "Emergency-Capable Hospital Care",
            "facility": "Sub-District Hospital / District Hospital / Trauma Centre",
            "timeline": "Immediate (Emergency Transport)",
            "action": "Emergency physician evaluation, airway & hemodynamic stabilization."
        }
    }


def generate_handoff(
    patient: Dict[str, Any],
    module_results: Dict[str, Any],
    final_result: Dict[str, Any],
    referral: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    hospital_name: str = "District General Hospital"
) -> str:
    """Generate structured SBAR clinical handoff documentation.

    Handles missing fields gracefully.
    """
    patient = patient or {}
    module_results = module_results or {}
    final_result = final_result or {}
    history = history or []

    urgency = final_result.get("final_urgency", "LOW")
    score = final_result.get("final_score", 0.0)

    # Referral details
    if not referral:
        all_refs = load_referral_levels()
        referral = all_refs.get(urgency, all_refs["LOW"])

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p_name = patient.get("name", "Unknown Patient")
    p_id = patient.get("patient_id", "PAT-UNASSIGNED")
    p_age = patient.get("age", "N/A")
    p_sex = patient.get("sex", "N/A")
    complaint = patient.get("complaint", patient.get("symptoms", "No complaint recorded"))

    # Prior visit summary from registry
    history_summary = "No prior recorded visits across network hospitals."
    if history:
        prior_visits = []
        for v in history[:3]:
            h_name = v.get("hospital_name", "Hospital")
            v_urg = v.get("final_urgency", "N/A")
            v_time = v.get("timestamp", "").split()[0]
            prior_visits.append(f"{v_time} at {h_name} ({v_urg} Urgency)")
        history_summary = "; ".join(prior_visits)

    # Special population background
    special_pop = module_results.get("special_populations", {})
    special_notes = []
    if special_pop and special_pop.get("details"):
        sp_details = special_pop["details"]
        if sp_details.get("is_pregnant"):
            special_notes.append(f"Pregnancy ({sp_details.get('gestation_weeks', 'unknown')} weeks)")
        if sp_details.get("is_child"):
            special_notes.append(f"Pediatric patient ({sp_details.get('age_months', 0) // 12} yrs)")
        for flag in sp_details.get("danger_signs_triggered", []):
            special_notes.append(f"Danger Sign: {flag}")

    special_bg_str = ", ".join(special_notes) if special_notes else "Standard adult triage"

    # Module assessments
    mod_a = module_results.get("risk_screening", {})
    mod_b = module_results.get("vitals_anomaly", {})
    mod_c = module_results.get("symptom_triage", {})

    lines = [
        "================================================================================",
        f" CLINICAL SBAR HANDOFF SUMMARY — {hospital_name.upper()}",
        "================================================================================",
        f"Generated At: {timestamp_str} | Patient ID: {p_id}",
        "",
        "--------------------------------------------------------------------------------",
        " [S] SITUATION",
        "--------------------------------------------------------------------------------",
        f"• Patient: {p_name} ({p_age} yrs, {p_sex})",
        f"• Presenting Chief Complaint: {complaint}",
        f"• Triage Urgency Level: {urgency} (Calibrated Score: {score:.0%})",
        f"• Safety Override Active: {'YES' if final_result.get('safety_override') else 'NO'}",
        f"• Priority Lane Status: {'YES (Immediate Care Required)' if urgency == 'HIGH' else 'Standard Triage Queue'}",
        "",
        "--------------------------------------------------------------------------------",
        " [B] BACKGROUND",
        "--------------------------------------------------------------------------------",
        f"• Patient Category & Special Population: {special_bg_str}",
        f"• Chronic Risk Baseline: {mod_a.get('explanation', 'Not evaluated')}",
        f"• Multi-Hospital Longitudinal History: {history_summary}",
        "",
        "--------------------------------------------------------------------------------",
        " [A] CLINICAL ASSESSMENT",
        "--------------------------------------------------------------------------------",
        f"• Module C (Symptom NLP): {mod_c.get('urgency', 'N/A')} Urgency ({mod_c.get('score', 0):.0%})",
        f"  Rationale: {mod_c.get('explanation', 'N/A')}",
        f"• Module B (Vitals Telemetry): {mod_b.get('urgency', 'N/A')} Urgency ({mod_b.get('score', 0):.0%})",
        f"  Telemetry Findings: {mod_b.get('explanation', 'N/A')}",
        f"• Module A (Tabular Chronic Risk): {mod_a.get('urgency', 'N/A')} Urgency ({mod_a.get('score', 0):.0%})",
        f"• Multi-Modal Clinical Synthesis: {final_result.get('clinical_synthesis', 'Assessment pending')}",
        "",
        "--------------------------------------------------------------------------------",
        " [R] RECOMMENDATION & REFERRAL DIRECTIVE",
        "--------------------------------------------------------------------------------",
        f"• Recommended Care Level: {referral.get('level', 'Clinical Consultation')}",
        f"• Target Healthcare Facility: {referral.get('facility', hospital_name)}",
        f"• Referral Timeline: {referral.get('timeline', 'Immediate')}",
        f"• Clinical Action Plan: {referral.get('action', 'Evaluate patient according to standard triage guidelines.')}",
        "",
        "--------------------------------------------------------------------------------",
        "REGULATORY & CLINICAL DISCLAIMER:",
        "AI decision-support output. Not a medical diagnosis. Requires review by a qualified clinician.",
        "================================================================================"
    ]

    return "\n".join(lines)
