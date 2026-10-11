"""Multi-Modal AI Medical Triage Dashboard.

A unified Streamlit application combining tabular chronic risk, time-series
physiological vitals telemetry, NLP free-text symptom triage, maternal/pediatric
danger signs, and multi-patient hospital queue into ONE explained urgency score.
Targets rural healthcare access gaps (UN SDG 3: Good Health and Well-Being).
"""

import html
import io
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import config
from dashboard.components import (
    highlight_symptom_tokens,
    plot_vitals_telemetry,
    render_urgency_banner,
    render_urgency_chip,
)
from dashboard.mode import (
    get_mode,
    get_mode_config,
    is_home,
    is_hospital,
    render_mode_selector,
)
from dashboard.home_support import (
    collect_concerns,
    render_contact_alert,
    render_nearby_hospitals,
    render_wellness_plan,
)
from dashboard.override import override_summary, render_clinician_override
from dashboard.pdf_export import generate_home_pdf, generate_triage_pdf
from dashboard.pdf_simple import history_report_pdf, text_to_pdf
from dashboard.queue import add_to_queue, render_queue_ui
from dashboard.styles import CUSTOM_CSS
from modules.fusion.engine import fuse_triage_modalities
from modules.fusion.handoff import generate_handoff, load_referral_levels
from modules import contact_alert, facilities, reference_ranges, wellness
from modules.registry import (
    LocalRegistryProvider,
    authenticate_doctor,
    check_clinical_deterioration,
    generate_otp,
    list_doctors,
    mask_mobile,
    validate_mobile,
    verify_otp,
)
from modules.risk_screening.explain import plot_shap_waterfall_bar
from modules.risk_screening.model import RiskScreeningModel
from modules.special_populations import (
    detect_patient_category,
    evaluate_special_population,
    load_pediatric_data,
)
from modules.symptom_triage.model import SymptomTriageModel
from modules.vitals_anomaly.generator import generate_synthetic_vitals
from modules.vitals_anomaly.model import VitalsAnomalyDetector

# ==========================================
# STREAMLIT PAGE CONFIG & STYLES
# ==========================================
st.set_page_config(
    page_title="AI Medical Triage Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Permanently fixed sidebar: hide every collapse/expand control variant.
st.markdown(
    """
    <style>
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"],
    button[kind="headerNoPadding"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# RESOURCE CACHING
# ==========================================
@st.cache_resource
def get_symptom_model():
    return SymptomTriageModel()


@st.cache_resource
def get_risk_model():
    return RiskScreeningModel()


@st.cache_resource
def get_vitals_model():
    return VitalsAnomalyDetector()


@st.cache_resource
def get_registry_provider():
    return LocalRegistryProvider()


symptom_model = get_symptom_model()
risk_model = get_risk_model()
vitals_model = get_vitals_model()
registry = get_registry_provider()

_AUDIT_READER_NAMES = (
    "get_audit_log", "get_audit_logs", "get_audit_trail", "get_patient_audit_log",
    "get_access_log", "get_audit_history", "list_audit_log", "list_audit_logs",
)


def get_audit_entries(patient_id):
    """Read audit entries, whatever the registry names its reader method.

    Returns a list of dicts, or None if no reader method could be found/used.
    """
    sources = [registry]
    try:
        from modules.registry import db as _registry_db
        sources.append(_registry_db)
    except ImportError:
        pass
    for source in sources:
        for name in _AUDIT_READER_NAMES:
            fn = getattr(source, name, None)
            if not callable(fn):
                continue
            try:
                rows = fn(patient_id)
            except Exception:
                continue
            entries = []
            for r in rows or []:
                try:
                    entries.append(dict(r))
                except (TypeError, ValueError):
                    continue
            return entries
    return None


def format_audit_line(a, with_role=True):
    """One readable line for an audit entry; tolerant of different key names."""
    when = a.get("timestamp", "")
    who = a.get("who", "unknown")
    role = a.get("role", "")
    place = a.get("hospital_name") or a.get("hospital") or a.get("hospital_id") or ""
    action = a.get("action", "")
    who_part = f"{who} ({role})" if with_role and role else who
    place_part = f" at {place}" if place else ""
    action_part = f" - *{action}*" if action else ""
    return f"• **{when}** | {who_part}{place_part}{action_part}"


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "usage_mode" not in st.session_state:
    st.session_state["usage_mode"] = config.DEFAULT_MODE

if "patient_name" not in st.session_state:
    st.session_state["patient_name"] = "Sunita Devi"
if "patient_country_code" not in st.session_state:
    st.session_state["patient_country_code"] = "+91"
if "patient_mobile_raw" not in st.session_state:
    st.session_state["patient_mobile_raw"] = "9876543210"
if "patient_consent" not in st.session_state:
    st.session_state["patient_consent"] = True
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = ""

if "symptom_text" not in st.session_state:
    st.session_state["symptom_text"] = "Mild scratchy throat and runny nose for 2 days, no fever"
if "patient_age" not in st.session_state:
    st.session_state["patient_age"] = 28
if "patient_age_months" not in st.session_state:
    st.session_state["patient_age_months"] = 336
if "patient_sex" not in st.session_state:
    st.session_state["patient_sex"] = "Female"
# Vitals + chronic baseline start at STANDARD adult reference values
# (see modules/reference_ranges.py for the values and their sources).
for _key, _val in reference_ranges.session_defaults().items():
    if _key not in st.session_state:
        st.session_state[_key] = _val
# Version counter baked into the vitals/baseline widget keys. Bumping it forces
# Streamlit to rebuild those boxes from session_state (used by Reset + presets).
if "_std_ver" not in st.session_state:
    st.session_state["_std_ver"] = 0


def _reset_standard_values():
    """Button callback: put every vital / baseline box back to the standard value."""
    reference_ranges.reset_to_standard(st.session_state)
    st.session_state["vitals_anomaly_type"] = "none"
    st.session_state["_std_ver"] += 1


# Maternal & Pediatric states
if "is_pregnant" not in st.session_state:
    st.session_state["is_pregnant"] = False
if "gestational_weeks" not in st.session_state:
    st.session_state["gestational_weeks"] = 24
if "is_postpartum" not in st.session_state:
    st.session_state["is_postpartum"] = False
if "pregnancy_red_flags" not in st.session_state:
    st.session_state["pregnancy_red_flags"] = []
if "child_danger_signs" not in st.session_state:
    st.session_state["child_danger_signs"] = []
if "vitals_anomaly_type" not in st.session_state:
    st.session_state["vitals_anomaly_type"] = "none"

# (Manual Home-mode vitals manual_hr / manual_spo2 / manual_temp / manual_rr
#  are initialised above from reference_ranges.session_defaults().)

# Active Clinician context in Hospital mode
if "current_hospital_id" not in st.session_state:
    st.session_state["current_hospital_id"] = "HOSP-001"
if "active_doctor_name" not in st.session_state:
    st.session_state["active_doctor_name"] = "Dr. Priya Sharma (Triage Lead)"
if "otp_verified_mobile" not in st.session_state:
    st.session_state["otp_verified_mobile"] = ""


# ==========================================
# START SCREEN (Home / Hospital choice, shown before anything else)
# ==========================================
HOSPITAL_CHOICES = [
    ("HOSP-001", "Apex District Hospital, Jaipur"),
    ("HOSP-002", "Alwar Community Health Centre"),
    ("HOSP-003", "St. Jude Rural Clinic, Bangalore"),
    ("HOSP-004", "Metro General Hospital"),
]

if "mode_chosen" not in st.session_state:
    st.session_state["mode_chosen"] = False

if not st.session_state["mode_chosen"]:
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none !important;}</style>",
        unsafe_allow_html=True,
    )
    st.title("🩺 " + config.PROJECT_TITLE)
    st.caption("Decision-support prototype. Not a medical diagnosis.")
    st.subheader("How are you using this today?")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("### 🏠 Home Use\nPlain-language result, simple guidance, emergency help.")
        if st.button("Continue as Home Use", type="primary", use_container_width=True, key="_start_home"):
            st.session_state["usage_mode"] = config.MODE_HOME if hasattr(config, "MODE_HOME") else "home"
            st.session_state["mode_chosen"] = True
            st.rerun()
    with sc2:
        st.markdown("### 🏥 Hospital\nClinical detail, patient registry, triage queue, SBAR handoff.")
        hosp_ids = [h[0] for h in HOSPITAL_CHOICES]
        cur = st.session_state.get("current_hospital_id", "HOSP-001")
        start_hosp = st.selectbox(
            "Hospital location:",
            HOSPITAL_CHOICES,
            index=hosp_ids.index(cur) if cur in hosp_ids else 0,
            format_func=lambda x: x[1],
            key="_start_hosp_select",
        )
        if st.button("Continue as Hospital", type="primary", use_container_width=True, key="_start_hospital"):
            st.session_state["usage_mode"] = config.MODE_HOSPITAL if hasattr(config, "MODE_HOSPITAL") else "hospital"
            st.session_state["current_hospital_id"] = start_hosp[0]
            st.session_state["mode_chosen"] = True
            st.rerun()
    st.stop()


# ==========================================
# BENCHMARK PATIENT PRESETS LOADER
# ==========================================
def load_benchmark_patient(preset_key: str):
    """Load benchmark patient scenarios covering adult, pediatric, maternal, and multi-hospital flows."""
    st.session_state["_std_ver"] = st.session_state.get("_std_ver", 0) + 1  # refresh vitals/baseline boxes
    if preset_key == "ADULT_LOW":
        st.session_state["patient_name"] = "Aarav Sharma"
        st.session_state["patient_country_code"] = "+91"
        st.session_state["patient_mobile_raw"] = "9812345671"
        st.session_state["symptom_text"] = "Mild runny nose, sneezing for 2 days, no fever, no chest pain"
        st.session_state["patient_age"] = 28
        st.session_state["patient_sex"] = "Male"
        st.session_state["patient_bp"] = 115
        st.session_state["patient_bp_dia"] = 75
        st.session_state["patient_chol"] = 175
        st.session_state["patient_glucose"] = 88
        st.session_state["patient_bmi"] = 21.8
        st.session_state["is_pregnant"] = False
        st.session_state["pregnancy_red_flags"] = []
        st.session_state["child_danger_signs"] = []
        st.session_state["vitals_anomaly_type"] = "none"
        st.session_state["manual_hr"] = 72
        st.session_state["manual_spo2"] = 98.5
        st.session_state["manual_temp"] = 36.6
        st.session_state["manual_rr"] = 15

    elif preset_key == "ADULT_MEDIUM":
        st.session_state["patient_name"] = "Vikram Singh"
        st.session_state["patient_country_code"] = "+91"
        st.session_state["patient_mobile_raw"] = "9812345672"
        st.session_state["symptom_text"] = "High fever of 101F for 3 days with nausea and body ache"
        st.session_state["patient_age"] = 54
        st.session_state["patient_sex"] = "Male"
        st.session_state["patient_bp"] = 142
        st.session_state["patient_bp_dia"] = 88
        st.session_state["patient_chol"] = 245
        st.session_state["patient_glucose"] = 175
        st.session_state["patient_bmi"] = 29.5
        st.session_state["is_pregnant"] = False
        st.session_state["pregnancy_red_flags"] = []
        st.session_state["child_danger_signs"] = []
        st.session_state["vitals_anomaly_type"] = "none"
        st.session_state["manual_hr"] = 86
        st.session_state["manual_spo2"] = 96.0
        st.session_state["manual_temp"] = 38.2
        st.session_state["manual_rr"] = 18

    elif preset_key == "ADULT_HIGH":
        st.session_state["patient_name"] = "Baldev Raj"
        st.session_state["patient_country_code"] = "+91"
        st.session_state["patient_mobile_raw"] = "9812345673"
        st.session_state["symptom_text"] = "Crushing chest pain radiating down left arm, gasping for air, sweating cold"
        st.session_state["patient_age"] = 66
        st.session_state["patient_sex"] = "Male"
        st.session_state["patient_bp"] = 178
        st.session_state["patient_bp_dia"] = 105
        st.session_state["patient_chol"] = 295
        st.session_state["patient_glucose"] = 210
        st.session_state["patient_bmi"] = 33.0
        st.session_state["is_pregnant"] = False
        st.session_state["pregnancy_red_flags"] = []
        st.session_state["child_danger_signs"] = []
        st.session_state["vitals_anomaly_type"] = "combined_critical"
        st.session_state["manual_hr"] = 145
        st.session_state["manual_spo2"] = 88.0
        st.session_state["manual_temp"] = 37.0
        st.session_state["manual_rr"] = 28

    elif preset_key == "MATERNAL_PREECLAMPSIA":
        st.session_state["patient_name"] = "Meena Devi"
        st.session_state["patient_country_code"] = "+91"
        st.session_state["patient_mobile_raw"] = "9812345674"
        st.session_state["symptom_text"] = "Severe pounding headache that won't go away, seeing spots and blurry vision, swollen face"
        st.session_state["patient_age"] = 26
        st.session_state["patient_sex"] = "Female"
        st.session_state["patient_bp"] = 158
        st.session_state["patient_bp_dia"] = 102
        st.session_state["patient_chol"] = 190
        st.session_state["patient_glucose"] = 96
        st.session_state["patient_bmi"] = 25.0
        st.session_state["is_pregnant"] = True
        st.session_state["gestational_weeks"] = 32
        st.session_state["pregnancy_red_flags"] = ["preg_severe_headache_bp", "preg_blurred_vision", "preg_swelling_face_hands"]
        st.session_state["child_danger_signs"] = []
        st.session_state["vitals_anomaly_type"] = "none"
        st.session_state["manual_hr"] = 96
        st.session_state["manual_spo2"] = 97.0
        st.session_state["manual_temp"] = 37.1
        st.session_state["manual_rr"] = 20

    elif preset_key == "PEDIATRIC_DANGER":
        st.session_state["patient_name"] = "Baby Rohan"
        st.session_state["patient_country_code"] = "+91"
        st.session_state["patient_mobile_raw"] = "9812345675"
        st.session_state["symptom_text"] = "Child is very sleepy, refuses to drink any breastmilk, breathing very rapidly and pulling chest inward"
        st.session_state["patient_age"] = 2
        st.session_state["patient_age_months"] = 24
        st.session_state["patient_sex"] = "Male"
        st.session_state["patient_bp"] = 90
        st.session_state["patient_bp_dia"] = 60
        st.session_state["patient_chol"] = 150
        st.session_state["patient_glucose"] = 80
        st.session_state["patient_bmi"] = 16.0
        st.session_state["is_pregnant"] = False
        st.session_state["pregnancy_red_flags"] = []
        st.session_state["child_danger_signs"] = ["unable_to_drink", "lethargic_unconscious", "chest_indrawing"]
        st.session_state["vitals_anomaly_type"] = "hypoxia"
        st.session_state["manual_hr"] = 150
        st.session_state["manual_spo2"] = 91.0
        st.session_state["manual_temp"] = 39.2
        st.session_state["manual_rr"] = 52

    elif preset_key == "RETURNING_RAMESH":
        st.session_state["patient_name"] = "Ramesh Kumar"
        st.session_state["patient_country_code"] = "+91"
        st.session_state["patient_mobile_raw"] = "9876543210"
        st.session_state["symptom_text"] = "Worsening shortness of breath walking up steps, mild chest tightness in morning"
        st.session_state["patient_age"] = 58
        st.session_state["patient_sex"] = "Male"
        st.session_state["patient_bp"] = 152
        st.session_state["patient_bp_dia"] = 94
        st.session_state["patient_chol"] = 260
        st.session_state["patient_glucose"] = 185
        st.session_state["patient_bmi"] = 28.4
        st.session_state["is_pregnant"] = False
        st.session_state["pregnancy_red_flags"] = []
        st.session_state["child_danger_signs"] = []
        st.session_state["vitals_anomaly_type"] = "none"
        st.session_state["manual_hr"] = 88
        st.session_state["manual_spo2"] = 95.0
        st.session_state["manual_temp"] = 36.7
        st.session_state["manual_rr"] = 19


# ==========================================
# 1. TOP INTERACTION BAR & USAGE MODE SELECTOR
# ==========================================
render_mode_selector()
mode_cfg = get_mode_config()

# Top Banner Header
if is_hospital():
    st.markdown(
        """
        <div style="background: #1e3a8a; color: white; padding: 12px 18px; border-radius: 8px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; font-size: 1.25rem;">🏥 Clinician Workstation — Hospital Emergency & OPD Triage</h3>
                <small style="opacity: 0.9;">Multi-Modal Diagnostic Support | SQLite Priority Queue | SBAR Referral Gateway</small>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.82rem;">Clinician Authenticated</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="background: #065f46; color: white; padding: 12px 18px; border-radius: 8px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; font-size: 1.25rem;">🏠 Home Health Assessment & Emergency Guide</h3>
                <small style="opacity: 0.9;">Simple Plain-Language Decision Support for Families & Rural Health Posts | UN SDG 3</small>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.82rem;">Confidential & Safe</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# SIDEBAR CONTROLS: DEMOS, REGISTRY, INPUTS
# ==========================================
st.sidebar.title("🎛️ Triage Controls")
if st.sidebar.button("↩ Change mode", use_container_width=True, key="_change_mode"):
    st.session_state["mode_chosen"] = False  # entered data is kept in session_state
    st.rerun()

# Hospital Environment Switcher (Hospital Mode only)
if is_hospital():
    st.sidebar.subheader("🏥 Operating Facility")
    hosp_options = [
        ("HOSP-001", "Apex District Hospital, Jaipur"),
        ("HOSP-002", "Alwar Community Health Centre"),
        ("HOSP-003", "St. Jude Rural Clinic, Bangalore"),
        ("HOSP-004", "Metro General Hospital"),
    ]
    cur_h_idx = 0
    for idx, h in enumerate(hosp_options):
        if h[0] == st.session_state["current_hospital_id"]:
            cur_h_idx = idx
    selected_hosp = st.sidebar.selectbox(
        "Current Facility:",
        hosp_options,
        index=cur_h_idx,
        format_func=lambda x: x[1],
        key="_hosp_facility_select",
    )
    st.session_state["current_hospital_id"] = selected_hosp[0]
    st.sidebar.caption(f"Logged in as: **{st.session_state['active_doctor_name']}**")
    st.sidebar.markdown("---")

# Quick Load Benchmark Patient Buttons
st.sidebar.subheader("⚡ Benchmark Scenarios")
b_c1, b_c2, b_c3 = st.sidebar.columns(3)
with b_c1:
    if st.button("🟢 Adult Low", use_container_width=True):
        load_benchmark_patient("ADULT_LOW")
        st.rerun()
with b_c2:
    if st.button("🟡 Adult Med", use_container_width=True):
        load_benchmark_patient("ADULT_MEDIUM")
        st.rerun()
with b_c3:
    if st.button("🔴 Adult High", use_container_width=True):
        load_benchmark_patient("ADULT_HIGH")
        st.rerun()

b_c4, b_c5, b_c6 = st.sidebar.columns(3)
with b_c4:
    if st.button("🤰 Pregnant", use_container_width=True, help="Load preeclampsia red-flag case"):
        load_benchmark_patient("MATERNAL_PREECLAMPSIA")
        st.rerun()
with b_c5:
    if st.button("👶 Under-5", use_container_width=True, help="Load child danger signs"):
        load_benchmark_patient("PEDIATRIC_DANGER")
        st.rerun()
with b_c6:
    if st.button("📋 Returning", use_container_width=True, help="Load Ramesh Kumar with 3 past visits"):
        load_benchmark_patient("RETURNING_RAMESH")
        st.rerun()

st.sidebar.markdown("---")

# Patient Identification & Country Mobile
st.sidebar.subheader("👤 Patient Identity & Verification")
p_name = st.sidebar.text_input("Full Name:", value=st.session_state["patient_name"])
st.session_state["patient_name"] = p_name

col_cc, col_num = st.sidebar.columns([1.2, 2.0])
with col_cc:
    country_codes = list(config.COUNTRY_PHONE_CONFIG.keys())
    c_idx = country_codes.index(st.session_state["patient_country_code"]) if st.session_state["patient_country_code"] in country_codes else 0
    p_code = st.selectbox("Code:", country_codes, index=c_idx, key="_cc_select")
    st.session_state["patient_country_code"] = p_code
with col_num:
    p_num = st.text_input("Mobile Number:", value=st.session_state["patient_mobile_raw"], key="_mobile_raw_input")
    st.session_state["patient_mobile_raw"] = p_num

# Validate mobile
is_valid_phone, full_phone, phone_err = validate_mobile(p_code, p_num)
if not is_valid_phone:
    st.sidebar.warning(f"⚠️ {phone_err}")
else:
    st.sidebar.caption(f"Masked display: `{mask_mobile(full_phone)}`")

# Consent Checkbox
consent_checked = st.sidebar.checkbox(
    "Clinical Continuity Consent (DPDP Act 2023 / HIPAA)",
    value=st.session_state["patient_consent"],
    help="Authorizes recording triage records across network clinics for continuity of care.",
)
st.session_state["patient_consent"] = consent_checked

# Search existing patient history button
if st.sidebar.button("🔍 Search Patient History", use_container_width=True):
    if is_valid_phone:
        existing = registry.find_patient_by_mobile(full_phone)
        if existing:
            st.session_state["patient_id"] = existing["patient_id"]
            st.session_state["patient_name"] = existing["name"]
            st.session_state["patient_age"] = existing["age"]
            st.session_state["patient_sex"] = existing["sex"]
            st.sidebar.success(f"Found record: {existing['name']} ({existing['patient_id']})")
        else:
            st.sidebar.info("No prior clinic records found for this number.")
    else:
        st.sidebar.error("Please enter a valid mobile number first.")

st.sidebar.markdown("---")

# ------------------------------------------
# HOME MODE: nearby hospitals + emergency contact
# ------------------------------------------
home_city = None
sec_full, sec_valid, sec_relation, share_consent = "", False, "", False

if is_home():
    st.sidebar.subheader("📍 Your Location")
    _cities = facilities.list_cities()
    st.session_state.setdefault("home_city", _cities[0] if _cities else "")
    if _cities:
        _ci = _cities.index(st.session_state["home_city"]) if st.session_state["home_city"] in _cities else 0
        home_city = st.sidebar.selectbox(
            "Nearest city (to suggest hospitals):", _cities, index=_ci, key="_home_city_select",
        )
        st.session_state["home_city"] = home_city

    st.sidebar.subheader("👥 Emergency Contact")
    st.sidebar.caption("A friend, guardian or relative who should know about your result.")
    st.session_state.setdefault("sec_relation", contact_alert.RELATIONS[0])
    st.session_state.setdefault("sec_country_code", "+91")
    st.session_state.setdefault("sec_mobile_raw", "")
    st.session_state.setdefault("share_consent", False)

    _ri = contact_alert.RELATIONS.index(st.session_state["sec_relation"]) if st.session_state["sec_relation"] in contact_alert.RELATIONS else 0
    sec_relation = st.sidebar.selectbox("Who is this person?", contact_alert.RELATIONS, index=_ri, key="_sec_relation_select")
    st.session_state["sec_relation"] = sec_relation

    _sc1, _sc2 = st.sidebar.columns([1.2, 2.0])
    with _sc1:
        _codes = list(config.COUNTRY_PHONE_CONFIG.keys())
        _cidx = _codes.index(st.session_state["sec_country_code"]) if st.session_state["sec_country_code"] in _codes else 0
        sec_code = st.selectbox("Code:", _codes, index=_cidx, key="_sec_cc_select")
        st.session_state["sec_country_code"] = sec_code
    with _sc2:
        sec_num = st.text_input("Contact's Mobile:", value=st.session_state["sec_mobile_raw"], key="_sec_mobile_input")
        st.session_state["sec_mobile_raw"] = sec_num

    if sec_num.strip():
        sec_valid, sec_full, _sec_err = contact_alert.validate_secondary(sec_code, sec_num, full_phone if is_valid_phone else "")
        if sec_valid:
            st.sidebar.caption(f"Contact: `{contact_alert.mask_number(sec_full)}`")
        else:
            st.sidebar.warning(f"⚠️ {_sec_err}")

    share_consent = st.sidebar.checkbox(
        "I agree to share my result (urgency level and main concerns) with this contact.",
        value=st.session_state["share_consent"],
        key="_share_consent_checkbox",
    )
    st.session_state["share_consent"] = share_consent
    st.sidebar.markdown("---")

# Physical limits used to hide unsuitable exercises/yoga (both modes)
st.session_state.setdefault("wellness_limits", [])
wellness_limits = st.sidebar.multiselect(
    "Any of these? (hides unsuitable exercises)",
    options=list(wellness.LIMIT_OPTIONS.keys()),
    default=[x for x in st.session_state["wellness_limits"] if x in wellness.LIMIT_OPTIONS],
    format_func=lambda k: wellness.LIMIT_OPTIONS[k],
    key="_wellness_limits_select",
)
st.session_state["wellness_limits"] = wellness_limits
st.sidebar.markdown("---")


# ==========================================
# CLINICAL INPUT SECTIONS
# ==========================================
# Section 1: Demographics & Special Populations
st.sidebar.subheader("1. Demographics & Category")
c_d1, c_d2 = st.sidebar.columns(2)
with c_d1:
    _today = date.today()

    def _age_from_dob(d: date) -> int:
        return _today.year - d.year - ((_today.month, _today.day) < (d.month, d.day))

    def _approx_dob(age_years: int) -> date:
        try:
            return _today.replace(year=_today.year - int(age_years))
        except ValueError:  # 29 Feb
            return _today.replace(year=_today.year - int(age_years), day=28)

    # Benchmark presets change patient_age; resync the DOB when they do.
    if (
        "patient_dob" not in st.session_state
        or _age_from_dob(st.session_state["patient_dob"]) != int(st.session_state["patient_age"])
    ):
        st.session_state["patient_dob"] = _approx_dob(st.session_state["patient_age"])
    p_dob = st.date_input(
        "Date of Birth:",
        value=st.session_state["patient_dob"],
        min_value=date(1900, 1, 1),
        max_value=_today,
        format="DD/MM/YYYY",
    )
    st.session_state["patient_dob"] = p_dob
    p_age = max(0, min(105, _age_from_dob(p_dob)))
    st.session_state["patient_age"] = p_age
    st.caption(f"Age: **{p_age}** years")
with c_d2:
    sex_options = ["Female", "Male"]
    s_idx = 0 if st.session_state["patient_sex"] == "Female" else 1
    p_sex = st.selectbox("Biological Sex:", sex_options, index=s_idx)
    st.session_state["patient_sex"] = p_sex

# Child age in months if under 5
p_age_months = float(p_age * 12.0)
if p_age < 5:
    p_age_months = st.sidebar.number_input(
        "Child Age (Months):",
        min_value=1,
        max_value=59,
        value=min(max(1, int(st.session_state.get("patient_age_months", p_age * 12))), 59),
    )
    st.session_state["patient_age_months"] = p_age_months

# Auto-detect category
category_detected = detect_patient_category(
    p_age, p_sex, st.session_state.get("is_pregnant", False), st.session_state.get("is_postpartum", False)
)
st.sidebar.info(f"**Patient Category:** `{category_detected}`")

# Maternal Danger Signs (if female 12-55 or pregnant checked)
preg_red_flags_selected = []
if p_sex == "Female" and (12 <= p_age <= 55 or st.session_state.get("is_pregnant")):
    with st.sidebar.expander("🤰 Maternal & Pregnancy Safety", expanded=st.session_state.get("is_pregnant", False)):
        is_preg = st.checkbox("Currently Pregnant", value=st.session_state.get("is_pregnant", False))
        st.session_state["is_pregnant"] = is_preg
        if is_preg:
            g_weeks = st.slider("Gestational Weeks:", 1, 42, int(st.session_state.get("gestational_weeks", 24)))
            st.session_state["gestational_weeks"] = g_weeks
            is_post = st.checkbox("Post-delivery (within 6 weeks)", value=st.session_state.get("is_postpartum", False))
            st.session_state["is_postpartum"] = is_post

            ped_data = load_pediatric_data()
            flag_opts = {f["id"]: f["label"] for f in ped_data.get("pregnancy_red_flags", [])}
            selected_flag_ids = st.multiselect(
                "Maternal Danger Signs / Red Flags:",
                options=list(flag_opts.keys()),
                default=st.session_state.get("pregnancy_red_flags", []),
                format_func=lambda x: flag_opts[x],
            )
            st.session_state["pregnancy_red_flags"] = selected_flag_ids
            preg_red_flags_selected = selected_flag_ids

# Child IMNCI Danger Signs (if under 18, especially under 5)
child_signs_selected = []
if p_age < 18:
    with st.sidebar.expander("👶 Child Danger Signs (WHO IMNCI)", expanded=(p_age < 5)):
        ped_data = load_pediatric_data()
        imnci_opts = {s["id"]: s["label"] for s in ped_data.get("who_imnci_danger_signs_under5", [])}
        selected_child_ids = st.multiselect(
            "Select Observed Danger Signs:",
            options=list(imnci_opts.keys()),
            default=st.session_state.get("child_danger_signs", []),
            format_func=lambda x: imnci_opts[x],
        )
        st.session_state["child_danger_signs"] = selected_child_ids
        child_signs_selected = selected_child_ids

st.sidebar.markdown("---")

# Section 2: Symptoms Input (NLP)
st.sidebar.subheader("2. Typed Symptoms (NLP)")
symptom_input = st.sidebar.text_area(
    "Describe symptoms in plain text:",
    value=st.session_state["symptom_text"],
    height=90,
    help="Enter free-text complaints. Negations and common typos are handled automatically.",
)
st.session_state["symptom_text"] = symptom_input

st.sidebar.markdown("---")

# Section 3: Vitals Telemetry / Measurements
st.sidebar.subheader("3. Physiological Vital Signs")
if is_home():
    st.sidebar.caption("Enter measurements from home thermometer, pulse oximeter, or BP cuff:")
    m_v1, m_v2 = st.sidebar.columns(2)
    with m_v1:
        v_hr = st.number_input("Pulse / HR (bpm)", min_value=30, max_value=220, value=int(st.session_state["manual_hr"]), key=f"w_manual_hr_{st.session_state['_std_ver']}",
                               help=reference_ranges.range_text("heart_rate"))
        v_temp = st.number_input("Temp (°C)", min_value=33.0, max_value=43.0, value=float(st.session_state["manual_temp"]), step=0.1, key=f"w_manual_temp_{st.session_state['_std_ver']}",
                                 help=reference_ranges.range_text("temperature"))
    with m_v2:
        v_spo2 = st.number_input("SpO2 Oxygen (%)", min_value=60.0, max_value=100.0, value=float(st.session_state["manual_spo2"]), step=0.5, key=f"w_manual_spo2_{st.session_state['_std_ver']}",
                                 help=reference_ranges.range_text("spo2"))
        v_rr = st.number_input("Breaths / min", min_value=8, max_value=80, value=int(st.session_state["manual_rr"]), key=f"w_manual_rr_{st.session_state['_std_ver']}",
                               help=reference_ranges.range_text("resp_rate"))

    st.session_state["manual_hr"] = v_hr
    st.session_state["manual_spo2"] = v_spo2
    st.session_state["manual_temp"] = v_temp
    st.session_state["manual_rr"] = v_rr

    # Compare with the standard adult ranges (adults only; children use pediatric tables)
    if p_age >= 18:
        _vit_off = reference_ranges.outside_range(
            {"heart_rate": v_hr, "spo2": v_spo2, "temperature": v_temp, "resp_rate": v_rr}
        )
        if _vit_off:
            st.sidebar.caption("⚠️ Outside the standard adult range:\n" + reference_ranges.format_outside(_vit_off))
        else:
            st.sidebar.caption("✅ All vitals are within the standard adult range.")
    else:
        st.sidebar.caption("ℹ️ Adult reference ranges do not apply to under-18 patients; pediatric ranges are used by the safety rules.")

    # Generate synthetic telemetry from home inputs
    vitals_df = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=30, freq="10s"),
        "heart_rate": np.clip(np.random.normal(v_hr, 2.0, 30), 30, 220),
        "spo2": np.clip(np.random.normal(v_spo2, 0.5, 30), 60.0, 100.0),
        "temperature": np.full(30, v_temp),
        "respiratory_rate": np.full(30, v_rr),
    })
    model_type_key = "isolation_forest"

else:
    vitals_source = st.sidebar.radio("Telemetry Source:", ["Simulate Telemetry Stream", "Upload CSV"], index=0)
    vitals_model_choice = st.sidebar.selectbox("Model Architecture:", ["Isolation Forest (Default)", "Reconstruction Autoencoder"], index=0)
    model_type_key = "isolation_forest" if "Isolation" in vitals_model_choice else "autoencoder"

    if vitals_source == "Simulate Telemetry Stream":
        _anomaly_options = [
            ("none", "Normal Resting Baseline"),
            ("hypoxia", "Acute Hypoxia (SpO2 < 90%)"),
            ("tachycardia", "Severe Tachycardia (HR > 140)"),
            ("bradycardia", "Severe Bradycardia (HR < 40)"),
            ("septic_fever", "Septic Fever Spike + Tachypnea"),
            ("combined_critical", "Critical Hypoxia + Tachycardia"),
        ]
        _anomaly_ids = [o[0] for o in _anomaly_options]
        anomaly_choice = st.sidebar.selectbox(
            "Inject Physiological Pattern:",
            _anomaly_options,
            index=_anomaly_ids.index(st.session_state["vitals_anomaly_type"])
            if st.session_state["vitals_anomaly_type"] in _anomaly_ids else 0,
            format_func=lambda x: x[1],
            key=f"w_anomaly_{st.session_state['_std_ver']}",
        )[0]
        st.session_state["vitals_anomaly_type"] = anomaly_choice

        vitals_df = generate_synthetic_vitals(
            duration_minutes=45,
            sampling_interval_sec=10,
            anomaly_type=anomaly_choice,
            seed=config.RANDOM_SEED,
        )
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Vitals CSV:", type=["csv"])
        if uploaded_file is not None:
            try:
                vitals_df = pd.read_csv(uploaded_file)
            except Exception:
                vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none")
        else:
            vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none")

st.sidebar.markdown("---")

# Section 4: Tabular Chronic Risk Parameters
st.sidebar.subheader("4. Chronic Risk Baseline")
c_b1, c_b2 = st.sidebar.columns(2)
with c_b1:
    p_bp = st.number_input("Resting BP Sys (mmHg)", min_value=70, max_value=240, value=int(st.session_state["patient_bp"]), key=f"w_patient_bp_{st.session_state['_std_ver']}",
                           help=reference_ranges.range_text("bp_systolic"))
    p_bp_dia = st.number_input("Resting BP Dia (mmHg)", min_value=40, max_value=140, value=int(st.session_state["patient_bp_dia"]), key=f"w_patient_bp_dia_{st.session_state['_std_ver']}",
                               help=reference_ranges.range_text("bp_diastolic"))
    p_chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=480, value=int(st.session_state["patient_chol"]), key=f"w_patient_chol_{st.session_state['_std_ver']}",
                             help=reference_ranges.range_text("cholesterol"))
with c_b2:
    p_glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=400, value=int(st.session_state["patient_glucose"]), key=f"w_patient_glucose_{st.session_state['_std_ver']}",
                                help=reference_ranges.range_text("glucose"))
    p_bmi = st.number_input("BMI", min_value=12.0, max_value=55.0, value=float(st.session_state["patient_bmi"]), step=0.1, key=f"w_patient_bmi_{st.session_state['_std_ver']}",
                            help=reference_ranges.range_text("bmi"))

st.session_state["patient_bp"] = p_bp
st.session_state["patient_bp_dia"] = p_bp_dia
st.session_state["patient_chol"] = p_chol
st.session_state["patient_glucose"] = p_glucose
st.session_state["patient_bmi"] = p_bmi

# Compare with the standard adult ranges (adults only)
if p_age >= 18:
    _base_off = reference_ranges.outside_range({
        "bp_systolic": p_bp, "bp_diastolic": p_bp_dia, "cholesterol": p_chol,
        "glucose": p_glucose, "bmi": p_bmi,
    })
    if _base_off:
        st.sidebar.caption("⚠️ Outside the standard adult range:\n" + reference_ranges.format_outside(_base_off))
    else:
        st.sidebar.caption("✅ All baseline values are within the standard adult range.")

st.sidebar.button(
    "↺ Reset to standard values",
    key="btn_reset_standard",
    on_click=_reset_standard_values,
    disabled=(p_age < 18),
    use_container_width=True,
    help="Puts every vital-sign and baseline box back to the standard adult reference value. "
         "(Disabled for under-18 patients - children have different normal ranges.)",
)

with st.sidebar.expander("ℹ️ Where do the standard values come from?"):
    st.markdown(
        "Boxes open pre-filled with typical **normal adult resting values**. "
        "Hover the **?** next to any box to see its normal range.\n\n"
        + "\n".join(f"- [{name}]({url})" for name, url in reference_ranges.SOURCES)
        + "\n\n*Reference ranges for screening and education only - not a diagnosis.*"
    )

if p_age < 18:
    st.sidebar.caption("ℹ️ *Chronic disease models are adult-only and automatically excluded for under-18 patients.*")


# ==========================================
# MODEL INFERENCES & MULTI-MODAL FUSION
# ==========================================
# 1. Module C: Symptom Triage (NLP)
symptom_res = symptom_model.predict(st.session_state["symptom_text"])

# 2. Module B: Vitals Anomaly (Time Series)
vitals_res = vitals_model.detect(vitals_df, model_type=model_type_key)

# 3. Special Populations Rule-Based Safety Layer
special_pop_input = {
    "age": p_age,
    "age_months": p_age_months,
    "sex": p_sex,
    "is_pregnant": st.session_state.get("is_pregnant", False),
    "is_postpartum": st.session_state.get("is_postpartum", False),
    "gestational_weeks": st.session_state.get("gestational_weeks"),
    "systolic_bp": p_bp,
    "diastolic_bp": p_bp_dia,
    "child_danger_signs": child_signs_selected,
    "pregnancy_red_flags": preg_red_flags_selected,
    "vitals": {
        "hr": float(vitals_df["heart_rate"].mean()),
        "rr": float(vitals_df["respiratory_rate"].mean()) if "respiratory_rate" in vitals_df else 18.0,
        "spo2": float(vitals_df["spo2"].min()),
        "temperature": float(vitals_df["temperature"].mean()) if "temperature" in vitals_df else 37.0,
    },
}
special_pop_res = evaluate_special_population(special_pop_input)

# 4. Module A: Chronic Risk Screening (Adult only)
is_pediatric = p_age < 18.0
if is_pediatric:
    risk_res = {
        "module": "Risk Screening",
        "urgency": "LOW",
        "score": 0.0,
        "explanation": "Not applicable: Adult chronic disease model excluded for pediatric patient (<18y).",
        "details": {
            "applicable": False,
            "status": "Not applicable (adult model)",
            "heart_disease_risk": 0.0,
            "diabetes_risk": 0.0,
            "lead_shap_factors": [],
        },
    }
else:
    risk_features = {
        "age": p_age,
        "sex": 1 if p_sex == "Male" else 0,
        "resting_bp": p_bp,
        "trestbps": p_bp,
        "cholesterol": p_chol,
        "chol": p_chol,
        "glucose": p_glucose,
        "bmi": p_bmi,
        "thalach": int(vitals_df["heart_rate"].mean()),
        "symptom_text": st.session_state["symptom_text"],
    }
    risk_res = risk_model.predict(risk_features)

# 5. Master Fusion Engine (Mode-Invariant Triage Calculation)
fusion_res = fuse_triage_modalities(
    risk_result=risk_res,
    vitals_result=vitals_res,
    symptom_result=symptom_res,
    patient_features={
        "age": p_age,
        "sex": p_sex,
        "systolic_bp": p_bp,
        "diastolic_bp": p_bp_dia,
        "glucose": p_glucose,
        "is_pregnant": st.session_state.get("is_pregnant", False),
    },
    emergency_country="India",
    special_population_result=special_pop_res,
)

final_urgency = fusion_res["final_urgency"]
final_score = fusion_res["final_score"]
# Clinician override (Hospital mode) can change this; the AI result stays untouched.
effective_urgency = final_urgency
active_override = None
ref_levels = load_referral_levels()  # needed by the save step in both modes


# ==========================================
# MAIN PAGE WORKSPACE (HOSPITAL VS HOME)
# ==========================================
if is_hospital():
    # Hospital Mode: Multi-Tab Workstation
    tab_assessment, tab_queue, tab_registry = st.tabs([
        "🩺 Clinical Triage Assessment",
        "📋 Hospital Triage Queue",
        "🗂️ Patient Registry & History",
    ])
    tab_my_records = None
else:
    # Home Mode: Simple Clean Layout
    tab_assessment, tab_my_records = st.tabs([
        "🩺 My Health Assessment",
        "📖 My Past Clinic Records & Privacy",
    ])
    tab_queue = None
    tab_registry = None


# -------------------------------------------------------------
# TAB 1: TRIAGE ASSESSMENT (COMMON TO BOTH MODES)
# -------------------------------------------------------------
with tab_assessment:
    # 1. Top Urgency Banner
    st.markdown(
        render_urgency_banner(
            urgency=final_urgency,
            score=final_score,
            synthesis=fusion_res["clinical_synthesis"],
            is_override=fusion_res["safety_override"],
            simple_language=is_home(),
        ),
        unsafe_allow_html=True,
    )

    # Maternal / Child Danger Signs Alert Banner (if triggered)
    if special_pop_res["urgency"] == config.TIER_HIGH:
        spec_details = special_pop_res.get("details", {})
        reasons_list = spec_details.get("danger_signs_detected", []) + spec_details.get("red_flags_detected", [])
        st.error(
            f"""
            ### 🚨 SPECIAL POPULATION CRITICAL ALERT: {spec_details.get('category')}
            **Urgent signs detected:** {', '.join(reasons_list) if reasons_list else 'Severe physiological instability'}
            """
        )
        if spec_details.get("preeclampsia_risk"):
            st.warning("⚠️ **Suspected Preeclampsia / Eclampsia Risk:** Elevated blood pressure with neurological or visual symptoms.")

        # Show emergency instructions
        if spec_details.get("emergency_instructions"):
            st.markdown("#### 🚨 Immediate Safety Instructions:")
            for inst in spec_details["emergency_instructions"]:
                st.markdown(f"- **{inst}**")

    # Mode-Specific Presentation: Hospital vs Home
    if is_hospital():
        # --- HOSPITAL MODE: FULL CLINICAL BREAKDOWN ---
        st.markdown("### 📊 Multi-Modal Diagnostic Breakdown")
        col1, col2, col3 = st.columns(3)

        # Card 1: Module C
        with col1:
            st.markdown('<div class="module-card">', unsafe_allow_html=True)
            st.markdown(f"### 💬 Symptom Triage {render_urgency_chip(symptom_res['urgency'])}", unsafe_allow_html=True)
            st.metric("NLP Urgency Score", f"{symptom_res['score']:.0%}")
            if symptom_res["details"].get("red_flag_triggered"):
                st.error("🚨 Red-Flag Safety Trigger Active")
            st.markdown(f"**Clinical Finding:** {symptom_res['explanation']}")
            st.markdown("---")
            st.markdown("**Highlighted Key Phrases:**")
            st.markdown(
                highlight_symptom_tokens(st.session_state["symptom_text"], symptom_res["details"].get("top_tokens", [])),
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Card 2: Module B
        with col2:
            st.markdown('<div class="module-card">', unsafe_allow_html=True)
            st.markdown(f"### 💓 Vitals Telemetry {render_urgency_chip(vitals_res['urgency'])}", unsafe_allow_html=True)
            st.metric("Telemetry Anomaly Score", f"{vitals_res['score']:.0%}")
            if vitals_res["details"].get("safety_override"):
                st.error("⚠️ Hard Clinical Safety Override")
            mean_hr_val = vitals_res["details"].get("mean_hr", int(vitals_df["heart_rate"].mean()))
            min_spo2_val = vitals_res["details"].get("min_spo2", float(vitals_df["spo2"].min()))
            st.markdown(f"**Mean HR:** `{mean_hr_val} bpm` | **Min SpO2:** `{min_spo2_val}%`")
            st.markdown(f"**Telemetry Assessment:** {vitals_res['explanation']}")
            st.markdown("---")
            st.caption(f"Engine: {vitals_res['details'].get('model_used', 'Isolation Forest')}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Card 3: Module A
        with col3:
            st.markdown('<div class="module-card">', unsafe_allow_html=True)
            st.markdown(f"### 📊 Risk Screening {render_urgency_chip(risk_res['urgency'])}", unsafe_allow_html=True)
            if is_pediatric:
                st.info("👶 **Not Applicable (Pediatric Model)**\nAdult chronic risk screening excluded.")
                st.markdown(f"**Patient Age:** {p_age} years (< 18)")
            else:
                st.metric("Tabular Risk Score", f"{risk_res['score']:.0%}")
                h_risk = risk_res["details"].get("heart_disease_risk", 0.0)
                d_risk = risk_res["details"].get("diabetes_risk", 0.0)
                st.markdown(f"**Heart Risk:** `{h_risk:.0%}` | **Diabetes Risk:** `{d_risk:.0%}`")
                st.markdown(f"**Primary Driver:** {risk_res['details'].get('primary_driver', 'Cardiovascular')}")
                st.markdown(f"**Risk Evaluation:** {risk_res['explanation']}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Hospital Diagnostic Visualizations
        st.markdown("### 📈 Diagnostic Telemetry & Explainability")
        v_col, s_col = st.columns([1.1, 0.9])
        with v_col:
            st.markdown("**Vitals Time-Series Stream (Heart Rate & SpO2)**")
            telemetry_fig = plot_vitals_telemetry(vitals_df, vitals_res["details"].get("anomaly_indices"))
            st.pyplot(telemetry_fig, use_container_width=True)

        with s_col:
            st.markdown("**Tabular Risk Feature Attribution (SHAP Analysis)**")
            if is_pediatric:
                st.info("SHAP analysis not applicable for pediatric age group.")
            else:
                lead_shap_factors = risk_res["details"].get("lead_shap_factors", [])
                if lead_shap_factors:
                    shap_fig = plot_shap_waterfall_bar(lead_shap_factors, title="SHAP Drivers of Chronic Liability")
                    if shap_fig:
                        st.pyplot(shap_fig, use_container_width=True)
                    else:
                        st.info("Baseline markers are aligned with standard population distributions.")
                else:
                    st.info("Risk indicators evaluated via multi-variate statistical calibration.")

        # Clinician override of the AI tier (recorded in audit log)
        effective_urgency, active_override = render_clinician_override(
            ai_urgency=final_urgency,
            patient_key=st.session_state.get("patient_id") or st.session_state.get("patient_name") or "unknown",
            patient_id=st.session_state.get("patient_id"),
            doctor_name=st.session_state["active_doctor_name"],
            hospital_id=st.session_state.get("current_hospital_id", "HOSP-001"),
            log_audit=registry.log_audit,
        )

        # SBAR Doctor Handoff Summary
        st.markdown("### 📝 SBAR Doctor Handoff & Referral Protocol")
        ref_levels = load_referral_levels()
        designated_ref = ref_levels.get(effective_urgency, ref_levels.get("LOW"))

        sbar_text = generate_handoff(
            patient={
                "patient_id": st.session_state.get("patient_id") or "PAT-UNREG",
                "name": st.session_state["patient_name"],
                "age": p_age,
                "sex": p_sex,
                "mobile_masked": mask_mobile(full_phone) if is_valid_phone else "N/A",
                "chief_complaint": st.session_state["symptom_text"],
            },
            module_results={
                "risk_screening": risk_res,
                "vitals_anomaly": vitals_res,
                "symptom_triage": symptom_res,
                "special_population": special_pop_res,
            },
            final_result=fusion_res,
            referral=designated_ref,
            hospital_name=selected_hosp[1],
        )

        if active_override:
            sbar_text += "\n\n" + override_summary(active_override)

        with st.expander("📄 View Structured SBAR Clinical Handoff Note", expanded=(final_urgency == "HIGH")):
            st.code(sbar_text, language="text")
            col_sbar1, col_sbar2 = st.columns([1, 3])
            with col_sbar1:
                st.download_button(
                    "📥 Download SBAR Handoff (PDF)",
                    data=text_to_pdf("SBAR Clinical Handoff", sbar_text),
                    file_name=f"sbar_handoff_{p_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        # Queue Quick Action (Add Current Patient to Queue)
        st.markdown("### ➕ Patient Queue Management")
        q_act1, q_act2 = st.columns([1.5, 3.5])
        with q_act1:
            if st.button("🚨 Add Patient to Hospital Queue", type="primary", use_container_width=True):
                q_pat_id = st.session_state.get("patient_id") or f"PAT-{datetime.now().strftime('%M%S')}"
                res_q = add_to_queue(
                    patient_id=q_pat_id,
                    hospital_id=st.session_state["current_hospital_id"],
                    name=st.session_state["patient_name"],
                    urgency=effective_urgency,
                    score=final_score,
                    complaint=st.session_state["symptom_text"][:120],
                )
                if res_q.get("is_priority"):
                    st.error(f"Patient added to 🚨 PRIORITY LANE ({res_q.get('action')})!")
                else:
                    st.success(f"Patient added to waiting queue ({res_q.get('action')}).")

    else:
        # --- HOME USE MODE: PLAIN LANGUAGE & CALM GUIDANCE ---
        st.markdown("### 💡 What This Means For You")
        if final_urgency == "HIGH":
            st.error(
                """
                ### 🚨 Urgent Attention Required
                Your symptoms or vitals indicate a potentially serious condition that needs immediate medical evaluation.
                **Please do not wait.** Contact emergency services or have someone take you to the nearest hospital right away.
                """
            )
            col_em1, col_em2 = st.columns(2)
            with col_em1:
                st.markdown("#### 📞 Emergency Phone Numbers")
                st.markdown("- **Ambulance (India):** `108`")
                st.markdown("- **National Emergency Helpline:** `112`")
                st.markdown("- **Women Helpline:** `1091`")
            with col_em2:
                st.markdown("#### 🏃 Immediate Next Steps")
                st.markdown("1. Stay calm and sit or lie in a comfortable resting position.")
                st.markdown("2. Keep warm and loosen any tight clothing.")
                st.markdown("3. Have someone stay with you at all times.")
                st.markdown("4. Do not take self-prescribed medicines without a doctor.")

        elif final_urgency == "MEDIUM":
            st.warning(
                """
                ### 🟡 Medical Check-Up Recommended
                You have noticeable symptoms that require a formal medical check-up within the next 24 to 48 hours.
                Visit your local Primary Health Centre (PHC), community dispensary, or family doctor.
                """
            )
        else:
            st.success(
                """
                ### 🟢 Stable & Suitable for Home Care
                Your symptoms appear mild and self-limiting, and your vital signs are in normal ranges.
                Rest, stay well-hydrated, and practice supportive home care.
                """
            )

        # Expandable Technical Details for Home Users
        with st.expander("🔬 View Detailed Physiological Charts & Findings"):
            st.markdown(f"**Mean Heart Rate:** `{int(vitals_df['heart_rate'].mean())} bpm` | **Oxygen SpO2:** `{float(vitals_df['spo2'].min()):.1f}%`")
            st.markdown(f"**Symptom Findings:** {symptom_res['explanation']}")
            telemetry_fig = plot_vitals_telemetry(vitals_df, vitals_res["details"].get("anomaly_indices"))
            st.pyplot(telemetry_fig, use_container_width=True)

    # 3b. Home-only: nearby hospitals (with busy-hospital redirect) and contact alert
    if is_home():
        render_nearby_hospitals(effective_urgency, home_city)
        render_contact_alert(
            patient_name=st.session_state["patient_name"],
            urgency=effective_urgency,
            concerns=collect_concerns(special_pop_res, symptom_res, st.session_state["symptom_text"]),
            contact_full=sec_full,
            contact_valid=sec_valid,
            share_consent=share_consent,
            relation=sec_relation,
        )

    # 4. Actionable Lifestyle, Diet & Supportive Care (Common to both)
    st.markdown("### 🌿 Supportive Wellness & Care Guidance")
    recs = fusion_res["recommendations"]

    if recs.get("is_emergency"):
        st.info(f"**Primary Medical Directive:** {recs.get('action')}")
    else:
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("#### 🧘 Gentle Breathing & Yoga")
            for asana in recs.get("yoga_asanas", []):
                st.markdown(f"""
                <div class="asana-card">
                    <h4 style="margin: 0 0 6px 0; color: #0369a1;">{asana.get('name')}</h4>
                    <p style="margin: 0 0 4px 0; font-size: 0.9rem;"><b>Practice:</b> {asana.get('how_to')}</p>
                    <p style="margin: 0; font-size: 0.88rem; color: #0284c7;"><b>Benefit:</b> {asana.get('benefit')}</p>
                </div>
                """, unsafe_allow_html=True)

        with r_col2:
            st.markdown("#### 🥗 Dietary & Hydration Suggestions")
            for diet in recs.get("dietary_guidelines", []):
                tips_html = "".join([f"<li style='margin-bottom: 4px;'>{t}</li>" for t in diet.get("tips", [])])
                st.markdown(f"""
                <div class="diet-card">
                    <h4 style="margin: 0 0 6px 0; color: #15803d;">{diet.get('focus')}</h4>
                    <ul style="margin: 0; padding-left: 18px; font-size: 0.88rem;">
                        {tips_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            if recs.get("exercise_plan"):
                st.markdown("#### 🏃 Physical Activity Guidance")
                st.info(recs.get("exercise_plan"))

    # 4b. Detailed personalised wellness plan (not shown for HIGH urgency)
    def _num(x):
        return float(x) if isinstance(x, (int, float)) else None

    render_wellness_plan(
        wellness.build_plan(
            age=int(p_age),
            urgency=effective_urgency,
            sex=p_sex,
            systolic_bp=p_bp,
            diastolic_bp=p_bp_dia,
            glucose=p_glucose,
            bmi=p_bmi,
            is_pregnant=st.session_state.get("is_pregnant", False),
            is_postpartum=st.session_state.get("is_postpartum", False),
            heart_risk=_num(risk_res.get("details", {}).get("heart_disease_risk")),
            diabetes_risk=_num(risk_res.get("details", {}).get("diabetes_risk")),
            symptom_text=st.session_state["symptom_text"],
            limits=wellness_limits,
        )
    )

    # 5. Dual Mode PDF Report Download
    st.markdown("---")
    p_info = {
        "name": st.session_state["patient_name"],
        "age": p_age,
        "sex": p_sex,
        "resting_bp": p_bp,
        "glucose": p_glucose,
        "mobile_masked": mask_mobile(full_phone) if is_valid_phone else "N/A",
    }
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    col_down, col_save = st.columns([1.5, 2.5])
    with col_down:
        if is_hospital():
            pdf_data = generate_triage_pdf(
                fusion_result=fusion_res,
                risk_result=risk_res,
                vitals_result=vitals_res,
                symptom_result=symptom_res,
                patient_info=p_info,
                special_population_result=special_pop_res,
            )
            st.download_button(
                label="📄 Download Clinical Triage PDF",
                data=pdf_data,
                file_name=f"clinical_triage_{timestamp_str}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            home_pdf_data = generate_home_pdf(
                fusion_result=fusion_res,
                patient_info=p_info,
                special_population_result=special_pop_res,
            )
            st.download_button(
                label="📄 Download At-Home Health Summary (PDF)",
                data=home_pdf_data,
                file_name=f"home_health_summary_{timestamp_str}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_save:
        # Save assessment to registry if consent given
        if consent_checked and is_valid_phone:
            if st.button("💾 Save Assessment to Multi-Hospital Health Record", use_container_width=True):
                # Ensure patient is registered
                pat_rec = registry.find_patient_by_mobile(full_phone)
                if not pat_rec:
                    pat_rec = registry.register_patient(
                        name=st.session_state["patient_name"],
                        age=int(p_age),
                        sex=p_sex,
                        raw_mobile=full_phone,
                        consent_given=True,
                    )
                st.session_state["patient_id"] = pat_rec["patient_id"]

                # Record visit
                v_id = registry.add_visit(
                    patient_id=pat_rec["patient_id"],
                    hospital_id=st.session_state.get("current_hospital_id", "HOSP-001"),
                    inputs={
                        "symptoms": st.session_state["symptom_text"],
                        "vitals": {"hr": int(vitals_df["heart_rate"].mean()), "spo2": float(vitals_df["spo2"].min())},
                        "resting_bp": p_bp,
                        "glucose": p_glucose,
                    },
                    module_results={
                        "risk_screening": risk_res,
                        "vitals_anomaly": vitals_res,
                        "symptom_triage": symptom_res,
                    },
                    final_urgency=effective_urgency,
                    final_score=final_score,
                    referral_level=ref_levels.get(effective_urgency, {}).get("level", "Primary"),
                    notes=f"Triage assessment performed via {get_mode()}"
                    + (f" | {override_summary(active_override)}" if active_override else ""),
                )
                st.success(f"Assessment securely saved to patient health record ({pat_rec['patient_id']}, Visit: {v_id})!")


# -------------------------------------------------------------
# TAB 2: HOSPITAL TRIAGE QUEUE (HOSPITAL MODE ONLY)
# -------------------------------------------------------------
if tab_queue is not None:
    with tab_queue:
        render_queue_ui(st.session_state["current_hospital_id"])


# -------------------------------------------------------------
# TAB 3: PATIENT REGISTRY & HISTORY (HOSPITAL MODE ONLY)
# -------------------------------------------------------------
if tab_registry is not None:
    with tab_registry:
        st.subheader("🗂️ Patient Longitudinal Health Registry")
        st.caption("Secure, encrypted multi-hospital longitudinal health records with audit logging and FHIR export.")

        c_reg_s1, c_reg_s2 = st.columns([2, 1])
        with c_reg_s1:
            search_phone_input = st.text_input(
                "Search by Mobile Number (+91 9876543210):",
                value=full_phone if is_valid_phone else "+91 9876543210",
                key="_reg_search_phone",
            )
        with c_reg_s2:
            st.write("")
            st.write("")
            do_search = st.button("Search Registry", type="primary", use_container_width=True)

        target_pat = None
        if do_search or st.session_state.get("patient_id"):
            target_pat = registry.find_patient_by_mobile(search_phone_input)

        if target_pat:
            p_id = target_pat["patient_id"]
            # Audit log search
            registry.log_audit(
                who=st.session_state["active_doctor_name"],
                role="Medical Officer",
                hospital_id=st.session_state["current_hospital_id"],
                patient_id=p_id,
                action="VIEW_PATIENT_HISTORY",
            )

            st.markdown(f"""
            <div style="background-color: #f1f5f9; border-left: 5px solid #2563eb; padding: 14px 18px; border-radius: 8px; margin-bottom: 14px;">
                <h3 style="margin: 0; color: #1e3a8a;">{target_pat['name']} ({p_id})</h3>
                <p style="margin: 4px 0 0 0; font-size: 0.95rem; color: #334155;">
                    <b>Age / Sex:</b> {target_pat['age']} yrs / {target_pat['sex']} &nbsp;|&nbsp;
                    <b>Registered Mobile:</b> {target_pat['mobile_masked']} &nbsp;|&nbsp;
                    <b>Consent:</b> {'Active' if target_pat['consent_given'] else 'Revoked'}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Retrieve history
            history = registry.get_history(p_id)

            # Check clinical deterioration
            is_worse, det_msg = False, ""
            if len(history) >= 2:
                is_worse, det_msg = check_clinical_deterioration(
                    history[0]["final_urgency"], history[0]["final_score"], history[1:]
                )
            elif len(history) == 1:
                is_worse, det_msg = check_clinical_deterioration(
                    final_urgency, final_score, history
                )

            if is_worse:
                st.error(f"""
                ### ⚠️ CLINICAL DETERIORATION WARNING
                **{det_msg}**
                """)

            # Render Timeline
            st.markdown(f"#### 📅 Multi-Hospital Visit Timeline ({len(history)} Visits Recorded)")
            if not history:
                st.info("No prior visits recorded for this patient.")
            else:
                for idx, v in enumerate(history, 1):
                    u_chip = render_urgency_chip(v["final_urgency"])
                    with st.container():
                        st.markdown(f"""
                        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0; color: #0f172a;">Visit #{len(history) - idx + 1}: {v['hospital_name']}</h4>
                                <div>{u_chip} &nbsp; <b>Score: {v['final_score']:.0%}</b></div>
                            </div>
                            <small style="color: #64748b;">Date: {v['timestamp']} | Care Level: {v['referral_level']}</small>
                            <p style="margin: 6px 0 2px 0; font-size: 0.92rem;"><b>Presenting Complaint:</b> {v['inputs'].get('symptoms', 'N/A')}</p>
                            <small style="color: #475569;">Recorded Vitals: {v['inputs'].get('vitals', {})}</small>
                        </div>
                        """, unsafe_allow_html=True)

            # Actions: FHIR R4 Export & Audit Log
            st.markdown("---")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.download_button(
                    "📥 Export Patient History (PDF)",
                    data=history_report_pdf(
                        {**dict(target_pat), "patient_id": p_id},
                        history,
                        det_msg if is_worse else None,
                    ),
                    file_name=f"patient_history_{p_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with col_f2:
                with st.expander("🛡️ Clinical Audit Trail (Who Accessed Records)"):
                    audits = get_audit_entries(p_id)
                    if audits is None:
                        st.warning("Audit trail could not be read: no audit reader method found in the registry.")
                    elif not audits:
                        st.caption("No access events recorded yet.")
                    else:
                        for a in audits:
                            st.caption(format_audit_line(a))
        else:
            st.info("Enter a registered phone number to load historical clinic records.")


# -------------------------------------------------------------
# TAB: MY PAST RECORDS & PRIVACY (HOME MODE ONLY)
# -------------------------------------------------------------
if tab_my_records is not None:
    with tab_my_records:
        st.subheader("📖 My Past Health Assessment Records")
        st.markdown("To protect your privacy under the Digital Personal Data Protection (DPDP) Act, enter your mobile number and one-time verification passcode (OTP).")

        h_col1, h_col2 = st.columns([2, 1])
        with h_col1:
            otp_mobile_input = st.text_input("Mobile Number (+91 9876543210):", value=full_phone if is_valid_phone else "+91 9876543210", key="_home_otp_mobile")
        with h_col2:
            st.write("")
            st.write("")
            if st.button("Send Verification Code", use_container_width=True):
                otp_code = generate_otp(otp_mobile_input)
                st.info(f"📱 Passcode sent to your phone! *(Demo Passcode: `{otp_code}`)*")

        otp_val = st.text_input("Enter 6-Digit Verification Code:", key="_otp_val_input", help="Universal demo code: 123456")
        if st.button("Verify & Show My Records", type="primary"):
            if verify_otp(otp_mobile_input, otp_val):
                st.session_state["otp_verified_mobile"] = otp_mobile_input
                st.success("Identity verified successfully!")
            else:
                st.error("Invalid or expired verification code. Use code: 123456.")

        # Show records if OTP verified
        if st.session_state.get("otp_verified_mobile"):
            ver_pat = registry.find_patient_by_mobile(st.session_state["otp_verified_mobile"])
            if ver_pat:
                p_id = ver_pat["patient_id"]
                st.markdown(f"### Welcome back, {ver_pat['name']}!")
                history = registry.get_history(p_id)

                if not history:
                    st.info("You do not have any past clinic visits recorded yet.")
                else:
                    st.markdown(f"**Found {len(history)} previous visit(s):**")
                    for v in history:
                        u_color = "#fee2e2" if v["final_urgency"] == "HIGH" else ("#fef3c7" if v["final_urgency"] == "MEDIUM" else "#d1fae5")
                        st.markdown(f"""
                        <div style="background-color: {u_color}; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <b>Facility: {v['hospital_name']}</b>
                                <b>Priority: {v['final_urgency']}</b>
                            </div>
                            <small>Date: {v['timestamp']}</small>
                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;">Symptoms checked: {v['inputs'].get('symptoms', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")
                # Privacy Rights: Who viewed my data & Delete my data
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    with st.expander("👁️ Who viewed my health records?"):
                        audits = get_audit_entries(p_id)
                        if audits is None:
                            st.warning("Access history could not be read right now.")
                        elif not audits:
                            st.caption("No one has viewed your records yet.")
                        else:
                            for a in audits:
                                st.caption(format_audit_line(a, with_role=False))
                with p_col2:
                    if st.button("🗑️ Delete My Data (Right to Erasure)", help="Purge personal identifying data in compliance with DPDP Act"):
                        if registry.delete_patient_data(p_id):
                            st.session_state["otp_verified_mobile"] = ""
                            st.warning("Your personal data has been erased from the registry.")
                            st.rerun()
            else:
                st.info("No prior records associated with this verified phone number.")


# ==========================================
# FOOTER & REGULATORY DISCLAIMER
# ==========================================
st.markdown("---")
st.markdown(
    f"""
    <div class="disclaimer-box">
        <b>Regulatory & Clinical Safety Notice:</b> {fusion_res.get('disclaimer')}
    </div>
    """,
    unsafe_allow_html=True,
)