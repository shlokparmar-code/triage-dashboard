"""Multi-Modal AI Medical Triage Dashboard.

A single-page Streamlit application combining tabular chronic risk, time-series
physiological vitals telemetry, and NLP free-text symptom triage into ONE explained urgency score.
Targets rural healthcare access gaps (UN SDG 3: Good Health and Well-Being).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import config
from modules.risk_screening.model import RiskScreeningModel
from modules.risk_screening.explain import plot_shap_waterfall_bar
from modules.vitals_anomaly.model import VitalsAnomalyDetector
from modules.vitals_anomaly.generator import generate_synthetic_vitals
from modules.symptom_triage.model import SymptomTriageModel
from modules.fusion.engine import fuse_triage_modalities
from dashboard.styles import CUSTOM_CSS
from dashboard.components import (
    render_urgency_banner,
    highlight_symptom_tokens,
    plot_vitals_telemetry,
    render_urgency_chip
)
from dashboard.pdf_export import generate_triage_pdf

# Streamlit Page Config
st.set_page_config(
    page_title="AI Medical Triage Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom stylesheet
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Cache Model Instances
@st.cache_resource
def get_symptom_model():
    return SymptomTriageModel()


@st.cache_resource
def get_risk_model():
    return RiskScreeningModel()


@st.cache_resource
def get_vitals_model():
    return VitalsAnomalyDetector()


symptom_model = get_symptom_model()
risk_model = get_risk_model()
vitals_model = get_vitals_model()


# Initialize Session State
if "symptom_text" not in st.session_state:
    st.session_state["symptom_text"] = "Mild scratchy throat and runny nose for 2 days, no fever"
if "patient_age" not in st.session_state:
    st.session_state["patient_age"] = 35
if "patient_sex" not in st.session_state:
    st.session_state["patient_sex"] = "Female"
if "patient_bp" not in st.session_state:
    st.session_state["patient_bp"] = 118
if "patient_chol" not in st.session_state:
    st.session_state["patient_chol"] = 185
if "patient_glucose" not in st.session_state:
    st.session_state["patient_glucose"] = 92
if "patient_bmi" not in st.session_state:
    st.session_state["patient_bmi"] = 22.4
if "vitals_anomaly_type" not in st.session_state:
    st.session_state["vitals_anomaly_type"] = "none"


# Preset Patient Profiles Handler
def load_sample_patient(tier: str):
    if tier == "LOW":
        st.session_state["symptom_text"] = "Mild runny nose, occasional sneeze for 2 days, no chest pain, no fever"
        st.session_state["patient_age"] = 28
        st.session_state["patient_sex"] = "Female"
        st.session_state["patient_bp"] = 115
        st.session_state["patient_chol"] = 175
        st.session_state["patient_glucose"] = 88
        st.session_state["patient_bmi"] = 21.8
        st.session_state["vitals_anomaly_type"] = "none"
    elif tier == "MEDIUM":
        st.session_state["symptom_text"] = "High fever of 102F for 3 days with persistent lower abdominal cramping and nausea"
        st.session_state["patient_age"] = 54
        st.session_state["patient_sex"] = "Male"
        st.session_state["patient_bp"] = 142
        st.session_state["patient_chol"] = 245
        st.session_state["patient_glucose"] = 175
        st.session_state["patient_bmi"] = 29.5
        st.session_state["vitals_anomaly_type"] = "septic_fever"
    elif tier == "HIGH":
        st.session_state["symptom_text"] = "Crushing chest pain radiating down left arm, gasping for air, sweating cold"
        st.session_state["patient_age"] = 66
        st.session_state["patient_sex"] = "Male"
        st.session_state["patient_bp"] = 178
        st.session_state["patient_chol"] = 295
        st.session_state["patient_glucose"] = 210
        st.session_state["patient_bmi"] = 33.0
        st.session_state["vitals_anomaly_type"] = "combined_critical"


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🏥 Patient Inputs")
st.sidebar.caption("Rural Decision-Support Portal | UN SDG 3")

# Accessibility / Localization Toggles
simple_language = st.sidebar.toggle("🗣️ Simple-Language Mode", value=False, help="Use simplified plain language for rural health workers or patients.")
emergency_country = st.sidebar.selectbox("📍 Emergency Region", ["India", "US", "UK", "Universal"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Load Benchmark Patients")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)
with col_p1:
    if st.button("🟢 LOW", use_container_width=True, help="Load mild routine patient"):
        load_sample_patient("LOW")
        st.rerun()
with col_p2:
    if st.button("🟡 MED", use_container_width=True, help="Load urgent clinic patient"):
        load_sample_patient("MEDIUM")
        st.rerun()
with col_p3:
    if st.button("🔴 HIGH", use_container_width=True, help="Load emergency patient"):
        load_sample_patient("HIGH")
        st.rerun()

st.sidebar.markdown("---")

# --- SECTION 1: SYMPTOMS (Module C) ---
st.sidebar.subheader("1. Typed Symptoms (NLP)")
symptom_input = st.sidebar.text_area(
    "Describe symptoms in plain text:",
    value=st.session_state["symptom_text"],
    height=90,
    help="Enter free text complaints. Negations ('no chest pain') and common misspellings are handled automatically."
)
st.session_state["symptom_text"] = symptom_input

# Preset Symptom Snippets
st.sidebar.caption("Preset Examples:")
s_cols = st.sidebar.columns(2)
with s_cols[0]:
    if st.button("💔 Chest Pain", use_container_width=True):
        st.session_state["symptom_text"] = "Crushing chest pain radiating to left arm and cannot breathe"
        st.rerun()
    if st.button("🤧 Mild Cold", use_container_width=True):
        st.session_state["symptom_text"] = "Mild runny nose, sneezing for 2 days, no fever"
        st.rerun()
with s_cols[1]:
    if st.button("🔥 High Fever", use_container_width=True):
        st.session_state["symptom_text"] = "Fever of 103F for three days with persistent shivering and nausea"
        st.rerun()
    if st.button("🛡️ Negation Test", use_container_width=True):
        st.session_state["symptom_text"] = "Mild tension headache, but no chest pain and denies trouble breathing"
        st.rerun()

st.sidebar.markdown("---")

# --- SECTION 2: VITALS (Module B) ---
st.sidebar.subheader("2. Vitals Telemetry (Time Series)")
vitals_source = st.sidebar.radio("Vitals Telemetry Source:", ["Simulate Telemetry Stream", "Upload CSV"], index=0)

vitals_model_choice = st.sidebar.selectbox(
    "Anomaly Model Architecture:",
    ["Isolation Forest (Default)", "Reconstruction Autoencoder"],
    index=0
)
model_type_key = "isolation_forest" if "Isolation" in vitals_model_choice else "autoencoder"

if vitals_source == "Simulate Telemetry Stream":
    anomaly_choice = st.sidebar.selectbox(
        "Inject Physiological Pattern:",
        [
            ("none", "Normal Resting Baseline"),
            ("hypoxia", "Acute Hypoxia (SpO2 < 90%)"),
            ("tachycardia", "Severe Tachycardia (HR > 140)"),
            ("bradycardia", "Severe Bradycardia (HR < 40)"),
            ("septic_fever", "Septic Fever Spike + Tachypnea"),
            ("combined_critical", "Critical Hypoxia + Tachycardia")
        ],
        index=0 if st.session_state["vitals_anomaly_type"] == "none" else (
            1 if st.session_state["vitals_anomaly_type"] == "hypoxia" else (
                2 if st.session_state["vitals_anomaly_type"] == "tachycardia" else (
                    4 if st.session_state["vitals_anomaly_type"] == "septic_fever" else 5
                )
            )
        ),
        format_func=lambda x: x[1]
    )[0]
    st.session_state["vitals_anomaly_type"] = anomaly_choice

    vitals_df = generate_synthetic_vitals(
        duration_minutes=45,
        sampling_interval_sec=10,
        anomaly_type=anomaly_choice,
        seed=config.RANDOM_SEED
    )
else:
    uploaded_file = st.sidebar.file_uploader("Upload Vitals CSV (columns: heart_rate, spo2):", type=["csv"])
    if uploaded_file is not None:
        try:
            vitals_df = pd.read_csv(uploaded_file)
        except Exception:
            st.sidebar.error("Error reading CSV. Loading default synthetic stream.")
            vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none")
    else:
        vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none")

st.sidebar.markdown("---")

# --- SECTION 3: CHRONIC RISK (Module A) ---
st.sidebar.subheader("3. Patient Baseline (Tabular)")
c_a1, c_a2 = st.sidebar.columns(2)
with c_a1:
    p_age = st.number_input("Age (years)", min_value=18, max_value=95, value=st.session_state["patient_age"])
    p_sex = st.selectbox("Sex", ["Female", "Male"], index=0 if st.session_state["patient_sex"] == "Female" else 1)
    p_bp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=230, value=st.session_state["patient_bp"])
with c_a2:
    p_chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=480, value=st.session_state["patient_chol"])
    p_glucose = st.number_input("Blood Glucose (mg/dL)", min_value=60, max_value=350, value=st.session_state["patient_glucose"])
    p_bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=float(st.session_state["patient_bmi"]), step=0.1)

st.session_state["patient_age"] = p_age
st.session_state["patient_sex"] = p_sex
st.session_state["patient_bp"] = p_bp
st.session_state["patient_chol"] = p_chol
st.session_state["patient_glucose"] = p_glucose
st.session_state["patient_bmi"] = p_bmi

# ==========================================
# MODEL INFERENCES
# ==========================================
# 1. Module C: Symptom Triage
symptom_res = symptom_model.predict(st.session_state["symptom_text"])

# 2. Module B: Vitals Anomaly
vitals_res = vitals_model.detect(vitals_df, model_type=model_type_key)

# 3. Module A: Chronic Risk Screening
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
    "symptom_text": st.session_state["symptom_text"]
}
risk_res = risk_model.predict(risk_features)

# 4. Master Fusion Engine
fusion_res = fuse_triage_modalities(
    risk_result=risk_res,
    vitals_result=vitals_res,
    symptom_result=symptom_res,
    patient_features=risk_features,
    emergency_country=emergency_country
)

final_urgency = fusion_res["final_urgency"]
final_score = fusion_res["final_score"]

# ==========================================
# MAIN DASHBOARD PAGE
# ==========================================
st.title("🏥 Multi-Modal AI Medical Triage Dashboard")
st.markdown(
    "**Decision-Support Prototype for Rural Healthcare Access** • *UN SDG 3: Good Health and Well-Being (Target 3.8)*"
)

# 1. TOP-LINE URGENCY BANNER
st.markdown(
    render_urgency_banner(
        urgency=final_urgency,
        score=final_score,
        synthesis=fusion_res["clinical_synthesis"],
        is_override=fusion_res["safety_override"],
        simple_language=simple_language
    ),
    unsafe_allow_html=True
)

# 2. THREE COMPONENT CARDS (Side-by-Side)
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
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Card 2: Module B
with col2:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.markdown(f"### 💓 Vitals Anomaly {render_urgency_chip(vitals_res['urgency'])}", unsafe_allow_html=True)
    st.metric("Telemetry Anomaly Score", f"{vitals_res['score']:.0%}")
    if vitals_res["details"].get("safety_override"):
        st.error("⚠️ Hard Clinical Safety Override")
    mean_hr_val = vitals_res['details'].get('mean_hr', int(vitals_df['heart_rate'].mean()))
    min_spo2_val = vitals_res['details'].get('min_spo2', float(vitals_df['spo2'].min()))
    st.markdown(f"**Mean HR:** `{mean_hr_val} bpm` | **Min SpO2:** `{min_spo2_val}%`")
    st.markdown(f"**Telemetry Assessment:** {vitals_res['explanation']}")
    st.markdown("---")
    st.caption(f"Engine: {vitals_res['details'].get('model_used', 'Isolation Forest')}")
    st.markdown('</div>', unsafe_allow_html=True)

# Card 3: Module A
with col3:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.markdown(f"### 📊 Risk Screening {render_urgency_chip(risk_res['urgency'])}", unsafe_allow_html=True)
    st.metric("Tabular Risk Score", f"{risk_res['score']:.0%}")
    h_risk = risk_res['details'].get('heart_disease_risk', 0.0)
    d_risk = risk_res['details'].get('diabetes_risk', 0.0)
    st.markdown(f"**Heart Risk:** `{h_risk:.0%}` | **Diabetes Risk:** `{d_risk:.0%}`")
    st.markdown(f"**Primary Driver:** {risk_res['details'].get('primary_driver', 'Cardiovascular')}")
    st.markdown(f"**Risk Evaluation:** {risk_res['explanation']}")
    st.markdown('</div>', unsafe_allow_html=True)

# 3. INTERACTIVE VISUALIZATIONS SECTION
st.markdown("### 📈 Diagnostic Explainability & Telemetry")
v_col, s_col = st.columns([1.1, 0.9])

with v_col:
    st.markdown("**Vitals Time-Series Stream (Heart Rate & SpO2)**")
    telemetry_fig = plot_vitals_telemetry(vitals_df, vitals_res["details"].get("anomaly_indices"))
    st.pyplot(telemetry_fig, use_container_width=True)

with s_col:
    st.markdown("**Tabular Risk Feature Attribution (SHAP Analysis)**")
    lead_shap_factors = risk_res["details"].get("lead_shap_factors", [])
    if lead_shap_factors:
        shap_fig = plot_shap_waterfall_bar(lead_shap_factors, title="SHAP Drivers of Chronic Liability")
        if shap_fig:
            st.pyplot(shap_fig, use_container_width=True)
        else:
            st.info("Baseline markers are aligned with standard population distributions.")
    else:
        st.info("Risk indicators evaluated via multi-variate statistical calibration.")

# 4. ACTIONABLE CARE & RECOMMENDATIONS
st.markdown("### 🌿 Care Recommendations & Action Plan")
recs = fusion_res["recommendations"]

if recs.get("is_emergency"):
    st.error(f"### 🚨 {recs.get('title')}")
    st.markdown(f"**Primary Medical Directive:** {recs.get('action')}")
    st.markdown(f"📞 **Emergency Dispatch Hotline ({emergency_country}):** `{recs.get('emergency_number')}`")
    st.markdown("**Actionable Steps While Awaiting Medical Assistance:**")
    for step in recs.get("while_waiting_steps", []):
        st.markdown(f"- {step}")
else:
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("#### 🧘 Targeted Yoga Asana & Pranayama")
        for asana in recs.get("yoga_asanas", []):
            st.markdown(f"""
            <div class="asana-card">
                <h4 style="margin: 0 0 6px 0; color: #0369a1;">{asana.get('name')}</h4>
                <p style="margin: 0 0 4px 0; font-size: 0.9rem;"><b>How-To:</b> {asana.get('how_to')}</p>
                <p style="margin: 0; font-size: 0.88rem; color: #0284c7;"><b>Clinical Benefit:</b> {asana.get('benefit')}</p>
            </div>
            """, unsafe_allow_html=True)

    with r_col2:
        st.markdown("#### 🥗 Personalized Dietary Guidelines")
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
        
        st.markdown("#### 🏃 Tailored Physical Activity Plan")
        st.info(recs.get("exercise_plan", "30 minutes daily moderate brisk walking."))

# 5. DOWNLOAD PDF REPORT & REGULATORY DISCLAIMER
st.markdown("---")
col_down, col_space = st.columns([1, 2])
with col_down:
    # Generate PDF
    pdf_data = generate_triage_pdf(
        fusion_result=fusion_res,
        risk_result=risk_res,
        vitals_result=vitals_res,
        symptom_result=symptom_res,
        patient_info={
            "age": p_age,
            "sex": p_sex,
            "resting_bp": p_bp,
            "glucose": p_glucose
        }
    )
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="📄 Download Patient Triage Summary (PDF)",
        data=pdf_data,
        file_name=f"triage_summary_{timestamp_str}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.markdown(f"""
<div class="disclaimer-box">
    <b>Regulatory & Clinical Safety Notice:</b> {fusion_res.get('disclaimer')}
</div>
""", unsafe_allow_html=True)
