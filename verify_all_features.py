"""Verification script for all 7 additions & 6 benchmark patient profiles."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from modules.risk_screening.model import RiskScreeningModel
from modules.vitals_anomaly.model import VitalsAnomalyDetector
from modules.vitals_anomaly.generator import generate_synthetic_vitals
from modules.symptom_triage.model import SymptomTriageModel
from modules.special_populations.rules import evaluate_special_population
from modules.fusion.engine import fuse_triage_modalities
from modules.fusion.handoff import generate_handoff, load_referral_levels
from dashboard.queue import add_to_queue, get_queue, call_next, clear_queue
from dashboard.pdf_export import generate_triage_pdf, generate_home_pdf
from modules.registry.provider import LocalRegistryProvider, check_clinical_deterioration
from modules.registry.crypto import mask_mobile, validate_mobile


def main():
    print("=" * 70)
    print("MULTI-MODAL AI MEDICAL TRIAGE DASHBOARD - VERIFICATION OF ALL 7 FEATURES")
    print("=" * 70)

    symptom_model = SymptomTriageModel()
    vitals_model = VitalsAnomalyDetector()
    risk_model = RiskScreeningModel()
    registry = LocalRegistryProvider()

    # -------------------------------------------------------------
    # 1. BENCHMARK PATIENT 1: Adult LOW
    # -------------------------------------------------------------
    print("\n--- 1. BENCHMARK PATIENT 1: Adult Routine Care (LOW) ---")
    s_sym = "Mild runny nose, sneezing for 2 days, no fever, no chest pain"
    v_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none")
    r_feat = {"age": 28, "sex": 1, "resting_bp": 115, "trestbps": 115, "cholesterol": 175, "chol": 175, "glucose": 88, "bmi": 21.8, "thalach": 72, "symptom_text": s_sym}
    spec = evaluate_special_population({"age": 28, "sex": "Male"})

    res_sym = symptom_model.predict(s_sym)
    res_vit = vitals_model.detect(v_df)
    res_rsk = risk_model.predict(r_feat)
    res_fus = fuse_triage_modalities(res_rsk, res_vit, res_sym, r_feat, special_population_result=spec)

    print(f"Urgency: {res_fus['final_urgency']} (Score: {res_fus['final_score']:.1%})")
    assert res_fus["final_urgency"] == "LOW", f"Expected LOW, got {res_fus['final_urgency']}"
    print("  [PASS] Adult LOW tier correctly evaluated.")

    # -------------------------------------------------------------
    # 2. BENCHMARK PATIENT 2: Adult Urgent Outpatient (MEDIUM)
    # -------------------------------------------------------------
    print("\n--- 2. BENCHMARK PATIENT 2: Adult Urgent Outpatient (MEDIUM) ---")
    s_sym = "High fever of 101F for 3 days with nausea and body ache"
    v_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none")
    r_feat = {"age": 54, "sex": 1, "resting_bp": 140, "trestbps": 140, "cholesterol": 230, "chol": 230, "glucose": 150, "bmi": 28.0, "thalach": 86, "symptom_text": s_sym}
    spec = evaluate_special_population({"age": 54, "sex": "Male"})

    res_sym = symptom_model.predict(s_sym)
    res_vit = vitals_model.detect(v_df)
    res_rsk = risk_model.predict(r_feat)
    res_fus = fuse_triage_modalities(res_rsk, res_vit, res_sym, r_feat, special_population_result=spec)

    print(f"Urgency: {res_fus['final_urgency']} (Score: {res_fus['final_score']:.1%})")
    assert res_fus["final_urgency"] == "MEDIUM", f"Expected MEDIUM, got {res_fus['final_urgency']}"
    print("  [PASS] Adult MEDIUM tier correctly evaluated.")

    # -------------------------------------------------------------
    # 3. BENCHMARK PATIENT 3: Adult Emergency (HIGH)
    # -------------------------------------------------------------
    print("\n--- 3. BENCHMARK PATIENT 3: Adult Acute Emergency (HIGH) ---")
    s_sym = "Crushing chest pain radiating down left arm, gasping for air, sweating cold"
    v_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="combined_critical")
    r_feat = {"age": 66, "sex": 1, "resting_bp": 178, "trestbps": 178, "cholesterol": 295, "chol": 295, "glucose": 210, "bmi": 33.0, "thalach": 145, "symptom_text": s_sym}
    spec = evaluate_special_population({"age": 66, "sex": "Male"})

    res_sym = symptom_model.predict(s_sym)
    res_vit = vitals_model.detect(v_df)
    res_rsk = risk_model.predict(r_feat)
    res_fus = fuse_triage_modalities(res_rsk, res_vit, res_sym, r_feat, special_population_result=spec)

    print(f"Urgency: {res_fus['final_urgency']} (Score: {res_fus['final_score']:.1%})")
    assert res_fus["final_urgency"] == "HIGH", f"Expected HIGH, got {res_fus['final_urgency']}"
    assert res_fus["safety_override"] is True
    print("  [PASS] Adult HIGH emergency correctly triggered.")

    # -------------------------------------------------------------
    # 4. BENCHMARK PATIENT 4: Maternal Preeclampsia (Forced HIGH)
    # -------------------------------------------------------------
    print("\n--- 4. BENCHMARK PATIENT 4: Maternal Preeclampsia Red Flag (HIGH) ---")
    preg_pat = {
        "age": 26,
        "sex": "Female",
        "is_pregnant": True,
        "gestational_weeks": 32,
        "systolic_bp": 158,
        "diastolic_bp": 102,
        "pregnancy_red_flags": ["preg_severe_headache_bp", "preg_blurred_vision", "preg_swelling_face_hands"]
    }
    s_sym = "Severe persistent headache and blurred vision with facial swelling"
    spec_preg = evaluate_special_population(preg_pat)

    print(f"Special Population Urgency: {spec_preg['urgency']}")
    print(f"Preeclampsia Risk: {spec_preg['details']['preeclampsia_risk']}")
    assert spec_preg["urgency"] == "HIGH"
    assert spec_preg["details"]["preeclampsia_risk"] is True

    # Fuse with mild vitals & symptoms to verify safety gate override
    res_vit_norm = {"module": "Vitals Anomaly", "urgency": "LOW", "score": 0.10, "explanation": "Vitals normal"}
    res_sym_norm = {"module": "Symptom Triage", "urgency": "LOW", "score": 0.15, "explanation": "Headache"}
    res_rsk_norm = {"module": "Risk Screening", "urgency": "LOW", "score": 0.20, "explanation": "Baseline normal"}

    res_fus_preg = fuse_triage_modalities(
        res_rsk_norm, res_vit_norm, res_sym_norm, preg_pat, special_population_result=spec_preg
    )
    print(f"Fused Urgency: {res_fus_preg['final_urgency']} (Override: {res_fus_preg['safety_override']})")
    assert res_fus_preg["final_urgency"] == "HIGH"
    assert res_fus_preg["safety_override"] is True
    assert "Special Population" in str(res_fus_preg["overriding_modules"])
    print("  [PASS] Maternal preeclampsia red flag successfully forces HIGH urgency.")

    # -------------------------------------------------------------
    # 5. BENCHMARK PATIENT 5: Pediatric IMNCI Danger Signs (Forced HIGH)
    # -------------------------------------------------------------
    print("\n--- 5. BENCHMARK PATIENT 5: Child Under-5 WHO IMNCI Danger Sign (HIGH) ---")
    child_pat = {
        "age": 2,
        "age_months": 24,
        "sex": "Male",
        "child_danger_signs": ["unable_to_drink", "lethargic_unconscious", "chest_indrawing"],
        "vitals": {"rr": 55, "hr": 140, "spo2": 92}
    }
    spec_child = evaluate_special_population(child_pat)
    print(f"Category: {spec_child['details']['category']}")
    print(f"Module A Applicable: {spec_child['details']['module_a_applicable']}")
    print(f"Danger Signs: {spec_child['details']['danger_signs_detected']}")
    assert spec_child["details"]["module_a_applicable"] is False
    assert spec_child["urgency"] == "HIGH"

    res_fus_child = fuse_triage_modalities(
        res_rsk_norm, res_vit_norm, res_sym_norm, child_pat, special_population_result=spec_child
    )
    print(f"Child Fused Urgency: {res_fus_child['final_urgency']}")
    print(f"Module A Status in Fusion: {res_fus_child['module_contributions']['risk_screening']['status']}")
    assert res_fus_child["final_urgency"] == "HIGH"
    assert res_fus_child["module_contributions"]["risk_screening"]["applicable"] is False
    assert "Not applicable" in res_fus_child["module_contributions"]["risk_screening"]["status"]
    print("  [PASS] Pediatric danger signs force HIGH and exclude adult Module A.")

    # -------------------------------------------------------------
    # 6. BENCHMARK PATIENT 6: Existing Patient (Ramesh Kumar)
    # -------------------------------------------------------------
    print("\n--- 6. BENCHMARK PATIENT 6: Existing Multi-Hospital Patient (Ramesh Kumar) ---")
    pat_ramesh = registry.find_patient_by_mobile("+91 9876543210")
    assert pat_ramesh is not None, "Ramesh Kumar record must exist in seeded DB"
    print(f"Found Patient: {pat_ramesh['name']} (ID: {pat_ramesh['patient_id']})")
    print(f"Masked Phone: {pat_ramesh['mobile_masked']}")

    history = registry.get_history(pat_ramesh["patient_id"])
    print(f"Total Historical Visits: {len(history)} across network hospitals")
    assert len(history) >= 3, "Expected at least 3 historical visits"
    for v in history:
        print(f"  - [{v['timestamp']}] Facility: {v['hospital_name']} | Urgency: {v['final_urgency']} ({v['final_score']:.0%})")

    is_worse, det_msg = check_clinical_deterioration(
        history[0]["final_urgency"], history[0]["final_score"], history[1:]
    )
    print(f"Deterioration Alert: {is_worse} ({det_msg})")
    assert is_worse is True
    print("  [PASS] Multi-hospital history and deterioration detection verified.")

    # -------------------------------------------------------------
    # 7. PRIORITY LANE & QUEUE VERIFICATION
    # -------------------------------------------------------------
    print("\n--- 7. TRIAGE QUEUE & PRIORITY LANE VERIFICATION ---")
    hosp_test = "HOSP-001"
    clear_queue(hosp_test)

    # Add Low patient, then Medium patient, then High patient
    add_to_queue("P-LOW", hosp_test, "Low Patient", "LOW", 0.20, "Mild sneeze")
    add_to_queue("P-MED", hosp_test, "Med Patient", "MEDIUM", 0.50, "Fever")
    add_to_queue("P-HIGH", hosp_test, "High Patient", "HIGH", 0.95, "Chest pain")

    pri_lane, reg_queue = get_queue(hosp_test)
    print(f"Priority Lane count: {len(pri_lane)} (Expected: 1)")
    print(f"Regular Queue count: {len(reg_queue)} (Expected: 2)")
    assert len(pri_lane) == 1
    assert pri_lane[0]["name"] == "High Patient"

    # Call Next must pick High patient first
    called = call_next(hosp_test)
    print(f"First patient called: {called['name']} ({called['urgency']})")
    assert called["name"] == "High Patient"

    # Next called must be P-LOW (FCFS regular queue)
    called_2 = call_next(hosp_test)
    print(f"Second patient called: {called_2['name']} ({called_2['urgency']})")
    assert called_2["name"] == "Low Patient"
    print("  [PASS] Priority Lane precedence & FCFS queue verified.")

    # -------------------------------------------------------------
    # 8. SBAR HANDOFF & DUAL PDF VERIFICATION
    # -------------------------------------------------------------
    print("\n--- 8. SBAR DOCTOR HANDOFF & DUAL PDF EXPORT VERIFICATION ---")
    sbar = generate_handoff(
        patient=pat_ramesh,
        module_results={"risk_screening": res_rsk, "vitals_anomaly": res_vit, "symptom_triage": res_sym},
        final_result=res_fus,
        hospital_name="Apex District Hospital"
    )
    assert "SITUATION" in sbar and "BACKGROUND" in sbar and "ASSESSMENT" in sbar and "RECOMMENDATION" in sbar
    print("  [PASS] SBAR documentation generated with all 4 required sections.")

    hosp_pdf = generate_triage_pdf(res_fus, res_rsk, res_vit, res_sym, pat_ramesh, spec)
    assert len(hosp_pdf) > 1000
    home_pdf = generate_home_pdf(res_fus, pat_ramesh, spec)
    assert len(home_pdf) > 1000
    print(f"  [PASS] Hospital PDF ({len(hosp_pdf)} bytes) & Home PDF ({len(home_pdf)} bytes) generated successfully.")

    print("\n" + "=" * 70)
    print("ALL 7 FEATURES AND 6 BENCHMARK PATIENTS VERIFIED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
