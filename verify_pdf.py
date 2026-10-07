"""Verification script to generate sample patient PDFs and render pages to PNG images."""

import io
from pathlib import Path
import pypdfium2 as pdfium

import config
from modules.risk_screening.model import RiskScreeningModel
from modules.vitals_anomaly.model import VitalsAnomalyDetector
from modules.vitals_anomaly.generator import generate_synthetic_vitals
from modules.symptom_triage.model import SymptomTriageModel
from modules.fusion.engine import fuse_triage_modalities
from dashboard.pdf_export import generate_triage_pdf

OUTPUT_DIR = config.BASE_DIR / "tests" / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_sample_patient():
    print("Generating Sample Patient 1 (Age 18, Male, BP 118, Glucose 92, high temp + body ache, normal vitals)...")
    symptom_model = SymptomTriageModel()
    vitals_model = VitalsAnomalyDetector()
    risk_model = RiskScreeningModel()

    symptom_text = "High temperature of 102F and persistent body ache with shivering for 2 days"
    sym_res = symptom_model.predict(symptom_text)

    vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="none", seed=42)
    vit_res = vitals_model.detect(vitals_df)

    risk_features = {
        "age": 18,
        "sex": 1,
        "resting_bp": 118,
        "cholesterol": 165,
        "glucose": 92,
        "bmi": 21.0,
        "thalach": 160,
        "symptom_text": symptom_text
    }
    risk_res = risk_model.predict(risk_features)

    fusion_res = fuse_triage_modalities(
        risk_result=risk_res,
        vitals_result=vit_res,
        symptom_result=sym_res,
        patient_features=risk_features,
        emergency_country="India"
    )

    print(f"Urgency: {fusion_res['final_urgency']}, Score: {fusion_res['final_score']}")
    print(f"Synthesis: {fusion_res['clinical_synthesis']}")

    pdf_bytes = generate_triage_pdf(
        fusion_result=fusion_res,
        risk_result=risk_res,
        vitals_result=vit_res,
        symptom_result=sym_res,
        patient_info={
            "age": 18,
            "sex": "Male",
            "resting_bp": 118,
            "glucose": 92
        }
    )

    pdf_path = OUTPUT_DIR / "sample_patient_triage.pdf"
    pdf_path.write_bytes(pdf_bytes)
    print(f"Saved PDF to {pdf_path}")

    # Render pages to PNG
    pdf = pdfium.PdfDocument(pdf_bytes)
    print(f"Total pages in sample patient PDF: {len(pdf)}")
    for i, page in enumerate(pdf):
        image = page.render(scale=2.0).to_pil()
        img_path = OUTPUT_DIR / f"sample_patient_page_{i+1}.png"
        image.save(img_path)
        print(f"Rendered page {i+1} to {img_path}")


def test_high_urgency_patient():
    print("\nGenerating High-Urgency Patient with very long symptoms and long Module A explanation...")
    symptom_model = SymptomTriageModel()
    vitals_model = VitalsAnomalyDetector()
    risk_model = RiskScreeningModel()

    symptom_text = (
        "Severe crushing chest pain radiating to left arm and neck, gasping for breath, cold profuse sweating, "
        "dizziness, intense nausea, accompanied by feeling of impending doom and sudden extreme weakness "
        "that began approximately 45 minutes ago after light physical exertion."
    )
    sym_res = symptom_model.predict(symptom_text)

    vitals_df = generate_synthetic_vitals(duration_minutes=30, anomaly_type="hypoxia", seed=42)
    vit_res = vitals_model.detect(vitals_df)

    risk_features = {
        "age": 72,
        "sex": 1,
        "resting_bp": 185,
        "cholesterol": 310,
        "glucose": 240,
        "bmi": 36.5,
        "thalach": 110,
        "symptom_text": symptom_text
    }
    risk_res = risk_model.predict(risk_features)

    fusion_res = fuse_triage_modalities(
        risk_result=risk_res,
        vitals_result=vit_res,
        symptom_result=sym_res,
        patient_features=risk_features,
        emergency_country="India"
    )

    pdf_bytes = generate_triage_pdf(
        fusion_result=fusion_res,
        risk_result=risk_res,
        vitals_result=vit_res,
        symptom_result=sym_res,
        patient_info={
            "age": 72,
            "sex": "Male",
            "resting_bp": 185,
            "glucose": 240
        }
    )

    pdf_path = OUTPUT_DIR / "high_urgency_patient_triage.pdf"
    pdf_path.write_bytes(pdf_bytes)
    print(f"Saved High-Urgency PDF to {pdf_path}")

    pdf = pdfium.PdfDocument(pdf_bytes)
    print(f"Total pages in high urgency patient PDF: {len(pdf)}")
    for i, page in enumerate(pdf):
        image = page.render(scale=2.0).to_pil()
        img_path = OUTPUT_DIR / f"high_urgency_page_{i+1}.png"
        image.save(img_path)
        print(f"Rendered page {i+1} to {img_path}")


if __name__ == "__main__":
    test_sample_patient()
    test_high_urgency_patient()
