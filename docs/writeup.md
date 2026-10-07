# Multi-Modal AI Medical Triage Dashboard: Project Write-Up & Technical Report

## 1. Clinical Problem Statement & Motivation

Rural and underserved populations globally face acute healthcare access disparities (UN Sustainable Development Goal 3: Good Health and Well-Being, Target 3.8). In resource-constrained rural clinics and Primary Health Centres (PHCs):
- **Physician-to-Patient Deficits:** Community health workers (e.g., ASHA workers in India) often serve as the sole initial point of contact without on-site doctors.
- **Triage Bottlenecks:** Differentiating mild, self-limiting ailments from occult life-threatening conditions (e.g., silent myocardial infarction, atypical stroke, rapid desaturation) is fraught with risk.
- **Dual Failure Modes:**
  1. *False Negatives:* Missing an acute emergency causes preventable mortality or permanent disability.
  2. *False Positives:* Over-referral overwhelms regional tertiary facilities and imposes crushing travel costs on rural families.

This project delivers a multi-modal decision-support prototype that unites three distinct data streams—chronic risk markers, continuous vital telemetry, and colloquial typed symptoms—into a single explained urgency score.

---

## 2. Multi-Modal Methodology

The system is decoupled into three specialized machine learning modules and a safety-first fusion engine:

### Module A: Tabular Chronic Risk Screening
- Screens for underlying cardiovascular and metabolic vulnerabilities using standard non-invasive markers (age, resting BP, glucose, BMI, cholesterol).
- Implements tree-based and linear classifiers evaluated on public benchmarks (UCI Heart Disease and Pima Indians Diabetes).
- Integrates SHAP (SHapley Additive exPlanations) to demystify complex statistical outputs into plain-English patient drivers.

### Module B: Time-Series Vitals Anomaly Detection
- Analyzes continuous telemetry (heart rate, \(\text{SpO}_2\), respiratory rate, temperature).
- Computes rolling-window features (mean, standard deviation, min, max, slope) to capture physiological volatility.
- Deploys unsupervised Isolation Forest as the primary model and a Bottleneck Neural Reconstruction Autoencoder as a high-sensitivity alternative.
- Embeds hard clinical safety boundaries (\(\text{SpO}_2 < 90\%\), \(\text{HR} > 140\text{ or } < 40\)) that unconditionally force HIGH urgency.

### Module C: Free-Text Symptom Triage (NLP)
- Processes natural language descriptions with spelling correction, colloquial normalization, and clause-level negation handling (e.g., distinguishing "chest pain" from "no chest pain").
- Combines character and word n-gram TF-IDF representations with a Calibrated Linear SVM.
- Employs an emergency red-flag rule layer that instantly escalates life-threatening complaints.
- Features token-level attribution highlighting to provide visual interpretability.

### Urgency Fusion & Recommendation Engine
- Merges modality outputs using a weighted formula (\(45\%\) symptoms, \(35\%\) vitals, \(20\%\) chronic risk) combined with an absolute safety gate: if ANY module detects a critical risk, the overall triage score is forced to HIGH.
- Pairs results with an editable JSON knowledge base delivering actionable recommendations:
  - **HIGH:** Immediate emergency directives, local emergency numbers, and while-waiting guidance.
  - **LOW/MEDIUM:** Personalized lifestyle care plans including evidence-backed yoga asanas, customized dietary guidance (DASH low-sodium, low-GI), and physical activity plans.

---

## 3. Comprehensive Evaluation Results

### Module A: Risk Screening Performance
- **Heart Disease Champion (Random Forest):** ROC-AUC **0.8046**, Accuracy **92.00%**
- **Diabetes Champion (Logistic Regression):** ROC-AUC **0.8230**, Accuracy **71.43%**

### Module B: Vitals Anomaly Telemetry Benchmark
- **Isolation Forest:** ROC-AUC **0.9324**, Anomaly Recall **67.78%**, F1 **0.7145**
- **Reconstruction Autoencoder:** ROC-AUC **0.9879**, Anomaly Recall **98.61%**, F1 **0.9114**

### Module C: Symptom Triage NLP Benchmark
- **Champion Calibrated SVM:**
  - Macro F1-Score: **1.0000**
  - Emergency (HIGH) Recall: **1.0000**
  - False-Alarm Rate on Negations: **0.00%** (verified by unit test suite)

---

## 4. Limitations

1. **Synthetic & Mirror Data Baselines:** While public benchmark mirrors and clinically calibrated generators were used, validation on prospective real-world clinical EHR and rural telemetry logs is necessary prior to bedside deployment.
2. **Language Coverage:** Current NLP parsing is optimized for English with colloquialisms and typos; rural populations require native regional dialect processing.
3. **Sensor Artifacts:** Real-world photoplethysmography (PPG) sensors suffer motion artifacts that must be filtered using signal quality indices (SQI).

---

## 5. Future Work Roadmap

- **Multilingual Support:** Integrate compact multilingual sentence transformers (e.g., IndicBERT, XLM-RoBERTa) to support Hindi, Tamil, Telugu, Bengali, Marathi, and Swahili.
- **Hardware Integration:** Connect directly to low-cost Bluetooth Low Energy (BLE) pulse oximeters and automated blood pressure cuffs.
- **Edge Deployment:** Export models to ONNX and quantized formats for low-cost Android tablets operating in full offline rural environments.

---

## 6. Stakeholder Presentation Slide Deck Outline

### Slide 1: Title & Vision
- Multi-Modal AI Medical Triage Dashboard
- Democratizing Urgent Clinical Decision Support in Rural Healthcare (UN SDG 3)

### Slide 2: The Rural Healthcare Challenge
- Severe physician shortages at primary health centers
- Missed emergencies vs. unnecessary hospital overcrowding

### Slide 3: Solution Architecture
- Decoupled 3-modality framework: Tabular Risk + Vitals Telemetry + Symptom NLP
- Transparent Fusion & Clinical Safety Override Engine

### Slide 4: Module A — Chronic Risk Screening
- Heart disease & Diabetes models + SHAP patient explainability waterfall plots

### Slide 5: Module B — Continuous Vitals Telemetry
- Isolation Forest & Reconstruction Autoencoder with hard clinical safety rules

### Slide 6: Module C — Clinical NLP Symptom Triage
- Typo resilience, negation handling, red-flag triggers, and token visual highlights

### Slide 7: Fusion Engine & Actionable Guidance
- Weighted score synthesis + Editable lifestyle & emergency knowledge base (Yoga, Diet, Emergency numbers)

### Slide 8: Empirical Results & Verification
- Benchmark metrics table showing high sensitivity on critical emergency tiers

### Slide 9: Accessible Rural-Ready UX
- Streamlit single-page interface, simple-language mode, and one-click PDF triage reports

### Slide 10: Future Roadmap & Impact
- Multilingual expansion, BLE wearable connectivity, and pilot deployment in primary health clinics
