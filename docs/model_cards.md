# Model Cards & Clinical Validation

## Model Card 1: Tabular Chronic Risk Screening (Module A)

### Model Details
- **Developer:** Multi-Modal Medical Triage Engineering Team
- **Model Architecture:**
  - **Cardiovascular Disease:** Random Forest Classifier (150 estimators, max depth 6)
  - **Type 2 Diabetes Mellitus:** L2-Regularized Logistic Regression Pipeline with Standard Scaling
- **Input Features:** Age, Sex, Resting BP (trestbps), Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate (thalach), Exercise Angina, ST Depression (oldpeak), BMI, Plasma Glucose.
- **Explainability:** SHAP (SHapley Additive exPlanations) TreeExplainer and LinearExplainer with background population reference baselines.

### Evaluation Metrics
| Condition | Champion Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Heart Disease** | Random Forest | 0.9200 | 1.0000 | 0.1111 | 0.2000 | **0.8046** |
| **Diabetes** | Logistic Regression | 0.7143 | 0.6087 | 0.5185 | 0.5600 | **0.8230** |

### Limitations & Ethical Considerations
- **Demographic Representation:** The Pima Indians Diabetes dataset represents a specific subpopulation; generalization across broader rural populations requires local calibration.
- **Scope:** Intended strictly as a chronic background liability index, not an acute emergency diagnostic tool.

---

## Model Card 2: Vitals Anomaly Telemetry (Module B)

### Model Details
- **Architecture:**
  - **Primary:** Isolation Forest (unsupervised tree ensemble, contamination=0.02)
  - **Alternative:** Multi-layer Bottleneck Reconstruction Autoencoder (Hidden layers: 32 -> 8 -> 32)
- **Features Extracted:** Rolling-window (window size 6) mean, standard deviation, minimum, maximum, and rate-of-change slope across continuous physiological signals:
  - Heart Rate (bpm)
  - Blood Oxygen Saturation (\(\text{SpO}_2\), %)
  - Respiratory Rate (breaths/min)
  - Core Body Temperature (°C)
- **Safety Overrides:** Deterministic physiological threshold rules enforce HIGH urgency:
  - \(\text{SpO}_2 < 90\%\) (Severe Hypoxemia)
  - \(\text{Heart Rate} > 140\text{ bpm}\) (Severe Tachycardia)
  - \(\text{Heart Rate} < 40\text{ bpm}\) (Severe Bradycardia)
  - \(\text{Respiratory Rate} < 8 \text{ or } > 32\text{ breaths/min}\)

### Benchmark Anomaly Detection Metrics
Evaluated across synthetic injected anomaly sequences (Hypoxia, Tachycardia, Bradycardia, Septic Fever):
| Model | ROC-AUC | Recall on Anomalies | F1-Score |
| :--- | :--- | :--- | :--- |
| **Isolation Forest** | 0.9324 | 0.6778 | 0.7145 |
| **Reconstruction Autoencoder** | **0.9879** | **0.9861** | **0.9114** |

---

## Model Card 3: Free-Text Symptom Triage NLP (Module C)

### Model Details
- **Architecture:** Word (1-2) + Character (3-5) n-gram TF-IDF Vectorizer coupled to an Ensemble Calibrated Linear Support Vector Machine (`CalibratedClassifierCV`).
- **Safety Architecture:**
  - **Negation-Aware Red-Flag Rule Engine:** Scans clinical clauses for acute emergency signs (crushing chest pain, severe dyspnea, facial droop/stroke, hematemesis, anaphylaxis, suicidal ideation) while preserving negations ("no chest pain", "denies shortness of breath").
  - **Edge-Case Validation:** Guards against empty inputs, random character repetition (gibberish), and non-medical conversational inputs.
  - **Low-Confidence Fallback:** Automatically escalates low-confidence predictions to MEDIUM for human review.
- **Training Dataset:** 1,950 clinically categorized symptom sentences across LOW (Self-Care), MEDIUM (Urgent / 24-48h), and HIGH (Emergency).

### Evaluation Metrics
| Urgency Tier | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **HIGH (Emergency)** | 1.0000 | **1.0000** | 1.0000 | 130 |
| **MEDIUM (Urgent)** | 1.0000 | **1.0000** | 1.0000 | 130 |
| **LOW (Routine)** | 1.0000 | **1.0000** | 1.0000 | 130 |
| **Overall Macro Avg** | 1.0000 | **1.0000** | 1.0000 | 390 |

### Explainability
Calculates token-level coefficient log-odds attributions, mapping words to:
- **Escalating Tokens (Red):** Words elevating clinical urgency (e.g., "crushing", "chest", "bleeding").
- **De-escalating Tokens (Green):** Words indicating benign or mild symptoms (e.g., "mild", "slight", "no_fever").
