# Multi-Modal AI Medical Triage Dashboard: System Architecture

## 1. High-Level Architecture Overview

The Multi-Modal AI Medical Triage Dashboard is engineered as a decoupled, multi-modal decision-support system. It integrates three independent machine learning modules (tabular chronic disease screening, time-series physiological anomaly telemetry, and NLP free-text symptom triage) and synthesizes their predictions into a single, calibrated urgency tier and clinical explanation.

```mermaid
flowchart TD
    subgraph INPUTS["Multi-Modal Clinical Inputs"]
        InA["Patient Demographics & Labs<br/>(Age, BP, Glucose, Chol, BMI)"]
        InB["Continuous Telemetry Stream<br/>(Heart Rate, SpO2, RR, Temp)"]
        InC["Typed Free-Text Symptoms<br/>(Colloquial complaints, typos)"]
    end

    subgraph MODULE_A["Module A: Risk Screening (Tabular)"]
        A_Prep["Feature Standardization & Imputation"]
        A_Model["Champion Models<br/>- Heart: Random Forest<br/>- Diabetes: Logistic Regression"]
        A_SHAP["SHAP Feature Attribution<br/>(Waterfall / Bar Insights)"]
        InA --> A_Prep --> A_Model --> A_SHAP
    end

    subgraph MODULE_B["Module B: Vitals Telemetry (Time-Series)"]
        B_Safety{"Clinical Safety Rules<br/>SpO2 < 90% or HR > 140 or < 40?"}
        B_Roll["Rolling Window Features<br/>(Mean, Std, Min, Max, Slope)"]
        B_Model["Anomaly Detection<br/>- Isolation Forest (Primary)<br/>- Reconstruction Autoencoder"]
        InB --> B_Safety
        B_Safety -- No --> B_Roll --> B_Model
        B_Safety -- Yes --> B_Override["Forced HIGH Urgency<br/>(Critical Physiological Distress)"]
    end

    subgraph MODULE_C["Module C: Symptom Triage (NLP)"]
        C_RedFlag{"Emergency Red-Flags<br/>(Chest pain, Dyspnea, Stroke)?"}
        C_Clean["Negation Tagging & Typo Normalizer<br/>('no chest pain' -> 'not_chest_pain')"]
        C_Model["Champion NLP Classifier<br/>(Char+Word TF-IDF + Calibrated Linear SVM)"]
        C_Explain["Token Attribution Weights<br/>(Escalating vs Calming tokens)"]
        InC --> C_RedFlag
        C_RedFlag -- No --> C_Clean --> C_Model --> C_Explain
        C_RedFlag -- Yes --> C_Override["Forced HIGH Urgency<br/>(Life-Threatening Red-Flag)"]
    end

    subgraph FUSION["Fusion & Recommendation Engine"]
        SafetyGate{"ANY Module = HIGH?"}
        WeightedAvg["Transparent Weighted Fusion<br/>Score = 0.45*Sym + 0.35*Vit + 0.20*Rsk"]
        Synth["Clinical Synthesis Generator<br/>(2-3 Plain English Sentences)"]
        Recs["Editable Knowledge Base<br/>(Yoga Asanas, DASH/Low-GI Diet, Emergency Dispatch)"]
    end

    subgraph OUTPUT["Single-Page Streamlit Dashboard"]
        Banner["Top-Line Urgency Banner<br/>(LOW / MEDIUM / HIGH)"]
        Cards["Three Independent Module Cards"]
        Plots["SHAP Bar + Vitals Telemetry Visualizations"]
        PDF["Downloadable Clinical Summary (PDF)"]
    end

    A_SHAP --> SafetyGate
    B_Model --> SafetyGate
    B_Override --> SafetyGate
    C_Explain --> SafetyGate
    C_Override --> SafetyGate

    SafetyGate -- Yes --> ForceHigh["Final Urgency: HIGH (Emergency Protocol)"]
    SafetyGate -- No --> WeightedAvg
    WeightedAvg --> CalibrateTier["Final Tier Mapping: LOW / MEDIUM / HIGH"]

    ForceHigh --> Synth --> Recs --> Banner
    CalibrateTier --> Synth --> Recs --> Banner
    Recs --> Cards
    A_SHAP --> Plots
    B_Model --> Plots
    Recs --> PDF
```

---

## 2. Standard Module Contract

Every module enforces a clean, decoupled interface returning a standard dictionary:

```python
{
    "module": str,                # "risk_screening" | "vitals_anomaly" | "symptom_triage"
    "urgency": "LOW" | "MEDIUM" | "HIGH",
    "score": float,               # 0.0 to 1.0 calibrated probability / anomaly index
    "explanation": str,           # Plain-English clinical rationale
    "details": dict               # Granular sub-scores, feature contributions, raw metrics
}
```

This strict schema guarantees that teammates can work independently on individual modules without merge friction or contract breaks.

---

## 3. Mathematical Fusion Specification

The final urgency score \( S_{\text{final}} \) combines the three modalities:

$$S_{\text{weighted}} = w_{\text{symptom}} \cdot S_{\text{symptom}} + w_{\text{vitals}} \cdot S_{\text{vitals}} + w_{\text{risk}} \cdot S_{\text{risk}}$$

Where the clinically prioritized weights are:
- \( w_{\text{symptom}} = 0.45 \) (Subjective acute clinical complaints)
- \( w_{\text{vitals}} = 0.35 \) (Objective real-time physiological status)
- \( w_{\text{risk}} = 0.20 \) (Long-term chronic background liability)

### Hard Clinical Safety Rule (Fail-Safe Override)
To prevent catastrophic algorithmic omissions:

$$\text{If } \max(U_{\text{symptom}}, U_{\text{vitals}}, U_{\text{risk}}) = \text{HIGH} \implies U_{\text{final}} = \text{HIGH}$$

$$S_{\text{final}} = \max(S_{\text{weighted}}, 0.85, S_{\text{symptom}}, S_{\text{vitals}}, S_{\text{risk}})$$
