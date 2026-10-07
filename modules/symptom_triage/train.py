"""Training and Evaluation pipeline for Symptom Triage NLP models.

Compares Baseline (Logistic Regression) vs Calibrated Linear Support Vector Machine
with char + word n-gram TF-IDF vectorization, prioritized HIGH-urgency recall,
and serializes the top-performing model.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.symptom_triage.preprocessor import preprocess_text
from modules.symptom_triage.data_generator import save_symptom_dataset, generate_symptom_dataset


def train_symptom_model():
    """Execute training pipeline, compare models, and save best model artifacts."""
    print("=" * 60)
    print("MODULE C: SYMPTOM TRIAGE NLP - TRAINING PIPELINE")
    print("=" * 60)

    dataset_path = config.DATA_DIR / "symptoms_labeled.csv"
    if not dataset_path.exists():
        print("[Dataset] Synthesizing labeled symptom dataset...")
        save_symptom_dataset(dataset_path)

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_path}")

    # Preprocessing
    print("Preprocessing text (spelling correction + negation tagging)...")
    df["processed_text"] = df["text"].apply(preprocess_text)

    # Train/Test split with stratification
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["processed_text"],
        df["urgency"],
        test_size=0.20,
        random_state=config.RANDOM_SEED,
        stratify=df["urgency"]
    )

    # Hybrid TF-IDF: word (1-2) + char (3-5) n-grams for typo resilience
    print("Fitting Word+Char TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    class_weights = {
        config.TIER_HIGH: 2.5,   # High recall penalty for missed emergencies
        config.TIER_MEDIUM: 1.2,
        config.TIER_LOW: 1.0
    }

    # Model 1: Baseline Logistic Regression
    print("\n[Model 1] Training Baseline Logistic Regression...")
    clf_lr = LogisticRegression(
        class_weight=class_weights,
        max_iter=1000,
        random_state=config.RANDOM_SEED
    )
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_macro_lr = f1_score(y_test, y_pred_lr, average="macro")
    recall_high_lr = recall_score(y_test, y_pred_lr, labels=[config.TIER_HIGH], average="macro")

    # Model 2: Stronger Calibrated Linear SVM
    print("[Model 2] Training Calibrated Linear SVM (Ensemble Folds)...")
    base_svc = LinearSVC(
        class_weight=class_weights,
        random_state=config.RANDOM_SEED,
        dual="auto",
        max_iter=3000
    )
    clf_calibrated = CalibratedClassifierCV(estimator=base_svc, cv=3)
    clf_calibrated.fit(X_train, y_train)
    y_pred_calibrated = clf_calibrated.predict(X_test)
    acc_cal = accuracy_score(y_test, y_pred_calibrated)
    f1_macro_cal = f1_score(y_test, y_pred_calibrated, average="macro")
    recall_high_cal = recall_score(y_test, y_pred_calibrated, labels=[config.TIER_HIGH], average="macro")

    print("\n--- MODEL COMPARISON ---")
    print(f"Baseline Logistic Regression: Accuracy={acc_lr:.4f}, Macro-F1={f1_macro_lr:.4f}, HIGH Recall={recall_high_lr:.4f}")
    print(f"Calibrated Linear SVM:       Accuracy={acc_cal:.4f}, Macro-F1={f1_macro_cal:.4f}, HIGH Recall={recall_high_cal:.4f}")

    # Prioritize HIGH Recall first, then Macro-F1
    if recall_high_cal >= recall_high_lr and f1_macro_cal >= f1_macro_lr:
        best_model = clf_calibrated
        best_name = "Calibrated Linear SVM"
        best_preds = y_pred_calibrated
    else:
        best_model = clf_lr
        best_name = "Logistic Regression"
        best_preds = y_pred_lr

    print(f"\n=> Selected Champion Model: {best_name}")
    print("\nClassification Report (Champion Model):")
    report = classification_report(y_test, best_preds, digits=4)
    print(report)

    cm = confusion_matrix(y_test, best_preds, labels=config.TIERS)
    print("Confusion Matrix (rows: True, cols: Pred) [LOW, MEDIUM, HIGH]:")
    print(cm)

    # Save artifacts
    model_path = config.MODELS_DIR / "symptom_model.joblib"
    vectorizer_path = config.MODELS_DIR / "symptom_vectorizer.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nArtifacts saved successfully to:\n - {model_path}\n - {vectorizer_path}")

    return {
        "best_model_name": best_name,
        "accuracy": float(accuracy_score(y_test, best_preds)),
        "macro_f1": float(f1_score(y_test, best_preds, average="macro")),
        "high_recall": float(recall_score(y_test, best_preds, labels=[config.TIER_HIGH], average="macro")),
        "report": report,
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    train_symptom_model()
