"""Training and evaluation pipeline for Module A: Tabular Risk Screening."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.risk_screening.data_loader import load_heart_disease_data, load_diabetes_data


def evaluate_binary_model(model, X_test, y_test, name: str) -> dict:
    """Compute complete metric suite for a binary classification model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[{name}] Metrics:")
    print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }


def train_risk_models():
    """Train, compare, and save models for Heart Disease and Diabetes risk."""
    print("=" * 60)
    print("MODULE A: TABULAR RISK SCREENING - TRAINING PIPELINE")
    print("=" * 60)

    # 1. HEART DISEASE
    print("\n--- 1. Training Cardiovascular Disease Risk Models ---")
    heart_df, heart_source = load_heart_disease_data()
    print(f"Heart Data Source: {heart_source} (rows: {len(heart_df)})")

    X_heart = heart_df.drop(columns=["target"])
    y_heart = heart_df["target"].astype(int)

    X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
        X_heart, y_heart, test_size=0.20, random_state=config.RANDOM_SEED, stratify=y_heart
    )

    # Model A1: Logistic Regression Pipeline
    pipe_lr_h = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000))
    ])
    pipe_lr_h.fit(X_h_train, y_h_train)
    res_lr_h = evaluate_binary_model(pipe_lr_h, X_h_test, y_h_test, "Heart: Logistic Regression")

    # Model A2: Random Forest
    rf_h = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=config.RANDOM_SEED)
    rf_h.fit(X_h_train, y_h_train)
    res_rf_h = evaluate_binary_model(rf_h, X_h_test, y_h_test, "Heart: Random Forest")

    # Select champion (prioritizing ROC-AUC then F1)
    if res_rf_h["roc_auc"] >= res_lr_h["roc_auc"]:
        champion_heart = rf_h
        champion_h_res = res_rf_h
    else:
        champion_heart = pipe_lr_h
        champion_h_res = res_lr_h

    print(f"\n=> Heart Champion Model: {champion_h_res['name']} (ROC-AUC: {champion_h_res['roc_auc']:.4f})")
    heart_model_path = config.MODELS_DIR / "heart_disease_model.joblib"
    joblib.dump(champion_heart, heart_model_path)
    joblib.dump(X_h_train.sample(min(80, len(X_h_train)), random_state=config.RANDOM_SEED), config.MODELS_DIR / "heart_background.joblib")

    # 2. DIABETES
    print("\n--- 2. Training Diabetes Risk Models ---")
    diab_df, diab_source = load_diabetes_data()
    print(f"Diabetes Data Source: {diab_source} (rows: {len(diab_df)})")

    X_diab = diab_df.drop(columns=["Outcome"])
    y_diab = diab_df["Outcome"].astype(int)

    X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
        X_diab, y_diab, test_size=0.20, random_state=config.RANDOM_SEED, stratify=y_diab
    )

    # Model D1: Logistic Regression Pipeline
    pipe_lr_d = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000))
    ])
    pipe_lr_d.fit(X_d_train, y_d_train)
    res_lr_d = evaluate_binary_model(pipe_lr_d, X_d_test, y_d_test, "Diabetes: Logistic Regression")

    # Model D2: Random Forest
    rf_d = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=config.RANDOM_SEED)
    rf_d.fit(X_d_train, y_d_train)
    res_rf_d = evaluate_binary_model(rf_d, X_d_test, y_d_test, "Diabetes: Random Forest")

    if res_rf_d["roc_auc"] >= res_lr_d["roc_auc"]:
        champion_diab = rf_d
        champion_d_res = res_rf_d
    else:
        champion_diab = pipe_lr_d
        champion_d_res = res_lr_d

    print(f"\n=> Diabetes Champion Model: {champion_d_res['name']} (ROC-AUC: {champion_d_res['roc_auc']:.4f})")
    diabetes_model_path = config.MODELS_DIR / "diabetes_model.joblib"
    joblib.dump(champion_diab, diabetes_model_path)
    joblib.dump(X_d_train.sample(min(80, len(X_d_train)), random_state=config.RANDOM_SEED), config.MODELS_DIR / "diabetes_background.joblib")

    print("\n[Artifacts] Module A models and SHAP backgrounds persisted successfully to models/")
    return {
        "heart_champion": champion_h_res,
        "diabetes_champion": champion_d_res
    }


if __name__ == "__main__":
    train_risk_models()
