"""Training and calibration pipeline for Module B: Vitals Anomaly Detection."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.vitals_anomaly.generator import generate_synthetic_vitals
from modules.vitals_anomaly.features import extract_rolling_features
from modules.vitals_anomaly.lstm_model import VitalsAutoencoder


def train_vitals_models():
    """Train Isolation Forest and Autoencoder models on baseline normal vitals and evaluate on anomaly benchmarks."""
    print("=" * 60)
    print("MODULE B: VITALS ANOMALY DETECTION - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Synthesize normal physiological baseline datasets
    print("[1] Generating baseline normal vitals for model training...")
    normal_runs = []
    for s in range(12):
        df_normal = generate_synthetic_vitals(duration_minutes=90, sampling_interval_sec=10, anomaly_type="none", seed=100 + s)
        feat_df, cols = extract_rolling_features(df_normal)
        normal_runs.append(feat_df)

    X_train_raw = pd.concat(normal_runs, ignore_index=True)
    print(f"Total baseline training feature vectors: {len(X_train_raw)} across {len(cols)} rolling features.")

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw.values)

    # 2. Train Primary Model: Isolation Forest
    print("\n[2] Training Isolation Forest...")
    isoforest = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )
    isoforest.fit(X_train_scaled)

    # 3. Train Alternative Model: Reconstruction Autoencoder
    print("\n[3] Training Neural Reconstruction Autoencoder...")
    autoencoder = VitalsAutoencoder(hidden_layer_sizes=(32, 8, 32), max_iter=250, random_state=config.RANDOM_SEED)
    autoencoder.fit(X_train_raw.values)

    # 4. Evaluation on Benchmark Synthetic Test Sets with Anomaly Ground Truth
    print("\n[4] Evaluating models on injected anomaly benchmarks...")
    test_types = ["tachycardia", "hypoxia", "bradycardia", "septic_fever"]
    
    y_true_all = []
    iso_scores_all = []
    ae_scores_all = []

    for t_type in test_types:
        df_test = generate_synthetic_vitals(duration_minutes=60, sampling_interval_sec=10, anomaly_type=t_type, seed=999)
        feat_test, _ = extract_rolling_features(df_test)
        
        # Isolation Forest scoring
        X_test_scaled = scaler.transform(feat_test.values)
        raw_iso = isoforest.decision_function(X_test_scaled)
        iso_scores = 1.0 / (1.0 + np.exp(7.0 * raw_iso))

        # Autoencoder scoring
        ae_scores = autoencoder.compute_anomaly_scores(feat_test.values)

        y_true_all.extend(df_test["ground_truth_anomaly"].values)
        iso_scores_all.extend(iso_scores)
        ae_scores_all.extend(ae_scores)

    y_true_arr = np.array(y_true_all)
    iso_scores_arr = np.array(iso_scores_all)
    ae_scores_arr = np.array(ae_scores_all)

    # Metrics
    auc_iso = roc_auc_score(y_true_arr, iso_scores_arr)
    bin_iso = (iso_scores_arr > 0.50).astype(int)
    f1_iso = f1_score(y_true_arr, bin_iso, zero_division=0)
    rec_iso = recall_score(y_true_arr, bin_iso, zero_division=0)

    auc_ae = roc_auc_score(y_true_arr, ae_scores_arr)
    bin_ae = (ae_scores_arr > 0.50).astype(int)
    f1_ae = f1_score(y_true_arr, bin_ae, zero_division=0)
    rec_ae = recall_score(y_true_arr, bin_ae, zero_division=0)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Isolation Forest:        ROC-AUC: {auc_iso:.4f} | Recall: {rec_iso:.4f} | F1: {f1_iso:.4f}")
    print(f"Reconstruction Autoenc:  ROC-AUC: {auc_ae:.4f} | Recall: {rec_ae:.4f} | F1: {f1_ae:.4f}")

    # 5. Persist Artifacts
    iso_path = config.MODELS_DIR / "vitals_isoforest.joblib"
    scaler_path = config.MODELS_DIR / "vitals_scaler.joblib"
    ae_path = config.MODELS_DIR / "vitals_autoencoder.joblib"

    joblib.dump(isoforest, iso_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(autoencoder, ae_path)

    print(f"\nArtifacts saved successfully:\n - {iso_path}\n - {scaler_path}\n - {ae_path}")

    return {
        "isoforest_metrics": {"roc_auc": float(auc_iso), "f1": float(f1_iso), "recall": float(rec_iso)},
        "autoencoder_metrics": {"roc_auc": float(auc_ae), "f1": float(f1_ae), "recall": float(rec_ae)}
    }


if __name__ == "__main__":
    train_vitals_models()
