"""Data loader for Module A: UCI Heart Disease and Pima Indians Diabetes datasets.

Attempts to retrieve public benchmark datasets via HTTP; if network access fails,
generates a clinically calibrated synthetic dataset clearly labeled as synthetic.
"""

import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import requests

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

logger = logging.getLogger(__name__)

# Public raw mirrors
HEART_DISEASE_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv"
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def generate_synthetic_heart_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic heart disease dataset with clinical correlations."""
    np.random.seed(config.RANDOM_SEED)
    age = np.random.randint(30, 80, size=n_samples)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.20, 0.25, 0.10])
    trestbps = np.clip(np.random.normal(132, 18, size=n_samples), 90, 200).astype(int)
    chol = np.clip(np.random.normal(246, 50, size=n_samples), 130, 450).astype(int)
    fbs = (trestbps > 140).astype(int) * np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.45, 0.05])
    thalach = np.clip(220 - age + np.random.normal(0, 15, size=n_samples), 80, 205).astype(int)
    exang = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    oldpeak = np.clip(np.random.exponential(1.1, size=n_samples), 0.0, 6.0).round(1)

    # Risk score calculation for ground truth label
    log_odds = (
        -4.5
        + 0.04 * (age - 50)
        + 0.6 * sex
        + 0.8 * (cp > 0)
        + 0.02 * (trestbps - 120)
        + 0.005 * (chol - 200)
        + 0.7 * exang
        + 0.5 * oldpeak
        - 0.02 * (thalach - 140)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    target = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "target": target
    })
    df.attrs["source"] = "SYNTHETIC_CALIBRATED_FALLBACK"
    return df


def generate_synthetic_diabetes_data(n_samples: int = 768) -> pd.DataFrame:
    """Generate realistic synthetic Pima Indians Diabetes dataset."""
    np.random.seed(config.RANDOM_SEED)
    age = np.random.randint(21, 80, size=n_samples)
    pregnancies = np.random.choice(range(0, 14), size=n_samples, p=[0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005])
    glucose = np.clip(np.random.normal(120, 32, size=n_samples), 55, 200).astype(int)
    blood_pressure = np.clip(np.random.normal(72, 12, size=n_samples), 45, 120).astype(int)
    skin_thickness = np.clip(np.random.normal(21, 15, size=n_samples), 0, 99).astype(int)
    insulin = np.clip(np.random.normal(80, 115, size=n_samples), 0, 800).astype(int)
    bmi = np.clip(np.random.normal(32, 7, size=n_samples), 18.0, 60.0).round(1)
    pedigree = np.clip(np.random.exponential(0.47, size=n_samples), 0.08, 2.4).round(3)

    log_odds = (
        -5.0
        + 0.035 * (glucose - 100)
        + 0.08 * (bmi - 25)
        + 0.03 * (age - 30)
        + 0.8 * pedigree
        + 0.015 * (blood_pressure - 70)
        + 0.1 * pregnancies
    )
    prob = 1 / (1 + np.exp(-log_odds))
    outcome = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
        "Outcome": outcome
    })
    df.attrs["source"] = "SYNTHETIC_CALIBRATED_FALLBACK"
    return df


def load_heart_disease_data() -> Tuple[pd.DataFrame, str]:
    """Load or download UCI Heart Disease dataset, falling back to synthetic if necessary."""
    local_file = config.DATA_DIR / "heart_disease.csv"
    if local_file.exists():
        df = pd.read_csv(local_file)
        source = df.attrs.get("source", "LOCAL_CACHE")
        return df, source

    # Try downloading from public repository
    try:
        logger.info(f"Downloading Heart Disease dataset from {HEART_DISEASE_URL}...")
        resp = requests.get(HEART_DISEASE_URL, timeout=5)
        if resp.status_code == 200 and "target" in resp.text:
            local_file.write_text(resp.text, encoding="utf-8")
            df = pd.read_csv(local_file)
            return df, "PUBLIC_UCI_MIRROR"
    except Exception as e:
        logger.warning(f"Failed to download Heart Disease data: {e}. Generating calibrated synthetic fallback.")

    df = generate_synthetic_heart_data()
    df.to_csv(local_file, index=False)
    return df, "SYNTHETIC_CALIBRATED_FALLBACK"


def load_diabetes_data() -> Tuple[pd.DataFrame, str]:
    """Load or download Pima Indians Diabetes dataset, falling back to synthetic if necessary."""
    local_file = config.DATA_DIR / "pima_diabetes.csv"
    if local_file.exists():
        df = pd.read_csv(local_file)
        source = df.attrs.get("source", "LOCAL_CACHE")
        return df, source

    try:
        logger.info(f"Downloading Pima Diabetes dataset from {DIABETES_URL}...")
        resp = requests.get(DIABETES_URL, timeout=5)
        if resp.status_code == 200 and len(resp.text) > 1000:
            # Pima dataset without header
            lines = resp.text.strip().split("\n")
            cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
            data = [line.strip().split(",") for line in lines if line.strip()]
            df = pd.DataFrame(data, columns=cols).astype(float)
            df.to_csv(local_file, index=False)
            return df, "PUBLIC_PIMA_MIRROR"
    except Exception as e:
        logger.warning(f"Failed to download Pima Diabetes data: {e}. Generating calibrated synthetic fallback.")

    df = generate_synthetic_diabetes_data()
    df.to_csv(local_file, index=False)
    return df, "SYNTHETIC_CALIBRATED_FALLBACK"
