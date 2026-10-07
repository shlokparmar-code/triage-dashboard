"""Synthetic physiological vitals generator with clinical anomaly injection."""

import random
from datetime import datetime, timedelta
from typing import Generator, Optional
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


def generate_synthetic_vitals(
    duration_minutes: int = 60,
    sampling_interval_sec: int = 10,
    anomaly_type: str = "none",
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate multi-variate vital signs time-series with realistic physiology and optional anomalies.

    Args:
        duration_minutes: Length of recording in minutes.
        sampling_interval_sec: Time between readings (default 10s).
        anomaly_type: 'none' | 'tachycardia' | 'hypoxia' | 'bradycardia' | 'septic_fever' | 'combined_critical'.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: ['timestamp', 'heart_rate', 'spo2', 'respiratory_rate', 'temperature', 'ground_truth_anomaly']
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_points = int((duration_minutes * 60) / sampling_interval_sec)
    base_time = datetime.now() - timedelta(minutes=duration_minutes)
    timestamps = [base_time + timedelta(seconds=i * sampling_interval_sec) for i in range(n_points)]

    # Physiological baselines with smooth drift (circadian / autonomic fluctuation)
    t = np.linspace(0, 4 * np.pi, n_points)
    
    # Heart Rate: mean ~72 bpm, range ~62-84 bpm
    hr = 74.0 + 5.0 * np.sin(t) + np.random.normal(0, 1.8, n_points)
    
    # SpO2: mean ~98.2%, physiological ceiling at 100%
    spo2 = 98.4 + 0.4 * np.cos(t * 0.7) + np.random.normal(0, 0.35, n_points)
    spo2 = np.clip(spo2, 94.0, 100.0)
    
    # Respiratory Rate: mean ~16 breaths/min
    rr = 15.8 + 1.2 * np.sin(t * 1.3) + np.random.normal(0, 0.8, n_points)
    
    # Temperature: mean ~36.8 C
    temp = 36.8 + 0.15 * np.cos(t * 0.3) + np.random.normal(0, 0.08, n_points)

    anomaly_mask = np.zeros(n_points, dtype=int)

    # Inject anomaly window in middle 30% of recording
    start_idx = int(n_points * 0.40)
    end_idx = int(n_points * 0.65)
    w_len = end_idx - start_idx

    if anomaly_type == "tachycardia":
        # Acute tachycardia: HR jumps to 145-165 bpm
        hr[start_idx:end_idx] += np.linspace(35, 75, w_len) + np.random.normal(0, 2.0, w_len)
        anomaly_mask[start_idx:end_idx] = 1

    elif anomaly_type == "hypoxia":
        # Desaturation event: SpO2 plummets below 90%
        desat_curve = np.sin(np.linspace(0, np.pi, w_len)) * 12.0
        spo2[start_idx:end_idx] -= desat_curve
        spo2[start_idx:end_idx] = np.clip(spo2[start_idx:end_idx], 75.0, 100.0)
        # Reflex tachycardia during hypoxia
        hr[start_idx:end_idx] += (desat_curve * 1.5)
        anomaly_mask[start_idx:end_idx] = 1

    elif anomaly_type == "bradycardia":
        # Severe bradycardia: HR drops to 32-38 bpm
        hr[start_idx:end_idx] = 36.0 + np.random.normal(0, 1.5, w_len)
        anomaly_mask[start_idx:end_idx] = 1

    elif anomaly_type == "septic_fever":
        # Pyrexia spike with tachypnea and tachycardia
        temp[start_idx:end_idx] += np.linspace(0.5, 2.8, w_len)
        hr[start_idx:end_idx] += 38.0
        rr[start_idx:end_idx] += 12.0
        anomaly_mask[start_idx:end_idx] = 1

    elif anomaly_type == "combined_critical":
        # Severe hypoxia (SpO2 85%) + severe tachycardia (150 bpm)
        spo2[start_idx:end_idx] -= 12.5
        hr[start_idx:end_idx] += 70.0
        rr[start_idx:end_idx] += 14.0
        anomaly_mask[start_idx:end_idx] = 1

    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": np.round(hr, 1),
        "spo2": np.round(spo2, 1),
        "respiratory_rate": np.round(rr, 1),
        "temperature": np.round(temp, 2),
        "ground_truth_anomaly": anomaly_mask
    })
    return df


def stream_vitals_generator(
    df: pd.DataFrame,
    chunk_size: int = 1
) -> Generator[pd.DataFrame, None, None]:
    """Yield successive rows/chunks to emulate real-time patient monitor telemetry."""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]
