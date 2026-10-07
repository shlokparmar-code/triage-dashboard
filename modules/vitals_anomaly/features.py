"""Rolling window feature engineering for multi-variate physiological time series."""

from typing import List, Tuple
import numpy as np
import pandas as pd


PHYSIOLOGICAL_CHANNELS = ["heart_rate", "spo2", "respiratory_rate", "temperature"]


def extract_rolling_features(
    df: pd.DataFrame,
    window_size: int = 6
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute rolling statistics (mean, std, min, max, slope) across physiological channels.

    Args:
        df: Input DataFrame containing vital signs columns.
        window_size: Number of time steps for rolling window calculation.

    Returns:
        Tuple of (feature_df: pd.DataFrame, feature_column_names: List[str]).
    """
    feat_df = pd.DataFrame(index=df.index)
    feature_cols = []

    available_channels = [c for c in PHYSIOLOGICAL_CHANNELS if c in df.columns]
    if not available_channels:
        # Fallback to heart_rate and spo2 if present under aliases
        if "hr" in df.columns:
            df["heart_rate"] = df["hr"]
            available_channels.append("heart_rate")
        if "oxygen" in df.columns or "o2" in df.columns:
            df["spo2"] = df.get("oxygen", df.get("o2"))
            available_channels.append("spo2")

    for ch in available_channels:
        series = df[ch].astype(float)
        
        # Raw value
        feat_df[f"{ch}_raw"] = series
        feature_cols.append(f"{ch}_raw")

        # Rolling mean
        rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
        feat_df[f"{ch}_mean"] = rolling_mean
        feature_cols.append(f"{ch}_mean")

        # Rolling standard deviation (instability indicator)
        rolling_std = series.rolling(window=window_size, min_periods=1).std().fillna(0.0)
        feat_df[f"{ch}_std"] = rolling_std
        feature_cols.append(f"{ch}_std")

        # Rolling min & max
        rolling_min = series.rolling(window=window_size, min_periods=1).min()
        rolling_max = series.rolling(window=window_size, min_periods=1).max()
        feat_df[f"{ch}_min"] = rolling_min
        feat_df[f"{ch}_max"] = rolling_max
        feature_cols.extend([f"{ch}_min", f"{ch}_max"])

        # Rolling slope (rate of acute change)
        slope = (series - series.shift(window_size)).fillna(0.0) / max(window_size, 1)
        feat_df[f"{ch}_slope"] = slope
        feature_cols.append(f"{ch}_slope")

    # Forward & backward fill any residual boundary NaNs
    feat_df = feat_df.bfill().ffill().fillna(0.0)
    return feat_df, feature_cols
