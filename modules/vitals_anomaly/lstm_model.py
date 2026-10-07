"""Reconstruction-based Autoencoder for Time-Series Vitals Anomaly Detection.

Learns normal physiological manifolds by compressing rolling vitals into a lower-dimensional
latent bottleneck and reconstructing the input. Anomalies exhibit high Mean Squared Error (MSE).
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class VitalsAutoencoder:
    """Bottleneck Neural Autoencoder for physiological reconstruction anomaly scoring."""

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (32, 8, 32), max_iter: int = 250, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="tanh",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15
        )
        self.threshold = 0.50
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "VitalsAutoencoder":
        """Train autoencoder on normal baseline physiological features."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, X_scaled)
        
        # Calibrate baseline reconstruction error on training data (98th percentile as threshold)
        preds = self.model.predict(X_scaled)
        train_mse = np.mean(np.square(X_scaled - preds), axis=1)
        self.threshold = float(np.percentile(train_mse, 98.0))
        self.is_fitted = True
        return self

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized reconstruction error [0, 1] for each time window."""
        if not self.is_fitted:
            # Fallback if not fitted
            return np.zeros(len(X))

        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)

        # Scale MSE relative to calibrated baseline threshold
        # Score = 1 / (1 + exp(-4 * (mse - threshold) / threshold))
        normalized_score = 1.0 / (1.0 + np.exp(-3.5 * (mse - self.threshold) / max(self.threshold, 1e-4)))
        return np.clip(normalized_score, 0.0, 1.0)
