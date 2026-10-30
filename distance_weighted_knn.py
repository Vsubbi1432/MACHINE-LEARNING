"""Distance-weighted k-NN regression, from scratch (UNIT II).

Weights are ``w_i = 1 / (d_i + eps)^2`` over the k nearest neighbours — the
textbook distance-weighted variant that fixes k-NN's discretisation of the
prediction function. Supports Euclidean and Manhattan distances.
Pure Python + numpy only.
"""
from __future__ import annotations

import numpy as np


class DistanceWeightedKNN:
    """k-NN regressor with inverse-square distance weights."""

    def __init__(self, k: int = 7, eps: float = 1e-9, metric: str = "euclidean"):
        self.k = k
        self.eps = eps
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> DistanceWeightedKNN:
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)
        return self

    def _neighbours(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.metric == "euclidean":
            d = np.sqrt(np.sum((self._X - x) ** 2, axis=1))
        else:  # manhattan
            d = np.sum(np.abs(self._X - x), axis=1)
        idx = np.argsort(d, kind="stable")[: self.k]
        return idx, d[idx]

    def predict_one(self, x: np.ndarray) -> float:
        idx, d = self._neighbours(np.asarray(x, dtype=float))
        weights = 1.0 / (d + self.eps) ** 2
        return float(np.sum(weights * self._y[idx]) / np.sum(weights))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self.predict_one(row) for row in np.asarray(X, dtype=float)])
