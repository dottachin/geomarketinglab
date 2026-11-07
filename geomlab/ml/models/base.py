"""Common model interface + attribution helper.

Every model scores rows ``(city, h3_id, business_type)`` given ``CellData`` and returns
raw probabilities; ``attribute`` returns one SHAP-style value per (row, feature) in
log-odds space, so drivers are comparable across model families:

* LightGBM: exact TreeSHAP (``pred_contrib=True``).
* MLP / GNN: Expected Gradients (Erion et al. 2021) — the gradient-based SHAP
  approximation also used by ``shap.GradientExplainer``: integrated gradients averaged
  over baselines drawn from a background sample of training cells.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from geomlab.ml.data import CellData


class SuitabilityModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, cells: CellData, rows: pd.DataFrame, seed: int) -> SuitabilityModel: ...

    @abstractmethod
    def predict(self, cells: CellData, rows: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def attribute(self, cells: CellData, rows: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_features) contributions to the logit, in ``cells.feature_cols`` order."""


def expected_gradients(f_grad, x: np.ndarray, background: np.ndarray, n_samples: int = 32,
                       seed: int = 0) -> np.ndarray:
    """Expected gradients for a batch.

    ``f_grad(z) -> (output, grad)`` takes a (B, F) float32 array and returns d logit / dz.
    For each row we draw ``n_samples`` (baseline, alpha) pairs and average
    ``(x - baseline) * grad(baseline + alpha (x - baseline))``.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros_like(x, dtype=np.float64)
    for _ in range(n_samples):
        base = background[rng.integers(0, len(background), size=len(x))]
        alpha = rng.uniform(0, 1, size=(len(x), 1)).astype(np.float32)
        z = (base + alpha * (x - base)).astype(np.float32)
        _, grad = f_grad(z)
        total += (x - base) * grad
    return total / n_samples
