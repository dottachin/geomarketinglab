"""Feature-drift and score-distribution monitoring (doc §6.4).

Population Stability Index per feature between the training sample stored in the
champion bundle and the current gold features, plus the KS statistic. Results are logged
to the champion's MLflow run; features above ``drift.psi_alert`` raise an alert in the
scoring DAG (and can trigger retraining). Evidently AI can render the same comparison as
an HTML report; PSI/KS are computed here directly to keep the dependency footprint small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from geomlab.config import model_config


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in reference.columns:
        if col not in current:
            continue
        r = reference[col].to_numpy(dtype=float)
        c = current[col].to_numpy(dtype=float)
        ks = ks_2samp(r[~np.isnan(r)], c[~np.isnan(c)]).statistic if np.isfinite(r).any() and np.isfinite(c).any() else np.nan
        rows.append({"feature": col, "psi": psi(r, c), "ks": float(ks)})
    rep = pd.DataFrame(rows)
    rep["alert"] = rep["psi"] > float(model_config()["drift"]["psi_alert"])
    return rep.sort_values("psi", ascending=False)
