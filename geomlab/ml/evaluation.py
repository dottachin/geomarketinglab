"""Spatial cross-validation and ranking metrics (doc §6.3).

Folds group cells by their H3 parent at a coarse resolution (spatial blocks), never at
random. Training cells within ``buffer_rings`` of any test cell are dropped from the
training fold, so neighbouring-cell similarity cannot leak into the score.
"""

from __future__ import annotations

from collections.abc import Iterator

import h3
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


def spatial_blocks(rows: pd.DataFrame, parent_resolution: int) -> pd.Series:
    return rows["city"] + ":" + rows["h3_id"].map(lambda c: h3.cell_to_parent(c, parent_resolution))


def spatial_folds(rows: pd.DataFrame, n_folds: int, parent_resolution: int,
                  buffer_rings: int = 1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    blocks = spatial_blocks(rows, parent_resolution)
    n_groups = blocks.nunique()
    k = max(2, min(n_folds, n_groups))
    for train_idx, test_idx in GroupKFold(n_splits=k).split(rows, groups=blocks):
        if buffer_rings > 0:
            test_cells = set(zip(rows["city"].iloc[test_idx], rows["h3_id"].iloc[test_idx], strict=True))
            buffer = {(c, n) for c, h in test_cells for n in h3.grid_disk(h, buffer_rings)}
            keys = list(zip(rows["city"].iloc[train_idx], rows["h3_id"].iloc[train_idx], strict=True))
            train_idx = train_idx[[key not in buffer for key in keys]]
        yield train_idx, test_idx


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    return float(y[np.argsort(-s)[:k]].mean()) if k else float("nan")


def ndcg_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    if k == 0 or y.sum() == 0:
        return float("nan")
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (y[np.argsort(-s)[:k]] * disc).sum()
    idcg = (np.sort(y)[::-1][:k] * disc).sum()
    return float(dcg / idcg)


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "mean_pred": float(p[m].mean()),
                        "frac_pos": float(y[m].mean()), "n": int(m.sum())})
    return out


def evaluate(rows: pd.DataFrame, scores: np.ndarray, ks: list[int]) -> dict[str, float]:
    """Metrics per business type (pooled over cities), ranking metrics per (city, type);
    returns per-type values and their macro average."""
    df = rows[["city", "business_type", "label"]].assign(score=scores)
    out: dict[str, float] = {}
    per_type = []
    for bt, g in df.groupby("business_type"):
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        m = {"pr_auc": np.nan, "roc_auc": np.nan, "brier": float(brier_score_loss(y, np.clip(s, 0, 1)))}
        if 0 < y.sum() < len(y):
            m["pr_auc"] = float(average_precision_score(y, s))
            m["roc_auc"] = float(roc_auc_score(y, s))
            m["pr_auc_lift"] = m["pr_auc"] / y.mean()
        for k in ks:
            p = [precision_at_k(c["label"].to_numpy(), c["score"].to_numpy(), k) for _, c in g.groupby("city")]
            n = [ndcg_at_k(c["label"].to_numpy(), c["score"].to_numpy(), k) for _, c in g.groupby("city")]
            m[f"precision_at_{k}"] = float(np.nanmean(p))
            m[f"ndcg_at_{k}"] = float(np.nanmean(n)) if not np.all(np.isnan(n)) else np.nan
        for name, v in m.items():
            out[f"{bt}.{name}"] = v
        per_type.append(m)
    for name in per_type[0] if per_type else []:
        vals = [m[name] for m in per_type if name in m]
        out[name] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
    return out
