"""Training/scoring data: gold tables → arrays, plus the preprocessing every model shares.

``CellData`` holds ALL cells (the GNN needs non-candidate neighbours for message passing);
``rows`` is the (city, h3_id, business_type, label) frame the models are fitted/scored on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import h3
import numpy as np
import pandas as pd

from geomlab.config import business_type_index
from geomlab.features.dictionary import model_feature_columns
from geomlab.lakehouse import Lake, get_lake


@dataclass
class CellData:
    frame: pd.DataFrame                 # gold.cell_features, index = node id (0..n-1)
    feature_cols: list[str]
    node_of: dict[tuple[str, str], int] = field(default_factory=dict)
    _edges: np.ndarray | None = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, feature_cols: list[str] | None = None) -> CellData:
        frame = frame.reset_index(drop=True)
        cols = feature_cols or model_feature_columns(frame)
        for c in cols:  # live/what-if frames may lack a column → explicit NaN
            if c not in frame:
                frame[c] = np.nan
        node_of = {(c, h): i for i, (c, h) in enumerate(zip(frame["city"], frame["h3_id"], strict=True))}
        return cls(frame=frame, feature_cols=list(cols), node_of=node_of)

    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_cols].to_numpy(dtype="float64")

    def nodes(self, rows: pd.DataFrame) -> np.ndarray:
        return np.array([self.node_of[(c, h)] for c, h in zip(rows["city"], rows["h3_id"], strict=True)])

    @property
    def edges(self) -> np.ndarray:
        """Undirected H3 adjacency (distance 1) within each city, shape (2, E)."""
        if self._edges is None:
            src, dst = [], []
            for (city, cell), i in self.node_of.items():
                for nb in h3.grid_ring(cell, 1):
                    j = self.node_of.get((city, nb))
                    if j is not None:
                        src.append(i)
                        dst.append(j)
            self._edges = np.array([src, dst], dtype=np.int64)
        return self._edges


def type_ids(rows: pd.DataFrame) -> np.ndarray:
    idx = business_type_index()
    return rows["business_type"].map(idx).to_numpy(dtype=np.int64)


class Preprocessor:
    """Median imputation, signed log1p for heavy-tailed non-negative features, z-scoring.
    Fitted on training cells only (no information from test folds)."""

    def fit(self, X: np.ndarray) -> Preprocessor:
        with warnings.catch_warnings():  # all-NaN columns (e.g. no GEE yet) → median 0
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median_ = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
        Xi = np.where(np.isnan(X), self.median_, X)
        nonneg = (Xi >= 0).all(axis=0)
        p99, p50 = np.percentile(Xi, 99, axis=0), np.percentile(Xi, 50, axis=0)
        self.log_ = nonneg & (p99 > 10 * np.maximum(p50, 1e-9)) & (p99 > 10)
        Xt = self._log(Xi)
        self.mean_ = Xt.mean(axis=0)
        self.std_ = Xt.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def _log(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, self.log_] = np.log1p(np.clip(X[:, self.log_], 0, None))
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xi = np.where(np.isnan(X), self.median_, X)
        return ((self._log(Xi) - self.mean_) / self.std_).astype(np.float32)


def load_gold(lake: Lake | None = None, version: dict[str, int] | None = None,
              cities: list[str] | None = None) -> tuple[CellData, pd.DataFrame, dict[str, int]]:
    """Read gold.cell_features + gold.training_set (optionally at pinned Delta versions)."""
    lake = lake or get_lake()
    version = version or {}
    flt = [("city", "in", cities)] if cities else None
    cells = lake.read("gold", "cell_features", version=version.get("cell_features"), filters=flt)
    rows = lake.read("gold", "training_set", version=version.get("training_set"), filters=flt)
    used = {"cell_features": version.get("cell_features", lake.version("gold", "cell_features")),
            "training_set": version.get("training_set", lake.version("gold", "training_set"))}
    rows = rows[rows["business_type"].isin(business_type_index())].reset_index(drop=True)
    return CellData.from_frame(cells), rows, used
