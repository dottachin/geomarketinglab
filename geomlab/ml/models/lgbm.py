"""Baseline: one LightGBM classifier per business type (doc §6.2 "mandatory baseline")."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from geomlab.ml.data import CellData
from geomlab.ml.models.base import SuitabilityModel


class LightGBMModel(SuitabilityModel):
    name = "lightgbm"

    def __init__(self, **params):
        self.params = params
        self.models: dict[str, lgb.LGBMClassifier] = {}
        self.prior: dict[str, float] = {}

    def fit(self, cells: CellData, rows: pd.DataFrame, seed: int) -> LightGBMModel:
        X = cells.X
        for bt, g in rows.groupby("business_type"):
            y = g["label"].to_numpy()
            self.prior[bt] = float(y.mean())
            if y.min() == y.max():  # a single class: constant predictor
                continue
            clf = lgb.LGBMClassifier(random_state=seed, verbose=-1, subsample_freq=1, **self.params)
            clf.fit(X[cells.nodes(g)], y)
            self.models[bt] = clf
        return self

    def predict(self, cells: CellData, rows: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(rows))
        X = cells.X
        for bt, g in rows.groupby("business_type"):
            pos = rows.index.get_indexer(g.index)
            m = self.models.get(bt)
            out[pos] = m.predict_proba(X[cells.nodes(g)])[:, 1] if m else self.prior.get(bt, 0.0)
        return out

    def attribute(self, cells: CellData, rows: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(rows), len(cells.feature_cols)))
        X = cells.X
        for bt, g in rows.groupby("business_type"):
            m = self.models.get(bt)
            if m is None:
                continue
            pos = rows.index.get_indexer(g.index)
            contrib = m.predict_proba(X[cells.nodes(g)], pred_contrib=True)
            out[pos] = contrib[:, :-1]  # last column = expected value (bias)
        return out

    def feature_importance(self, feature_cols: list[str]) -> pd.Series:
        if not self.models:
            return pd.Series(dtype=float)
        gain = np.mean([m.booster_.feature_importance("gain") for m in self.models.values()], axis=0)
        return pd.Series(gain, index=feature_cols).sort_values(ascending=False)
