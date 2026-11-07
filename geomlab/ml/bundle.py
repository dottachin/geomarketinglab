"""The deployable artifact: a fitted model + calibrators + metadata, logged to MLflow as a
pyfunc so it can be versioned in the registry and loaded by batch scoring (mode A) and by
the live what-if tool (mode B)."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from geomlab.ml.data import CellData
from geomlab.ml.models.base import SuitabilityModel


@dataclass
class SuitabilityBundle:
    model: SuitabilityModel
    feature_cols: list[str]
    calibrators: dict[str, IsotonicRegression]
    metadata: dict = field(default_factory=dict)  # gold versions, data_date, metrics, trained_at
    model_version: str = "unregistered"

    @property
    def family(self) -> str:
        return self.model.name

    def cells(self, frame: pd.DataFrame) -> CellData:
        return CellData.from_frame(frame, self.feature_cols)

    def predict(self, cells: CellData, rows: pd.DataFrame) -> pd.DataFrame:
        """Raw and calibrated suitability for each row."""
        raw = self.model.predict(cells, rows)
        cal = raw.copy()
        for bt, g in rows.groupby("business_type"):
            c = self.calibrators.get(bt)
            if c is not None:
                pos = rows.index.get_indexer(g.index)
                cal[pos] = c.predict(raw[pos])
        return pd.DataFrame({"raw_score": raw, "suitability": np.clip(cal, 0, 1)}, index=rows.index)

    def attribute(self, cells: CellData, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.attribute(cells, rows), columns=self.feature_cols, index=rows.index)


def fit_calibrators(rows: pd.DataFrame, oof: np.ndarray, min_pos: int = 10) -> dict[str, IsotonicRegression]:
    """Isotonic calibration per business type on out-of-fold scores (pooled fallback)."""
    pooled = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, rows["label"])
    out = {}
    for bt, g in rows.groupby("business_type"):
        pos = rows.index.get_indexer(g.index)
        if g["label"].sum() >= min_pos:
            out[bt] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[pos], g["label"])
        else:
            out[bt] = pooled
    return out


class BundlePyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper. ``predict`` takes a gold.cell_features frame with an extra
    ``business_type`` column (one row per cell × type) and returns calibrated scores."""

    def __init__(self, bundle: SuitabilityBundle):
        self.bundle = bundle

    def predict(self, context, model_input: pd.DataFrame, params=None):  # noqa: ARG002
        rows = model_input[["city", "h3_id", "business_type"]].reset_index(drop=True)
        frame = model_input.drop(columns=["business_type"]).drop_duplicates(["city", "h3_id"])
        return self.bundle.predict(self.bundle.cells(frame), rows)["suitability"].to_numpy()
