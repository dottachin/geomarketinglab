"""Batch inference (doc §2.1 step 6, §7.4 mode A): score every candidate cell × business
type with the champion, add the saturation penalty, SHAP drivers, confidence flags and
the model_version stamp, then publish to PostGIS (serving) and to gold/scores (Delta).

    expected_capacity = pop_catchment / residents_per_business
    saturation        = competitors / (competitors + expected_capacity)        ∈ [0, 1)
    final_score       = suitability × (1 − λ_type × saturation)

So a cell with demand but no competitor keeps its full suitability, and a cell where
competitors already exceed what the catchment population supports is pushed down.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from geomlab.config import business_types, model_config
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger
from geomlab.ml.bundle import SuitabilityBundle

log = get_logger(__name__)


def saturation(competitors: pd.Series, pop_catchment: pd.Series, residents_per_business: float) -> pd.Series:
    capacity = pop_catchment / residents_per_business
    capacity = capacity.fillna(capacity.median() if capacity.notna().any() else 1.0).clip(lower=0.1)
    return (competitors / (competitors + capacity)).clip(0, 1)


def confidence_reasons(cells: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    cfg = model_config()["scoring"]["low_confidence"]
    c = cells.set_index(["city", "h3_id"])
    key = pd.MultiIndex.from_arrays([rows["city"], rows["h3_id"]])
    osm = c["osm_pois_1km"].reindex(key).to_numpy()
    has_gee = c["has_gee"].reindex(key).fillna(False).to_numpy(dtype=bool)
    has_hcp = c["has_hcp"].reindex(key).fillna(False).to_numpy(dtype=bool)
    positives = rows.groupby(["city", "business_type"])["label"].transform("sum").to_numpy()
    reasons = []
    for i in range(len(rows)):
        r = []
        if not osm[i] >= cfg["min_osm_pois_1km"]:
            r.append("sparse_osm_coverage")
        if not has_gee[i]:
            r.append("missing_satellite_data")
        if not has_hcp[i]:
            r.append("missing_census_data")
        if positives[i] < 10:
            r.append("few_training_examples")
        reasons.append(r)
    return pd.Series(reasons, index=rows.index)


def score(bundle: SuitabilityBundle, cells_frame: pd.DataFrame, rows: pd.DataFrame,
          with_drivers: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure scoring: returns (scores, drivers) frames ready for serving."""
    rows = rows.reset_index(drop=True)
    cells = bundle.cells(cells_frame)
    pred = bundle.predict(cells, rows)
    out = rows[["city", "h3_id", "business_type"]].copy()
    out["suitability"] = pred["suitability"].to_numpy()
    out["raw_score"] = pred["raw_score"].to_numpy()
    out["saturation"] = 0.0
    out["final_score"] = 0.0
    for bt_slug, g in rows.groupby("business_type"):
        bt = business_types()[bt_slug]
        sat = saturation(g["comp_catchment"].astype(float), g["pop_catchment"].astype(float), bt.residents_per_business)
        out.loc[g.index, "saturation"] = sat.to_numpy()
        out.loc[g.index, "final_score"] = out.loc[g.index, "suitability"] * (1 - bt.saturation_lambda * sat.to_numpy())
    grp = out.groupby(["city", "business_type"])["final_score"]
    out["rank"] = grp.rank(ascending=False, method="first").astype(int)
    out["percentile"] = grp.rank(pct=True).astype(float)
    out["n_existing"] = rows["n_in_cell"].astype(int)
    out["competitors_catchment"] = rows["comp_catchment"].astype(int)
    out["nearest_competitor_m"] = rows["nearest_comp_m"].astype(float)
    out["pop_catchment"] = rows["pop_catchment"].astype(float)
    q75 = out.groupby(["city", "business_type"])["suitability"].transform(lambda s: s.quantile(0.75))
    out["opportunity"] = (out["suitability"] >= q75) & (out["n_existing"] == 0)
    reasons = confidence_reasons(cells_frame, rows)
    out["confidence_reasons"] = reasons
    out["low_confidence"] = reasons.map(bool)
    out["model_version"] = bundle.model_version
    out["data_date"] = pd.to_datetime(cells_frame["data_date"].max()).date()
    out["scored_at"] = datetime.now(UTC)

    drivers = pd.DataFrame()
    if with_drivers:
        drivers = top_drivers(bundle, cells, rows, out)
    return out, drivers


def top_drivers(bundle: SuitabilityBundle, cells, rows: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    n_top = int(model_config()["scoring"]["top_drivers"])
    attr = bundle.attribute(cells, rows).to_numpy()
    raw = cells.frame[bundle.feature_cols].to_numpy()[cells.nodes(rows)]
    order = np.argsort(-np.abs(attr), axis=1)[:, :n_top]
    recs = []
    for i in range(len(rows)):
        for r, j in enumerate(order[i]):
            v = raw[i, j]
            recs.append((rows.at[i, "h3_id"], rows.at[i, "business_type"], bundle.model_version, r + 1,
                         bundle.feature_cols[j], None if np.isnan(v) else float(v), float(attr[i, j])))
    return pd.DataFrame(recs, columns=["h3_id", "business_type", "model_version", "rank", "feature", "value", "shap"])


def run(lake: Lake | None = None, publish_db: bool = True) -> dict:
    from geomlab.ml.train import load_champion

    lake = lake or get_lake()
    bundle = load_champion()
    cells_frame = lake.read("gold", "cell_features")
    rows = lake.read("gold", "training_set")
    scores, drivers = score(bundle, cells_frame, rows)
    lake.write("gold", "scores", scores.assign(confidence_reasons=scores["confidence_reasons"].map(json.dumps)))
    lake.write("gold", "shap_drivers", drivers)
    if publish_db:
        from geomlab.serving import db

        grid = lake.read_geo("silver", "grid")
        db.init_schema()
        db.load_cells(grid)
        poi = lake.read_geo("silver", "poi")
        for city in poi["city"].unique():
            db.load_pois(poi[poi["city"] == city], city)
        db.load_gold_features(cells_frame)
        active = {"model_version": bundle.model_version, "family": bundle.family,
                  "data_date": scores["data_date"].iloc[0] if len(scores) else date.today(),
                  "metrics": json.dumps({k: v for k, v in bundle.metadata.get("cv", {}).items() if "mean" in k})}
        db.replace_scores(scores, drivers, bundle.model_version, active)
    log.info("scored rows=%d model_version=%s", len(scores), bundle.model_version)
    return {"rows": len(scores), "model_version": bundle.model_version}
