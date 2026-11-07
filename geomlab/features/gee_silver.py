"""Silver: GEE raster statistics per H3 cell (bronze reduceRegions records → typed table).

  silver.gee_cells  city, h3_id, pop_cell, ndvi_mean, ndbi_mean, ndwi_mean, builtup_ratio,
                    trees_ratio, grass_ratio, water_ratio, bare_ratio, nightlight_mean,
                    nightlight_trend (radiance change per year, OLS over the monthly series),
                    feature_date
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomlab.ingestion import gee as gee_ingest
from geomlab.lakehouse import Lake, get_lake, read_bytes_table
from geomlab.logging import get_logger

log = get_logger(__name__)
RATIO_COLS = ["builtup_ratio", "trees_ratio", "grass_ratio", "water_ratio", "bare_ratio"]
INDEX_COLS = ["ndvi_mean", "ndbi_mean", "ndwi_mean"]


def nightlight_trend(monthly: pd.DataFrame) -> pd.Series:
    """OLS slope of radiance over time, per row, in units per year. Columns ``nl_YYYY_MM``."""
    cols = sorted(monthly.columns)
    if len(cols) < 3:
        return pd.Series(np.nan, index=monthly.index)
    t = np.array([int(c[3:7]) + (int(c[8:10]) - 1) / 12 for c in cols])
    y = monthly[cols].to_numpy(dtype="float64")
    tc = t - t.mean()
    mask = ~np.isnan(y)
    yc = np.where(mask, y - np.nanmean(y, axis=1, keepdims=True), 0.0)
    denom = (mask * tc**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (yc * tc).sum(axis=1) / denom
    slope[mask.sum(axis=1) < 3] = np.nan
    return pd.Series(slope, index=monthly.index)


def clean(raw: pd.DataFrame, city: str) -> pd.DataFrame:
    nl_cols = [c for c in raw.columns if c.startswith("nl_")]
    out = pd.DataFrame({"city": city, "h3_id": raw["h3_id"].astype(str)})
    out["pop_cell"] = pd.to_numeric(raw.get("worldpop_sum"), errors="coerce").clip(lower=0)
    for c in INDEX_COLS:
        out[c] = pd.to_numeric(raw.get(c), errors="coerce").clip(-1, 1)
    for c in RATIO_COLS:
        out[c] = pd.to_numeric(raw.get(c), errors="coerce").clip(0, 1)
    nl = raw[nl_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0) if nl_cols else pd.DataFrame(index=raw.index)
    out["nightlight_mean"] = nl.mean(axis=1) if nl_cols else np.nan
    out["nightlight_trend"] = nightlight_trend(nl) if nl_cols else np.nan
    out["feature_date"] = str(raw["extracted_at"].iloc[0])[:10] if "extracted_at" in raw else None
    return out.drop_duplicates("h3_id")


def run(city_slug: str, lake: Lake | None = None) -> int | None:
    lake = lake or get_lake()
    rows = lake.manifest(gee_ingest.SOURCE)
    rows = rows[rows["object_key"].str.endswith(f"/{city_slug}_cells.parquet")]
    if rows.empty:
        log.warning("gee_silver city=%s no bronze extract; GEE features will be missing", city_slug)
        return None
    df = clean(read_bytes_table(lake.get_raw(rows.iloc[-1]["object_key"])), city_slug)
    return lake.write("silver", "gee_cells", df, predicate=f"city = '{city_slug}'")
