"""Google Earth Engine ingestion: zonal statistics per H3 cell (doc §3.1).

Ported from pfe-old ``src/data/extract_gee.py`` and extended with the VIIRS monthly series
(for the night-light *trend* feature, doc §5.2).

Datasets
  * Sentinel-2 SR harmonized, cloud-masked median   → NDVI, NDBI, NDWI
  * Dynamic World V1 mean class probabilities        → built-up / trees / grass / water / bare
  * VIIRS DNB monthly (12 months, one band per month) → night-light mean + trend
  * WorldPop 100 m, most recent year                 → residents per cell

Change detection: the token is ``<latest VIIRS month>|<WorldPop year>|<S2 window month>``.
Sentinel-2 revisits every ~5 days but a 12-month median only moves meaningfully monthly,
VIIRS is monthly and WorldPop yearly, so most daily runs are skipped (doc §3.1 remark).

Output: ``bronze/gee/ingest_date=…/<city>_cells.parquet`` — the reduceRegions records
exactly as Earth Engine returned them (one row per cell).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from geomlab.lakehouse import Lake, df_to_parquet_bytes, get_lake
from geomlab.logging import get_logger
from geomlab.settings import get_settings

log = get_logger(__name__)
SOURCE = "gee"
CHUNK = 200  # cells per reduceRegions().getInfo() call
DW_BANDS = {"built": "builtup_ratio", "trees": "trees_ratio", "grass": "grass_ratio",
            "water": "water_ratio", "bare": "bare_ratio"}
VIIRS = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
WORLDPOP = "WorldPop/GP/100m/R2025A/v1/pop"  # supersedes WorldPop/GP/100m/pop (ended 2020)


def init_ee():
    import ee

    s = get_settings()
    if not (s.earthengine_project or s.gee_service_account):
        raise RuntimeError("Earth Engine not configured: set EARTHENGINE_PROJECT (and optionally "
                           "GEE_SERVICE_ACCOUNT + GEE_SERVICE_ACCOUNT_KEY_PATH) in .env")
    if s.gee_service_account and s.gee_service_account_key_path:
        creds = ee.ServiceAccountCredentials(s.gee_service_account, s.gee_service_account_key_path)
        ee.Initialize(creds, project=s.earthengine_project)
    else:
        ee.Initialize(project=s.earthengine_project)
    return ee


def _worldpop(ee):
    """Morocco WorldPop images up to the current year (R2025A also ships projections to 2030;
    projected years are never used as observations)."""
    return (ee.ImageCollection(WORLDPOP).filter(ee.Filter.eq("country", "MAR"))
            .filter(ee.Filter.lte("year", date.today().year)))


def change_token(ee) -> str:
    viirs_last = ee.Date(ee.ImageCollection(VIIRS).aggregate_max("system:time_start")).format("YYYY-MM")
    wp_year = _worldpop(ee).aggregate_max("year")
    s2_month = date.today().strftime("%Y-%m")
    return f"{viirs_last.getInfo()}|{int(wp_year.getInfo())}|{s2_month}"


def _feature_image(ee, aoi, start: str, end: str):
    def mask_s2(img):
        good = img.select("SCL").remap([4, 5, 6, 7, 11], [1, 1, 1, 1, 1], 0)
        return img.updateMask(good)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask_s2).median())
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi_mean")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("ndbi_mean")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("ndwi_mean")
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi).filterDate(start, end)
          .select(list(DW_BANDS)).mean().rename(list(DW_BANDS.values())))
    return ndvi.addBands([ndbi, ndwi, dw])


def _viirs_stack(ee, months: int = 12):
    """One band per month (``nl_YYYY_MM``) for the last ``months`` available months."""
    col = ee.ImageCollection(VIIRS).sort("system:time_start", False).limit(months).select("avg_rad")
    def rename(img):
        return img.rename(ee.String("nl_").cat(ee.Date(img.get("system:time_start")).format("YYYY_MM")))
    return col.map(rename).toBands().regexpRename("^[0-9]+_", "")


def _reduce(ee, image, feats: list, scale: int, reducer) -> list[dict]:
    rows: list[dict] = []
    for i in range(0, len(feats), CHUNK):
        fc = ee.FeatureCollection(feats[i : i + CHUNK])
        out = image.reduceRegions(collection=fc, reducer=reducer, scale=scale, tileScale=4)
        rows.extend(f["properties"] for f in out.getInfo()["features"])
    return rows


def extract_city(ee, cells: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """``cells`` needs ``h3_id`` and ``geometry`` (shapely polygons, EPSG:4326)."""
    from shapely.geometry import mapping

    end = date.today()
    start = end - timedelta(days=30 * months)
    feats = [ee.Feature(ee.Geometry(mapping(g)), {"h3_id": h}) for h, g in zip(cells["h3_id"], cells["geometry"], strict=True)]
    aoi = ee.FeatureCollection(feats).geometry().bounds()

    base = pd.DataFrame(_reduce(ee, _feature_image(ee, aoi, start.isoformat(), end.isoformat()),
                                feats, 10, ee.Reducer.mean())).set_index("h3_id")
    nl = pd.DataFrame(_reduce(ee, _viirs_stack(ee, months), feats, 500, ee.Reducer.mean())).set_index("h3_id")
    worldpop = _worldpop(ee).sort("year", False).first().rename("population")
    pop = pd.DataFrame(_reduce(ee, worldpop, feats, 100, ee.Reducer.sum())).set_index("h3_id")
    pop = pop.rename(columns={"sum": "worldpop_sum"})[["worldpop_sum"]]
    out = base.join([nl, pop], how="left").reset_index()
    return out.drop(columns=[c for c in out.columns if c.startswith("system:")], errors="ignore")


def ingest(cells_by_city: dict[str, pd.DataFrame], lake: Lake | None = None, force: bool = False) -> dict:
    """Run zonal stats for every city when the change token moved. ``cells_by_city`` comes
    from ``silver.grid`` (the grid must exist before GEE can be reduced per cell)."""
    lake = lake or get_lake()
    ee = init_ee()
    token = change_token(ee)
    status = {}
    for city, cells in cells_by_city.items():
        fname = f"{city}_cells.parquet"
        if not force and lake.last_token(SOURCE, prefix=fname) == token:
            status[city] = "unchanged"
            continue
        df = extract_city(ee, cells)
        df["extracted_at"] = datetime.now(UTC).isoformat()
        df["change_token"] = token
        lake.put_raw(SOURCE, fname, df_to_parquet_bytes(df), url=json.dumps({"datasets": [VIIRS, WORLDPOP]}),
                     change_token=token)
        status[city] = f"extracted:{len(df)}"
        log.info("gee city=%s cells=%d token=%s", city, len(df), token)
    return status
