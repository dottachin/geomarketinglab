"""Silver: human place names for H3 cells (UI labels, never model features).

  silver.cell_names  city, h3_id, neighbourhood, landmark, landmark_m

neighbourhood  nearest OSM place node (silver.places) to the cell centre, by tier: fine places
               (neighbourhood/quarter) within a small radius win over a suburb within a larger
               one; null when none (the UI falls back to the district).
landmark       named POI from a prominent category (silver.poi) near the centre, chosen by
               ``distance_m - bonus_m[category]`` so a mall beats a slightly closer school.

Radii, place types and bonuses come from config/naming.yaml. Distances are measured in the
city's projected CRS via geomlab.geo. Deterministic: re-running yields identical rows.
"""

from __future__ import annotations

import re

import geopandas as gpd
import pandas as pd

from geomlab import geo
from geomlab.config import get_city, naming_config
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger

log = get_logger(__name__)

# Tifinagh, Arabic (+ supplement, presentation forms A/B) and invisible bidi/zero-width marks.
_NON_LATIN = re.compile(
    "[ⴰ-⵿؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿​-‏‪-‮⁦-⁩]+"
)
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|«\s*»")
_SEP = r"[-–—/|·,;:]"
_REPEATED_SEP = re.compile(rf"\s*({_SEP})(?:\s*{_SEP})+\s*")
_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
_TRIM = " \t\r\n-–—/|·,;:.\"'"

NAME_COLUMNS = ["city", "h3_id", "neighbourhood", "landmark", "landmark_m"]


def _trim_brackets(text: str) -> str:
    """Drop unbalanced edge brackets and a pair wrapping the whole text ("(Hassan II)" → "Hassan II")."""
    while text:
        if text[0] == "(" and text[-1] == ")" and text[1:-1].count("(") == text[1:-1].count(")") == 0:
            text = text[1:-1]
        elif text[0] == "(" and text.count("(") > text.count(")"):
            text = text[1:]
        elif text[-1] == ")" and text.count(")") > text.count("("):
            text = text[:-1]
        else:
            break
        text = text.strip(_TRIM)
    return text


def _latin_part(value: str | None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = _NON_LATIN.sub(" ", str(value))
    text = _EMPTY_BRACKETS.sub(" ", text)
    text = _REPEATED_SEP.sub(r" \1 ", text)
    text = _trim_brackets(re.sub(r"\s+", " ", text).strip(_TRIM))
    return text if _LATIN_LETTER.search(text) else None


def clean_name(name: str | None, name_en: str | None = None, name_fr: str | None = None) -> str | None:
    """Display-ready Latin name: ``name:en``, then ``name:fr``, then ``name`` without its
    Tifinagh/Arabic segments. When nothing Latin is left, the original ``name`` (Arabic only)."""
    for candidate in (name_en, name_fr, name):
        latin = _latin_part(candidate)
        if latin:
            return latin
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    original = re.sub(r"\s+", " ", str(name)).strip()
    return original or None


def clean_names(df: pd.DataFrame) -> pd.Series:
    """Vectorised :func:`clean_name` over ``name`` / ``name_en`` / ``name_fr`` columns (any may be missing)."""
    cols = [df[c] if c in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
            for c in ("name", "name_en", "name_fr")]
    return pd.Series([clean_name(n, en, fr) for n, en, fr in zip(*cols, strict=True)], index=df.index, dtype=object)


def _points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=geo.WGS84)


def pick_neighbourhood(cents: gpd.GeoDataFrame, places: pd.DataFrame, tiers: list[dict], epsg: int) -> pd.Series:
    """Name of the nearest place of the first tier that has one within its radius (else None),
    indexed like ``cents``. ``places`` has ``place``, ``name``, ``lon``, ``lat``."""
    out = pd.Series([None] * len(cents), index=cents.index, dtype=object)
    places = places[places["name"].notna()] if len(places) else places
    for tier in tiers:
        todo = cents[out.isna()]
        cand = places[places["place"].isin(tier["places"])] if len(places) else places
        if todo.empty or cand.empty:
            continue
        cand = _points(cand.reset_index(drop=True))
        pairs = geo.pairs_within_m(todo, cand, tier["radius_m"], epsg)
        if pairs.empty:
            continue
        best = pairs.sort_values(["distance_m", "target"]).drop_duplicates("source")
        out.loc[best["source"].to_numpy()] = cand.loc[best["target"].to_numpy(), "name"].to_numpy()
    return out


def pick_landmark(
    cents: gpd.GeoDataFrame, poi: pd.DataFrame, bonus_m: dict[str, float], radius_m: float, epsg: int
) -> pd.DataFrame:
    """``landmark`` (display name) and ``landmark_m`` (distance to the centre) per cell, indexed
    like ``cents``; the lowest ``distance_m - bonus_m[category]`` among named POIs of the listed
    categories within ``radius_m`` wins."""
    out = pd.DataFrame({"landmark": pd.Series([None] * len(cents), index=cents.index, dtype=object),
                        "landmark_m": pd.Series(float("nan"), index=cents.index, dtype="float64")})
    if len(poi) == 0:
        return out
    cand = poi[poi["category"].isin(list(bonus_m))].copy()
    cand["display"] = clean_names(cand)
    cand = cand[cand["display"].notna()].reset_index(drop=True)
    if cand.empty:
        return out
    cand = _points(cand)
    pairs = geo.pairs_within_m(cents, cand, radius_m, epsg)
    if pairs.empty:
        return out
    cats = cand.loc[pairs["target"].to_numpy(), "category"].to_numpy()
    pairs["score"] = pairs["distance_m"].to_numpy() - pd.Series(cats).map(bonus_m).to_numpy()
    best = pairs.sort_values(["score", "distance_m", "target"]).drop_duplicates("source")
    out.loc[best["source"].to_numpy(), "landmark"] = cand.loc[best["target"].to_numpy(), "display"].to_numpy()
    out.loc[best["source"].to_numpy(), "landmark_m"] = best["distance_m"].round(1).to_numpy()
    return out


def name_cells(grid: pd.DataFrame, places: pd.DataFrame, poi: pd.DataFrame, epsg: int,
               cfg: dict | None = None) -> pd.DataFrame:
    """Pure core of :func:`run`: one row per grid cell with the ``NAME_COLUMNS``."""
    cfg = cfg or naming_config()
    cents = _points(pd.DataFrame(grid[["city", "h3_id", "lon", "lat"]]).reset_index(drop=True))
    lm = pick_landmark(cents, poi, cfg["landmark_bonus_m"], cfg["landmark_radius_m"], epsg)
    out = pd.DataFrame({
        "city": cents["city"].to_numpy(),
        "h3_id": cents["h3_id"].to_numpy(),
        "neighbourhood": pick_neighbourhood(cents, places, cfg["tiers"], epsg).to_numpy(),
        "landmark": lm["landmark"].to_numpy(),
        "landmark_m": lm["landmark_m"].to_numpy(),
    })
    return out.astype({"city": "string", "h3_id": "string", "neighbourhood": "string", "landmark": "string",
                       "landmark_m": "float64"})


def run(city_slug: str, lake: Lake | None = None) -> int:
    """Write silver.cell_names for one city (predicate overwrite); return the row count."""
    lake = lake or get_lake()
    flt = [("city", "=", city_slug)]
    grid = lake.read("silver", "grid", filters=flt, columns=["city", "h3_id", "lon", "lat"])
    places = (lake.read("silver", "places", filters=flt) if lake.exists("silver", "places")
              else pd.DataFrame(columns=["place", "name", "lon", "lat"]))
    poi = lake.read("silver", "poi", filters=flt) if lake.exists("silver", "poi") else pd.DataFrame()
    names = name_cells(grid, places, poi, get_city(city_slug).crs_projected)
    lake.write("silver", "cell_names", names, predicate=f"city = '{city_slug}'")
    log.info("cell_names city=%s cells=%d neighbourhood=%d landmark=%d", city_slug, len(names),
             int(names["neighbourhood"].notna().sum()), int(names["landmark"].notna().sum()))
    return len(names)
