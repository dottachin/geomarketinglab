"""Silver: HCP indicators per administrative unit, and disaggregated to H3 (doc §3.2).

Parsing and HCP↔OSM name matching are ported from pfe-old ``src/data/extract_hcp.py``
(geo-code normalisation, per-file parsers, alias table, apostrophe/diacritic handling).

What changed versus the old implementation: the old code copied each arrondissement's
values onto every hexagon whose centre fell inside it. Now, as the architecture asks,
values are *disaggregated*:

* extensive variables (counts: population, households, establishments, jobs) are split
  across the hexagons overlapping a unit proportionally to WorldPop population in the
  overlap (dasymetric, population-weighted); if WorldPop is missing, by overlap area;
* intensive variables (rates, percentages) are averaged over the units overlapping a
  hexagon, weighted by overlap area.

Tables:
  silver.hcp_units   geo_code, name_fr, <all parsed indicators>, hcp_source
  silver.hcp_cells   city, h3_id, hcp_<var> …, hcp_coverage (share of the cell covered by a matched unit)
"""

from __future__ import annotations

import io
import unicodedata
from typing import Any

import geopandas as gpd
import numpy as np
import openpyxl
import pandas as pd

from geomlab.config import get_city, hcp_config
from geomlab.ingestion import hcp as hcp_ingest
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------- parsing
def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()


def normalise_geo_code(val: Any, strip_leading_zero: bool = False) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        code = str(int(val))
    except (ValueError, TypeError):
        code = str(val).strip().rstrip(".").replace(".", "")
    if strip_leading_zero and code.startswith("0") and len(code) > 1:
        code = code.lstrip("0") or "0"
    return code if code.isdigit() else None


def _rows(data: bytes, sheet: str) -> list[tuple]:
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    return list(wb[sheet].iter_rows(values_only=True))


def _positional(rows: list[tuple], start: int, code_col: int, name_col: int, col_map: dict,
                strip_zero: bool = False) -> pd.DataFrame:
    records = []
    for row in rows[start:]:
        code = normalise_geo_code(row[code_col], strip_leading_zero=strip_zero)
        if code is None:
            continue
        rec = {"geo_code": code, "name_fr": str(row[name_col]) if row[name_col] else None}
        for pos, feat in col_map.items():
            if feat in ("geo_code", "geo_code_str", "name_fr", "name_ar"):
                continue
            pos = int(pos)
            rec[feat] = row[pos] if pos < len(row) else None
        records.append(rec)
    return pd.DataFrame(records).drop_duplicates("geo_code")


def parse_population(data: bytes, spec: dict) -> pd.DataFrame:
    df = _positional(_rows(data, spec["sheet"]), spec["data_start_row"], 6, 0, spec["columns"])
    df = df[~df["name_fr"].fillna("").str.lower().str.contains("milieu|dont le centre")]
    pop = pd.to_numeric(df["population"], errors="coerce")
    hh = pd.to_numeric(df["menages"], errors="coerce")
    df["mean_household_size"] = (pop / hh).where(hh > 0)
    return df


def build_units_table(files: dict[str, bytes]) -> pd.DataFrame:
    cfg = hcp_config()["files"]
    if "population" not in files:
        raise FileNotFoundError("HCP population file missing from bronze")
    master = parse_population(files["population"], cfg["population"])
    master = master[master["geo_code"].str.len() >= 7]  # communes / arrondissements only
    layout = {  # name: (code column, name column, strip leading zero)
        "housing": (0, 1, False), "transport": (0, 1, False),
        "economic": (5, 6, True), "socioeco": (0, 1, False),
    }
    for name, (code_col, name_col, strip) in layout.items():
        if name not in files:
            log.warning("hcp file %s missing; its indicators will be null", name)
            continue
        spec = cfg[name]
        part = _positional(_rows(files[name], spec["sheet"]), spec["data_start_row"], code_col,
                           name_col, spec["columns"], strip)
        part = part.drop(columns=[c for c in part.columns if c in master.columns and c != "geo_code"])
        master = master.merge(part, on="geo_code", how="left")
    num = [c for c in master.columns if c not in ("geo_code", "name_fr")]
    master[num] = master[num].apply(pd.to_numeric, errors="coerce")
    master["hcp_source"] = "HCP RGPH 2024"
    return master.reset_index(drop=True)


# ----------------------------------------------------------------------------- name matching
_PREFIXES = ("arrondissement de ", "arrondissement d'", "arrondissement du ", "arrondissement des ",
             "commune de ", "commune d'", "municipalite de ", "municipalite d'",
             "prefecture d'arrondissements de ", "prefecture d'arrondissements d'", "pachalik de ",
             "arrondissement ", "commune ")  # bare forms last (OSM Rabat: "Arrondissement Agdal-Riyad")


def leaf_key(name: str) -> str:
    s = strip_diacritics(name)
    for ch in ("’", "‘", "ʼ", "`"):
        s = s.replace(ch, "'")
    for p in _PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    return " ".join(s.replace("-", " ").split())


def match_units(units: pd.DataFrame, admin: gpd.GeoDataFrame, aliases: dict[str, str]) -> gpd.GeoDataFrame:
    """Attach HCP rows to OSM admin polygons by name (alias table first, then leaf keys).
    When several HCP rows map to one polygon, prefer the most specific (leaf) code."""
    alias_keys = {leaf_key(k): leaf_key(v) for k, v in aliases.items()}
    units = units.copy()
    units["_key"] = units["name_fr"].fillna("").map(leaf_key).map(lambda k: alias_keys.get(k, k))
    units["_leaf"] = (units["geo_code"].str[-1] != "0").astype(int)
    units["_len"] = units["geo_code"].str.len()
    units = units.sort_values(["_leaf", "_len"], ascending=False).drop_duplicates("_key")
    admin = admin.copy()
    admin["_key"] = admin["unit_name"].map(leaf_key).map(lambda k: alias_keys.get(k, k))
    matched = admin.merge(units.drop(columns=["_leaf", "_len"]), on="_key", how="inner")
    missing = sorted(set(admin["unit_name"]) - set(matched["unit_name"]))
    if missing:
        log.warning("hcp_unmatched_admin_units n=%d sample=%s", len(missing), missing[:10])
    return gpd.GeoDataFrame(matched.drop(columns="_key"), geometry="geometry", crs=admin.crs)


# ----------------------------------------------------------------------------- disaggregation
def disaggregate(
    units: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    pop_cell: pd.Series | None,
    extensive: list[str],
    intensive: list[str],
    epsg: int,
) -> pd.DataFrame:
    """Distribute unit values to cells. ``pop_cell`` is indexed by h3_id (may be None)."""
    u = units.to_crs(epsg=epsg)[["geo_code", "geometry", *[c for c in extensive + intensive if c in units]]]
    g = grid.to_crs(epsg=epsg)[["h3_id", "geometry"]].assign(cell_area=lambda d: d.area)
    pieces = gpd.overlay(g, u, how="intersection", keep_geom_type=True)
    if pieces.empty:
        return pd.DataFrame({"h3_id": grid["h3_id"], "hcp_coverage": 0.0})
    pieces["area"] = pieces.area
    pieces["frac_of_cell"] = pieces["area"] / pieces["cell_area"]
    # population living in each piece ≈ cell population × share of the cell's area
    if pop_cell is not None and pop_cell.notna().any():
        pieces["w"] = pieces["h3_id"].map(pop_cell).fillna(0.0) * pieces["frac_of_cell"]
    else:
        pieces["w"] = 0.0
    tot_w = pieces.groupby("geo_code")["w"].transform("sum")
    tot_a = pieces.groupby("geo_code")["area"].transform("sum")
    pieces["share"] = np.where(tot_w > 0, pieces["w"] / tot_w.replace(0, np.nan), pieces["area"] / tot_a)

    out = pd.DataFrame({"h3_id": grid["h3_id"]}).set_index("h3_id")
    for col in extensive:
        if col in pieces:
            out[f"hcp_{col}"] = (pieces[col] * pieces["share"]).groupby(pieces["h3_id"]).sum(min_count=1)
    for col in intensive:
        if col in pieces:
            valid = pieces[col].notna()
            num = (pieces.loc[valid, col] * pieces.loc[valid, "area"]).groupby(pieces.loc[valid, "h3_id"]).sum()
            den = pieces.loc[valid, "area"].groupby(pieces.loc[valid, "h3_id"]).sum()
            out[f"hcp_{col}"] = num / den
    out["hcp_coverage"] = pieces.groupby("h3_id")["frac_of_cell"].sum().clip(upper=1.0)
    out["hcp_coverage"] = out["hcp_coverage"].fillna(0.0)
    return out.reset_index()


def run_units(lake: Lake | None = None) -> int:
    lake = lake or get_lake()
    units = build_units_table(hcp_ingest.latest_files(lake))
    log.info("hcp_units rows=%d cols=%d", len(units), units.shape[1])
    return lake.write("silver", "hcp_units", units)


def run_cells(city_slug: str, lake: Lake | None = None) -> int:
    lake = lake or get_lake()
    cfg = hcp_config()
    city = get_city(city_slug)
    units = lake.read("silver", "hcp_units")
    prefixes = (cfg.get("city_geo_code_prefixes") or {}).get(city_slug)
    if prefixes:
        units = units[units["geo_code"].map(lambda c: any(c.startswith(p) for p in prefixes))]
    admin = lake.read_geo("silver", "admin_units", filters=[("city", "=", city_slug)])
    grid = lake.read_geo("silver", "grid", filters=[("city", "=", city_slug)])
    matched = match_units(units, admin, (cfg.get("name_aliases") or {}).get(city_slug, {}))
    pop = None
    if lake.exists("silver", "gee_cells"):
        gee = lake.read("silver", "gee_cells", filters=[("city", "=", city_slug)], columns=["h3_id", "pop_cell"])
        pop = gee.set_index("h3_id")["pop_cell"]
    d = cfg["disaggregation"]
    cells = disaggregate(matched, grid, pop, d["extensive"], d["intensive"], city.crs_projected)
    cells["city"] = city_slug
    log.info("hcp_cells city=%s matched_units=%d/%d covered_cells=%.0f%%", city_slug, len(matched),
             len(admin), 100 * (cells["hcp_coverage"] > 0.5).mean())
    return lake.write("silver", "hcp_cells", cells, predicate=f"city = '{city_slug}'")
