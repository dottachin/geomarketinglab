"""Silver: city boundary, admin units, H3 grid and H3 neighbour table (doc §3.2).

Ported from pfe-old ``src/data/build_grid.py`` (clip to boundary, drop slivers, district
assignment), now reading the boundaries from bronze instead of calling OSMnx inline.

Tables (all keyed by ``city``, rewritten per city with a predicate so tasks are idempotent):
  silver.city_boundary   city, geometry
  silver.admin_units     city, unit_name, admin_level, geometry
  silver.grid            city, h3_id, district, area_m2, lon, lat, geometry (clipped hexagon)
  silver.h3_neighbors    city, h3_id, neighbor_id, distance (1..2)
"""

from __future__ import annotations

import re

import geopandas as gpd
import h3
import pandas as pd

from geomlab import geo
from geomlab.config import City, get_city
from geomlab.ingestion import boundaries
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger

log = get_logger(__name__)
MIN_AREA_FRACTION = 0.05  # drop boundary slivers smaller than 5 % of a full cell
NEIGHBOR_K = 2

_ARABIC_TIFINAGH = re.compile(r"[؀-ۿⴰ-⵿]+")


def clean_admin_name(raw: str | None) -> str | None:
    """Keep the French part of multilingual OSM names ("Arrondissement d'Anfa مقاطعة أنفا")."""
    if not raw or not isinstance(raw, str):
        return None
    parts = [p.strip() for p in _ARABIC_TIFINAGH.split(raw) if p.strip()]
    return parts[0] if parts else raw.strip()


def pick_admin_level(admin: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Most granular level in 8..10 (arrondissements/communes), avoiding quartiers (11+)."""
    levels = sorted((lv for lv in admin["admin_level"].dropna().unique() if 8 <= lv <= 10), reverse=True)
    if not levels:
        all_levels = sorted(admin["admin_level"].dropna().unique())
        if not all_levels:
            return admin.iloc[0:0]
        levels = [all_levels[-2] if len(all_levels) >= 2 else all_levels[-1]]
    return admin[admin["admin_level"] == levels[0]]


def build_grid(city: City, boundary: gpd.GeoDataFrame, admin: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = boundary.geometry.union_all()
    cells = geo.polygon_to_cells(geom, city.h3_resolution)
    if not cells:
        raise RuntimeError(f"no H3 cells for {city.slug}; check boundary / resolution")
    grid = geo.cells_to_gdf(cells)
    full_area = geo.area_m2(grid, city.crs_projected)
    grid["geometry"] = grid.geometry.intersection(geom)
    grid = grid[~grid.geometry.is_empty].copy()
    clipped = geo.area_m2(grid, city.crs_projected)
    keep = clipped >= full_area.loc[clipped.index] * MIN_AREA_FRACTION
    grid = grid[keep].copy()
    grid["area_m2"] = clipped[keep].to_numpy()
    latlng = [h3.cell_to_latlng(c) for c in grid["h3_id"]]
    grid["lat"] = [p[0] for p in latlng]
    grid["lon"] = [p[1] for p in latlng]
    grid["city"] = city.slug

    # district = admin unit containing the cell centre (best effort; used for CV and UI)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid["lon"], grid["lat"]), index=grid.index, crs=geo.WGS84)
    if len(admin):
        j = gpd.sjoin(pts, admin[["unit_name", "geometry"]], how="left", predicate="within")
        grid["district"] = j[~j.index.duplicated()]["unit_name"].reindex(grid.index)
    else:
        grid["district"] = None
    return grid[["city", "h3_id", "district", "area_m2", "lon", "lat", "geometry"]].reset_index(drop=True)


def run(city_slug: str, lake: Lake | None = None) -> int:
    lake = lake or get_lake()
    city = get_city(city_slug)
    boundary = boundaries.load_latest(city_slug, "boundary", lake)
    raw_admin = boundaries.load_latest(city_slug, "admin", lake)

    admin = pick_admin_level(raw_admin.assign(admin_level=pd.to_numeric(raw_admin["admin_level"], errors="coerce")))
    name_col = "name:fr" if "name:fr" in admin.columns and admin["name:fr"].notna().any() else "name"
    admin = gpd.GeoDataFrame(
        {"city": city_slug, "unit_name": admin[name_col].map(clean_admin_name),
         "admin_level": admin["admin_level"].astype("float64")},
        geometry=admin.geometry.values, crs=geo.WGS84,
    ).dropna(subset=["unit_name"])

    grid = build_grid(city, boundary, admin)
    neighbors = geo.neighbor_pairs(grid["h3_id"], NEIGHBOR_K).assign(city=city_slug)

    pred = f"city = '{city_slug}'"
    lake.write("silver", "city_boundary", boundary[["city", "geometry"]], predicate=pred)
    lake.write("silver", "admin_units", admin, predicate=pred)
    lake.write("silver", "h3_neighbors", neighbors, predicate=pred)
    version = lake.write("silver", "grid", grid, predicate=pred)
    log.info("grid city=%s cells=%d admin_units=%d", city_slug, len(grid), len(admin))
    return version


def load_grid(city_slug: str, lake: Lake | None = None) -> gpd.GeoDataFrame:
    lake = lake or get_lake()
    return lake.read_geo("silver", "grid", filters=[("city", "=", city_slug)])
