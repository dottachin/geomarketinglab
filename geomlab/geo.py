"""CRS discipline and H3 helpers (ported from pfe-old ``src/utils/geo.py``).

Rule: geometries are STORED in EPSG:4326; every distance/area is computed in the city's
projected CRS (doc §4.2). Stage code calls these helpers and never measures in degrees.
H3 functions use the h3 v4 API.
"""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

WGS84 = "EPSG:4326"


# --- metric operations ------------------------------------------------------------
def area_m2(gdf: gpd.GeoDataFrame | gpd.GeoSeries, epsg: int) -> pd.Series:
    return gdf.to_crs(epsg=epsg).area


def nearest_distance_m(
    sources: gpd.GeoDataFrame, targets: gpd.GeoDataFrame, epsg: int, max_m: float | None = None
) -> pd.Series:
    """Distance (m) from each source to its nearest target; NaN when there is none."""
    if len(targets) == 0 or len(sources) == 0:
        return pd.Series(np.nan, index=sources.index, dtype="float64")
    src = sources.to_crs(epsg=epsg)[["geometry"]]
    tgt = targets.to_crs(epsg=epsg)[["geometry"]]
    joined = gpd.sjoin_nearest(src, tgt, how="left", distance_col="_d", max_distance=max_m)
    return joined.groupby(level=0)["_d"].min().reindex(sources.index)


def count_within_m(
    centers: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    radius_m: float,
    epsg: int,
    weights: pd.Series | None = None,
) -> pd.Series:
    """(Weighted) number of ``points`` within ``radius_m`` of each center."""
    if len(points) == 0 or len(centers) == 0:
        return pd.Series(0.0, index=centers.index)
    c = centers.to_crs(epsg=epsg)[["geometry"]].copy()
    c["geometry"] = c.geometry.buffer(radius_m)
    p = points.to_crs(epsg=epsg)[["geometry"]].copy()
    p["_w"] = 1.0 if weights is None else weights.to_numpy(dtype="float64")
    joined = gpd.sjoin(c, p, how="inner", predicate="intersects")
    return joined.groupby(level=0)["_w"].sum().reindex(centers.index).fillna(0.0)


# --- H3 --------------------------------------------------------------------------
def _to_latlngpoly(poly: Polygon) -> h3.LatLngPoly:
    outer = [(lat, lon) for lon, lat in poly.exterior.coords]
    holes = [[(lat, lon) for lon, lat in ring.coords] for ring in poly.interiors]
    return h3.LatLngPoly(outer, *holes)


def polygon_to_cells(geometry: BaseGeometry, resolution: int) -> set[str]:
    """H3 cells whose centre lies in ``geometry`` (EPSG:4326)."""
    polys: Iterable[Polygon]
    if isinstance(geometry, MultiPolygon):
        polys = geometry.geoms
    elif isinstance(geometry, Polygon):
        polys = [geometry]
    else:
        polys = [geometry.convex_hull]
    cells: set[str] = set()
    for poly in polys:
        if not poly.is_empty:
            cells.update(h3.polygon_to_cells(_to_latlngpoly(poly), resolution))
    return cells


def cell_polygon(cell: str) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])


def cell_centroid(cell: str) -> Point:
    lat, lon = h3.cell_to_latlng(cell)
    return Point(lon, lat)


def cells_to_gdf(cells: Iterable[str], geometry: str = "polygon") -> gpd.GeoDataFrame:
    """GeoDataFrame of H3 cells with ``h3_id`` + polygon (or centroid) geometry."""
    cells = list(cells)
    make = cell_polygon if geometry == "polygon" else cell_centroid
    return gpd.GeoDataFrame({"h3_id": cells}, geometry=[make(c) for c in cells], crs=WGS84)


def points_to_cells(lon: np.ndarray, lat: np.ndarray, resolution: int) -> list[str]:
    return [h3.latlng_to_cell(float(la), float(lo), resolution) for lo, la in zip(lon, lat, strict=True)]


def neighbor_pairs(cells: Iterable[str], k: int) -> pd.DataFrame:
    """Rows ``(h3_id, neighbor_id, distance)`` for every neighbour within k rings that is
    itself in ``cells`` (distance 1..k). Used for spatial context features and the GNN graph."""
    cell_set = set(cells)
    rows = []
    for c in cell_set:
        for n in h3.grid_disk(c, k):
            if n != c and n in cell_set:
                rows.append((c, n, h3.grid_distance(c, n)))
    return pd.DataFrame(rows, columns=["h3_id", "neighbor_id", "distance"])


def parent_block(cell: str, resolution: int) -> str:
    return h3.cell_to_parent(cell, resolution)
