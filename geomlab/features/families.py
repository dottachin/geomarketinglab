"""Silver: spatial aggregation of every source to the H3 grid, one table per feature family
(doc §5.2). dbt then joins them, adds spatial context (k-ring averages) and labels in gold.

All distances/areas are computed in the city's projected CRS. Feature functions are pure
(DataFrames in → DataFrame keyed by h3_id out) so they are unit-testable.

  silver.cell_demand         pop_500m, pop_1km                       (WorldPop rings)
  silver.cell_economic       building_count, building_footprint_ratio (OSM buildings)
  silver.cell_attractors     att_<category>_<ring>m, att_total_<ring>m, osm_pois_1km
  silver.cell_accessibility  road_len_<class>_m, road_density_km_km2, intersection_density_km2,
                             dist_main_road_m, walk_pop_10min
  silver.cell_environment    landuse_entropy, landuse_share_<class>, dist_center_m
  silver.cell_competition    (long) business_type, n_in_cell, comp_<ring>m, comp_catchment,
                             nearest_comp_m, pop_catchment   ← NOT a model input (leakage)
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from geomlab import geo
from geomlab.config import business_types, get_city, poi_taxonomy
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger

log = get_logger(__name__)

WALK_SPEED_MPS = 4.8 / 3.6
WALK_MINUTES = 10
MAX_SNAP_M = 300.0
LANDUSE_CLASSES = ["residential", "commercial", "industrial", "green", "agricultural", "other"]
VEHICLE_CLASSES = ["primary", "secondary", "tertiary", "local"]


def centroids(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"h3_id": grid["h3_id"].to_numpy()},
                            geometry=gpd.points_from_xy(grid["lon"], grid["lat"]), crs=geo.WGS84)


def _points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=geo.WGS84)


# ----------------------------------------------------------------------------- demand
def demand_features(cents: gpd.GeoDataFrame, pop_cell: pd.Series, epsg: int) -> pd.DataFrame:
    pts = cents.assign(pop=cents["h3_id"].map(pop_cell).fillna(0.0).to_numpy())
    out = pd.DataFrame({"h3_id": cents["h3_id"]})
    for r in (500, 1000):
        out[f"pop_{r}m" if r < 1000 else "pop_1km"] = geo.count_within_m(cents, pts, r, epsg, weights=pts["pop"]).to_numpy()
    if pop_cell.isna().all():
        out[["pop_500m", "pop_1km"]] = np.nan
    return out


# ----------------------------------------------------------------------------- economic
def economic_features(grid: gpd.GeoDataFrame, buildings: pd.DataFrame) -> pd.DataFrame:
    agg = buildings.groupby("h3_id")["footprint_m2"].agg(["count", "sum"])
    out = pd.DataFrame({"h3_id": grid["h3_id"], "area_m2": grid["area_m2"]}).set_index("h3_id")
    out["building_count"] = agg["count"].reindex(out.index).fillna(0)
    out["building_footprint_ratio"] = (agg["sum"].reindex(out.index).fillna(0) / out["area_m2"]).clip(0, 1)
    return out.drop(columns="area_m2").reset_index()


# ----------------------------------------------------------------------------- attractors
def attractor_features(cents: gpd.GeoDataFrame, poi: pd.DataFrame, epsg: int) -> pd.DataFrame:
    tax = poi_taxonomy()
    pts = _points(poi)
    out = pd.DataFrame({"h3_id": cents["h3_id"]})
    att = pts[pts["role"] == "attractor"]
    for r in tax.rings_m:
        suffix = f"{r}m"
        for cat in tax.attractor:
            out[f"att_{cat}_{suffix}"] = geo.count_within_m(cents, att[att["category"] == cat], r, epsg).to_numpy()
        out[f"att_total_{suffix}"] = geo.count_within_m(cents, att, r, epsg).to_numpy()
    # coverage signal for the low-confidence flag (all mapped POIs, any role)
    out["osm_pois_1km"] = geo.count_within_m(cents, pts, 1000, epsg).to_numpy()
    return out


# ----------------------------------------------------------------------------- accessibility
def walking_reach_population(
    roads: gpd.GeoDataFrame, cents: gpd.GeoDataFrame, pop_cell: pd.Series, epsg: int,
    minutes: float = WALK_MINUTES, chunk: int = 256,
) -> pd.Series:
    """Residents reachable within ``minutes`` of walking on the OSM network from each cell
    centre (network isochrone, doc §5.2 "walking-time isochrones"). Index: h3_id."""
    limit_m = WALK_SPEED_MPS * 60 * minutes
    walk = roads[roads["highway"].map(lambda h: not str(h).startswith("motorway"))]
    if walk.empty or pop_cell.isna().all():
        return pd.Series(np.nan, index=cents["h3_id"])
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    node_index: dict[int, int] = {}
    xy: list[tuple[float, float]] = []
    rows, cols, lens = [], [], []
    for refs, line in zip(walk["node_refs"], walk.geometry, strict=True):
        coords = list(line.coords)
        if len(coords) != len(refs):
            continue
        px, py = tf.transform([c[0] for c in coords], [c[1] for c in coords])
        ids = []
        for ref, x, y in zip(refs, px, py, strict=True):
            if ref not in node_index:
                node_index[ref] = len(xy)
                xy.append((x, y))
            ids.append(node_index[ref])
        for a, b, x0, y0, x1, y1 in zip(ids[:-1], ids[1:], px[:-1], py[:-1], px[1:], py[1:], strict=True):
            d = float(np.hypot(x1 - x0, y1 - y0))
            rows += [a, b]
            cols += [b, a]
            lens += [d, d]
    n = len(xy)
    graph = sparse.csr_matrix((lens, (rows, cols)), shape=(n, n))
    xy_arr = np.asarray(xy)

    # node → cell membership matrix, and cell population vector
    cell_ids = cents["h3_id"].to_numpy()
    cell_pos = {c: i for i, c in enumerate(cell_ids)}
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(xy_arr[:, 0], xy_arr[:, 1])
    node_cells = geo.points_to_cells(np.asarray(lon), np.asarray(lat), _resolution(cell_ids[0]))
    member = [(i, cell_pos[c]) for i, c in enumerate(node_cells) if c in cell_pos]
    m = sparse.csr_matrix((np.ones(len(member)), ([a for a, _ in member], [b for _, b in member])),
                          shape=(n, len(cell_ids)))
    pop_vec = pd.Series(cell_ids).map(pop_cell).fillna(0.0).to_numpy()

    cx, cy = tf.transform(cents.geometry.x.to_numpy(), cents.geometry.y.to_numpy())
    dist, src = cKDTree(xy_arr).query(np.column_stack([cx, cy]))
    result = pop_vec.copy()  # fallback: only the cell itself
    ok = np.where(dist <= MAX_SNAP_M)[0]
    for start in range(0, len(ok), chunk):
        idx = ok[start : start + chunk]
        d = dijkstra(graph, directed=False, indices=src[idx], limit=limit_m)
        reach = sparse.csr_matrix(np.isfinite(d).astype(np.float32))
        cells_reached = (reach @ m) > 0
        result[idx] = cells_reached @ pop_vec
    return pd.Series(result, index=cell_ids)


def _resolution(cell: str) -> int:
    import h3

    return h3.get_resolution(cell)


def accessibility_features(
    grid: gpd.GeoDataFrame, cents: gpd.GeoDataFrame, roads: gpd.GeoDataFrame,
    nodes: pd.DataFrame, pop_cell: pd.Series, epsg: int,
) -> pd.DataFrame:
    out = pd.DataFrame({"h3_id": grid["h3_id"], "area_km2": grid["area_m2"] / 1e6}).set_index("h3_id")
    vehicle = roads[roads["road_class"].isin(VEHICLE_CLASSES)]
    if len(vehicle):
        pieces = gpd.overlay(vehicle[["road_class", "geometry"]].to_crs(epsg=epsg),
                             grid[["h3_id", "geometry"]].to_crs(epsg=epsg), how="intersection",
                             keep_geom_type=True)
        pieces["len"] = pieces.length
        lengths = pieces.pivot_table(index="h3_id", columns="road_class", values="len", aggfunc="sum")
    else:
        lengths = pd.DataFrame()
    for cls in VEHICLE_CLASSES:
        out[f"road_len_{cls}_m"] = lengths.get(cls, pd.Series(dtype=float)).reindex(out.index).fillna(0.0)
    total = out[[f"road_len_{c}_m" for c in VEHICLE_CLASSES]].sum(axis=1)
    out["road_density_km_km2"] = (total / 1000.0) / out["area_km2"]
    junctions = nodes[nodes["street_count"] >= 3].groupby("h3_id").size()
    out["intersection_density_km2"] = junctions.reindex(out.index).fillna(0) / out["area_km2"]
    main = roads[roads["road_class"].isin(["primary", "secondary"])]
    out["dist_main_road_m"] = geo.nearest_distance_m(cents, main, epsg).to_numpy()
    out["walk_pop_10min"] = walking_reach_population(roads, cents, pop_cell, epsg).reindex(out.index).to_numpy()
    return out.drop(columns="area_km2").reset_index()


# ----------------------------------------------------------------------------- environment
def environment_features(grid: gpd.GeoDataFrame, cents: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame,
                         pop_cell: pd.Series, epsg: int) -> pd.DataFrame:
    out = pd.DataFrame({"h3_id": grid["h3_id"]}).set_index("h3_id")
    if len(landuse):
        pieces = gpd.overlay(grid[["h3_id", "geometry"]].to_crs(epsg=epsg),
                             landuse[["landuse_class", "geometry"]].to_crs(epsg=epsg), how="intersection",
                             keep_geom_type=True)
        pieces["a"] = pieces.area
        area = pieces.pivot_table(index="h3_id", columns="landuse_class", values="a", aggfunc="sum")
        area = area.reindex(columns=LANDUSE_CLASSES).fillna(0.0)
    else:
        area = pd.DataFrame(0.0, index=out.index, columns=LANDUSE_CLASSES)
    area = area.reindex(out.index).fillna(0.0)
    shares = area.div(area.sum(axis=1).replace(0, np.nan), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(shares * np.log(shares.where(shares > 0))).sum(axis=1, min_count=1)
    out["landuse_entropy"] = (ent / np.log(len(LANDUSE_CLASSES))).fillna(0.0)
    for cls in ("residential", "commercial", "green"):
        out[f"landuse_share_{cls}"] = shares[cls].fillna(0.0)
    # urban vs peri-urban: distance to the population-weighted centre (boundary centroid fallback)
    proj = cents.to_crs(epsg=epsg)
    w = cents["h3_id"].map(pop_cell).fillna(0.0).to_numpy()
    if w.sum() > 0:
        cx, cy = np.average(proj.geometry.x, weights=w), np.average(proj.geometry.y, weights=w)
    else:
        cx, cy = proj.geometry.x.mean(), proj.geometry.y.mean()
    out["dist_center_m"] = np.hypot(proj.geometry.x.to_numpy() - cx, proj.geometry.y.to_numpy() - cy)
    return out.reset_index()


# ----------------------------------------------------------------------------- competition
def competition_table(cents: gpd.GeoDataFrame, poi: pd.DataFrame, pop_cell: pd.Series, epsg: int) -> pd.DataFrame:
    pts = _points(poi[poi["role"] == "business"])
    pop_pts = cents.assign(pop=cents["h3_id"].map(pop_cell).fillna(0.0).to_numpy())
    rings = poi_taxonomy().rings_m
    frames = []
    for slug, bt in business_types().items():
        comp = pts[pts["category"] == slug]
        f = pd.DataFrame({"h3_id": cents["h3_id"], "business_type": slug})
        f["n_in_cell"] = f["h3_id"].map(comp.groupby("h3_id").size()).fillna(0).astype("int64").to_numpy()
        for r in rings:
            f[f"comp_{r}m"] = geo.count_within_m(cents, comp, r, epsg).to_numpy()
        f["comp_catchment"] = geo.count_within_m(cents, comp, bt.catchment_m, epsg).to_numpy()
        f["nearest_comp_m"] = geo.nearest_distance_m(cents, comp, epsg).to_numpy()
        f["pop_catchment"] = geo.count_within_m(cents, pop_pts, bt.catchment_m, epsg, weights=pop_pts["pop"]).to_numpy()
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- orchestration
def run(city_slug: str, lake: Lake | None = None) -> dict[str, int]:
    lake = lake or get_lake()
    epsg = get_city(city_slug).crs_projected
    flt = [("city", "=", city_slug)]
    grid = lake.read_geo("silver", "grid", filters=flt)
    cents = centroids(grid)
    poi = lake.read("silver", "poi", filters=flt)
    roads = lake.read_geo("silver", "roads", filters=flt)
    nodes = lake.read("silver", "road_nodes", filters=flt)
    landuse = lake.read_geo("silver", "landuse", filters=flt)
    buildings = lake.read("silver", "buildings", filters=flt)
    if lake.exists("silver", "gee_cells"):
        gee = lake.read("silver", "gee_cells", filters=flt, columns=["h3_id", "pop_cell"])
        pop_cell = gee.set_index("h3_id")["pop_cell"]
    else:
        pop_cell = pd.Series(np.nan, index=grid["h3_id"])

    tables = {
        "cell_demand": demand_features(cents, pop_cell, epsg),
        "cell_economic": economic_features(grid, buildings),
        "cell_attractors": attractor_features(cents, poi, epsg),
        "cell_accessibility": accessibility_features(grid, cents, roads, nodes, pop_cell, epsg),
        "cell_environment": environment_features(grid, cents, landuse, pop_cell, epsg),
        "cell_competition": competition_table(cents, poi, pop_cell, epsg),
    }
    versions = {}
    for name, df in tables.items():
        df["city"] = city_slug
        versions[name] = lake.write("silver", name, df, predicate=f"city = '{city_slug}'")
    log.info("families city=%s cells=%d", city_slug, len(grid))
    return versions
