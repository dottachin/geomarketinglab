"""Synthetic city generator: writes realistic-looking silver tables for 'testville' so the
feature → dbt → training → scoring path can be tested end to end without network access."""

from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon, box

from geomlab import geo
from geomlab.config import business_types, poi_taxonomy
from geomlab.features import grid as grid_mod
from geomlab.lakehouse import Lake

CITY = "testville"
LON0, LAT0 = -7.60, 33.57
SIZE_DEG = 0.05  # ~4.6 km x 5.5 km


def write_city(lake: Lake, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    boundary = gpd.GeoDataFrame({"city": [CITY]}, geometry=[box(LON0, LAT0, LON0 + SIZE_DEG, LAT0 + SIZE_DEG)],
                                crs=geo.WGS84)
    half = SIZE_DEG / 2
    admin = gpd.GeoDataFrame(
        {"city": CITY, "unit_name": ["Arrondissement Nord", "Arrondissement Sud"], "admin_level": [10.0, 10.0]},
        geometry=[box(LON0, LAT0 + half, LON0 + SIZE_DEG, LAT0 + SIZE_DEG), box(LON0, LAT0, LON0 + SIZE_DEG, LAT0 + half)],
        crs=geo.WGS84)
    from geomlab.config import City

    city = City(CITY, "Testville", None, None, 32629, 9)
    grid = grid_mod.build_grid(city, boundary, admin)
    pred = f"city = '{CITY}'"
    lake.write("silver", "city_boundary", boundary, predicate=pred)
    lake.write("silver", "admin_units", admin, predicate=pred)
    lake.write("silver", "grid", grid, predicate=pred)
    lake.write("silver", "h3_neighbors", geo.neighbor_pairs(grid["h3_id"], 2).assign(city=CITY), predicate=pred)

    # population: radial decay from the centre + noise
    cx, cy = LON0 + half, LAT0 + half
    r = np.hypot(grid["lon"] - cx, grid["lat"] - cy) / half
    pop = np.clip(2500 * np.exp(-2 * r) * rng.lognormal(0, 0.3, len(grid)), 0, None)
    gee = pd.DataFrame({
        "city": CITY, "h3_id": grid["h3_id"], "pop_cell": pop,
        "ndvi_mean": rng.uniform(0.05, 0.4, len(grid)), "ndbi_mean": rng.uniform(-0.1, 0.3, len(grid)),
        "ndwi_mean": rng.uniform(-0.3, 0.0, len(grid)), "builtup_ratio": np.clip(1 - r + rng.normal(0, .1, len(grid)), 0, 1),
        "trees_ratio": rng.uniform(0, .2, len(grid)), "grass_ratio": rng.uniform(0, .2, len(grid)),
        "water_ratio": np.zeros(len(grid)), "bare_ratio": rng.uniform(0, .2, len(grid)),
        "nightlight_mean": 40 * np.exp(-1.5 * r) + rng.normal(0, 1, len(grid)),
        "nightlight_trend": rng.normal(0.5, 0.2, len(grid)), "feature_date": "2026-10-01",
    })
    lake.write("silver", "gee_cells", gee, predicate=pred)

    # POIs: attractors scattered with density ~ population; businesses depend on attractors
    tax = poi_taxonomy()
    rows = []
    weights = pop / pop.sum()
    def sample_points(n):
        idx = rng.choice(len(grid), size=n, p=weights)
        return grid["lon"].to_numpy()[idx] + rng.normal(0, 4e-4, n), grid["lat"].to_numpy()[idx] + rng.normal(0, 4e-4, n)
    for cat in tax.attractor:
        lon, lat = sample_points(int(rng.integers(15, 60)))
        rows += [(f"n{len(rows)+i}", "attractor", cat, None, lo, la) for i, (lo, la) in enumerate(zip(lon, lat, strict=True))]
    for bt in business_types():
        lon, lat = sample_points(int(rng.integers(25, 70)))
        rows += [(f"n{len(rows)+i}", "business", bt, f"{bt} {i}", lo, la) for i, (lo, la) in enumerate(zip(lon, lat, strict=True))]
    poi = pd.DataFrame(rows, columns=["osm_id", "role", "category", "name", "lon", "lat"])
    poi["h3_id"] = geo.points_to_cells(poi["lon"].to_numpy(), poi["lat"].to_numpy(), 9)
    poi = gpd.GeoDataFrame(poi.assign(city=CITY), geometry=gpd.points_from_xy(poi.lon, poi.lat), crs=geo.WGS84)
    lake.write("silver", "poi", poi, predicate=pred)

    # roads: a lattice every ~0.002° with shared node ids at intersections
    step, n = 0.002, int(SIZE_DEG / 0.002) + 1
    xs = [LON0 + i * step for i in range(n)]
    ys = [LAT0 + j * step for j in range(n)]
    nid = lambda i, j: 1_000_000 + i * n + j  # noqa: E731
    roads = []
    for i, x in enumerate(xs):
        cls = "primary" if i == n // 2 else "local"
        roads.append((f"wv{i}", "primary" if cls == "primary" else "residential", cls,
                      [nid(i, j) for j in range(n)], LineString([(x, y) for y in ys])))
    for j, y in enumerate(ys):
        cls = "secondary" if j == n // 2 else "local"
        roads.append((f"wh{j}", "secondary" if cls == "secondary" else "residential", cls,
                      [nid(i, j) for i in range(n)], LineString([(x, y) for x in xs])))
    roads = gpd.GeoDataFrame(pd.DataFrame(roads, columns=["osm_id", "highway", "road_class", "node_refs", "geometry"]),
                             geometry="geometry", crs=geo.WGS84).assign(city=CITY)
    lake.write("silver", "roads", roads, predicate=pred)
    nodes = pd.DataFrame([(nid(i, j), 4, x, y) for i, x in enumerate(xs) for j, y in enumerate(ys)],
                         columns=["node_id", "street_count", "lon", "lat"])
    nodes["h3_id"] = geo.points_to_cells(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), 9)
    lake.write("silver", "road_nodes", nodes.assign(city=CITY), predicate=pred)

    lu = gpd.GeoDataFrame({"landuse_class": ["residential", "commercial", "green"], "city": CITY},
                          geometry=[box(LON0, LAT0, LON0 + half, LAT0 + half),
                                    box(LON0 + half, LAT0 + half, LON0 + SIZE_DEG, LAT0 + SIZE_DEG),
                                    Polygon([(LON0, LAT0 + half), (LON0 + half, LAT0 + half), (LON0, LAT0 + SIZE_DEG)])],
                          crs=geo.WGS84)
    lake.write("silver", "landuse", lu, predicate=pred)
    nb = rng.poisson(pop / 40)
    bld = pd.DataFrame({"city": CITY, "h3_id": np.repeat(grid["h3_id"].to_numpy(), nb),
                        "footprint_m2": rng.uniform(60, 400, int(nb.sum()))})
    lake.write("silver", "buildings", bld, predicate=pred)
    assert h3.is_valid_cell(grid["h3_id"].iloc[0])
