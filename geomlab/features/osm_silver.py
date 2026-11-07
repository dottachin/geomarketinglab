"""Silver: OSM layers per city, parsed from the Geofabrik working copy with pyosmium.

One pass over the PBF produces, for every configured city (filtered by bbox, then by
boundary polygon):

  silver.poi         city, osm_id, role, category, name, lon, lat, h3_id
                     (categories from config/poi_categories.yaml, normalised, deduplicated)
  silver.roads       city, osm_id, highway, road_class, node_refs, geometry (LineString)
  silver.road_nodes  city, node_id, street_count, lon, lat, h3_id  (street_count ≥ 3 → junction)
  silver.landuse     city, landuse_class, geometry (Polygon)
  silver.buildings   city, h3_id, footprint_m2  (one row per building, geometry dropped)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon

from geomlab import geo
from geomlab.config import PoiTaxonomy, cities, poi_taxonomy
from geomlab.ingestion import osm as osm_ingest
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger

log = get_logger(__name__)

ROAD_CLASS = {
    **dict.fromkeys(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"], "primary"),
    **dict.fromkeys(["secondary", "secondary_link"], "secondary"),
    **dict.fromkeys(["tertiary", "tertiary_link"], "tertiary"),
    **dict.fromkeys(["residential", "living_street", "unclassified", "service"], "local"),
    **dict.fromkeys(["pedestrian", "footway"], "pedestrian"),
}
LANDUSE_CLASS = {
    "residential": "residential", "commercial": "commercial", "retail": "commercial",
    "industrial": "industrial", "park": "green", "grass": "green", "forest": "green",
    "recreation_ground": "green", "cemetery": "other", "construction": "other",
    "religious": "other", "education": "other", "military": "other", "farmland": "agricultural",
}


@dataclass
class _Acc:
    pois: list = field(default_factory=list)
    roads: list = field(default_factory=list)
    landuse: list = field(default_factory=list)
    buildings: list = field(default_factory=list)
    street_count: Counter = field(default_factory=Counter)
    node_xy: dict = field(default_factory=dict)


def _make_handler(bboxes: dict[str, tuple[float, float, float, float]], taxonomy: PoiTaxonomy):
    import osmium

    keys = taxonomy.keys | {"highway", "building", "landuse", "leisure"}

    def city_of(lon: float, lat: float) -> str | None:
        for slug, (w, s, e, n) in bboxes.items():
            if w <= lon <= e and s <= lat <= n:
                return slug
        return None

    class Handler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.acc = {slug: _Acc() for slug in bboxes}

        def node(self, n):
            if not n.tags or not any(k in n.tags for k in taxonomy.keys):
                return
            loc = n.location
            if not loc.valid():
                return
            slug = city_of(loc.lon, loc.lat)
            if slug is None:
                return
            tags = {t.k: t.v for t in n.tags}
            hit = taxonomy.classify(tags)
            if hit:
                self.acc[slug].pois.append((f"n{n.id}", *hit, tags.get("name"), loc.lon, loc.lat))

        def way(self, w):
            if not any(k in w.tags for k in keys):
                return
            coords, refs = [], []
            for nd in w.nodes:
                if nd.location.valid():
                    coords.append((nd.lon, nd.lat))
                    refs.append(nd.ref)
            if len(coords) < 2:
                return
            slug = city_of(*coords[0])
            if slug is None:
                return
            acc = self.acc[slug]
            tags = {t.k: t.v for t in w.tags}
            closed = refs[0] == refs[-1] and len(coords) >= 4

            hw = tags.get("highway")
            if hw in ROAD_CLASS:
                acc.roads.append((f"w{w.id}", hw, ROAD_CLASS[hw], refs, LineString(coords).wkb))
                if ROAD_CLASS[hw] != "pedestrian":
                    # incident street segments: endpoints +1, interior nodes +2 (OSMnx street_count)
                    acc.street_count[refs[0]] += 1
                    acc.street_count[refs[-1]] += 1
                    for r in refs[1:-1]:
                        acc.street_count[r] += 2
                    for r, xy in zip(refs, coords, strict=True):
                        acc.node_xy[r] = xy
            if closed and "building" in tags and tags["building"] != "no":
                acc.buildings.append(Polygon(coords).wkb)
            lu = LANDUSE_CLASS.get(tags.get("landuse") or tags.get("leisure") or "")
            if closed and lu:
                acc.landuse.append((lu, Polygon(coords).wkb))
            hit = taxonomy.classify(tags)
            if hit:
                arr = np.asarray(coords)
                acc.pois.append((f"w{w.id}", *hit, tags.get("name"), float(arr[:, 0].mean()), float(arr[:, 1].mean())))

    return Handler()


def _within(gdf: gpd.GeoDataFrame, boundary) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    return gdf[gdf.intersects(boundary)]


def run(city_slugs: list[str] | None = None, lake: Lake | None = None) -> dict[str, int]:
    from shapely import from_wkb

    lake = lake or get_lake()
    city_slugs = city_slugs or list(cities())
    pbf = osm_ingest.ensure_working_copy(lake)
    bounds = lake.read_geo("silver", "city_boundary")
    bboxes = {}
    boundary_geom = {}
    for slug in city_slugs:
        g = bounds[bounds["city"] == slug].geometry.union_all()
        if g.is_empty:
            raise FileNotFoundError(f"no silver.city_boundary for {slug}; run features.grid first")
        boundary_geom[slug] = g
        bboxes[slug] = g.bounds  # (minx, miny, maxx, maxy) = (w, s, e, n)

    handler = _make_handler(bboxes, poi_taxonomy())
    handler.apply_file(str(pbf), locations=True, idx="flex_mem")

    counts = {}
    for slug in city_slugs:
        acc, city, b = handler.acc[slug], cities()[slug], boundary_geom[slug]
        pred = f"city = '{slug}'"

        poi = pd.DataFrame(acc.pois, columns=["osm_id", "role", "category", "name", "lon", "lat"])
        poi = gpd.GeoDataFrame(poi, geometry=gpd.points_from_xy(poi.lon, poi.lat), crs=geo.WGS84)
        poi = _within(poi, b).drop_duplicates("osm_id")
        # Same shop mapped as node + building outline: drop same-category duplicates < 15 m apart.
        poi = _dedupe_near(poi, city.crs_projected, 15.0)
        poi["h3_id"] = geo.points_to_cells(poi["lon"].to_numpy(), poi["lat"].to_numpy(), city.h3_resolution)
        poi["city"] = slug
        lake.write("silver", "poi", poi, predicate=pred)

        roads = pd.DataFrame(acc.roads, columns=["osm_id", "highway", "road_class", "node_refs", "wkb"])
        roads = gpd.GeoDataFrame(roads.drop(columns="wkb"), geometry=from_wkb(roads["wkb"].to_numpy()), crs=geo.WGS84)
        roads = _within(roads, b).assign(city=slug)
        lake.write("silver", "roads", roads, predicate=pred)

        nodes = pd.DataFrame(
            [(r, c, *acc.node_xy[r]) for r, c in acc.street_count.items() if r in acc.node_xy],
            columns=["node_id", "street_count", "lon", "lat"],
        )
        nodes["h3_id"] = geo.points_to_cells(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), city.h3_resolution)
        nodes["city"] = slug
        lake.write("silver", "road_nodes", nodes, predicate=pred)

        lu = pd.DataFrame(acc.landuse, columns=["landuse_class", "wkb"])
        lu = gpd.GeoDataFrame(lu.drop(columns="wkb"), geometry=from_wkb(lu["wkb"].to_numpy()), crs=geo.WGS84)
        lu = _within(lu[lu.is_valid], b).assign(city=slug)
        lake.write("silver", "landuse", lu, predicate=pred)

        bld = gpd.GeoSeries(from_wkb(np.asarray(acc.buildings, dtype=object)), crs=geo.WGS84)
        bld = bld[bld.is_valid]
        cent = bld.centroid
        bdf = pd.DataFrame({
            "city": slug,
            "h3_id": [h3.latlng_to_cell(p.y, p.x, city.h3_resolution) for p in cent],
            "footprint_m2": bld.to_crs(epsg=city.crs_projected).area.to_numpy(),
        })
        lake.write("silver", "buildings", bdf, predicate=pred)

        counts[slug] = len(poi)
        log.info("osm_silver city=%s pois=%d roads=%d junctions=%d landuse=%d buildings=%d", slug,
                 len(poi), len(roads), int((nodes.street_count >= 3).sum()), len(lu), len(bdf))
    return counts


def _dedupe_near(poi: gpd.GeoDataFrame, epsg: int, tol_m: float) -> gpd.GeoDataFrame:
    if poi.empty:
        return poi
    proj = poi.to_crs(epsg=epsg)
    keep = np.ones(len(proj), dtype=bool)
    for _, grp in proj.groupby("category"):
        if len(grp) < 2:
            continue
        pairs = gpd.sjoin(grp[["geometry"]], grp[["geometry"]].assign(geometry=grp.buffer(tol_m)),
                          predicate="within")
        pairs = pairs[pairs.index < pairs["index_right"]]
        drop = proj.index.get_indexer(pairs["index_right"].unique())
        keep[drop] = False
    return poi[keep]


def load(table: str, city_slug: str, lake: Lake | None = None, geo_table: bool = False):
    lake = lake or get_lake()
    if geo_table:
        return lake.read_geo("silver", table, filters=[("city", "=", city_slug)])
    return lake.read("silver", table, filters=[("city", "=", city_slug)])
