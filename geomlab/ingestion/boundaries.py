"""City boundaries and administrative units (needed by the H3 grid and the HCP join).

Fetched from OpenStreetMap (Nominatim geocoding + Overpass via OSMnx — "Overpass API for
targeted queries", doc §3.1) and stored raw in bronze as GeoJSON. Boundaries change
rarely, so the change token is the content hash: unchanged downloads are not re-stored.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from geomlab.config import City, cities
from geomlab.geo import WGS84
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger

log = get_logger(__name__)
SOURCE = "boundaries"


def _osmnx():
    import osmnx as ox

    from geomlab.settings import get_settings

    ox.settings.cache_folder = str(get_settings().data_dir / "cache" / "osmnx")  # not the cwd
    return ox


def _city_boundary(city: City) -> gpd.GeoDataFrame:
    if city.bbox:
        s, w, n, e = city.bbox
        return gpd.GeoDataFrame({"city": [city.slug]}, geometry=[box(w, s, e, n)], crs=WGS84)
    ox = _osmnx()
    gdf = ox.geocode_to_gdf(city.osm_query)[["geometry"]].dissolve().reset_index(drop=True)
    gdf["city"] = city.slug
    return gdf.to_crs(WGS84)


def _admin_units(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ox = _osmnx()
    admin = ox.features_from_polygon(boundary.geometry.iloc[0], tags={"boundary": "administrative"})
    admin = admin[admin.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index()
    keep = [c for c in ("name", "name:fr", "admin_level", "ref") if c in admin.columns]
    admin = admin[keep + ["geometry"]].copy()
    admin["admin_level"] = pd.to_numeric(admin.get("admin_level"), errors="coerce")
    return admin.to_crs(WGS84)


def ingest(city_slugs: list[str] | None = None, lake: Lake | None = None) -> dict[str, str]:
    """Download boundary + admin units per city into bronze; return {city: status}."""
    lake = lake or get_lake()
    status = {}
    for slug in city_slugs or list(cities()):
        city = cities()[slug]
        boundary = _city_boundary(city)
        changed = False
        for name, gdf in (("boundary", boundary), ("admin", _admin_units(boundary))):
            payload = gdf.to_json().encode()
            fname = f"{slug}_{name}.geojson"
            import hashlib

            token = hashlib.sha256(payload).hexdigest()
            if lake.last_token(SOURCE, prefix=fname) == token:
                continue
            lake.put_raw(SOURCE, fname, payload, change_token=token)
            changed = True
        status[slug] = "changed" if changed else "unchanged"
        log.info("boundaries city=%s status=%s", slug, status[slug])
    return status


def load_latest(city_slug: str, kind: str, lake: Lake | None = None) -> gpd.GeoDataFrame:
    """Latest bronze GeoJSON for (city, kind in {boundary, admin})."""
    lake = lake or get_lake()
    rows = lake.manifest(SOURCE)
    rows = rows[rows["object_key"].str.endswith(f"/{city_slug}_{kind}.geojson")]
    if rows.empty:
        raise FileNotFoundError(f"no bronze {kind} for {city_slug}; run the boundaries ingestion")
    data = lake.get_raw(rows.iloc[-1]["object_key"])
    import io

    return gpd.read_file(io.BytesIO(data)).set_crs(WGS84, allow_override=True)
