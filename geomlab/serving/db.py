"""PostGIS/pgvector access: engines per role, schema bootstrap, and bulk loaders.

Roles (doc §9 "read-only DB roles for the LLM"):
  owner        DATABASE_URL     – pipelines (scoring, indexing, db-init)
  geomlab_ro   DATABASE_URL_RO  – the tool layer (SELECT on serving, rag, gold, analytics)
  geomlab_sql  DATABASE_URL_SQL – query_stats only (SELECT on analytics views, nothing else)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
from sqlalchemy import Engine, Text, create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY

from geomlab.logging import get_logger
from geomlab.settings import get_settings

log = get_logger(__name__)
SCHEMA_SQL = Path(__file__).with_name("schema.sql")


@lru_cache(maxsize=4)
def engine(role: str = "owner") -> Engine:
    s = get_settings()
    url = {"owner": s.database_url, "ro": s.database_url_ro, "sql": s.database_url_sql}[role]
    connect_args = {"options": "-c statement_timeout=5000"} if role in ("ro", "sql") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def init_schema() -> None:
    sql = SCHEMA_SQL.read_text().replace("{embedding_dim}", str(get_settings().embedding_dim))
    with engine().begin() as con:
        con.exec_driver_sql(sql)
        _grant(con)
    log.info("db_schema_ready")


def _grant(con) -> None:
    roles = {r for (r,) in con.execute(text("select rolname from pg_roles where rolname in ('geomlab_ro','geomlab_sql')"))}
    if "geomlab_ro" in roles:
        for schema in ("serving", "rag", "gold", "analytics"):
            con.exec_driver_sql(f"GRANT USAGE ON SCHEMA {schema} TO geomlab_ro")
            con.exec_driver_sql(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO geomlab_ro")
            con.exec_driver_sql(f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT ON TABLES TO geomlab_ro")
    if "geomlab_sql" in roles:
        con.exec_driver_sql("GRANT USAGE ON SCHEMA analytics TO geomlab_sql")
        con.exec_driver_sql("GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO geomlab_sql")
        # views run with the owner's rights, so geomlab_sql needs no access to serving.*
    else:
        log.warning("role geomlab_sql missing; query_stats will be unavailable (see infra/postgres/init)")


def _copy_geo(con, gdf: gpd.GeoDataFrame, staging: str, columns: list[str]) -> None:
    df = pd.DataFrame(gdf[columns])
    df["wkb"] = gdf.geometry.to_wkb(hex=True)
    df.to_sql(staging, con, schema="serving", if_exists="replace", index=False, method="multi", chunksize=2000)


def load_cells(grid: gpd.GeoDataFrame) -> None:
    with engine().begin() as con:
        _copy_geo(con, grid, "_stg_cells", ["h3_id", "city", "district", "area_m2", "lon", "lat"])
        con.exec_driver_sql("""
            INSERT INTO serving.cells (h3_id, city, district, area_m2, geom, centroid)
            SELECT h3_id, city, district, area_m2, ST_SetSRID(ST_GeomFromWKB(decode(wkb, 'hex')), 4326),
                   ST_SetSRID(ST_MakePoint(lon, lat), 4326)
            FROM serving._stg_cells
            ON CONFLICT (h3_id) DO UPDATE SET district = EXCLUDED.district, area_m2 = EXCLUDED.area_m2,
                geom = EXCLUDED.geom, centroid = EXCLUDED.centroid;
            DROP TABLE serving._stg_cells;""")


def load_pois(poi: gpd.GeoDataFrame, city: str) -> None:
    with engine().begin() as con:
        con.execute(text("DELETE FROM serving.pois WHERE city = :c"), {"c": city})
        _copy_geo(con, poi, "_stg_pois", ["city", "osm_id", "role", "category", "name", "h3_id"])
        con.exec_driver_sql("""
            INSERT INTO serving.pois (city, osm_id, role, category, name, h3_id, geom)
            SELECT city, osm_id, role, category, name, h3_id, ST_SetSRID(ST_GeomFromWKB(decode(wkb, 'hex')), 4326)
            FROM serving._stg_pois ON CONFLICT DO NOTHING;
            DROP TABLE serving._stg_pois;""")


def load_gold_features(features: pd.DataFrame) -> None:
    with engine().begin() as con:
        features.to_sql("cell_features", con, schema="gold", if_exists="replace", index=False,
                        method="multi", chunksize=1000)
        con.exec_driver_sql("CREATE INDEX IF NOT EXISTS cell_features_h3_idx ON gold.cell_features (h3_id)")
        _grant(con)


def replace_scores(scores: pd.DataFrame, drivers: pd.DataFrame, model_version: str, active: dict) -> None:
    """Atomically swap in a new scoring run and mark its model version active."""
    with engine().begin() as con:
        con.execute(text("DELETE FROM serving.scores WHERE model_version = :v"), {"v": model_version})
        con.execute(text("DELETE FROM serving.shap_drivers WHERE model_version = :v"), {"v": model_version})
        scores.to_sql("scores", con, schema="serving", if_exists="append", index=False, method="multi", chunksize=2000,
                      dtype={"confidence_reasons": ARRAY(Text)})
        drivers.to_sql("shap_drivers", con, schema="serving", if_exists="append", index=False, method="multi",
                       chunksize=5000)
        con.execute(text("""
            INSERT INTO serving.active_model (id, model_version, family, data_date, metrics, activated_at)
            VALUES (1, :model_version, :family, :data_date, CAST(:metrics AS jsonb), now())
            ON CONFLICT (id) DO UPDATE SET model_version = EXCLUDED.model_version, family = EXCLUDED.family,
                data_date = EXCLUDED.data_date, metrics = EXCLUDED.metrics, activated_at = now()"""), active)
        # keep only the active and the previous model version (reproducibility of recent answers)
        con.execute(text("""
            DELETE FROM serving.scores WHERE model_version NOT IN (
              SELECT model_version FROM (SELECT model_version, max(scored_at) m FROM serving.scores
                                         GROUP BY model_version ORDER BY m DESC LIMIT 2) k)"""))
        con.execute(text("""
            DELETE FROM serving.shap_drivers d WHERE NOT EXISTS (
              SELECT 1 FROM serving.scores s WHERE s.model_version = d.model_version)"""))
