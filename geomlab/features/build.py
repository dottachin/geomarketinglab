"""Feature pipeline orchestration: silver builders → dbt (gold) → quality checks → publish.

Every function is idempotent per city and is what the Airflow ``build_features`` DAG calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from geomlab.config import hcp_config, model_config
from geomlab.features import dictionary, families, gee_silver, hcp_silver
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger
from geomlab.settings import get_settings

log = get_logger(__name__)
GOLD_TABLES = {"gold_cell_features": "cell_features", "gold_training_set": "training_set"}


# ----------------------------------------------------------------------------- placeholders
def ensure_gee(city: str, lake: Lake) -> None:
    """If GEE was never extracted for ``city`` keep the pipeline running with explicit nulls
    (surfaced as ``has_gee = false`` → low-confidence flag)."""
    if lake.exists("silver", "gee_cells") and len(
        lake.read("silver", "gee_cells", filters=[("city", "=", city)], columns=["h3_id"])):
        return
    cells = lake.read("silver", "grid", filters=[("city", "=", city)], columns=["h3_id"])
    cols = ["pop_cell", *gee_silver.INDEX_COLS, *gee_silver.RATIO_COLS, "nightlight_mean", "nightlight_trend"]
    df = pd.DataFrame({"city": city, "h3_id": cells["h3_id"], **dict.fromkeys(cols, np.nan), "feature_date": None})
    df[cols] = df[cols].astype("float64")
    df["feature_date"] = df["feature_date"].astype("string")
    lake.write("silver", "gee_cells", df, predicate=f"city = '{city}'")
    log.warning("gee_placeholder city=%s (no Earth Engine extract)", city)


def ensure_hcp(city: str, lake: Lake) -> None:
    d = hcp_config()["disaggregation"]
    wanted = [f"hcp_{c}" for c in d["extensive"] + d["intensive"]]
    if lake.exists("silver", "hcp_cells"):
        cur = lake.read("silver", "hcp_cells", filters=[("city", "=", city)])
        if len(cur):
            missing = [c for c in wanted if c not in cur.columns]
            if missing:
                cur[missing] = np.nan
                lake.write("silver", "hcp_cells", cur, predicate=f"city = '{city}'")
            return
    cells = lake.read("silver", "grid", filters=[("city", "=", city)], columns=["h3_id"])
    df = pd.DataFrame({"h3_id": cells["h3_id"], **dict.fromkeys(wanted, np.nan), "hcp_coverage": 0.0, "city": city})
    df[wanted] = df[wanted].astype("float64")
    lake.write("silver", "hcp_cells", df, predicate=f"city = '{city}'")
    log.warning("hcp_placeholder city=%s (no HCP units matched)", city)


# ----------------------------------------------------------------------------- silver
def build_silver(city: str, lake: Lake | None = None) -> None:
    """Everything after OSM silver for one city (grid and OSM silver are separate tasks)."""
    lake = lake or get_lake()
    gee_silver.run(city, lake)
    ensure_gee(city, lake)
    try:
        hcp_silver.run_cells(city, lake)
    except Exception as exc:  # HCP is enrichment: a parsing/matching failure must not block
        log.error("hcp_cells_failed city=%s error=%s", city, exc)
    ensure_hcp(city, lake)
    families.run(city, lake)


# ----------------------------------------------------------------------------- dbt
def dbt_env() -> dict[str, str]:
    s = get_settings()
    env = {"LAKE_URI": s.lake_uri, "WAREHOUSE_DUCKDB": str(s.warehouse_duckdb), "DBT_PROFILES_DIR": str(s.dbt_dir)}
    if s.lake_is_s3:
        env |= {"S3_ENDPOINT_URL": s.s3_endpoint_url or "", "AWS_ACCESS_KEY_ID": s.aws_access_key_id or "",
                "AWS_SECRET_ACCESS_KEY": s.aws_secret_access_key or "", "AWS_REGION": s.aws_region}
    return env


def dbt_vars() -> dict:
    c = model_config()["candidates"]
    rings, feats = dictionary.context_spec()
    return {"min_population": c["min_population"], "min_attractors_1km": c["min_attractors_1km"],
            "min_builtup_ratio": c["min_builtup_ratio"], "context_rings": rings, "context_features": feats}


def run_dbt(command: str = "build", select: str | None = None) -> None:
    """Run dbt programmatically (``dbt build`` = run models + tests)."""
    import json

    from dbt.cli.main import dbtRunner

    get_settings().warehouse_duckdb.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(dbt_env())
    dbt_dir = get_settings().dbt_dir
    args = [command, "--project-dir", str(dbt_dir), "--profiles-dir", str(dbt_dir),
            "--vars", json.dumps(dbt_vars())]
    if select:
        args += ["--select", select]
    res = dbtRunner().invoke(args)
    if not res.success:
        raise RuntimeError(f"dbt {command} failed: {res.exception or 'see dbt logs (dbt/logs)'}")


# ----------------------------------------------------------------------------- gold
def read_warehouse(table: str, path: Path | None = None) -> pd.DataFrame:
    path = path or get_settings().warehouse_duckdb
    with duckdb.connect(str(path)) as con:  # same config as in-process dbt
        return con.execute(f"select * from gold.{table}").df()


def publish_gold(lake: Lake | None = None) -> dict[str, int]:
    """Copy dbt-built gold tables to Delta (new table version = new training dataset version)."""
    lake = lake or get_lake()
    versions = {}
    for wh_name, lake_name in GOLD_TABLES.items():
        df = read_warehouse(wh_name)
        versions[lake_name] = lake.write("gold", lake_name, df, description=f"dbt model {wh_name}")
    log.info("gold_published versions=%s", versions)
    return versions
