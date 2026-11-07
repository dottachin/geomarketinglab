"""Shared DAG settings: retries with exponential backoff, failure alerts, assets (doc §3.3).

Data-aware scheduling: each ingestion DAG publishes an Asset ONLY when its source changed
(an unchanged source raises AirflowSkipException, so no asset event is emitted). The
features DAG runs when any upstream asset updates, scoring runs when gold or the model
changes — so a day where nothing changed costs one cheap change check per source.
"""

from __future__ import annotations

import contextlib
import os
from datetime import timedelta

import httpx
from airflow.sdk import Asset

try:  # Airflow 3 task SDK
    from airflow.sdk.exceptions import AirflowSkipException  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from airflow.exceptions import AirflowSkipException  # noqa: F401

GRID = Asset("geomlab://silver/grid")
OSM = Asset("geomlab://silver/osm")
GEE = Asset("geomlab://silver/gee_cells")
HCP = Asset("geomlab://silver/hcp_units")
GOLD = Asset("geomlab://gold/cell_features")
MODEL = Asset("geomlab://mlflow/geomlab-suitability@champion")
SCORES = Asset("geomlab://postgis/serving.scores")
DOCS = Asset("geomlab://postgis/rag.chunks")


def notify_failure(context) -> None:
    """Alert on failure: Slack incoming webhook if SLACK_WEBHOOK_URL is set (else just log)."""
    ti = context.get("task_instance")
    msg = f":red_circle: GeoMarketingLab task failed: {ti.dag_id}.{ti.task_id} (try {ti.try_number})"
    url = os.getenv("SLACK_WEBHOOK_URL")
    if url:
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(url, json={"text": msg}, timeout=10)
    print(msg)


DEFAULT_ARGS = {
    "owner": "geomlab",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "on_failure_callback": notify_failure,
}


def cities() -> list[str]:
    from geomlab.config import cities as _c

    return list(_c())
