"""Daily check, monthly in practice: GEE zonal statistics per H3 cell when VIIRS / WorldPop /
the Sentinel-2 composite window changed."""

from datetime import datetime

from _common import DEFAULT_ARGS, GEE, AirflowSkipException, cities
from airflow.sdk import dag, task


@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT_ARGS,
     max_active_runs=1, tags=["geomlab", "ingestion"], doc_md=__doc__)
def ingest_gee():
    @task(outlets=[GEE], execution_timeout=None)
    def extract_and_clean() -> dict:
        from geomlab.features import gee_silver
        from geomlab.features.grid import load_grid
        from geomlab.ingestion import gee

        status = gee.ingest({c: load_grid(c) for c in cities()})
        changed = [c for c, s in status.items() if s != "unchanged"]
        if not changed:
            raise AirflowSkipException("GEE change token unchanged")
        for c in changed:
            gee_silver.run(c)
        return status

    extract_and_clean()


ingest_gee()
