"""Daily: download HCP RGPH tables, store in bronze only when the SHA-256 changed, parse to silver."""

from datetime import datetime

from _common import DEFAULT_ARGS, HCP, AirflowSkipException
from airflow.sdk import dag, task


@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT_ARGS,
     tags=["geomlab", "ingestion"], doc_md=__doc__)
def ingest_hcp():
    @task
    def fetch() -> dict:
        from geomlab.ingestion import hcp

        return hcp.ingest()

    @task(outlets=[HCP])
    def parse(status: dict) -> int:
        from geomlab.features import hcp_silver
        from geomlab.lakehouse import get_lake

        if "changed" not in status.values() and get_lake().exists("silver", "hcp_units"):
            raise AirflowSkipException("HCP files unchanged")
        return hcp_silver.run_units()

    parse(fetch())


ingest_hcp()
