"""Daily: Geofabrik replication state → diffs (or weekly full extract) → bronze → silver OSM layers."""

from datetime import datetime

from _common import DEFAULT_ARGS, OSM, AirflowSkipException, cities
from airflow.sdk import dag, task


@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT_ARGS,
     max_active_runs=1, tags=["geomlab", "ingestion"], doc_md=__doc__)
def ingest_osm():
    @task
    def fetch() -> dict:
        from geomlab.ingestion import osm

        return osm.ingest()

    @task(outlets=[OSM])
    def to_silver(result: dict) -> dict:
        if result["status"] == "unchanged":
            raise AirflowSkipException(f"OSM sequence {result['sequence']} unchanged")
        from geomlab.features import osm_silver

        return osm_silver.run(cities())

    to_silver(fetch())


ingest_osm()
