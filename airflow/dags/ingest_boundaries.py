"""Weekly: city boundaries + admin units (OSM/Nominatim) → bronze → silver grid + neighbours."""

from datetime import datetime

from _common import DEFAULT_ARGS, GRID, AirflowSkipException, cities
from airflow.sdk import dag, task


@dag(schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT_ARGS,
     tags=["geomlab", "ingestion"], doc_md=__doc__)
def ingest_boundaries():
    @task
    def fetch() -> dict:
        from geomlab.ingestion import boundaries

        return boundaries.ingest(cities())

    @task(outlets=[GRID])
    def build_grid(status: dict) -> None:
        from geomlab.features import grid
        from geomlab.lakehouse import get_lake

        changed = [c for c, s in status.items() if s == "changed" or not get_lake().exists("silver", "grid")]
        if not changed:
            raise AirflowSkipException("boundaries unchanged")
        for c in changed:
            grid.run(c)

    build_grid(fetch())


ingest_boundaries()
