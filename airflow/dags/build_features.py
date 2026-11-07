"""Asset-triggered: silver aggregation per city → dbt build (models + tests) → data-quality
gate → publish gold Delta tables (new version = new training dataset version)."""

from datetime import datetime

from _common import DEFAULT_ARGS, GEE, GOLD, GRID, HCP, OSM, cities
from airflow.sdk import dag, task


@dag(schedule=(GRID | OSM | GEE | HCP), start_date=datetime(2026, 1, 1), catchup=False,
     default_args=DEFAULT_ARGS, max_active_runs=1, tags=["geomlab", "features"], doc_md=__doc__)
def build_features():
    @task
    def city_list() -> list[str]:
        return cities()

    @task(max_active_tis_per_dag=1)
    def silver_city(city: str) -> str:
        from geomlab.features import build

        build.build_silver(city)
        return city

    @task
    def dbt_build(_done: list[str]) -> None:
        from geomlab.features import build

        build.run_dbt("build")

    @task
    def quality_gate() -> dict:
        from geomlab.features import build
        from geomlab.quality import checks

        report = checks.run(build.read_warehouse("gold_cell_features"))
        if not report.ok:
            raise ValueError(f"data-quality gate failed: {report.errors}")
        return {"warnings": report.warnings, "metrics": report.metrics}

    @task(outlets=[GOLD])
    def publish(_q: dict) -> dict:
        from geomlab.features import build

        return build.publish_gold()

    done = silver_city.expand(city=city_list())
    publish(dbt_build(done) >> quality_gate())


build_features()
