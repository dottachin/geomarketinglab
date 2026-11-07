"""Weekly retraining: spatial CV for LightGBM / MLP / GNN, MLflow tracking, champion registration."""

from datetime import datetime

from _common import DEFAULT_ARGS, MODEL
from airflow.sdk import dag, task


@dag(schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT_ARGS,
     max_active_runs=1, tags=["geomlab", "ml"], doc_md=__doc__)
def train_model():
    @task(outlets=[MODEL])
    def train() -> dict:
        from geomlab.ml.train import train as run

        out = run()
        return {k: out[k] for k in ("champion", "comparison", "run_id", "registered_version", "gold_versions") if k in out}

    train()


train_model()
