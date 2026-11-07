"""Batch inference after every feature refresh or retraining (doc §7.4 mode A): drift check,
scores + SHAP drivers + saturation penalty + model_version → PostGIS and gold/scores."""

from datetime import datetime

from _common import DEFAULT_ARGS, GOLD, MODEL, SCORES
from airflow.sdk import dag, task


@dag(schedule=(GOLD | MODEL), start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT_ARGS,
     max_active_runs=1, tags=["geomlab", "ml"], doc_md=__doc__)
def score_cells():
    @task
    def drift() -> dict:
        import mlflow
        import pandas as pd

        from geomlab.lakehouse import get_lake
        from geomlab.ml.monitoring import drift_report
        from geomlab.ml.train import load_champion

        bundle = load_champion()
        ref = pd.DataFrame(bundle.metadata["training_feature_sample"])
        rep = drift_report(ref, get_lake().read("gold", "cell_features", columns=list(ref.columns)))
        with mlflow.start_run(run_id=bundle.metadata["parent_run_id"]):
            mlflow.log_metric("drift_max_psi", float(rep["psi"].max()))
            mlflow.log_metric("drift_n_alerts", int(rep["alert"].sum()))
        alerts = rep[rep["alert"]]["feature"].tolist()
        if alerts:
            print(f"feature drift (PSI > threshold): {alerts} — consider retraining")
        return {"alerts": alerts}

    @task(outlets=[SCORES])
    def score(_d: dict) -> dict:
        from geomlab.ml.scoring import run

        return run()

    score(drift())


score_cells()
