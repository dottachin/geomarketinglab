"""Training DAG entry point (doc §6): spatial CV for every model family, MLflow tracking,
champion selection on mean AUC-PR, out-of-fold calibration, registry with alias
``champion``.

Thesis path (doc §6.2): LightGBM baseline → MLP + type embedding → GNN on the H3 graph.
Each family is a nested MLflow run, so the comparison/ablation table is one MLflow view.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from geomlab.config import model_config
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger
from geomlab.ml.bundle import BundlePyfunc, SuitabilityBundle, fit_calibrators
from geomlab.ml.data import CellData, load_gold
from geomlab.ml.evaluation import calibration_curve, evaluate, spatial_folds
from geomlab.ml.models.base import SuitabilityModel
from geomlab.ml.models.gnn import GNNModel
from geomlab.ml.models.lgbm import LightGBMModel
from geomlab.ml.models.mlp import MLPModel
from geomlab.settings import get_settings

log = get_logger(__name__)
EXPERIMENT = "geomlab-suitability"
FAMILIES = {"lightgbm": LightGBMModel, "mlp": MLPModel, "gnn": GNNModel}


def make_model(family: str) -> SuitabilityModel:
    return FAMILIES[family](**model_config()["models"].get(family, {}))


def cross_validate(family: str, cells: CellData, rows: pd.DataFrame, seed: int) -> tuple[np.ndarray, list[dict]]:
    cv = model_config()["spatial_cv"]
    ks = model_config()["metrics"]["k"]
    oof = np.full(len(rows), np.nan)
    fold_metrics = []
    for i, (tr, te) in enumerate(spatial_folds(rows, cv["n_folds"], cv["parent_resolution"], cv["buffer_rings"])):
        train_rows, test_rows = rows.iloc[tr], rows.iloc[te]
        if train_rows["label"].nunique() < 2:
            continue
        model = make_model(family).fit(cells, train_rows.reset_index(drop=True), seed + i)
        oof[te] = model.predict(cells, test_rows.reset_index(drop=True))
        m = evaluate(test_rows, oof[te], ks)
        m["fold"] = i
        fold_metrics.append(m)
        log.info("cv family=%s fold=%d pr_auc=%.3f n_train=%d n_test=%d", family, i, m.get("pr_auc", np.nan),
                 len(tr), len(te))
    return oof, fold_metrics


def _summarise(fold_metrics: list[dict]) -> dict[str, float]:
    df = pd.DataFrame(fold_metrics).drop(columns="fold")
    out = {f"cv_{k}_mean": float(v) for k, v in df.mean(numeric_only=True).items()}
    out |= {f"cv_{k}_std": float(v) for k, v in df.std(numeric_only=True).items() if "." not in k}
    return out


def _log_json(obj, name: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        mlflow.log_artifact(str(p))


def train(families: list[str] | None = None, cities: list[str] | None = None,
          lake: Lake | None = None, register: bool = True) -> dict:
    lake = lake or get_lake()
    cfg = model_config()
    seed = int(cfg["seed"])
    families = families or list(FAMILIES)
    cells, rows, gold_versions = load_gold(lake, cities=cities)
    if rows.empty:
        raise RuntimeError("gold.training_set is empty — run the feature pipeline first")
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    results: dict[str, dict] = {}
    with mlflow.start_run(run_name=f"train-{datetime.now(UTC):%Y%m%d-%H%M}") as parent:
        mlflow.log_params({"gold_cell_features_version": gold_versions["cell_features"],
                           "gold_training_set_version": gold_versions["training_set"],
                           "n_rows": len(rows), "n_cells": len(cells.frame), "n_features": len(cells.feature_cols),
                           "cities": ",".join(sorted(rows["city"].unique())), "seed": seed,
                           "positive_rate": round(float(rows["label"].mean()), 4)})
        mlflow.set_tag("data_date", str(cells.frame["data_date"].max()))
        _log_json(cells.feature_cols, "feature_columns.json")

        for fam in families:
            with mlflow.start_run(run_name=fam, nested=True):
                t0 = time.time()
                mlflow.log_params({"family": fam, **{f"hp_{k}": v for k, v in cfg["models"].get(fam, {}).items()}})
                oof, folds = cross_validate(fam, cells, rows, seed)
                if not folds:
                    log.warning("family=%s produced no valid folds", fam)
                    continue
                summary = _summarise(folds)
                ok = ~np.isnan(oof)
                calibrators = fit_calibrators(rows[ok].reset_index(drop=True), oof[ok])
                mlflow.log_metrics({k: v for k, v in summary.items() if np.isfinite(v)})
                _log_json(folds, "cv_folds.json")
                _log_json(calibration_curve(rows["label"].to_numpy()[ok], oof[ok]), "calibration_oof.json")
                final = make_model(fam).fit(cells, rows, seed)
                if isinstance(final, LightGBMModel):
                    _log_json(final.feature_importance(cells.feature_cols).head(30).to_dict(), "importance_gain.json")
                mlflow.log_metric("train_seconds", time.time() - t0)
                results[fam] = {"summary": summary, "model": final, "calibrators": calibrators}
                log.info("family=%s cv_pr_auc=%.3f±%.3f", fam, summary.get("cv_pr_auc_mean", np.nan),
                         summary.get("cv_pr_auc_std", np.nan))

        metric = f"cv_{cfg['selection']['metric']}_mean"
        champion = max(results, key=lambda f: np.nan_to_num(results[f]["summary"].get(metric, -1), nan=-1))
        comparison = {f: {k: r["summary"].get(k) for k in (metric, "cv_roc_auc_mean", "cv_ndcg_at_10_mean",
                                                            "cv_precision_at_10_mean", "cv_brier_mean")}
                      for f, r in results.items()}
        _log_json(comparison, "model_comparison.json")
        mlflow.set_tag("champion_family", champion)
        log.info("champion=%s comparison=%s", champion, comparison)

        best = results[champion]
        bundle = SuitabilityBundle(
            model=best["model"], feature_cols=cells.feature_cols, calibrators=best["calibrators"],
            metadata={"gold_versions": gold_versions, "data_date": str(cells.frame["data_date"].max()),
                      "trained_at": datetime.now(UTC).isoformat(), "cv": best["summary"],
                      "family": champion, "parent_run_id": parent.info.run_id,
                      "training_feature_sample": cells.frame[cells.feature_cols].sample(
                          min(2000, len(cells.frame)), random_state=seed).to_dict("list")},
        )
        info = mlflow.pyfunc.log_model(name="model", python_model=BundlePyfunc(bundle),
                                       pip_requirements=["geomlab"])
        out = {"champion": champion, "comparison": comparison, "run_id": parent.info.run_id,
               "model_uri": info.model_uri, "gold_versions": gold_versions}
        if register:
            name = cfg["selection"]["registered_model"]
            mv = mlflow.register_model(info.model_uri, name)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(name, "champion", mv.version)
            client.set_model_version_tag(name, mv.version, "family", champion)
            out["registered_version"] = int(mv.version)
            log.info("registered model=%s version=%s alias=champion", name, mv.version)
    return out


def load_champion() -> SuitabilityBundle:
    """Load the registered champion bundle (used by scoring and live inference)."""
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    name = model_config()["selection"]["registered_model"]
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    bundle: SuitabilityBundle = model.unwrap_python_model().bundle
    bundle.model_version = f"{name}:v{mv.version}:{bundle.family}"
    return bundle
