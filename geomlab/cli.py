"""Command line: every pipeline step, runnable without Airflow (the DAGs call the same functions).

  geomlab ingest boundaries|osm|gee|hcp [--city C]
  geomlab silver grid|osm|hcp-units|cells [--city C]
  geomlab features [--city C]            silver → dbt build → quality gate → publish gold
  geomlab train [--families lightgbm,mlp,gnn]
  geomlab score
  geomlab lake-init | db-init | index-docs [--force] | drift
  geomlab ask "Où ouvrir une pharmacie à Casablanca ?"
  geomlab run-all [--city C] [--skip-gee]  end-to-end local run
"""

from __future__ import annotations

import argparse
import json
import sys

from geomlab.config import cities


def _cities(arg: str | None) -> list[str]:
    return [arg] if arg else list(cities())


def cmd_ingest(a) -> dict:
    from geomlab.ingestion import boundaries, gee, hcp, osm

    if a.source == "boundaries":
        return boundaries.ingest(_cities(a.city))
    if a.source == "osm":
        return osm.ingest(force_full=a.force)
    if a.source == "hcp":
        return hcp.ingest()
    from geomlab.features.grid import load_grid

    return gee.ingest({c: load_grid(c) for c in _cities(a.city)}, force=a.force)


def cmd_silver(a) -> dict:
    from geomlab.features import build, grid, hcp_silver, osm_silver

    if a.step == "grid":
        return {c: grid.run(c) for c in _cities(a.city)}
    if a.step == "osm":
        return osm_silver.run(_cities(a.city))
    if a.step == "hcp-units":
        return {"version": hcp_silver.run_units()}
    for c in _cities(a.city):
        build.build_silver(c)
    return {"cities": _cities(a.city)}


def cmd_features(a) -> dict:
    from geomlab.features import build
    from geomlab.quality import checks

    for c in _cities(a.city):
        build.build_silver(c)
    build.run_dbt("build")
    report = checks.run(build.read_warehouse("gold_cell_features"))
    if not report.ok:
        raise SystemExit(f"data-quality gate failed: {report.errors}")
    return {"gold_versions": build.publish_gold(), "warnings": report.warnings}


def cmd_train(a) -> dict:
    from geomlab.ml.train import train

    out = train(families=a.families.split(",") if a.families else None)
    return {k: v for k, v in out.items() if k != "model"}


def cmd_score(_a) -> dict:
    from geomlab.ml.scoring import run

    return run()


def cmd_lake_init(_a) -> dict:
    """Create the object-storage buckets (lakehouse + MLflow artifacts) on RustFS/S3."""
    from geomlab.lakehouse import get_lake

    lake = get_lake()
    if not lake.is_s3:
        return {"status": "local lake, nothing to create", "uri": lake.uri}
    fs, root = lake._fs()
    created = []
    for bucket in {root.split("/")[0], "mlflow"}:
        fs.create_dir(bucket)
        created.append(bucket)
    return {"buckets": sorted(created)}


def cmd_db_init(_a) -> dict:
    from geomlab.serving.db import init_schema

    init_schema()
    return {"status": "ok"}


def cmd_index_docs(a) -> dict:
    from geomlab.genai.indexing import index_corpus
    from geomlab.serving.db import init_schema

    init_schema()
    return index_corpus(force=a.force)


def cmd_drift(_a) -> dict:
    import pandas as pd

    from geomlab.lakehouse import get_lake
    from geomlab.ml.monitoring import drift_report
    from geomlab.ml.train import load_champion

    bundle = load_champion()
    ref = pd.DataFrame(bundle.metadata["training_feature_sample"])
    cur = get_lake().read("gold", "cell_features", columns=list(ref.columns))
    rep = drift_report(ref, cur)
    return {"alerts": rep[rep["alert"]].to_dict("records"), "top": rep.head(10).to_dict("records")}


def cmd_ask(a) -> dict:
    from geomlab.genai.agent import ask

    return ask(a.question, provider=a.provider).to_dict()


def cmd_run_all(a) -> dict:
    """Local end-to-end run (what the Airflow DAGs do, in order)."""
    from geomlab.ingestion import boundaries, gee, hcp, osm

    city_list = _cities(a.city)
    boundaries.ingest(city_list)
    cmd_silver(argparse.Namespace(step="grid", city=a.city))
    osm.ingest()
    cmd_silver(argparse.Namespace(step="osm", city=a.city))
    if not a.skip_gee:
        from geomlab.features.grid import load_grid

        gee.ingest({c: load_grid(c) for c in city_list})
    try:
        hcp.ingest()
        cmd_silver(argparse.Namespace(step="hcp-units", city=None))
    except Exception as exc:  # noqa: BLE001 - HCP is enrichment
        print(f"HCP skipped: {exc}", file=sys.stderr)
    out = {"features": cmd_features(a), "train": cmd_train(argparse.Namespace(families=None))}
    out["score"] = cmd_score(a)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geomlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    s.add_argument("source", choices=["boundaries", "osm", "gee", "hcp"])
    s.add_argument("--city")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("silver")
    s.add_argument("step", choices=["grid", "osm", "hcp-units", "cells"])
    s.add_argument("--city")
    s.set_defaults(fn=cmd_silver)
    s = sub.add_parser("features")
    s.add_argument("--city")
    s.set_defaults(fn=cmd_features)
    s = sub.add_parser("train")
    s.add_argument("--families")
    s.set_defaults(fn=cmd_train)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    sub.add_parser("db-init").set_defaults(fn=cmd_db_init)
    sub.add_parser("lake-init").set_defaults(fn=cmd_lake_init)
    s = sub.add_parser("index-docs")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_index_docs)
    sub.add_parser("drift").set_defaults(fn=cmd_drift)
    s = sub.add_parser("ask")
    s.add_argument("question")
    s.add_argument("--provider", choices=["anthropic", "ollama"])
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("run-all")
    s.add_argument("--city")
    s.add_argument("--skip-gee", action="store_true")
    s.set_defaults(fn=cmd_run_all)
    a = p.parse_args(argv)
    print(json.dumps(a.fn(a), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
