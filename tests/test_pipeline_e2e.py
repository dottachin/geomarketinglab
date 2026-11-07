"""End to end on a synthetic city: silver families → dbt (models + tests) → quality gate →
gold Delta → spatial-CV training of all three model families → registry → scoring."""

import numpy as np
import yaml

from tests import synthetic


def test_full_pipeline(env):
    from tests.conftest import _reset_caches

    mc = yaml.safe_load((env / "config/model.yaml").read_text())
    mc["models"]["mlp"]["epochs"] = 5
    mc["models"]["gnn"]["epochs"] = 10
    mc["models"]["lightgbm"]["n_estimators"] = 50
    (env / "config/model.yaml").write_text(yaml.safe_dump(mc))
    _reset_caches()

    from geomlab.features import build, families
    from geomlab.lakehouse import get_lake
    from geomlab.ml import scoring, train
    from geomlab.quality import checks

    lake = get_lake()
    synthetic.write_city(lake)
    build.ensure_gee(synthetic.CITY, lake)
    build.ensure_hcp(synthetic.CITY, lake)
    families.run(synthetic.CITY, lake)
    build.run_dbt("build")
    feats = build.read_warehouse("gold_cell_features")
    assert not [c for c in feats.columns if c.startswith("comp_")]
    assert {"ctx1_pop_cell", "ctx2_nightlight_mean", "att_school_500m", "walk_pop_10min"} <= set(feats.columns)
    assert checks.run(feats, lake).ok
    versions = build.publish_gold(lake)
    assert versions == {"cell_features": 0, "training_set": 0}

    out = train.train(lake=lake)
    assert out["champion"] in {"lightgbm", "mlp", "gnn"} and out["registered_version"] == 1
    bundle = train.load_champion()
    assert bundle.model_version.startswith("geomlab-suitability:v1:")

    scores, drivers = scoring.score(bundle, lake.read("gold", "cell_features"), lake.read("gold", "training_set"))
    assert scores["final_score"].between(0, 1).all()
    assert (scores["final_score"] <= scores["suitability"] + 1e-9).all()  # penalty only lowers
    assert scores.groupby("business_type")["rank"].min().eq(1).all()
    assert scores["low_confidence"].all()  # no HCP in the synthetic city → flagged
    assert drivers.groupby(["h3_id", "business_type"]).size().max() == 8
    assert np.isfinite(drivers["shap"]).all()
