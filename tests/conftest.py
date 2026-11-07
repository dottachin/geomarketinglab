from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _reset_caches() -> None:
    from geomlab import config, lakehouse, settings
    from geomlab.features import dictionary

    for fn in (settings.get_settings, config.cities, config.poi_taxonomy, config.business_types,
               config.model_config, config.hcp_config, lakehouse.get_lake, dictionary._raw,
               dictionary.feature_dictionary):
        fn.cache_clear()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated config (with the synthetic 'testville' city), lake and warehouse."""
    cfg = tmp_path / "config"
    shutil.copytree(ROOT / "config", cfg)
    cities = yaml.safe_load((cfg / "cities.yaml").read_text())
    cities["testville"] = {"display_name": "Testville", "crs_projected": 32629, "h3_resolution": 9,
                           "bbox": [33.57, -7.60, 33.62, -7.55]}
    (cfg / "cities.yaml").write_text(yaml.safe_dump(cities))
    monkeypatch.setenv("CONFIG_DIR", str(cfg))
    monkeypatch.setenv("LAKE_URI", str(tmp_path / "lake"))
    monkeypatch.setenv("WAREHOUSE_DUCKDB", str(tmp_path / "warehouse.duckdb"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(tmp_path / "mlflow.db"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    _reset_caches()
    yield tmp_path
    _reset_caches()
