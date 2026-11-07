"""Feature dictionary: static entries from config/features.yaml + generated attractor and
spatial-context features. Single source for model inputs and human-readable explanations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import yaml

from geomlab.config import poi_taxonomy
from geomlab.settings import get_settings


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    family: str
    unit: str
    source: str
    direction: str
    description: str


@lru_cache(maxsize=1)
def _raw() -> dict:
    with (get_settings().config_dir / "features.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def feature_dictionary() -> dict[str, FeatureInfo]:
    raw = _raw()
    out: dict[str, FeatureInfo] = {n: FeatureInfo(name=n, **{k: str(v) for k, v in spec.items()})
                                   for n, spec in raw["features"].items()}
    tpl = raw["attractor_template"]
    tax = poi_taxonomy()
    for ring in tax.rings_m:
        for cat in [*tax.attractor, "total"]:
            name = f"att_{cat}_{ring}m"
            label = "attractor" if cat == "total" else cat.replace("_", " ")
            out[name] = FeatureInfo(name, tpl["family"], tpl["unit"], tpl["source"], tpl["direction"],
                                    tpl["description"].format(category=label, ring=ring))
    for k in raw["context"]["rings"]:
        for base in raw["context"]["features"]:
            b = out[base]
            out[f"ctx{k}_{base}"] = FeatureInfo(
                f"ctx{k}_{base}", "context", b.unit, b.source, b.direction,
                f"Average of '{b.description}' over neighbouring hexagons (≤{k} ring{'s' if k > 1 else ''})")
    return out


def context_spec() -> tuple[list[int], list[str]]:
    c = _raw()["context"]
    return list(c["rings"]), list(c["features"])


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns of ``df`` that are model inputs: in the dictionary, numeric, and never
    competition (``comp_*``/``n_in_cell``) — the no-leakage rule."""
    d = feature_dictionary()
    cols = [c for c in df.columns if c in d and pd.api.types.is_numeric_dtype(df[c])]
    leaked = [c for c in cols if c.startswith("comp_") or c == "n_in_cell"]
    if leaked:
        raise AssertionError(f"competition columns must never be model inputs: {leaked}")
    return sorted(cols)


def describe(feature: str) -> FeatureInfo:
    d = feature_dictionary()
    return d.get(feature) or FeatureInfo(feature, "other", "", "", "~", feature.replace("_", " "))
