"""Data-quality gate between dbt gold and publishing (doc §5.1).

dbt tests cover schema-level rules (not-null, uniqueness, accepted values, no-leakage).
These checks cover what is awkward in SQL: geometry validity, per-family null rates,
category coverage per city and row-count drift against the previously published gold
version (Delta time travel). Lightweight replacement for Great Expectations, documented
as such in docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from geomlab.config import business_types
from geomlab.features.dictionary import feature_dictionary
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger

log = get_logger(__name__)

MAX_NULL_RATE = {"demand": 0.6, "economic": 0.6, "attractors": 0.0, "accessibility": 0.2,
                 "environment": 0.6, "context": 0.2}
MIN_BUSINESS_POIS = 10
MAX_ROW_DRIFT = 0.10


@dataclass
class QualityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_null_rates(features: pd.DataFrame, report: QualityReport) -> None:
    d = feature_dictionary()
    cand = features[features["is_candidate"]] if "is_candidate" in features else features
    for family, limit in MAX_NULL_RATE.items():
        cols = [c for c in cand.columns if c in d and d[c].family == family]
        if not cols:
            continue
        rate = float(cand[cols].isna().mean().mean())
        report.metrics[f"null_rate_{family}"] = rate
        if rate > limit:
            msg = f"{family}: null rate {rate:.1%} > {limit:.0%}"
            # GEE/HCP may legitimately be missing (no credentials) → warning, not error
            (report.warnings if family in ("demand", "economic", "environment") else report.errors).append(msg)


def check_geometries(lake: Lake, report: QualityReport) -> None:
    grid = lake.read_geo("silver", "grid")
    invalid = int((~grid.is_valid).sum())
    report.metrics["invalid_cell_geometries"] = invalid
    if invalid:
        report.errors.append(f"{invalid} invalid hexagon geometries in silver.grid")


def check_category_coverage(lake: Lake, report: QualityReport) -> None:
    poi = lake.read("silver", "poi", columns=["city", "role", "category"])
    counts = poi[poi["role"] == "business"].groupby(["city", "category"]).size()
    for city in poi["city"].unique():
        for bt in business_types():
            n = int(counts.get((city, bt), 0))
            report.metrics[f"pois_{city}_{bt}"] = n
            if n < MIN_BUSINESS_POIS:
                report.warnings.append(f"{city}/{bt}: only {n} mapped businesses (weak labels)")


def check_row_drift(new: pd.DataFrame, lake: Lake, report: QualityReport) -> None:
    if not lake.exists("gold", "cell_features"):
        return
    prev = lake.read("gold", "cell_features", columns=["city"])
    for city, n_new in new.groupby("city").size().items():
        n_old = int((prev["city"] == city).sum())
        if n_old == 0:
            continue
        drift = abs(n_new - n_old) / n_old
        report.metrics[f"row_drift_{city}"] = drift
        if drift > MAX_ROW_DRIFT:
            report.errors.append(f"{city}: cell count changed {n_old} → {n_new} ({drift:.0%})")


def run(features: pd.DataFrame, lake: Lake | None = None) -> QualityReport:
    lake = lake or get_lake()
    report = QualityReport()
    check_null_rates(features, report)
    check_geometries(lake, report)
    check_category_coverage(lake, report)
    check_row_drift(features, lake, report)
    for w in report.warnings:
        log.warning("dq_warning %s", w)
    for e in report.errors:
        log.error("dq_error %s", e)
    return report
