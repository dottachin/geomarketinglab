"""The tool layer (doc §7.3–7.5): one implementation, exposed by FastAPI and the MCP server.

Rules from doc §7.4.2 implemented here, not left to the prompt:
  * numbers only come from these functions (scores, ranks, statistics);
  * every response carries ``model_version`` and ``data_date``;
  * explanations are returned with scores (SHAP drivers, competitors, neighbours);
  * uncertainty is exposed (``low_confidence`` + reasons, low retrieval similarity);
  * model evidence (``evidence: model``) and document evidence (``evidence: document``)
    are returned in separate shapes so they are cited separately;
  * inputs are validated against the allow-lists (cities, business types, features);
    invalid values raise ToolInputError listing the valid values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from geomlab.config import business_types, cities, get_business_type
from geomlab.features.dictionary import describe, feature_dictionary
from geomlab.serving.db import engine
from geomlab.tools import sql_guard

MAX_K = 50
KEY_STATS = ["pop_cell", "pop_1km", "walk_pop_10min", "att_total_1000m", "nightlight_mean",
             "builtup_ratio", "intersection_density_km2", "dist_main_road_m", "hcp_share_15_34",
             "hcp_employment_rate"]


class ToolInputError(ValueError):
    def __init__(self, message: str, field: str | None = None, valid: list[str] | None = None):
        super().__init__(message)
        self.field, self.valid = field, valid

    def to_dict(self) -> dict:
        return {"error": str(self), "field": self.field, "valid_values": self.valid}


class NotReadyError(RuntimeError):
    """No active model / no scores yet."""


# ----------------------------------------------------------------------------- validation
def _city(city: str) -> str:
    slug = (city or "").strip().lower()
    aliases = {c.display_name.lower(): s for s, c in cities().items()} | {"marrakesh": "marrakech",
                                                                          "casa": "casablanca"}
    slug = aliases.get(slug, slug)
    if slug not in cities():
        raise ToolInputError(f"unknown city '{city}'", "city", sorted(cities()))
    return slug


def _business(bt: str) -> str:
    slug = (bt or "").strip().lower()
    for s, b in business_types().items():
        if slug in {s, *(v.lower() for v in b.display_name.values())}:
            return s
    raise ToolInputError(f"unknown business type '{bt}'", "business_type", sorted(business_types()))


def _h3(h3_id: str) -> str:
    import h3

    if not h3.is_valid_cell(h3_id or ""):
        raise ToolInputError(f"'{h3_id}' is not a valid H3 cell id", "h3_id")
    return h3_id


def _active() -> dict:
    with engine("ro").connect() as con:
        row = con.execute(text("SELECT model_version, family, data_date, metrics FROM serving.active_model")).mappings().first()
    if row is None:
        raise NotReadyError("no model has been scored yet; run the training and scoring DAGs")
    return dict(row) | {"data_date": str(row["data_date"])}


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, np.generic):
        return _clean(v.item())
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


# ----------------------------------------------------------------------------- tools
def recommend_locations(business_type: str, city: str, k: int = 10, district: str | None = None,
                        max_competitors: int | None = None, min_population: float | None = None,
                        exclude_low_confidence: bool = False, opportunity_only: bool = False) -> dict:
    """Top-k hexagons for a business type in a city, ranked by final score."""
    bt, c = _business(business_type), _city(city)
    k = max(1, min(int(k), MAX_K))
    active = _active()
    where = ["s.model_version = :mv", "s.city = :city", "s.business_type = :bt"]
    params: dict[str, Any] = {"mv": active["model_version"], "city": c, "bt": bt, "k": k}
    if district:
        where.append("c.district ILIKE :district")
        params["district"] = f"%{district}%"
    if max_competitors is not None:
        where.append("s.competitors_catchment <= :maxc")
        params["maxc"] = int(max_competitors)
    if min_population is not None:
        where.append("coalesce(s.pop_catchment, 0) >= :minp")
        params["minp"] = float(min_population)
    if exclude_low_confidence:
        where.append("NOT s.low_confidence")
    if opportunity_only:
        where.append("s.opportunity")
    sql = f"""
        SELECT s.h3_id, c.district, s.rank AS city_rank, s.final_score, s.suitability, s.saturation,
               s.n_existing, s.competitors_catchment, s.nearest_competitor_m, s.pop_catchment,
               s.opportunity, s.low_confidence, s.confidence_reasons,
               ST_X(c.centroid) AS lon, ST_Y(c.centroid) AS lat
        FROM serving.scores s JOIN serving.cells c ON c.h3_id = s.h3_id
        WHERE {' AND '.join(where)}
        ORDER BY s.final_score DESC LIMIT :k"""
    with engine("ro").connect() as con:
        df = pd.read_sql(text(sql), con, params=params)
    if district and df.empty:
        with engine("ro").connect() as con:
            names = [r[0] for r in con.execute(text(
                "SELECT DISTINCT district FROM serving.cells WHERE city = :c AND district IS NOT NULL ORDER BY 1"),
                {"c": c})]
        raise ToolInputError(f"no scored cells match district '{district}' in {c}", "district", names)
    df.insert(0, "rank", range(1, len(df) + 1))
    for col in ("final_score", "suitability", "saturation"):
        df[col] = df[col].round(3)
    return {
        "evidence": "model",
        "business_type": bt, "city": c, "filters": {k_: v for k_, v in params.items() if k_ not in ("mv", "city", "bt", "k")},
        "results": _records(df),
        "model_version": active["model_version"], "model_family": active["family"], "data_date": active["data_date"],
        "score_definition": "final_score = calibrated suitability × (1 − λ·saturation); suitability is the "
                            "model's probability that this kind of hexagon hosts the business type",
        "caveat": "Scores rank locations by similarity to where such businesses already operate; "
                  "they are not a revenue forecast.",
    }


def explain_location(h3_id: str, business_type: str) -> dict:
    """Score, SHAP drivers, raw statistics, nearby competitors and neighbouring cells."""
    cell, bt = _h3(h3_id), _business(business_type)
    active = _active()
    mv = active["model_version"]
    catchment = get_business_type(bt).catchment_m
    with engine("ro").connect() as con:
        score = con.execute(text("""
            SELECT s.*, c.district, ST_X(c.centroid) AS lon, ST_Y(c.centroid) AS lat
            FROM serving.scores s JOIN serving.cells c ON c.h3_id = s.h3_id
            WHERE s.h3_id = :h AND s.business_type = :bt AND s.model_version = :mv"""),
            {"h": cell, "bt": bt, "mv": mv}).mappings().first()
        if score is None:
            raise ToolInputError(f"hexagon {cell} has no score for {bt} (not a candidate cell or outside the pilot cities)", "h3_id")
        drivers = pd.read_sql(text("""SELECT rank, feature, value, shap FROM serving.shap_drivers
            WHERE h3_id = :h AND business_type = :bt AND model_version = :mv ORDER BY rank"""), con,
            params={"h": cell, "bt": bt, "mv": mv})
        comps = pd.read_sql(text("""
            SELECT p.name, round(ST_Distance(p.geom::geography, c.centroid::geography)) AS distance_m
            FROM serving.pois p, serving.cells c
            WHERE c.h3_id = :h AND p.city = c.city AND p.category = :bt
              AND ST_DWithin(p.geom::geography, c.centroid::geography, :r)
            ORDER BY distance_m LIMIT 10"""), con, params={"h": cell, "bt": bt, "r": catchment})
        neigh = pd.read_sql(text("""
            SELECT s.h3_id, round(s.final_score::numeric, 3) AS final_score, s.rank AS city_rank, s.n_existing
            FROM serving.scores s JOIN serving.cells c ON c.h3_id = s.h3_id
            JOIN serving.cells me ON me.h3_id = :h
            WHERE s.business_type = :bt AND s.model_version = :mv AND s.h3_id <> :h
              AND ST_DWithin(c.centroid::geography, me.centroid::geography, 400)
            ORDER BY s.final_score DESC LIMIT 6"""), con, params={"h": cell, "bt": bt, "mv": mv})
        cols = ", ".join(f'"{c}"' for c in KEY_STATS)
        try:
            stats = con.execute(text(f"SELECT {cols} FROM gold.cell_features WHERE h3_id = :h"), {"h": cell}).mappings().first()
        except Exception:  # noqa: BLE001 - features table schema may lag behind KEY_STATS
            stats = None

    drv = []
    for r in drivers.itertuples():
        info = describe(r.feature)
        drv.append({"feature": r.feature, "label": info.description, "family": info.family, "unit": info.unit,
                    "source": info.source, "value": _clean(r.value), "shap": round(float(r.shap), 4),
                    "effect": "raises" if r.shap > 0 else "lowers"})
    key_stats = []
    for name in KEY_STATS:
        if stats is not None and name in stats and stats[name] is not None:
            info = describe(name)
            key_stats.append({"feature": name, "label": info.description, "value": _clean(stats[name]),
                              "unit": info.unit, "source": info.source})
    return {
        "evidence": "model",
        "h3_id": cell, "business_type": bt, "city": score["city"], "district": score["district"],
        "centroid": {"lon": score["lon"], "lat": score["lat"]},
        "final_score": round(score["final_score"], 3), "suitability": round(score["suitability"], 3),
        "saturation": round(score["saturation"], 3), "city_rank": score["rank"],
        "percentile": round(score["percentile"], 3), "opportunity": score["opportunity"],
        "existing_in_cell": score["n_existing"], "competitors_in_catchment": score["competitors_catchment"],
        "catchment_m": catchment, "population_in_catchment": _clean(score["pop_catchment"]),
        "low_confidence": score["low_confidence"], "confidence_reasons": list(score["confidence_reasons"]),
        "drivers": drv, "key_statistics": key_stats,
        "nearby_competitors": _records(comps), "neighbouring_cells": _records(neigh),
        "model_version": mv, "data_date": active["data_date"],
        "how_to_read_drivers": "SHAP values are contributions to the model's log-odds: positive raises the "
                               "suitability, negative lowers it, relative to an average cell.",
    }


def search_documents(query: str, city: str | None = None, doc_type: str | None = None,
                     year: int | None = None, k: int = 5) -> dict:
    """Hybrid (BM25-style full text + vector) retrieval over the document corpus."""
    from geomlab.genai.retrieval import hybrid_search

    if not query or not query.strip():
        raise ToolInputError("query must not be empty", "query")
    c = _city(city) if city else None
    hits, low = hybrid_search(query, k=max(1, min(int(k), 10)), city=c, doc_type=doc_type, year=year)
    return {
        "evidence": "document", "query": query, "low_confidence": low,
        "results": hits,
        "note": ("No sufficiently relevant passage was found; say so rather than answering from memory."
                 if low else "Cite as (document title, page)."),
    }


def query_stats(sql: str) -> dict:
    """Run a validated read-only SELECT on the analytics views."""
    try:
        safe = sql_guard.validate(sql)
    except sql_guard.SQLRejected as exc:
        raise ToolInputError(str(exc), "sql", sorted(sql_guard.ALLOWED_TABLES)) from None
    active = _active()
    with engine("sql").connect() as con:
        df = pd.read_sql(text(safe), con)
    return {"evidence": "model", "sql": safe, "columns": list(df.columns), "rows": _records(df.head(sql_guard.MAX_ROWS)),
            "row_count": len(df), "model_version": active["model_version"], "data_date": active["data_date"]}


@lru_cache(maxsize=1)
def _champion():
    from geomlab.ml.train import load_champion

    return load_champion()


def simulate_location(h3_id: str, business_type: str, changes: dict[str, float]) -> dict:
    """Mode B (doc §7.4): live inference with the registered model on a what-if scenario —
    e.g. 'what if a tram stop opened here' → {"att_tram_stop_500m": 1}."""
    from geomlab.ml.scoring import saturation

    cell, bt = _h3(h3_id), _business(business_type)
    d = feature_dictionary()
    bad = [f for f in changes if f not in d or f.startswith("ctx")]
    if bad:
        raise ToolInputError(f"cannot change {bad}", "changes", sorted(f for f in d if not f.startswith("ctx")))
    bundle = _champion()
    with engine("ro").connect() as con:
        base = pd.read_sql(text("SELECT * FROM gold.cell_features WHERE h3_id = :h"), con, params={"h": cell})
        sc = con.execute(text("""SELECT competitors_catchment, pop_catchment, n_existing FROM serving.scores
            WHERE h3_id = :h AND business_type = :bt ORDER BY scored_at DESC LIMIT 1"""), {"h": cell, "bt": bt}).mappings().first()
    if base.empty:
        raise ToolInputError(f"no features for hexagon {cell}", "h3_id")
    scenario = base.copy()
    for f, v in changes.items():
        scenario[f] = float(v)
    rows = pd.DataFrame({"city": base["city"], "h3_id": [cell], "business_type": [bt]})
    out = {}
    for name, frame in (("baseline", base), ("scenario", scenario)):
        cells = bundle.cells(frame)
        pred = bundle.predict(cells, rows)
        attr = bundle.attribute(cells, rows).iloc[0].sort_values(key=np.abs, ascending=False).head(5)
        suit = float(pred["suitability"].iloc[0])
        sat = 0.0
        if sc is not None:
            sat = float(saturation(pd.Series([sc["competitors_catchment"]], dtype=float),
                                   pd.Series([sc["pop_catchment"]], dtype=float),
                                   get_business_type(bt).residents_per_business).iloc[0])
        final = suit * (1 - get_business_type(bt).saturation_lambda * sat)
        out[name] = {"suitability": round(suit, 3), "final_score": round(final, 3),
                     "top_drivers": [{"feature": f, "label": describe(f).description, "shap": round(float(v), 4)}
                                     for f, v in attr.items()]}
    return {"evidence": "model", "mode": "live_inference", "h3_id": cell, "business_type": bt, "changes": changes,
            **out, "delta_final_score": round(out["scenario"]["final_score"] - out["baseline"]["final_score"], 3),
            "model_version": bundle.model_version, "data_date": bundle.metadata.get("data_date"),
            "caveat": "Single-cell what-if: neighbouring cells and competitors are held fixed (for the GNN, "
                      "neighbour features are not re-propagated in this mode)."}


# ----------------------------------------------------------------------------- resources
def list_cities() -> list[dict]:
    return [{"slug": s, "name": c.display_name, "h3_resolution": c.h3_resolution} for s, c in cities().items()]


def list_business_types() -> list[dict]:
    return [{"slug": s, "names": b.display_name, "catchment_m": b.catchment_m} for s, b in business_types().items()]


def feature_dictionary_resource() -> list[dict]:
    return [{"name": f.name, "family": f.family, "unit": f.unit, "source": f.source, "description": f.description}
            for f in feature_dictionary().values()]
