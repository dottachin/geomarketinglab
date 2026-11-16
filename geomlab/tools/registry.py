"""Tool registry: JSON schemas (for the LLM providers) + dispatch. The MCP server and the
FastAPI routes call ``geomlab.tools.core`` directly; the in-app agent uses this registry."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from geomlab.config import business_types, cities
from geomlab.tools import core, sql_guard


def _schemas() -> list[dict]:
    bts, cs = sorted(business_types()), sorted(cities())
    return [
        {"name": "recommend_locations",
         "description": "Rank the best hexagons (H3 cells) in a city to open a business type, from the deep "
                        "learning model's precomputed scores. Returns rank, the place names (district, neighbourhood, "
                        "nearby landmark), final_score, suitability, saturation, competitors, confidence flags, model_version and data_date. Use for any 'where should "
                        "I open…' question.",
         "input_schema": {"type": "object", "additionalProperties": False, "required": ["business_type", "city"],
                          "properties": {
                              "business_type": {"type": "string", "enum": bts},
                              "city": {"type": "string", "enum": cs},
                              "k": {"type": "integer", "minimum": 1, "maximum": core.MAX_K, "default": 10},
                              "district": {"type": "string", "description": "Area filter (partial match) on the arrondissement / commune "
                                                                         "or the neighbourhood name"},
                              "max_competitors": {"type": "integer", "description": "Max same-type businesses in the catchment"},
                              "min_population": {"type": "number", "description": "Min residents in the catchment"},
                              "exclude_low_confidence": {"type": "boolean"},
                              "opportunity_only": {"type": "boolean", "description": "Only high-suitability cells with no existing business"}}}},
        {"name": "explain_location",
         "description": "Explain one hexagon's score for a business type: place names (district, "
                        "neighbourhood, landmark), city rank out of n_cells, SHAP drivers (what raises/lowers it), "
                        "key statistics, nearby competitors with distances, neighbouring cells, confidence.",
         "input_schema": {"type": "object", "additionalProperties": False, "required": ["h3_id", "business_type"],
                          "properties": {"h3_id": {"type": "string"}, "business_type": {"type": "string", "enum": bts}}}},
        {"name": "search_documents",
         "description": "Search the document corpus (HCP reports, urban planning documents, commune regulations, "
                        "licensing procedures, methodology notes). Returns passages with document title and page. "
                        "Use for rules, procedures, context and methodology — never for scores.",
         "input_schema": {"type": "object", "additionalProperties": False, "required": ["query"],
                          "properties": {"query": {"type": "string"}, "city": {"type": "string", "enum": cs},
                                         "doc_type": {"type": "string"}, "year": {"type": "integer"},
                                         "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}}}},
        {"name": "query_stats",
         "description": "Run ONE read-only PostgreSQL SELECT on these views, for aggregate questions (counts, "
                        "averages per district…):\n" + sql_guard.schema_context(),
         "input_schema": {"type": "object", "additionalProperties": False, "required": ["sql"],
                          "properties": {"sql": {"type": "string"}}}},
        {"name": "simulate_location",
         "description": "What-if analysis with live model inference: change feature values of one hexagon "
                        "(e.g. {\"att_tram_stop_500m\": 1}) and compare baseline vs scenario scores.",
         "input_schema": {"type": "object", "additionalProperties": False,
                          "required": ["h3_id", "business_type", "changes"],
                          "properties": {"h3_id": {"type": "string"}, "business_type": {"type": "string", "enum": bts},
                                         "changes": {"type": "object", "additionalProperties": {"type": "number"}}}}},
    ]


FUNCTIONS: dict[str, Callable[..., dict]] = {
    "recommend_locations": core.recommend_locations,
    "explain_location": core.explain_location,
    "search_documents": core.search_documents,
    "query_stats": core.query_stats,
    "simulate_location": core.simulate_location,
}


def tool_schemas(enabled: list[str] | None = None) -> list[dict]:
    return [s for s in _schemas() if enabled is None or s["name"] in enabled]


def call(name: str, args: dict[str, Any]) -> tuple[dict, bool]:
    """Execute a tool; returns (payload, is_error). Errors are returned to the LLM so it can
    correct itself (doc §7.4.2 'invalid values return a clear error the LLM can correct')."""
    fn = FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}, True
    try:
        return fn(**args), False
    except core.ToolInputError as exc:
        return exc.to_dict(), True
    except core.NotReadyError as exc:
        return {"error": str(exc)}, True
    except TypeError as exc:
        return {"error": f"bad arguments: {exc}"}, True


def to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# Keys the LLM does not need to answer (coordinates, long definitions, provenance details).
# Dropping them keeps tool results small: a local CPU model reads ~15 tokens/s, so every
# kilobyte of JSON costs it about 20 seconds.
_LLM_DROP = {"lon", "lat", "centroid", "source_path", "filters", "score_definition", "how_to_read_drivers",
             "model_family", "family", "source", "evidence"}


def compact(value: Any, max_list: int = 10, max_text: int = 700) -> Any:
    """Shrink a tool payload for the LLM: drop unneeded keys, empty values and long tails,
    round floats. The full payload is still used for citations and the UI."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in _LLM_DROP:
                continue
            v = compact(v, max_list, max_text)
            if v is None or v == [] or v == {}:
                continue
            out[k] = v
        return out
    if isinstance(value, list):
        return [compact(v, max_list, max_text) for v in value[:max_list]]
    if isinstance(value, float):
        return round(value) if abs(value) >= 100 else round(value, 3)
    if isinstance(value, str) and len(value) > max_text:
        return value[:max_text] + "…"
    return value


def to_llm_json(payload: dict) -> str:
    return json.dumps(compact(payload), ensure_ascii=False, default=str, separators=(",", ":"))
