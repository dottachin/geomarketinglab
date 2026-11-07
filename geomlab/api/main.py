"""FastAPI backend (doc §8): serves scores to the web map and exposes the same tools the LLM
uses, plus the chat endpoint. Thin: every route delegates to ``geomlab.tools.core`` or
``geomlab.genai.agent``.

Security (doc §9): optional API keys (X-API-Key, API_KEYS env), rate limiting (slowapi),
read-only DB roles inside the tools, CORS restricted to the web origins.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from geomlab.api.schemas import ChatRequest, SearchRequest, SimulateRequest, StatsRequest
from geomlab.serving.db import engine
from geomlab.settings import get_settings
from geomlab.tools import core

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
app = FastAPI(title="GeoMarketingLab API", version="0.1.0",
              description="Business-location recommendations for Moroccan cities (H3 + deep learning + RAG). "
                          "Map data © OpenStreetMap contributors (ODbL); HCP RGPH 2024; Copernicus/VIIRS/WorldPop via GEE.")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    keys = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
    if keys and x_api_key not in keys:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


Auth = Depends(require_key)


@app.exception_handler(core.ToolInputError)
async def _bad_input(_: Request, exc: core.ToolInputError):
    return JSONResponse(status_code=422, content=exc.to_dict())


@app.exception_handler(core.NotReadyError)
async def _not_ready(_: Request, exc: core.NotReadyError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.get("/health")
def health() -> dict:
    try:
        with engine("ro").connect() as con:
            con.execute(text("select 1"))
        db = "ok"
    except Exception as exc:  # noqa: BLE001
        db = f"unavailable: {type(exc).__name__}"
    return {"status": "ok", "database": db}


# --- metadata / MCP-resource equivalents -----------------------------------------------
@app.get("/meta/cities", dependencies=[Auth])
def cities() -> list[dict]:
    return core.list_cities()


@app.get("/meta/business-types", dependencies=[Auth])
def business_types() -> list[dict]:
    return core.list_business_types()


@app.get("/meta/features", dependencies=[Auth])
def features() -> list[dict]:
    return core.feature_dictionary_resource()


@app.get("/meta/model", dependencies=[Auth])
def active_model() -> dict:
    return core._active()


# --- map data ------------------------------------------------------------------------
@app.get("/map/{city}/{business_type}", dependencies=[Auth])
@limiter.limit("30/minute")
def map_scores(request: Request, city: str, business_type: str) -> dict:  # noqa: ARG001 - slowapi needs request
    """All scored hexagons of a city for one business type (deck.gl H3HexagonLayer input)."""
    c, bt = core._city(city), core._business(business_type)
    active = core._active()
    with engine("ro").connect() as con:
        rows = con.execute(text("""
            SELECT s.h3_id, round(s.final_score::numeric, 4) AS final_score, round(s.suitability::numeric, 4) AS suitability,
                   round(s.saturation::numeric, 4) AS saturation, s.rank, s.n_existing, s.opportunity, s.low_confidence
            FROM serving.scores s WHERE s.model_version = :mv AND s.city = :c AND s.business_type = :bt"""),
            {"mv": active["model_version"], "c": c, "bt": bt}).mappings().all()
        comps = con.execute(text("""SELECT name, ST_X(geom) AS lon, ST_Y(geom) AS lat FROM serving.pois
            WHERE city = :c AND category = :bt"""), {"c": c, "bt": bt}).mappings().all()
    return {"city": c, "business_type": bt, "model_version": active["model_version"], "data_date": active["data_date"],
            "cells": [{k: (float(v) if hasattr(v, "is_finite") else v) for k, v in r.items()} for r in rows],
            "competitors": [dict(r) for r in comps]}


# --- tools -----------------------------------------------------------------------------
@app.get("/recommendations", dependencies=[Auth])
def recommendations(business_type: str, city: str, k: Annotated[int, Query(ge=1, le=50)] = 10,
                    district: str | None = None, max_competitors: int | None = None,
                    min_population: float | None = None, exclude_low_confidence: bool = False,
                    opportunity_only: bool = False) -> dict:
    return core.recommend_locations(business_type, city, k, district, max_competitors, min_population,
                                    exclude_low_confidence, opportunity_only)


@app.get("/cells/{h3_id}/explain", dependencies=[Auth])
def explain(h3_id: str, business_type: str) -> dict:
    return core.explain_location(h3_id, business_type)


@app.post("/cells/{h3_id}/simulate", dependencies=[Auth])
@limiter.limit("10/minute")
def simulate(request: Request, h3_id: str, body: SimulateRequest) -> dict:  # noqa: ARG001
    return core.simulate_location(h3_id, body.business_type, body.changes)


@app.post("/documents/search", dependencies=[Auth])
def search(body: SearchRequest) -> dict:
    return core.search_documents(body.query, body.city, body.doc_type, body.year, body.k)


@app.post("/stats/query", dependencies=[Auth])
@limiter.limit("20/minute")
def stats(request: Request, body: StatsRequest) -> dict:  # noqa: ARG001
    return core.query_stats(body.sql)


@app.post("/chat", dependencies=[Auth])
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest) -> dict:  # noqa: ARG001
    from geomlab.genai.agent import ask

    answer = ask(body.message, [t.model_dump() for t in body.history], provider=body.provider)
    return answer.to_dict()
