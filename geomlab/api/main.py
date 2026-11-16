"""FastAPI backend (doc §8): serves scores to the web map and exposes the same tools the LLM
uses, plus the chat endpoint. Thin: every route delegates to ``geomlab.tools.core`` or
``geomlab.genai.agent``.

Security (doc §9): optional API keys (X-API-Key, API_KEYS env), rate limiting (slowapi),
read-only DB roles inside the tools, CORS restricted to the web origins.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from geomlab.api.schemas import ChatRequest, SearchRequest, SimulateRequest, StatsRequest
from geomlab.logging import get_logger
from geomlab.serving.db import engine
from geomlab.settings import get_settings
from geomlab.tools import core

log = get_logger(__name__)

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
# declared before /map/{city}/{business_type}, which would otherwise capture it
@app.get("/map/{city}/pois", dependencies=[Auth])
@limiter.limit("30/minute")
def map_pois(request: Request, city: str) -> dict:  # noqa: ARG001
    """Points of interest of a city: existing businesses (role=business) and attractors."""
    return {"city": core._city(city), "pois": core.list_pois(city)}


@app.get("/map/{city}/{business_type}", dependencies=[Auth])
@limiter.limit("30/minute")
def map_scores(request: Request, city: str, business_type: str) -> dict:  # noqa: ARG001 - slowapi needs request
    """All scored hexagons of a city for one business type (deck.gl H3HexagonLayer input)."""
    return core.map_cells(city, business_type)


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


def _chat_provider(body: ChatRequest) -> str:
    provider = (body.provider or settings.llm_provider).lower()
    if provider == "anthropic" and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise HTTPException(status_code=503, detail="The assistant is not configured: set ANTHROPIC_API_KEY in .env "
                            "(or LLM_PROVIDER=ollama) and restart the api service.")
    return provider


def _llm_error(exc: Exception, provider: str) -> str:
    """A readable reason for a failed LLM call; full details go to the server log."""
    if isinstance(exc, httpx.ConnectError) and provider == "ollama":
        return (f"Ollama is not reachable at {settings.ollama_url}. Start it with "
                "`docker compose --profile local-llm up -d ollama` (and pull the model), or set LLM_PROVIDER=anthropic.")
    body_ = getattr(exc, "body", None)  # Anthropic errors (billing, rate limit, invalid model…)
    reason = body_.get("error", {}).get("message") if isinstance(body_, dict) else None
    return f"The language model call failed: {reason or str(exc) or type(exc).__name__}"


def _ask(body: ChatRequest, provider: str, on_event=None):
    from geomlab.genai.agent import ask

    return ask(body.message, [t.model_dump() for t in body.history], provider=provider,
               context=body.context.model_dump() if body.context else None, on_event=on_event)


@app.post("/chat", dependencies=[Auth])
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest) -> dict:  # noqa: ARG001
    provider = _chat_provider(body)
    try:
        answer = _ask(body, provider)
    except Exception as exc:  # LLM provider/network errors → readable 502, details in the server log
        log.exception("chat_failed provider=%s", provider)
        raise HTTPException(status_code=502, detail=_llm_error(exc, provider)) from exc
    return answer.to_dict()


@app.post("/chat/stream", dependencies=[Auth])
@limiter.limit("10/minute")
def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:  # noqa: ARG001
    """Same as /chat, streamed as NDJSON events: tool / tool_result / delta / reset, then one
    ``done`` (the /chat response) or ``error``. ``ping`` lines keep proxies from timing out
    while a local model reads a long prompt."""
    provider = _chat_provider(body)
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            events.put({"type": "done", **_ask(body, provider, on_event=events.put).to_dict()})
        except Exception as exc:  # noqa: BLE001 - reported to the client as an error event
            log.exception("chat_failed provider=%s", provider)
            events.put({"type": "error", "detail": _llm_error(exc, provider)})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            try:
                ev = events.get(timeout=10)
            except queue.Empty:
                ev = {"type": "ping"}
            if ev is None:
                return
            yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
