"""MCP server (doc §7.5): publishes the shared tool module so any MCP client (Claude Desktop,
Claude Code, the web app's own agent) can discover and call it. MCP is only the interface —
the model and the data stay in PostGIS / MLflow.

  geomlab-mcp                         stdio (local demos, Claude Desktop / Claude Code)
  geomlab-mcp --http --port 8765      streamable HTTP, bearer-token protected (MCP_AUTH_TOKEN)
"""

from __future__ import annotations

import argparse
import json

try:  # mcp >= 2: FastMCP was renamed MCPServer (same decorator API)
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from geomlab.tools import core

mcp = MCPServer("geomarketinglab", instructions=(
    "Business-location recommendations for Moroccan cities. Numbers (scores, ranks, statistics) must "
    "come from these tools; cite hexagons for model evidence and document/page for documents."))


def _safe(fn, **kw) -> dict:
    try:
        return fn(**kw)
    except core.ToolInputError as exc:
        return exc.to_dict()
    except core.NotReadyError as exc:
        return {"error": str(exc)}


@mcp.tool()
def recommend_locations(business_type: str, city: str, k: int = 10, district: str | None = None,
                        max_competitors: int | None = None, min_population: float | None = None,
                        exclude_low_confidence: bool = False, opportunity_only: bool = False) -> dict:
    """Top-k H3 cells to open this business type in a city (deep learning model scores,
    adjusted by a competition-saturation penalty). Stamped with model_version and data_date."""
    return _safe(core.recommend_locations, business_type=business_type, city=city, k=k, district=district,
                 max_competitors=max_competitors, min_population=min_population,
                 exclude_low_confidence=exclude_low_confidence, opportunity_only=opportunity_only)


@mcp.tool()
def explain_location(h3_id: str, business_type: str) -> dict:
    """Score, SHAP drivers, key statistics, nearby competitors and neighbouring cells for one H3 cell."""
    return _safe(core.explain_location, h3_id=h3_id, business_type=business_type)


@mcp.tool()
def search_documents(query: str, city: str | None = None, doc_type: str | None = None,
                     year: int | None = None, k: int = 5) -> dict:
    """Hybrid search over HCP reports, regulations, licensing procedures and methodology notes.
    Returns passages with document title, page and similarity."""
    return _safe(core.search_documents, query=query, city=city, doc_type=doc_type, year=year, k=k)


@mcp.tool()
def query_stats(sql: str) -> dict:
    """One read-only SELECT over analytics.scores, analytics.businesses, analytics.districts."""
    return _safe(core.query_stats, sql=sql)


@mcp.tool()
def simulate_location(h3_id: str, business_type: str, changes: dict[str, float]) -> dict:
    """What-if: change feature values of one cell and compare live model scores (mode B)."""
    return _safe(core.simulate_location, h3_id=h3_id, business_type=business_type, changes=changes)


@mcp.resource("geomlab://cities")
def cities_resource() -> str:
    """Supported cities (valid values for the `city` argument)."""
    return json.dumps(core.list_cities(), ensure_ascii=False)


@mcp.resource("geomlab://business-types")
def business_types_resource() -> str:
    """Supported business types (valid values for `business_type`)."""
    return json.dumps(core.list_business_types(), ensure_ascii=False)


@mcp.resource("geomlab://feature-dictionary")
def feature_dictionary_resource() -> str:
    """Meaning, unit and source of every model feature (to read SHAP drivers)."""
    return json.dumps(core.feature_dictionary_resource(), ensure_ascii=False)


@mcp.prompt()
def site_analysis_report(business_type: str, city: str, district: str = "") -> str:
    """Reusable site-analysis report template."""
    where = f" in the {district} area of {city}" if district else f" in {city}"
    return (f"Write a site-analysis report for opening a {business_type}{where}.\n"
            "1. Call recommend_locations (k=5"
            + (f", district='{district}'" if district else "") + ").\n"
            "2. Call explain_location for the top 3 cells.\n"
            f"3. Call search_documents for regulations and licensing rules for a {business_type}.\n"
            "Structure: Summary · Ranked locations (score, rank, 2–3 drivers each, competitors) · "
            "Regulatory notes (cited by document and page) · Data confidence and caveats · "
            "model_version and data date. Only use numbers returned by the tools.")


def http_app(token: str, host: str = "0.0.0.0"):
    """Streamable-HTTP app with bearer-token authentication (shared deployment)."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    app = mcp.streamable_http_app(host=host)

    class BearerAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.headers.get("authorization") != f"Bearer {token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    app.add_middleware(BearerAuth)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoMarketingLab MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.http:
        mcp.run(transport="stdio")
        return
    import uvicorn

    from geomlab.settings import get_settings

    token = get_settings().mcp_auth_token
    if not token:
        raise SystemExit("MCP_AUTH_TOKEN must be set for the HTTP transport")
    uvicorn.run(http_app(token, args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
