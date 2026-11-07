# GeoMarketingLab (pfe-msde)

Geospatial business-location recommendation platform for Moroccan cities: **which hexagons of
Casablanca, Rabat or Marrakech are the best places to open a pharmacy, café, bakery, gym or
supermarket?** Implementation of `../GeoMarketingLab_Architecture.docx`.

```
Sources            GEE (Sentinel-2, VIIRS, WorldPop, Dynamic World) · OSM (Geofabrik + diffs) · HCP RGPH 2024
Ingestion          Airflow 3 DAGs, one per source, daily + change detection        airflow/dags, geomlab/ingestion
Lakehouse          RustFS (S3) + Delta Lake, bronze/silver/gold, DuckDB             geomlab/lakehouse.py
Processing         H3 res 9 grid, feature families (Python) → dbt-duckdb gold       geomlab/features, dbt/
ML                 LightGBM baseline → MLP + type embedding → GraphSAGE, spatial CV, MLflow   geomlab/ml
Scoring            calibrated suitability × (1 − λ·saturation), SHAP drivers → PostGIS        geomlab/ml/scoring.py
Serving stores     PostgreSQL + PostGIS + pgvector                                  geomlab/serving
Tool layer         recommend / explain / search_documents / query_stats / simulate  geomlab/tools
                   exposed by FastAPI and an MCP server                             geomlab/api, geomlab/mcp_server.py
LLM + UI           tool-calling agent (Claude or Ollama) · React + MapLibre + deck.gl   geomlab/genai, web/
```

See [docs/architecture.md](docs/architecture.md) for the section-by-section mapping to the
architecture document (and the few deliberate deviations), and
[docs/reuse_from_pfe_old.md](docs/reuse_from_pfe_old.md) for what came from the old implementation.

## Quick start (Docker, whole stack)

```bash
cp .env.example .env          # fill in passwords, keys; see comments in the file
cd infra
docker compose --env-file ../.env up -d --build
```

| Service | URL |
|---|---|
| Web app (map + assistant) | http://localhost:8080 |
| API (OpenAPI docs) | http://localhost:8000/docs |
| Airflow | http://localhost:8081 |
| MLflow | http://localhost:5000 |
| RustFS console | http://localhost:9001 |
| MCP (streamable HTTP, bearer `MCP_AUTH_TOKEN`) | http://localhost:8765/mcp |

Then in Airflow, unpause the DAGs. `ingest_boundaries` builds the grid; the ingestion DAGs
publish assets that trigger `build_features` → `score_cells`. Run `train_model` once by hand
the first time (it is weekly afterwards), then `score_cells` runs automatically.

Earth Engine needs a Google Cloud project with the Earth Engine API enabled
(`EARTHENGINE_PROJECT`) and a service-account key in `secrets/key.json`. Without it the
pipeline still runs; satellite features are null and every score is flagged low-confidence.

## Local development (no Airflow)

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[all]"
cd infra && docker compose --env-file ../.env up -d postgres rustfs && cd ..   # or use a local lake (LAKE_URI unset)

.venv/bin/geomlab run-all --city casablanca            # boundaries → OSM → GEE → HCP → features → train → score
.venv/bin/geomlab index-docs                            # RAG corpus → pgvector
.venv/bin/geomlab ask "Où ouvrir une pharmacie à Maârif ?"
.venv/bin/uvicorn geomlab.api.main:app --reload         # API on :8000
cd web && npm install && npm run dev                    # UI on :5173 (needs Node 22)
```

Individual steps: `geomlab ingest {boundaries,osm,gee,hcp}`, `geomlab silver {grid,osm,hcp-units,cells}`,
`geomlab features`, `geomlab train`, `geomlab score`, `geomlab drift`, `geomlab lake-init`, `geomlab db-init`.

## Using the tools from Claude Desktop / Claude Code (MCP)

```json
{ "mcpServers": { "geomarketinglab": { "command": "/path/to/pfe-msde/.venv/bin/geomlab-mcp" } } }
```

Tools: `recommend_locations`, `explain_location`, `search_documents`, `query_stats`, `simulate_location`.
Resources: `geomlab://cities`, `geomlab://business-types`, `geomlab://feature-dictionary`.
Prompt: `site_analysis_report`.

## Quality gates

```bash
.venv/bin/ruff check geomlab tests airflow
.venv/bin/python -m pytest            # unit tests + an end-to-end run on a synthetic city (no network)
```

## Configuration over code

| To… | Edit |
|---|---|
| add a city | `config/cities.yaml` (+ HCP aliases in `config/hcp.yaml` if names differ) |
| add a business type | `config/poi_categories.yaml` (OSM selectors) + `config/business_types.yaml` |
| change features / context rings | `config/features.yaml` |
| change CV, models, thresholds | `config/model.yaml` |

## Honest framing

Scores rank how similar a place is to where such businesses already operate, adjusted for
competition. They are not revenue forecasts (survivor-bias assumption, see
`corpus/methodology/`). Map data © OpenStreetMap contributors (ODbL); census data © HCP;
satellite data: Copernicus Sentinel-2, NOAA VIIRS, WorldPop, Dynamic World via Google Earth Engine.
