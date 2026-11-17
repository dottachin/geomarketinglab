<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="GeoMarketingLab" width="480">
  </picture>
</p>

<p align="center">
  <img alt="License: All rights reserved" src="https://img.shields.io/badge/license-All%20rights%20reserved-red">
  <img alt="Status: Portfolio - view only" src="https://img.shields.io/badge/status-portfolio%20%C2%B7%20view%20only-0f172a">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
  <img alt="End-of-study project (PFE, Master MSDE)" src="https://img.shields.io/badge/PFE-Master%20MSDE-4f46e5">
</p>

> [!IMPORTANT]
> **This is proprietary software, shown for portfolio purposes only.**
> You may view the code, but you may not copy, run, modify, distribute or sell any part of it without written permission. See [LICENSE](LICENSE).

# GeoMarketingLab

**GeoMarketingLab** is a geospatial business-location recommendation platform for Moroccan cities. It answers one question: **which hexagons of Casablanca, Rabat or Marrakech are the best places to open a pharmacy, café, bakery, gym or supermarket?**

The city is cut into an H3 hexagon grid (resolution 9, about 0.1 km² per cell). Each cell is described by satellite imagery, OpenStreetMap points of interest and the 2024 census. A model learns where each type of business already operates, scores every cell, and lowers the score where competition is already dense. The results are shown on an interactive map, and an assistant answers questions in English, French or Arabic using only numbers that come from the platform's tools.

It was built as an end-of-study project (PFE, Master MSDE) and implements `../GeoMarketingLab_Architecture.docx`.

---

## Table of contents

1. [How it works (big picture)](#how-it-works-big-picture)
2. [Services](#services)
3. [Repository layout](#repository-layout)
4. [Features in detail](#features-in-detail)
5. [Prerequisites](#prerequisites)
6. [Configuration (environment variables)](#configuration-environment-variables)
7. [Quick start with Docker](#quick-start-with-docker)
8. [Local development (without Airflow)](#local-development-without-airflow)
9. [The data pipeline](#the-data-pipeline)
10. [How the model works](#how-the-model-works)
11. [API overview](#api-overview)
12. [Using the tools from Claude Desktop / Claude Code (MCP)](#using-the-tools-from-claude-desktop--claude-code-mcp)
13. [Configuration over code](#configuration-over-code)
14. [Testing and quality gates](#testing-and-quality-gates)
15. [Troubleshooting](#troubleshooting)
16. [Honest framing and data credits](#honest-framing-and-data-credits)
17. [License](#license)

---

## How it works (big picture)

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

```
                    ┌────────────────────────────────────────┐
  Browser  ───────▶ │ web (React + MapLibre + deck.gl, :8080)│
                    └───────────────┬────────────────────────┘
                                    │ HTTP + NDJSON chat stream
  Claude Desktop ──┐                ▼
  (MCP client)     │   ┌─────────────────────────┐      ┌────────────────────┐
                   └─▶ │ api (FastAPI) / mcp     │ ───▶ │ LLM: Claude or     │
                       │ one shared tool layer   │      │ Ollama (local)     │
                       └───────────┬─────────────┘      └────────────────────┘
                                   ▼
                ┌─────────────────────────────────────┐
                │ PostgreSQL + PostGIS + pgvector     │ ◀── scores, SHAP drivers, RAG chunks
                └─────────────────▲───────────────────┘
                                  │ score_cells / index_documents
  ┌────────────┐   ┌──────────────┴───────────────┐   ┌──────────────────┐
  │ GEE · OSM  │──▶│ Airflow 3 → RustFS + Delta   │──▶│ MLflow (champion │
  │ HCP · docs │   │ bronze → silver → gold (dbt) │   │ model registry)  │
  └────────────┘   └──────────────────────────────┘   └──────────────────┘
```

Every number the user sees (on the map, in the panels or in the assistant's answer) comes from the same functions in `geomlab/tools/core.py`. The LLM only chooses which tool to call and writes the explanation around the results; each result carries the `model_version` and `data_date` it was computed from.

See [docs/architecture.md](docs/architecture.md) for the section-by-section mapping to the architecture document (and the few deliberate deviations), and [docs/reuse_from_pfe_old.md](docs/reuse_from_pfe_old.md) for what came from the earlier implementation.

---

## Services

All services run from `infra/docker-compose.yml`.

| Service | Role | URL / port |
|---|---|---|
| `web` | Map and assistant UI (nginx serving the Vite build) | http://localhost:8080 (`WEB_PORT`) |
| `api` | FastAPI: map data, recommendations, explanations, chat | http://localhost:8000/docs (`API_PORT`) |
| `mcp` | MCP server over streamable HTTP, bearer `MCP_AUTH_TOKEN` | http://localhost:8765/mcp |
| `airflow-*` | Airflow 3 API server, scheduler, DAG processor, triggerer | http://localhost:8081 |
| `mlflow` | Experiment tracking and model registry (`champion` alias) | http://localhost:5000 |
| `rustfs` | S3-compatible object storage for the Delta lakehouse | :9000, console http://localhost:9001/rustfs/console/ |
| `postgres` | PostgreSQL + PostGIS + pgvector (serving schema, RAG index) | :5432 |
| `martin` | Vector tiles served from PostGIS | :3000 |
| `init` | One-shot: creates buckets, Delta tables, DB schema and roles | — |
| `ollama` | Optional local LLM (profile `local-llm`) | :11434 |

---

## Repository layout

```
pfe-msde/
├── airflow/dags/        one DAG per source + build_features, train_model, score_cells, index_documents
├── assets/              logo
├── config/              cities, business types, POI taxonomy, features, model settings (YAML)
├── corpus/              RAG documents: HCP publications, regulations, methodology
├── dbt/                 dbt-duckdb project: staging → intermediate → gold, with tests
├── docs/                architecture map, reuse notes
├── geomlab/
│   ├── ingestion/       boundaries, OSM, GEE, HCP collectors with change detection
│   ├── features/        H3 grid, silver builders, feature families, place names
│   ├── quality/         data-quality gate before gold is published
│   ├── ml/              training, spatial CV, models (LightGBM, MLP, GraphSAGE), scoring, drift
│   ├── serving/         PostGIS schema and loaders
│   ├── tools/           the shared tool layer, its JSON schemas and the SQL guard
│   ├── genai/           tool-calling agent, prompts, embeddings, indexing, hybrid retrieval
│   ├── api/             FastAPI app and Pydantic schemas
│   ├── mcp_server.py    MCP tools, resources and prompt
│   ├── geo.py           CRS and H3 helpers (the only place distances and areas are computed)
│   ├── lakehouse.py     RustFS / Delta bronze-silver-gold access
│   └── cli.py           `geomlab` command line (the DAGs call the same functions)
├── infra/               docker-compose.yml, Dockerfiles, Martin config
├── tests/               pytest suite + synthetic city "testville"
├── web/                 React 18 + TypeScript + Vite app
├── Makefile             shortcuts: install, test, lint, up, down, pipeline, api, web, mcp
└── pyproject.toml       package and optional extras (ingestion, ml, genai, api, dbt, all)
```

---

## Features in detail

**Map.** Pick a city and a business type and the map colours every hexagon by its final score (viridis scale). Existing businesses can be shown as points. Clicking a cell opens its panel: the score, its rank, the main SHAP drivers pushing it up or down, the competition around it and a human-readable place name. A top-k list shows the best cells.

**What-if simulation.** `simulate_location` re-scores a cell live with the champion model after a hypothetical change to its features (for example "what if a tram stop opened here") and returns the difference.

**Assistant.** A single, hand-written tool-calling agent (`geomlab/genai/agent.py`) works with Claude or with a local Ollama model. It answers in the user's language, receives the current map context automatically, and streams its tool steps and answer text to the UI. Tool results are compacted before they reach the model, and a time budget caps each answer.

**Documents (RAG).** HCP publications, regulations and methodology notes from `corpus/` are chunked by section, embedded (bge-m3) and stored in pgvector. Search is hybrid: vector similarity plus PostgreSQL full-text, fused with reciprocal rank fusion, with an optional reranker. The assistant cites model numbers and documents separately.

**Statistics.** `query_stats` runs read-only SQL against the serving schema through a SQL guard (parsed with sqlglot, allow-listed tables, a dedicated read-only database role).

**Security.** Optional API keys (`X-API-Key`, `API_KEYS`), per-route rate limits, restricted CORS, bearer token for the MCP server, and separate database roles for the API and for ad-hoc SQL.

---

## Prerequisites

| For | You need |
|---|---|
| The whole stack | Docker with Compose v2, about 8 GB of free RAM (more with Ollama) |
| Local development | Python 3.11+, Node 22 for the web app |
| Satellite features | A Google Cloud project with the Earth Engine API enabled and a service-account key |
| The assistant | An Anthropic API key with credit, **or** the Ollama container |

---

## Configuration (environment variables)

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored; secrets never go into the code or into `config/`. Local runs without Docker fall back to the defaults in `geomlab/settings.py` (a local lake in `./data/lake`, SQLite MLflow, Postgres on localhost).

| Group | Variables |
|---|---|
| Postgres | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `GEOMLAB_RO_PASSWORD`, `GEOMLAB_SQL_PASSWORD`, `DATABASE_URL`, `DATABASE_URL_RO`, `DATABASE_URL_SQL` |
| RustFS (S3) | `RUSTFS_ACCESS_KEY`, `RUSTFS_SECRET_KEY`, `RUSTFS_S3_PORT`, `RUSTFS_CONSOLE_PORT`, optional `LAKE_URI`, `S3_ENDPOINT_URL`, `AWS_*` for local runs |
| Airflow | `AIRFLOW_FERNET_KEY`, `AIRFLOW_JWT_SECRET`, `SLACK_WEBHOOK_URL` (failure alerts) |
| Earth Engine | `EARTHENGINE_PROJECT`, `GEE_SERVICE_ACCOUNT`, `GEE_SERVICE_ACCOUNT_KEY_PATH`, `GEE_KEY_DIR`, `GEE_KEY_FILE`, `HOST_GID` |
| LLM | `LLM_PROVIDER` (`anthropic` or `ollama`), `LLM_MODEL`, `ANTHROPIC_API_KEY`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `OLLAMA_THINK`, `CHAT_TIMEOUT_S` |
| RAG | `EMBEDDING_BACKEND` (`sentence-transformers`, `ollama` or `hash` for tests), `EMBEDDING_MODEL` |
| API / MCP | `API_KEYS` (comma-separated, empty = no auth), `MCP_AUTH_TOKEN`, `API_PORT`, `WEB_PORT` |

---

## Quick start with Docker

```bash
cp .env.example .env          # fill in passwords, keys; see comments in the file
cd infra
docker compose --env-file ../.env up -d --build
```

(`make up` does the same.) Then in Airflow (http://localhost:8081), unpause the DAGs. `ingest_boundaries` builds the grid; the ingestion DAGs publish assets that trigger `build_features` → `score_cells`. Run `train_model` once by hand the first time (it is weekly afterwards), then `score_cells` runs automatically.

Earth Engine needs a Google Cloud project with the Earth Engine API enabled (`EARTHENGINE_PROJECT`) and a service-account key in `secrets/key.json`. Without it the pipeline still runs; satellite features are null and every score is flagged low-confidence.

### Assistant (LLM)

| `LLM_PROVIDER` | Needs |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` with credit |
| `ollama` | the Ollama container: `docker compose --env-file ../.env --profile local-llm up -d ollama`, then `docker compose exec ollama ollama pull qwen3:8b` |

The UI uses `POST /chat/stream` (NDJSON: tool steps, answer text as it is generated, then `done`). On CPU, a local 8B model takes 1–3 minutes per answer; `OLLAMA_MODEL=qwen3:4b` is faster. Tool results are compacted before they reach the model and Ollama runs with `OLLAMA_NUM_CTX=8192` and `OLLAMA_THINK=false`; `CHAT_TIMEOUT_S` caps one answer.

Ports 8000/8080 taken by another stack: set `API_PORT` / `WEB_PORT` in `.env`.

---

## Local development (without Airflow)

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

Individual steps: `geomlab ingest {boundaries,osm,gee,hcp}`, `geomlab silver {grid,osm,hcp-units,cells}`, `geomlab features`, `geomlab train`, `geomlab score`, `geomlab drift`, `geomlab lake-init`, `geomlab db-init`.

Makefile shortcuts: `make install`, `make pipeline CITY=rabat`, `make api`, `make web`, `make mcp`, `make test`, `make lint`.

---

## The data pipeline

| Step | Where | What happens |
|---|---|---|
| 1. Ingest | `ingest_osm`, `ingest_gee`, `ingest_hcp` (daily), `ingest_boundaries` (weekly) | Download only what changed. OSM: Geofabrik sequence number; GEE: a token built from the latest VIIRS month, WorldPop year and Sentinel-2 window; HCP and boundaries: content hash. Nothing changed → the task is skipped and nothing downstream runs. |
| 2. Bronze | `Lake.put_raw` | Raw files stored immutably under an `ingest_date=` partition, with a manifest (URL, SHA-256, change token). |
| 3. Silver | `geomlab/features/*_silver.py`, `grid.py` | H3 grid clipped to the city, POIs mapped to the taxonomy, GEE reductions per cell, HCP census disaggregated to cells (population-weighted, mass-preserving). |
| 4. Gold | `build_features` → dbt → quality gate | Feature families computed per city, `dbt build` with tests, then the data-quality gate decides whether gold is published. |
| 5. Train | `train_model` (weekly) | Spatial CV, MLflow tracking, best model gets the `champion` alias. |
| 6. Score | `score_cells` | Drift check, calibrated scores, saturation penalty and SHAP drivers written to PostGIS and `gold/scores`. |
| 7. Serve | API, MCP, `index_documents` | Read from PostGIS; the RAG index is kept up to date. |

Geometries are stored in EPSG:4326. Distances and areas are only computed in the city's projected CRS (EPSG:32629) through `geomlab/geo.py`.

---

## How the model works

**Formulation.** For each (cell, business type), predict whether that type of business is present. Only plausibly urban cells are candidates (minimum population, nearby attractors and built-up ratio, see `config/model.yaml`), so the model does not learn the trivial "nothing in the desert" rule.

**Features.** Satellite (Sentinel-2 indices, night lights, built-up probability, population), OSM context rings (attractors, transport, amenities at several distances) and census variables. Competition columns (`comp_*`, `n_in_cell`) are **never** model inputs; a dbt test enforces it.

**Models.** A LightGBM baseline, an MLP with a business-type embedding, and GraphSAGE over the H3 neighbour graph. The champion is the best mean spatial-CV AUC-PR.

**Validation.** Spatial cross-validation only: cells are grouped into H3 parent blocks at resolution 7 (about 5 km²), 5 folds, and training cells within one ring of a test block are dropped to avoid leakage. Metrics: AUC-PR, Precision@k and NDCG@k (k = 10, 25, 50), Brier score and calibration.

**Scoring.**

```
final score = calibrated suitability × (1 − λ · saturation)
```

Suitability is calibrated with isotonic regression. Saturation is `competitors / (competitors + expected capacity)`, where the expected capacity is the population inside the catchment radius divided by the residents one business needs; `λ`, the catchment and the residents-per-business ratio come from `config/business_types.yaml`. Each cell keeps its top 8 SHAP drivers. Cells with sparse OSM coverage or a missing source are flagged low-confidence. Feature drift is monitored with PSI (alert above 0.2).

---

## API overview

Interactive documentation: http://localhost:8000/docs. When `API_KEYS` is set, every route except `/health` needs the `X-API-Key` header.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/meta/cities`, `/meta/business-types`, `/meta/features`, `/meta/model` | Reference data and current model info |
| GET | `/map/{city}/{business_type}` | Scores for every cell (map layer) |
| GET | `/map/{city}/pois` | Existing businesses |
| GET | `/recommendations?city=&business_type=&k=` | Top-k cells (k ≤ 50) |
| GET | `/cells/{h3_id}/explain?business_type=` | Score, drivers and context of one cell |
| POST | `/cells/{h3_id}/simulate` | What-if re-scoring |
| POST | `/documents/search` | Hybrid RAG search |
| POST | `/stats/query` | Guarded read-only SQL |
| POST | `/chat` | Assistant, single JSON answer |
| POST | `/chat/stream` | Assistant, NDJSON stream (used by the UI) |

Map, simulation, stats and chat routes are rate-limited (10 to 30 requests per minute).

---

## Using the tools from Claude Desktop / Claude Code (MCP)

```json
{ "mcpServers": { "geomarketinglab": { "command": "/path/to/pfe-msde/.venv/bin/geomlab-mcp" } } }
```

Or connect to the stack's server at `http://localhost:8765/mcp` with the bearer token `MCP_AUTH_TOKEN`.

Tools: `recommend_locations`, `explain_location`, `search_documents`, `query_stats`, `simulate_location`.
Resources: `geomlab://cities`, `geomlab://business-types`, `geomlab://feature-dictionary`.
Prompt: `site_analysis_report`.

---

## Configuration over code

| To… | Edit |
|---|---|
| add a city | `config/cities.yaml` (+ HCP aliases in `config/hcp.yaml` if names differ) |
| add a business type | `config/poi_categories.yaml` (OSM selectors) + `config/business_types.yaml` |
| change features / context rings | `config/features.yaml` |
| change CV, models, thresholds | `config/model.yaml` |

---

## Testing and quality gates

```bash
.venv/bin/ruff check geomlab tests airflow
.venv/bin/python -m pytest                                # unit tests + an end-to-end run on a synthetic city (no network)
docker build --target build -t geomlab-web-check web      # web type-check + bundle
```

The end-to-end test builds a synthetic city, `testville`, and runs the whole pipeline offline: silver, features, `dbt build` with its tests, training and scoring. Other tests cover CRS and H3 helpers, HCP disaggregation and matching, ingestion change detection, ML units (spatial folds, metrics), the SQL guard, the API and the agent with a fake LLM provider. GitHub Actions runs the same gates (`.github/workflows/ci.yml`).

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Every score is flagged low-confidence, satellite features are empty | Earth Engine is not configured. Set `EARTHENGINE_PROJECT` and the service-account key (chmod 640, `HOST_GID` set so Airflow can read it). |
| The map is empty after starting the stack | No model yet. Unpause the DAGs, run `train_model` once by hand, then let `score_cells` run. |
| API returns `401 invalid or missing X-API-Key` | `API_KEYS` is set; send one of its values in the `X-API-Key` header. |
| API returns `429` | Rate limit reached on that route; wait a minute. |
| The assistant answers with a billing or rate-limit error | The Anthropic key has no credit, or switch to `LLM_PROVIDER=ollama`. |
| The assistant is very slow or times out | Local model on CPU. Use `OLLAMA_MODEL=qwen3:4b`, or raise `CHAT_TIMEOUT_S`. |
| `docker compose up` fails with "port already allocated" | Another stack uses the port. Set `API_PORT`, `WEB_PORT`, `POSTGRES_PORT`, `RUSTFS_S3_PORT` or `RUSTFS_CONSOLE_PORT` in `.env`. |

---

## Honest framing and data credits

Scores rank how similar a place is to where such businesses already operate, adjusted for competition. They are not revenue forecasts (survivor-bias assumption, see `corpus/methodology/`).

Map data © OpenStreetMap contributors (ODbL); census data © HCP; satellite data: Copernicus Sentinel-2, NOAA VIIRS, WorldPop, Dynamic World via Google Earth Engine. Third-party data keeps its own licence and is not covered by the notice below.

---

## License

**Copyright © 2026 dottachin. All rights reserved.**

GeoMarketingLab is proprietary software. The source code is public **only so it can be viewed** as part of a portfolio. It is **not** open source.

Without prior written permission, you may not:

- copy or download the code, in whole or in part
- use, run, deploy or host it
- modify it or build derivative works from it
- distribute, sublicense or sell it
- use it to train AI models or to build competing products

The name "GeoMarketingLab" and its logo are also not licensed for use. The examination jury and the host institution may read and run the project to evaluate it. Read the full terms in [LICENSE](LICENSE).

For licensing or other inquiries, contact me through [my GitHub profile](https://github.com/dottachin).
