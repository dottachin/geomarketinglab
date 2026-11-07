# Architecture document → implementation map

Section numbers refer to `GeoMarketingLab_Architecture.docx`.

## §2 Layers

| # | Layer (doc) | Implemented with | Where |
|---|---|---|---|
| 1 | Sources | GEE (Sentinel-2 SR, Dynamic World, VIIRS DNB monthly, WorldPop), OSM (Geofabrik Morocco + replication diffs, Nominatim/Overpass for boundaries), HCP RGPH 2024 XLSX | `geomlab/ingestion/` |
| 2 | Ingestion & orchestration | Airflow 3 (TaskFlow, assets), one DAG per source, retries with exponential backoff, Slack alert callback | `airflow/dags/` |
| 3 | Lakehouse | **RustFS** (S3 API) + Delta Lake (`deltalake`), Parquet/WKB geometries, DuckDB | `geomlab/lakehouse.py` |
| 4 | Processing & features | GeoPandas/Shapely/h3/pyosmium (silver aggregation) + dbt-duckdb (gold, tests, docs) + Python DQ gate | `geomlab/features/`, `dbt/`, `geomlab/quality/` |
| 5 | ML & GenAI | LightGBM, PyTorch, PyTorch Geometric, MLflow registry; bge-m3 embeddings, pgvector | `geomlab/ml/`, `geomlab/genai/` |
| 6 | Serving | FastAPI + MCP (FastMCP), PostgreSQL/PostGIS/pgvector, Martin vector tiles, React + MapLibre + deck.gl | `geomlab/api/`, `geomlab/mcp_server.py`, `web/`, `infra/martin.yaml` |
| 7 | Platform & ops | Docker Compose, GitHub Actions, dbt tests + DQ checks, structured logs, PSI drift | `infra/`, `.github/workflows/ci.yml` |

## §2.1 Data flow → DAGs

| Step | DAG / task | Asset emitted |
|---|---|---|
| 1 daily pull of new/changed data | `ingest_osm`, `ingest_gee`, `ingest_hcp` (+ weekly `ingest_boundaries`) | `silver/osm`, `silver/gee_cells`, `silver/hcp_units`, `silver/grid` — only when something changed |
| 2 raw → bronze | `Lake.put_raw` (immutable, `ingest_date=` partition, `_manifest` Delta table with URL, SHA-256, change token) | |
| 3 silver | `features/grid.py`, `osm_silver.py`, `gee_silver.py`, `hcp_silver.py` | |
| 4 gold | `build_features`: per-city `families.py` → `dbt build` → quality gate → `publish_gold` | `gold/cell_features` |
| 5 training | `train_model` (weekly): spatial CV, MLflow, registry alias `champion` | `mlflow/...@champion` |
| 6 batch inference | `score_cells` (on new gold or new model): drift check, scores + SHAP + saturation → PostGIS + `gold/scores` | `postgis/serving.scores` |
| 7 serving | FastAPI + MCP read PostGIS; `index_documents` keeps pgvector current | |

**Change detection (§3.1 remark).** OSM: Geofabrik `state.txt` sequence number (daily diffs, full extract weekly).
GEE: token `<latest VIIRS month>|<WorldPop year>|<S2 window month>`. HCP: SHA-256 per file (manual upload
folder supported). Boundaries: content hash. Unchanged → `AirflowSkipException` → no asset event → nothing downstream runs.

## §3.2 H3 grid
Resolution 9, clipped to the city boundary, slivers < 5 % dropped, neighbour table up to k = 2.
HCP disaggregation: extensive variables population-weighted by WorldPop (area fallback), intensive
variables area-weighted (`features/hcp_silver.py::disaggregate`, unit-tested for mass preservation).

## §4 Lakehouse

| Zone | Content | Rules enforced |
|---|---|---|
| bronze | raw PBF + `.osc.gz` diffs, GEE reduceRegions output, HCP XLSX, boundary GeoJSON | never overwritten (`put_raw` refuses), manifest row per object |
| silver | `grid`, `h3_neighbors`, `admin_units`, `poi`, `roads`, `road_nodes`, `landuse`, `buildings`, `gee_cells`, `hcp_units`, `hcp_cells`, `cell_*` families | EPSG:4326 storage, EPSG:32629 for metric work, dedup (OSM node/outline duplicates < 15 m), per-city predicate overwrite |
| gold | `cell_features`, `training_set`, `scores`, `shap_drivers` | dbt tests + DQ gate before publish; Delta versions logged to MLflow |

Catalog: the Delta log itself plus the dbt sources file (a Unity/Nessie catalog would be the next step).

## §5.2 Feature families
All seven families are built (`config/features.yaml` documents every column): demand, economic activity,
competition (**penalty only, never a model input** — enforced in `model_feature_columns` and by a dbt
test), attractors (15 categories × 500 m/1 km rings), accessibility (road length by class, intersection
density, distance to main road, 10-minute **network** walking reach), environment (NDVI, land-use entropy,
distance to centre), spatial context (k ≤ 1 and k ≤ 2 neighbour means, in dbt).

## §6 Model
* Formulation: presence prediction per (cell, business type), candidate (urban) cells only.
* Models: LightGBM per type; MLP with type embedding (multi-task); GraphSAGE encoder on the H3 adjacency
  graph + the same type-embedding head (clean ablation tabular → graph). Multimodal CNN is a stretch goal
  (not implemented; pfe-old's patch extractor can be ported).
* Evaluation: GroupKFold on H3 res-7 parents with a 1-ring buffer; AUC-PR, ROC-AUC, Precision@k, NDCG@k,
  Brier, calibration curve on OOF predictions; champion = best mean AUC-PR.
* Calibration: isotonic per type on out-of-fold scores.
* Explainability: TreeSHAP for LightGBM, Expected Gradients (SHAP approximation) for MLP/GNN.
* MLOps: MLflow tracking + registry (alias `champion`), Delta time travel for dataset versions,
  weekly retraining DAG, PSI/KS drift logged at every scoring run.

## §7 LLM / RAG
* Tools exactly as §7.3 plus `simulate_location` (mode B live inference, §7.4).
* §7.4.2 rules implemented in code: numbers only from tools; drivers returned with scores;
  `model_version` + `data_date` on every result; `low_confidence` + reasons; model vs document
  evidence in separate shapes and separate citation lists; allow-list validation with valid values
  returned to the LLM.
* RAG: bge-m3 (1024-d) → pgvector HNSW; hybrid PostgreSQL full-text + vector with RRF; optional
  bge-reranker; metadata filters (city, type, year); low-similarity fallback flag.
* Agent: one tool-calling loop; Claude (`claude-opus-5-5`, server-side refusal fallback) or Ollama.
* MCP: tools, resources (cities, business types, feature dictionary), prompt (site-analysis report),
  stdio and bearer-protected streamable HTTP.

## Deliberate deviations (to defend or revisit)

| Doc says | Done | Why |
|---|---|---|
| MinIO | RustFS | requested; S3-compatible, MinIO community images are no longer published |
| Great Expectations | dbt tests + `geomlab/quality/checks.py` | same checks (nulls, geometry validity, coverage, row drift) without a heavy dependency |
| Evidently AI | PSI + KS in `ml/monitoring.py`, logged to MLflow | small, testable; Evidently can be added for HTML reports |
| LlamaIndex / LangGraph | hand-written loop | one loop serves Claude and Ollama; fewer moving parts to defend |
| OpenRouteService isochrones | network walking reach with SciPy Dijkstra on OSM ways | no API key / quota; same idea |
| DVC | Delta time travel | the doc lists either |
| Keycloak / JWT | optional API keys + rate limiting | doc marks auth optional |
| Prometheus + Grafana | not included | doc marks optional |
