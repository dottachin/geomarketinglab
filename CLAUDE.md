# GeoMarketingLab (pfe-msde) — guide for Claude Code

Implementation of `../GeoMarketingLab_Architecture.docx`. Map of doc sections → code: `docs/architecture.md`.

## Commands
- `.venv/bin/python -m pytest` — unit tests + offline end-to-end run on the synthetic city `testville`
- `.venv/bin/ruff check geomlab tests airflow`
- `.venv/bin/geomlab --help` — every pipeline step (the Airflow DAGs call the same functions)
- Stack: `cd infra && docker compose --env-file ../.env up -d` (RustFS, PostGIS+pgvector, Airflow 3, MLflow, API, MCP, Martin, web)

## Rules that matter
1. Store geometries in EPSG:4326; measure only in the city's projected CRS via `geomlab/geo.py`.
2. Competition columns (`comp_*`, `n_in_cell`) are never model inputs — saturation penalty only (`model_feature_columns`, dbt test).
3. Spatial CV only (H3 parent blocks + buffer); never random splits.
4. Bronze is immutable; every ingestion step has change detection and must be idempotent per city.
5. The LLM never produces numbers: they come from `geomlab/tools/core.py`, stamped with `model_version` and `data_date`.
6. Config over code: cities, business types, POI taxonomy, features, model settings live in `config/`.
7. Secrets only in `.env` (gitignored). Never print or commit them.
