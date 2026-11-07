PY ?= .venv/bin/python
CITY ?= casablanca

.PHONY: install test lint up down pipeline api web mcp
install:
	python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -e ".[all]"
test:
	$(PY) -m pytest
lint:
	.venv/bin/ruff check geomlab tests airflow
up:
	cd infra && docker compose --env-file ../.env up -d --build
down:
	cd infra && docker compose --env-file ../.env down
pipeline:
	.venv/bin/geomlab run-all --city $(CITY)
api:
	.venv/bin/uvicorn geomlab.api.main:app --reload --port 8000
web:
	cd web && npm install && npm run dev
mcp:
	.venv/bin/geomlab-mcp
