#!/bin/bash
# Runs once on first start of the Postgres volume. Passwords come from the environment (.env).
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
  CREATE EXTENSION IF NOT EXISTS postgis;
  CREATE EXTENSION IF NOT EXISTS vector;
  -- tool layer: read-only on serving / rag / gold / analytics (grants applied by geomlab db-init)
  CREATE ROLE geomlab_ro LOGIN PASSWORD '${GEOMLAB_RO_PASSWORD}';
  -- query_stats: read-only on analytics views only
  CREATE ROLE geomlab_sql LOGIN PASSWORD '${GEOMLAB_SQL_PASSWORD}';
  ALTER ROLE geomlab_ro SET statement_timeout = '5s';
  ALTER ROLE geomlab_sql SET statement_timeout = '5s';
  ALTER ROLE geomlab_sql SET default_transaction_read_only = on;
  REVOKE CREATE ON SCHEMA public FROM PUBLIC;
  CREATE DATABASE airflow;
  CREATE DATABASE mlflow;
SQL
