"""Runtime settings, read from the environment / ``.env`` only (doc §9: secrets via .env).

Nothing here is hardcoded per environment: the same code runs on a laptop (local lake
directory, DuckDB file) and in docker-compose (RustFS, PostGIS, MLflow) by switching
environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- paths ---------------------------------------------------------------------
    config_dir: Path = PROJECT_ROOT / "config"
    data_dir: Path = PROJECT_ROOT / "data"
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    dbt_dir: Path = PROJECT_ROOT / "dbt"

    # --- lakehouse (doc §4) --------------------------------------------------------
    # Local path ("./data/lake") for prototyping, "s3://lakehouse" for RustFS (S3-compatible object storage).
    lake_uri: str = str(PROJECT_ROOT / "data" / "lake")
    s3_endpoint_url: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    # DuckDB file holding the dbt-built silver views and gold tables.
    warehouse_duckdb: Path = PROJECT_ROOT / "data" / "warehouse.duckdb"

    # --- serving database (PostGIS + pgvector) -------------------------------------
    database_url: str = "postgresql+psycopg://geomlab:geomlab@localhost:5432/geomlab"
    # Read-only role used by the LLM tools (doc §9: read-only DB roles for the LLM).
    database_url_ro: str = "postgresql+psycopg://geomlab_ro:geomlab_ro@localhost:5432/geomlab"
    # Even narrower role for query_stats: SELECT on analytics.* views only.
    database_url_sql: str = "postgresql+psycopg://geomlab_sql:geomlab_sql@localhost:5432/geomlab"

    # --- MLOps -------------------------------------------------------------------
    mlflow_tracking_uri: str = "sqlite:///" + str(PROJECT_ROOT / "data" / "mlflow.db")

    # --- sources -----------------------------------------------------------------
    earthengine_project: str | None = None
    gee_service_account: str | None = None
    gee_service_account_key_path: str | None = None
    osm_pbf_url: str = "https://download.geofabrik.de/africa/morocco-latest.osm.pbf"
    osm_state_url: str = "https://download.geofabrik.de/africa/morocco-updates/state.txt"
    http_user_agent: str = "GeoMarketingLab/0.1 (academic project)"

    # --- GenAI -------------------------------------------------------------------
    llm_provider: str = Field("anthropic", description="anthropic | ollama")
    llm_model: str = "claude-opus-5-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    embedding_backend: str = Field("sentence-transformers", description="sentence-transformers | ollama | hash")
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str | None = None  # e.g. "BAAI/bge-reranker-v2-m3"; None disables reranking
    rag_min_similarity: float = 0.35  # below this → low-confidence fallback

    # --- API ---------------------------------------------------------------------
    api_keys: str = ""  # comma-separated; empty disables auth (local dev)
    rate_limit: str = "60/minute"
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    mcp_auth_token: str | None = None  # bearer token for streamable-HTTP MCP transport

    random_seed: int = 42
    log_level: str = "INFO"

    @property
    def lake_is_s3(self) -> bool:
        return self.lake_uri.startswith("s3://")

    def storage_options(self) -> dict[str, str]:
        """deltalake / object_store options (empty for a local lake)."""
        if not self.lake_is_s3:
            return {}
        opts = {
            "AWS_REGION": self.aws_region,
            # Single-writer pipeline (Airflow runs one task per table at a time), so the
            # unsafe-rename mode is acceptable; documented in docs/lakehouse.md.
            "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        }
        if self.s3_endpoint_url:
            opts["AWS_ENDPOINT_URL"] = self.s3_endpoint_url
            opts["AWS_ALLOW_HTTP"] = "true" if self.s3_endpoint_url.startswith("http://") else "false"
        if self.aws_access_key_id:
            opts["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            opts["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        return opts


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
