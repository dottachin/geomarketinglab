# Python image for the API, MCP server, MLflow server and one-off jobs.
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY geomlab ./geomlab
RUN pip install ".[ingestion,ml,genai,api,dbt]"
COPY config ./config
COPY dbt ./dbt
RUN useradd -m -u 10001 geomlab && mkdir -p /app/data && chown -R geomlab /app/data
USER geomlab
EXPOSE 8000
CMD ["uvicorn", "geomlab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
