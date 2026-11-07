# Airflow 3 + the geomlab package, so every task imports the same code as the CLI.
FROM apache/airflow:3.3.2-python3.12
USER airflow
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=airflow:root pyproject.toml README.md /opt/geomlab/
COPY --chown=airflow:root geomlab /opt/geomlab/geomlab
RUN pip install --no-cache-dir "/opt/geomlab[ingestion,ml,genai,dbt]"
