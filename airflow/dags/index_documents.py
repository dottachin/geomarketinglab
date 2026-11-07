"""Daily: (re)index changed documents of the RAG corpus into pgvector."""

from datetime import datetime

from _common import DEFAULT_ARGS, DOCS
from airflow.sdk import dag, task


@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT_ARGS,
     tags=["geomlab", "genai"], doc_md=__doc__)
def index_documents():
    @task(outlets=[DOCS])
    def index() -> dict:
        from geomlab.genai.indexing import index_corpus
        from geomlab.serving.db import init_schema

        init_schema()
        return index_corpus()

    index()


index_documents()
