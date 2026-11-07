"""Lakehouse access: medallion layers on RustFS/S3 (or a local directory) with Delta Lake.

Layout (doc §4.2)::

    <LAKE_URI>/bronze/<source>/ingest_date=YYYY-MM-DD/<file>   raw files, immutable
    <LAKE_URI>/bronze/_manifest                                 Delta: one row per raw object
    <LAKE_URI>/silver/<table>                                   Delta: cleaned, EPSG:4326, deduplicated
    <LAKE_URI>/gold/<table>                                     Delta: H3 feature tables, labels, scores

Geometries are stored as WKB in a binary ``geometry`` column (Delta has no geometry type);
``read_geo`` decodes them back into a GeoDataFrame in EPSG:4326.

Delta time travel is how training datasets are versioned (doc §6.4 "DVC or Delta time
travel"): training logs the gold table version to MLflow and can re-read it exactly.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import TableNotFoundError
from shapely import from_wkb, to_wkb

from geomlab.geo import WGS84
from geomlab.logging import get_logger
from geomlab.settings import get_settings

log = get_logger(__name__)

LAYERS = ("bronze", "silver", "gold")
MANIFEST = "_manifest"


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    object_key: str
    url: str | None
    sha256: str
    size_bytes: int
    change_token: str
    ingest_date: str
    ingested_at: str


class Lake:
    def __init__(self, uri: str | None = None, storage_options: dict[str, str] | None = None):
        s = get_settings()
        self.uri = (uri or s.lake_uri).rstrip("/")
        self.storage_options = storage_options if storage_options is not None else s.storage_options()
        self.is_s3 = self.uri.startswith("s3://")
        if not self.is_s3:
            Path(self.uri).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def table_uri(self, layer: str, name: str) -> str:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}")
        return f"{self.uri}/{layer}/{name}"

    # ------------------------------------------------------------------ Delta tables
    def write(
        self,
        layer: str,
        name: str,
        df: pd.DataFrame,
        *,
        mode: str = "overwrite",
        partition_by: list[str] | None = None,
        predicate: str | None = None,
        description: str | None = None,
    ) -> int:
        """Write a (Geo)DataFrame as a Delta table; return the new table version.

        ``predicate`` + ``mode="overwrite"`` replaces only matching rows (e.g. one city),
        which keeps per-city tasks idempotent.
        """
        table = _to_arrow(df)
        if predicate is not None and not self.exists(layer, name):
            predicate = None  # first write creates the table
        write_deltalake(
            self.table_uri(layer, name),
            table,
            mode=mode,
            partition_by=partition_by,
            predicate=predicate,
            schema_mode="overwrite" if (mode == "overwrite" and predicate is None) else "merge",
            storage_options=self.storage_options,
            description=description,
        )
        version = self.version(layer, name)
        log.info("delta_write table=%s/%s rows=%d version=%s", layer, name, len(df), version)
        return version

    def read(
        self,
        layer: str,
        name: str,
        *,
        version: int | None = None,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> pd.DataFrame:
        dt = DeltaTable(self.table_uri(layer, name), version=version, storage_options=self.storage_options)
        return dt.to_pyarrow_table(columns=columns, filters=filters).to_pandas()

    def read_geo(self, layer: str, name: str, **kw) -> gpd.GeoDataFrame:
        df = self.read(layer, name, **kw)
        geom = from_wkb(df.pop("geometry").to_numpy())
        return gpd.GeoDataFrame(df, geometry=geom, crs=WGS84)

    def exists(self, layer: str, name: str) -> bool:
        try:
            DeltaTable(self.table_uri(layer, name), storage_options=self.storage_options)
            return True
        except TableNotFoundError:
            return False

    def version(self, layer: str, name: str) -> int:
        return DeltaTable(self.table_uri(layer, name), storage_options=self.storage_options).version()

    def history(self, layer: str, name: str, limit: int = 10) -> list[dict]:
        dt = DeltaTable(self.table_uri(layer, name), storage_options=self.storage_options)
        return dt.history(limit)

    # ------------------------------------------------------------------ bronze objects
    @lru_cache(maxsize=1)  # noqa: B019 - one filesystem per Lake instance
    def _fs(self) -> tuple[pafs.FileSystem, str]:
        if not self.is_s3:
            return pafs.LocalFileSystem(), self.uri
        o = self.storage_options
        endpoint = o.get("AWS_ENDPOINT_URL")
        fs = pafs.S3FileSystem(
            access_key=o.get("AWS_ACCESS_KEY_ID"),
            secret_key=o.get("AWS_SECRET_ACCESS_KEY"),
            region=o.get("AWS_REGION", "us-east-1"),
            endpoint_override=endpoint.split("://", 1)[-1] if endpoint else None,
            scheme="http" if (endpoint or "").startswith("http://") else "https",
            allow_bucket_creation=True,
        )
        return fs, self.uri.removeprefix("s3://")

    def put_raw(
        self,
        source: str,
        filename: str,
        payload: bytes | Path,
        *,
        url: str | None = None,
        change_token: str | None = None,
        ingest_day: date | None = None,
    ) -> ManifestEntry:
        """Store a raw file in bronze (never overwritten) and append it to the manifest."""
        data = payload.read_bytes() if isinstance(payload, Path) else payload
        digest = hashlib.sha256(data).hexdigest()
        day = (ingest_day or date.today()).isoformat()
        key = f"bronze/{source}/ingest_date={day}/{filename}"
        fs, root = self._fs()
        path = f"{root}/{key}"
        fs.create_dir(path.rsplit("/", 1)[0], recursive=True)
        try:
            if fs.get_file_info(path).type != pafs.FileType.NotFound:
                existing = self.manifest(source)
                if (existing["object_key"] == key).any():
                    log.info("bronze_exists key=%s (immutable, skipping write)", key)
                    row = existing[existing["object_key"] == key].iloc[-1]
                    return ManifestEntry(**{k: row[k] for k in ManifestEntry.__dataclass_fields__})
        except FileNotFoundError:
            pass
        with fs.open_output_stream(path) as out:
            out.write(data)
        entry = ManifestEntry(
            source=source,
            object_key=key,
            url=url,
            sha256=digest,
            size_bytes=len(data),
            change_token=change_token or digest,
            ingest_date=day,
            ingested_at=datetime.now(UTC).isoformat(),
        )
        self.write("bronze", MANIFEST, pd.DataFrame([entry.__dict__]), mode="append")
        return entry

    def get_raw(self, object_key: str) -> bytes:
        fs, root = self._fs()
        with fs.open_input_stream(f"{root}/{object_key}") as fh:
            return fh.read()

    def manifest(self, source: str | None = None) -> pd.DataFrame:
        if not self.exists("bronze", MANIFEST):
            return pd.DataFrame(columns=list(ManifestEntry.__dataclass_fields__))
        df = self.read("bronze", MANIFEST)
        if source is not None:
            df = df[df["source"] == source]
        return df.sort_values("ingested_at")

    def last_token(self, source: str, prefix: str = "") -> str | None:
        """Latest change token recorded for objects of ``source`` whose filename starts with
        ``prefix`` — the basis of every DAG's "skip when nothing changed" check."""
        m = self.manifest(source)
        if prefix:
            m = m[m["object_key"].str.rsplit("/", n=1).str[-1].str.startswith(prefix)]
        return None if m.empty else str(m.iloc[-1]["change_token"])

    def latest_objects(self, source: str, prefix: str = "") -> pd.DataFrame:
        """Manifest rows of the most recent ingest_date for ``source`` (optionally filtered)."""
        m = self.manifest(source)
        if prefix:
            m = m[m["object_key"].str.rsplit("/", n=1).str[-1].str.startswith(prefix)]
        if m.empty:
            return m
        return m[m["ingest_date"] == m["ingest_date"].max()]


def _to_arrow(df: pd.DataFrame) -> pa.Table:
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.to_crs(WGS84) if df.crs is not None else df
        geom_name = gdf.geometry.name
        plain = pd.DataFrame(gdf.drop(columns=[geom_name]))
        plain["geometry"] = to_wkb(gdf.geometry.to_numpy())
        df = plain
    return pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)


def read_bytes_table(data: bytes) -> pd.DataFrame:
    """Parse a parquet blob stored in bronze."""
    return pd.read_parquet(io.BytesIO(data))


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


@lru_cache(maxsize=1)
def get_lake() -> Lake:
    return Lake()
