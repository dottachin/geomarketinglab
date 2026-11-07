"""HCP ingestion (doc §3.1): scheduled download of the RGPH 2024 tables with checksum
monitoring. There is no public API, so files are downloaded from the URLs in
``config/hcp.yaml``; a file is stored in bronze only when its SHA-256 changed.

A manual-upload path is supported too: drop a file into ``data/manual/hcp/<filename>``
and the next run picks it up instead of downloading (useful if the HCP site blocks bots).
"""

from __future__ import annotations

import hashlib

from geomlab.config import hcp_config
from geomlab.ingestion.http import fetch
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger
from geomlab.settings import get_settings

log = get_logger(__name__)
SOURCE = "hcp"


def ingest(lake: Lake | None = None) -> dict[str, str]:
    lake = lake or get_lake()
    manual_dir = get_settings().data_dir / "manual" / "hcp"
    status: dict[str, str] = {}
    for name, spec in hcp_config()["files"].items():
        fname = spec["filename"]
        manual = manual_dir / fname
        try:
            data = manual.read_bytes() if manual.exists() else fetch(spec["url"])
        except Exception as exc:  # one failing file must not block the others
            log.error("hcp_download_failed file=%s error=%s", fname, exc)
            status[name] = "failed"
            continue
        digest = hashlib.sha256(data).hexdigest()
        if lake.last_token(SOURCE, prefix=fname) == digest:
            status[name] = "unchanged"
            continue
        lake.put_raw(SOURCE, fname, data, url=spec["url"], change_token=digest)
        status[name] = "changed"
    log.info("hcp_ingest status=%s", status)
    if all(v == "failed" for v in status.values()):
        raise RuntimeError("every HCP download failed")
    return status


def latest_files(lake: Lake | None = None) -> dict[str, bytes]:
    """Latest bronze version of each configured HCP file, keyed by config name."""
    lake = lake or get_lake()
    m = lake.manifest(SOURCE)
    out = {}
    for name, spec in hcp_config()["files"].items():
        rows = m[m["object_key"].str.endswith("/" + spec["filename"])]
        if not rows.empty:
            out[name] = lake.get_raw(rows.iloc[-1]["object_key"])
    return out
