"""HTTP download with retries and exponential backoff (doc §3.3 good practices)."""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from geomlab.logging import get_logger
from geomlab.settings import get_settings

log = get_logger(__name__)


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": get_settings().http_user_agent},
    )


def fetch(url: str, *, retries: int = 4, timeout: float = 120.0) -> bytes:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with _client(timeout) as c:
                r = c.get(url)
                r.raise_for_status()
                return r.content
        except (httpx.HTTPError, OSError) as exc:
            if attempt == retries:
                raise
            log.warning("fetch_retry url=%s attempt=%d error=%s", url, attempt, exc)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def download_to(url: str, dest: Path, *, timeout: float = 1800.0) -> Path:
    """Stream a large file (e.g. the Geofabrik PBF) to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _client(timeout) as c, c.stream("GET", url) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest
