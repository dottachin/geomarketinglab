"""OpenStreetMap ingestion: Geofabrik Morocco extract + daily replication diffs (doc §3.1).

Change detection uses the replication sequence number published in Geofabrik's
``state.txt``:

* no base extract yet, or the base is older than ``FULL_REFRESH_DAYS`` → download the
  full ``morocco-latest.osm.pbf`` and store it in ``bronze/osm/``;
* otherwise, if the remote sequence number is ahead of ours → download only the missing
  ``.osc.gz`` diff files into bronze (a few hundred KB per day) and apply them to the
  local working copy with pyosmium;
* if the sequence number did not move → skip (nothing changed).

The working copy (``data/cache/osm/morocco.osm.pbf``) is a derived cache that can always
be rebuilt from bronze (full extract + diffs), so bronze stays the source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from geomlab.ingestion.http import download_to, fetch
from geomlab.lakehouse import Lake, get_lake
from geomlab.logging import get_logger
from geomlab.settings import get_settings

log = get_logger(__name__)
SOURCE = "osm"
FULL_REFRESH_DAYS = 7  # doc §3.1: "Daily diffs available; full extract weekly"


@dataclass(frozen=True)
class ReplicationState:
    sequence: int
    timestamp: str


def parse_state(text: str) -> ReplicationState:
    seq = re.search(r"sequenceNumber=(\d+)", text)
    ts = re.search(r"timestamp=(\S+)", text)
    if not seq:
        raise ValueError("state.txt has no sequenceNumber")
    return ReplicationState(int(seq.group(1)), (ts.group(1).replace("\\:", ":") if ts else ""))


def diff_url(base: str, seq: int) -> str:
    """Replication diff URL: sequence 4321 → ``000/004/321.osc.gz``."""
    s = f"{seq:09d}"
    return f"{base.rstrip('/')}/{s[0:3]}/{s[3:6]}/{s[6:9]}.osc.gz"


def working_copy() -> Path:
    return get_settings().data_dir / "cache" / "osm" / "morocco.osm.pbf"


def _local_sequence(lake: Lake) -> int | None:
    token = lake.last_token(SOURCE)
    return int(token) if token else None


def _last_full_day(lake: Lake) -> date | None:
    m = lake.manifest(SOURCE)
    m = m[m["object_key"].str.endswith(".osm.pbf")]
    return None if m.empty else date.fromisoformat(m.iloc[-1]["ingest_date"])


def ingest(lake: Lake | None = None, force_full: bool = False) -> dict:
    lake = lake or get_lake()
    s = get_settings()
    remote = parse_state(fetch(s.osm_state_url).decode())
    local_seq = _local_sequence(lake)
    last_full = _last_full_day(lake)
    wc = working_copy()

    needs_full = (
        force_full
        or local_seq is None
        or last_full is None
        or (date.today() - last_full) > timedelta(days=FULL_REFRESH_DAYS)
        or not wc.exists()
    )
    if needs_full:
        log.info("osm_full_download url=%s seq=%d", s.osm_pbf_url, remote.sequence)
        download_to(s.osm_pbf_url, wc)
        # Ask the extract itself which sequence it corresponds to (header), fall back to remote.
        seq = _pbf_sequence(wc) or remote.sequence
        lake.put_raw(SOURCE, "morocco-latest.osm.pbf", wc, url=s.osm_pbf_url, change_token=str(seq))
        return {"status": "full", "sequence": seq}

    if remote.sequence <= local_seq:
        log.info("osm_unchanged seq=%d", local_seq)
        return {"status": "unchanged", "sequence": local_seq}

    base = s.osm_state_url.rsplit("/", 1)[0]
    diffs: list[Path] = []
    for seq in range(local_seq + 1, remote.sequence + 1):
        url = diff_url(base, seq)
        data = fetch(url)
        lake.put_raw(SOURCE, f"diff-{seq:09d}.osc.gz", data, url=url, change_token=str(seq))
        p = wc.parent / f"diff-{seq:09d}.osc.gz"
        p.write_bytes(data)
        diffs.append(p)
    _apply_diffs(wc, diffs)
    for p in diffs:
        p.unlink(missing_ok=True)
    log.info("osm_diffs_applied from=%d to=%d", local_seq + 1, remote.sequence)
    return {"status": "diffs", "sequence": remote.sequence, "n_diffs": len(diffs)}


def _pbf_sequence(path: Path) -> int | None:
    import osmium

    header = osmium.io.Reader(str(path), osmium.osm.osm_entity_bits.NOTHING).header()
    value = header.get("osmosis_replication_sequence_number", "")
    return int(value) if value.isdigit() else None


def _apply_diffs(pbf: Path, diffs: list[Path]) -> None:
    """Merge .osc.gz change files into the working PBF (pyosmium MergeInputReader)."""
    import osmium

    reader = osmium.MergeInputReader()
    for d in diffs:
        reader.add_file(str(d))
    tmp = pbf.with_suffix(".new.pbf")
    tmp.unlink(missing_ok=True)
    writer = osmium.SimpleWriter(str(tmp))
    try:
        reader.apply_to_reader(osmium.io.Reader(str(pbf)), writer, with_history=False)
    finally:
        writer.close()
    tmp.replace(pbf)


def ensure_working_copy(lake: Lake | None = None) -> Path:
    """Rebuild the working PBF from bronze if the cache is missing (e.g. a fresh worker)."""
    wc = working_copy()
    if wc.exists():
        return wc
    lake = lake or get_lake()
    m = lake.manifest(SOURCE)
    fulls = m[m["object_key"].str.endswith(".osm.pbf")]
    if fulls.empty:
        raise FileNotFoundError("no OSM extract in bronze; run the OSM ingestion first")
    full = fulls.iloc[-1]
    wc.parent.mkdir(parents=True, exist_ok=True)
    wc.write_bytes(lake.get_raw(full["object_key"]))
    later = m[(m["ingested_at"] > full["ingested_at"]) & m["object_key"].str.contains("/diff-")]
    paths = []
    for _, row in later.iterrows():
        p = wc.parent / row["object_key"].rsplit("/", 1)[-1]
        p.write_bytes(lake.get_raw(row["object_key"]))
        paths.append(p)
    if paths:
        _apply_diffs(wc, paths)
    log.info("osm_working_copy_rebuilt diffs=%d at=%s", len(paths), datetime.now(UTC).isoformat())
    return wc
