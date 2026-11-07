"""Typed access to the YAML configuration in ``config/`` (config over code).

These objects are also the allow-lists the LLM tools validate against and expose as
MCP resources (doc §7.4.2 "Validate inputs").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from geomlab.settings import get_settings


def _read_yaml(name: str) -> dict[str, Any]:
    path: Path = get_settings().config_dir / name
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


@dataclass(frozen=True)
class City:
    slug: str
    display_name: str
    osm_query: str | None
    bbox: tuple[float, float, float, float] | None  # south, west, north, east
    crs_projected: int
    h3_resolution: int


@dataclass(frozen=True)
class BusinessType:
    slug: str
    display_name: dict[str, str]
    catchment_m: float
    residents_per_business: float
    saturation_lambda: float

    def label(self, lang: str = "en") -> str:
        return self.display_name.get(lang) or self.display_name.get("en") or self.slug


@dataclass(frozen=True)
class Selector:
    """All (key, value) pairs must match; value ``*`` matches any non-empty value."""

    tags: tuple[tuple[str, str], ...]

    def matches(self, tags: dict[str, str]) -> bool:
        for key, value in self.tags:
            got = tags.get(key)
            if got is None or got == "no":
                return False
            if value != "*" and got != value:
                return False
        return True


@dataclass(frozen=True)
class PoiTaxonomy:
    business: dict[str, list[Selector]]
    attractor: dict[str, list[Selector]]
    rings_m: tuple[int, ...]
    _order: list[tuple[str, str, list[Selector]]] = field(default_factory=list, repr=False)

    def classify(self, tags: dict[str, str]) -> tuple[str, str] | None:
        """Return ``(role, category)`` for an OSM tag dict, or None. First match wins;
        business categories are checked first so a pharmacy is never an attractor."""
        for role, cat, sels in self._order:
            if any(s.matches(tags) for s in sels):
                return role, cat
        return None

    @property
    def keys(self) -> set[str]:
        """OSM keys that matter (used to pre-filter elements while reading the PBF)."""
        out: set[str] = set()
        for _, _, sels in self._order:
            for s in sels:
                out.update(k for k, _ in s.tags)
        return out


def _parse_selectors(items: list[dict[str, Any]]) -> list[Selector]:
    return [Selector(tuple((str(k), str(v)) for k, v in item.items())) for item in items]


@lru_cache(maxsize=1)
def cities() -> dict[str, City]:
    out = {}
    for slug, c in _read_yaml("cities.yaml").items():
        bbox = c.get("bbox")
        out[slug] = City(
            slug=slug,
            display_name=c.get("display_name", slug.title()),
            osm_query=c.get("osm_query"),
            bbox=tuple(float(x) for x in bbox) if bbox else None,
            crs_projected=int(c.get("crs_projected", 32629)),
            h3_resolution=int(c.get("h3_resolution", 9)),
        )
    return out


@lru_cache(maxsize=1)
def poi_taxonomy() -> PoiTaxonomy:
    raw = _read_yaml("poi_categories.yaml")
    business = {k: _parse_selectors(v) for k, v in (raw.get("business") or {}).items()}
    attractor = {k: _parse_selectors(v) for k, v in (raw.get("attractor") or {}).items()}
    overlap = set(business) & set(attractor)
    if overlap:
        raise ValueError(f"categories cannot be both business and attractor: {overlap}")
    order = [("business", k, v) for k, v in business.items()] + [
        ("attractor", k, v) for k, v in attractor.items()
    ]
    return PoiTaxonomy(
        business=business,
        attractor=attractor,
        rings_m=tuple(int(r) for r in raw.get("rings_m", [500, 1000])),
        _order=order,
    )


@lru_cache(maxsize=1)
def business_types() -> dict[str, BusinessType]:
    raw = _read_yaml("business_types.yaml")
    taxonomy = poi_taxonomy()
    out = {}
    for slug, b in raw.items():
        if slug not in taxonomy.business:
            raise ValueError(f"business type '{slug}' has no OSM selectors in poi_categories.yaml")
        out[slug] = BusinessType(
            slug=slug,
            display_name=dict(b.get("display_name") or {"en": slug}),
            catchment_m=float(b.get("catchment_m", 500)),
            residents_per_business=float(b.get("residents_per_business", 3000)),
            saturation_lambda=float(b.get("saturation_lambda", 0.5)),
        )
    return out


@lru_cache(maxsize=1)
def model_config() -> dict[str, Any]:
    return _read_yaml("model.yaml")


@lru_cache(maxsize=1)
def hcp_config() -> dict[str, Any]:
    return _read_yaml("hcp.yaml")


def get_city(slug: str) -> City:
    try:
        return cities()[slug]
    except KeyError:
        raise KeyError(f"unknown city '{slug}'; valid: {sorted(cities())}") from None


def get_business_type(slug: str) -> BusinessType:
    try:
        return business_types()[slug]
    except KeyError:
        raise KeyError(f"unknown business type '{slug}'; valid: {sorted(business_types())}") from None


def business_type_index() -> dict[str, int]:
    """Stable integer id per business type (embedding index). Sorted for determinism."""
    return {slug: i for i, slug in enumerate(sorted(business_types()))}
