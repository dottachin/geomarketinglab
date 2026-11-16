"""Cell naming: multilingual OSM name cleaning, neighbourhood tiers with radius fallback,
landmark scoring, and serving.cells loading when silver.cell_names is missing."""

from contextlib import contextmanager

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from geomlab.features.naming import clean_name, name_cells, pick_landmark, pick_neighbourhood

EPSG = 32629
LON, LAT = -7.6, 33.58
DEG_LAT_M = 111_000.0  # 1° of latitude ≈ 111 km: good enough to place points at a given distance


def _north(m: float) -> float:
    return LAT + m / DEG_LAT_M


def _cents() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"h3_id": ["a"]}, geometry=[Point(LON, LAT)], crs=4326)


# --- clean_name --------------------------------------------------------------------
@pytest.mark.parametrize(("name", "en", "fr", "expected"), [
    ("Guéliz ⴳⵉⵍⵉⵣ گليز", None, None, "Guéliz"),
    ("Takaddoum ⵜⴰⵇⴰⴷⴷⵓⵎ التقدم", None, None, "Takaddoum"),
    ("Kennaria", None, None, "Kennaria"),
    ("دوار الحاجة", "Douar Hajja", None, "Douar Hajja"),
    ("Hay Riad ⵃⴰⵢ حي الرياض", "Hay Riad (EN)", "Hay Riad (FR)", "Hay Riad (EN)"),
    ("Maârif المعاريف", None, "Maârif FR", "Maârif FR"),
    ("Carré Eden - كاري عدن", None, None, "Carré Eden"),
    ("مسجد الحسن الثاني (Hassan II)", None, None, "Hassan II"),
    ("سيدي مومن", None, None, "سيدي مومن"),  # Arabic only: kept as is
    ("Ain Diab", "عين الذياب", None, "Ain Diab"),  # name:en with no Latin falls through
    (None, None, None, None),
])
def test_clean_name(name, en, fr, expected):
    assert clean_name(name, en, fr) == expected


# --- neighbourhood -----------------------------------------------------------------
TIERS = [{"places": ["neighbourhood", "quarter"], "radius_m": 800.0}, {"places": ["suburb"], "radius_m": 2000.0}]


def _places(rows):
    return pd.DataFrame(rows, columns=["place", "name", "lon", "lat"])


def test_neighbourhood_prefers_fine_place_over_closer_suburb():
    places = _places([("suburb", "Big Suburb", LON, _north(100)),
                      ("neighbourhood", "Far Hood", LON, _north(700)),
                      ("quarter", "Near Quarter", LON, _north(500))])
    assert pick_neighbourhood(_cents(), places, TIERS, EPSG).iloc[0] == "Near Quarter"


def test_neighbourhood_falls_back_to_suburb_then_null():
    places = _places([("neighbourhood", "Too Far", LON, _north(1200)), ("suburb", "Suburb", LON, _north(1500))])
    assert pick_neighbourhood(_cents(), places, TIERS, EPSG).iloc[0] == "Suburb"
    places = _places([("neighbourhood", "Too Far", LON, _north(1200)), ("suburb", "Very Far", LON, _north(2500))])
    assert pick_neighbourhood(_cents(), places, TIERS, EPSG).iloc[0] is None
    assert pick_neighbourhood(_cents(), _places([]), TIERS, EPSG).iloc[0] is None


# --- landmark ----------------------------------------------------------------------
BONUS = {"mall": 250.0, "school": 0.0, "mosque": 50.0}


def _poi(rows):
    return pd.DataFrame(rows, columns=["category", "name", "name_en", "name_fr", "lon", "lat"])


def test_landmark_prominence_beats_small_distance_gap():
    poi = _poi([("school", "École A", None, None, LON, _north(100)),
                ("mall", "Carré Eden ⴽⴰⵔⵉ كاري عدن", None, None, LON, _north(300)),
                ("cafe", "Closest Café", None, None, LON, _north(10))])  # not a landmark category
    lm = pick_landmark(_cents(), poi, BONUS, 400.0, EPSG)
    assert lm.loc[0, "landmark"] == "Carré Eden"
    assert lm.loc[0, "landmark_m"] == pytest.approx(300, abs=5)


def test_landmark_needs_a_name_and_the_radius():
    poi = _poi([("mall", None, None, None, LON, _north(50)),       # unnamed: skipped
                ("school", "École B", None, None, LON, _north(150)),
                ("mall", "Far Mall", None, None, LON, _north(600))])  # outside 400 m
    lm = pick_landmark(_cents(), poi, BONUS, 400.0, EPSG)
    assert lm.loc[0, "landmark"] == "École B"
    empty = pick_landmark(_cents(), _poi([]), BONUS, 400.0, EPSG)
    assert empty.loc[0, "landmark"] is None and pd.isna(empty.loc[0, "landmark_m"])


def test_name_cells_one_row_per_cell_with_nulls():
    grid = pd.DataFrame({"city": ["x", "x"], "h3_id": ["a", "b"], "lon": [LON, LON + 0.2], "lat": [LAT, LAT]})
    cfg = {"tiers": TIERS, "landmark_radius_m": 400.0, "landmark_bonus_m": BONUS}
    out = name_cells(grid, _places([("quarter", "Q", LON, _north(200))]),
                     _poi([("mosque", "Mosquée", None, None, LON, _north(20))]), EPSG, cfg)
    assert list(out.columns) == ["city", "h3_id", "neighbourhood", "landmark", "landmark_m"]
    assert out.loc[0, "neighbourhood"] == "Q" and out.loc[0, "landmark"] == "Mosquée"
    assert pd.isna(out.loc[1, "neighbourhood"]) and pd.isna(out.loc[1, "landmark"])


# --- serving load -----------------------------------------------------------------
def _grid():
    return gpd.GeoDataFrame({"h3_id": ["a", "b"], "city": "x", "district": "D", "area_m2": 1.0,
                             "lon": [LON, LON], "lat": [LAT, LAT]},
                            geometry=[Point(LON, LAT), Point(LON, LAT)], crs=4326)


@pytest.fixture()
def captured(monkeypatch):
    from geomlab.serving import db

    seen = {}

    class _Con:
        def exec_driver_sql(self, sql):
            seen["sql"] = sql

    class _Engine:
        @contextmanager
        def begin(self):
            yield _Con()

    monkeypatch.setattr(db, "engine", lambda role="owner": _Engine())
    monkeypatch.setattr(db, "_copy_geo", lambda con, gdf, staging, cols: seen.update(frame=gdf[cols].copy()))
    return seen


def test_load_cells_without_cell_names_gives_nulls(captured):
    from geomlab.serving import db

    db.load_cells(_grid())
    frame = captured["frame"]
    assert frame["neighbourhood"].isna().all() and frame["landmark"].isna().all()
    assert "neighbourhood = EXCLUDED.neighbourhood" in captured["sql"]


def test_load_cells_joins_partial_cell_names(captured):
    from geomlab.serving import db

    names = pd.DataFrame({"city": ["x"], "h3_id": ["b"], "neighbourhood": ["Guéliz"], "landmark": [pd.NA],
                          "landmark_m": [float("nan")]}).astype({"landmark": "string"})
    db.load_cells(_grid(), names)
    frame = captured["frame"].set_index("h3_id")
    assert frame.loc["b", "neighbourhood"] == "Guéliz"
    assert frame.loc["a", "neighbourhood"] is None and frame.loc["b", "landmark"] is None
