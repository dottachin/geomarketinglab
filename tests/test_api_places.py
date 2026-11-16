"""Place names on hexagons (neighbourhood, landmark, n_cells): API routes with a monkeypatched
core (no database) and the column-tolerant SELECT fragment."""

import pytest
from fastapi.testclient import TestClient

H3 = "89390cb1b4bffff"


def _cell(neighbourhood, landmark):
    return {"h3_id": H3, "district": "Arrondissement de Guéliz", "neighbourhood": neighbourhood,
            "landmark": landmark, "final_score": 0.81, "suitability": 0.9, "saturation": 0.1, "rank": 2,
            "n_existing": 0, "competitors_catchment": 3, "pop_catchment": 12000.0, "opportunity": True,
            "low_confidence": False}


@pytest.fixture()
def client(env):
    from geomlab.api import main

    return TestClient(main.app), main


@pytest.mark.parametrize(("neighbourhood", "landmark"), [("Guéliz", "Carré Eden"), (None, None)])
def test_map_cells_carry_place_names(client, monkeypatch, neighbourhood, landmark):
    c, main = client
    monkeypatch.setattr(main.core, "map_cells", lambda city, bt: {
        "city": city, "business_type": bt, "model_version": "m:v1", "data_date": "2026-11-01",
        "cells": [_cell(neighbourhood, landmark)], "competitors": []})
    r = c.get("/map/marrakech/cafe")
    assert r.status_code == 200
    cell = r.json()["cells"][0]
    assert cell["neighbourhood"] == neighbourhood and cell["landmark"] == landmark
    assert cell["district"] == "Arrondissement de Guéliz"


@pytest.mark.parametrize(("neighbourhood", "landmark"), [("Guéliz", "Carré Eden"), (None, None)])
def test_explain_carries_place_names_and_n_cells(client, monkeypatch, neighbourhood, landmark):
    c, main = client
    monkeypatch.setattr(main.core, "explain_location", lambda h3_id, bt: {
        "evidence": "model", "h3_id": h3_id, "business_type": bt, "city": "marrakech",
        "district": "Arrondissement de Guéliz", "neighbourhood": neighbourhood, "landmark": landmark,
        "city_rank": 2, "n_cells": 1888,
        "neighbouring_cells": [{"h3_id": "89390cb1b43ffff", "neighbourhood": neighbourhood,
                                "final_score": 0.7, "city_rank": 9, "n_existing": 1}],
        "model_version": "m:v1", "data_date": "2026-11-01"})
    r = c.get(f"/cells/{H3}/explain", params={"business_type": "cafe"})
    assert r.status_code == 200
    body = r.json()
    assert body["neighbourhood"] == neighbourhood and body["landmark"] == landmark
    assert body["n_cells"] == 1888 and body["city_rank"] == 2
    assert body["neighbouring_cells"][0]["neighbourhood"] == neighbourhood


def test_map_unknown_city_is_422_with_valid_values(client):
    c, _ = client
    r = c.get("/map/paris/cafe")
    assert r.status_code == 422
    assert r.json()["field"] == "city" and "marrakech" in r.json()["valid_values"]


def test_explain_without_scores_is_503(client, monkeypatch):
    c, main = client

    def not_ready(*_a, **_k):
        raise main.core.NotReadyError("no model has been scored yet")

    monkeypatch.setattr(main.core, "explain_location", not_ready)
    monkeypatch.setattr(main.core, "map_cells", not_ready)
    assert c.get(f"/cells/{H3}/explain", params={"business_type": "cafe"}).status_code == 503
    assert c.get("/map/marrakech/cafe").status_code == 503


class _Con:
    def __init__(self, present):
        self.present, self.params = present, None

    def execute(self, _sql, params):
        self.params = params
        return [(c,) for c in params["cols"] if c in self.present]


def test_label_cols_selects_existing_columns():
    from geomlab.tools import core

    con = _Con({"neighbourhood", "landmark"})
    assert core._label_cols(con) == "c.neighbourhood, c.landmark"
    assert con.params == {"cols": ["neighbourhood", "landmark"]}


def test_label_cols_falls_back_to_nulls_on_an_older_schema():
    from geomlab.tools import core

    assert core._label_cols(_Con(set())) == "NULL::text AS neighbourhood, NULL::text AS landmark"
    assert core._label_cols(_Con({"neighbourhood"}), cols=("neighbourhood",)) == "c.neighbourhood"


def test_prompt_names_places_by_neighbourhood_and_landmark(env):
    from geomlab.genai.prompts import system_prompt

    p = system_prompt()
    assert "neighbourhood" in p and "near <landmark>" in p and "Never lead with an h3_id" in p
