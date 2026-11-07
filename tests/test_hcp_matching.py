import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from geomlab.features.hcp_silver import leaf_key, match_units


def test_leaf_key_handles_osm_and_hcp_spellings():
    assert leaf_key("Arrondissement Agdal-Riyad") == leaf_key("Arrondissement d'Agdal Riyad") == "agdal riyad"
    assert leaf_key("Arrondissement de Marrakech-Médina") == leaf_key("Arrondissement de Marrakech-Medina")


def test_match_prefers_leaf_units_and_aliases():
    units = pd.DataFrame({"geo_code": ["4421010", "44210105", "61410199"],
                          "name_fr": ["Commune de Rabat", "Arrondissement de Hassan", "Arrondissement d'Assoukhour Assawda"],
                          "population": [1, 2, 3]})
    admin = gpd.GeoDataFrame({"unit_name": ["Arrondissement Hassan", "Arrondissement des Roches noires", "Bettana"]},
                             geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)], crs=4326)
    m = match_units(units, admin, {"assoukhour assawda": "Arrondissement des Roches noires"})
    assert dict(zip(m["unit_name"], m["geo_code"], strict=True)) == {
        "Arrondissement Hassan": "44210105", "Arrondissement des Roches noires": "61410199"}
