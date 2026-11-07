import geopandas as gpd
import h3
import pytest
from shapely.geometry import Point, box

from geomlab import geo


def test_polygon_to_cells_and_neighbors():
    cells = geo.polygon_to_cells(box(-7.60, 33.57, -7.58, 33.59), 9)
    assert len(cells) > 10 and all(h3.get_resolution(c) == 9 for c in cells)
    pairs = geo.neighbor_pairs(cells, 2)
    assert set(pairs["distance"]) <= {1, 2}
    assert set(pairs["neighbor_id"]) <= cells


def test_metric_operations_are_in_meters():
    # two points ~1 km apart along a meridian (0.009° lat ≈ 1000 m)
    a = gpd.GeoDataFrame(geometry=[Point(-7.6, 33.5)], crs=4326)
    b = gpd.GeoDataFrame(geometry=[Point(-7.6, 33.509)], crs=4326)
    d = geo.nearest_distance_m(a, b, 32629).iloc[0]
    assert 990 < d < 1010
    assert geo.count_within_m(a, b, 1100, 32629).iloc[0] == 1
    assert geo.count_within_m(a, b, 900, 32629).iloc[0] == 0


def test_taxonomy_business_before_attractor(env):
    from geomlab.config import poi_taxonomy

    t = poi_taxonomy()
    assert t.classify({"amenity": "pharmacy"}) == ("business", "pharmacy")
    assert t.classify({"healthcare": "pharmacy"}) == ("business", "pharmacy")  # not 'clinic'
    assert t.classify({"shop": "bakery"}) == ("business", "bakery")
    assert t.classify({"shop": "clothes"}) == ("attractor", "retail_other")
    assert t.classify({"amenity": "place_of_worship", "religion": "muslim"}) == ("attractor", "mosque")
    assert t.classify({"amenity": "place_of_worship"}) is None
    assert t.classify({"office": "no"}) is None


def test_business_types_need_selectors(env):
    import yaml

    from geomlab import config

    p = env / "config" / "business_types.yaml"
    data = yaml.safe_load(p.read_text())
    data["florist"] = {"display_name": {"en": "Florist"}}
    p.write_text(yaml.safe_dump(data))
    config.business_types.cache_clear()
    with pytest.raises(ValueError, match="florist"):
        config.business_types()
