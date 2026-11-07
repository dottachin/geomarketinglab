import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from geomlab import geo
from geomlab.features.hcp_silver import disaggregate


def _grid():
    cells = sorted(geo.polygon_to_cells(box(-7.60, 33.57, -7.58, 33.59), 9))
    g = geo.cells_to_gdf(cells)
    g["area_m2"] = geo.area_m2(g, 32629)
    return g


def test_extensive_totals_are_preserved_and_intensive_averaged():
    grid = _grid()
    west = box(-7.62, 33.56, -7.59, 33.60)
    east = box(-7.59, 33.56, -7.56, 33.60)
    units = gpd.GeoDataFrame({"geo_code": ["1", "2"], "population": [10000.0, 2000.0], "taux_chomage": [10.0, 30.0]},
                             geometry=[west, east], crs=4326)
    rng = np.random.default_rng(0)
    pop = pd.Series(rng.uniform(10, 1000, len(grid)), index=grid["h3_id"])
    out = disaggregate(units, grid, pop, ["population"], ["taux_chomage"], 32629)
    # cells fully inside the units' union → every unit's population is distributed exactly
    assert abs(out["hcp_population"].sum() - 12000.0) < 1e-6
    assert out["hcp_taux_chomage"].between(10 - 1e-9, 30 + 1e-9).all()
    assert (out["hcp_coverage"] > 0.99).all()


def test_area_weighting_when_no_worldpop():
    grid = _grid()
    unit = gpd.GeoDataFrame({"geo_code": ["1"], "population": [500.0]}, geometry=[box(-7.7, 33.5, -7.5, 33.7)], crs=4326)
    out = disaggregate(unit, grid, None, ["population"], [], 32629)
    share = out.set_index("h3_id")["hcp_population"] / 500.0
    expected = grid.set_index("h3_id")["area_m2"] / grid["area_m2"].sum()
    assert np.allclose(share.sort_index(), expected.sort_index(), rtol=1e-3)
