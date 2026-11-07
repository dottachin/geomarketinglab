import numpy as np
import pandas as pd

from geomlab.features import gee_silver, hcp_silver
from geomlab.ingestion import osm


def test_replication_state_and_diff_url():
    st = osm.parse_state("#comment\nsequenceNumber=4321\ntimestamp=2026-11-06T20\\:21\\:02Z\n")
    assert st.sequence == 4321 and st.timestamp == "2026-11-06T20:21:02Z"
    assert osm.diff_url("https://x/morocco-updates/", 4321) == "https://x/morocco-updates/000/004/321.osc.gz"


def test_put_raw_is_immutable_and_tracks_tokens(env):
    from geomlab.lakehouse import get_lake

    lake = get_lake()
    e1 = lake.put_raw("hcp", "pop.xlsx", b"v1", change_token="t1")
    e2 = lake.put_raw("hcp", "pop.xlsx", b"v2-same-day", change_token="t2")
    assert e1.object_key == e2.object_key and lake.get_raw(e1.object_key) == b"v1"
    assert lake.last_token("hcp", prefix="pop.xlsx") == "t1"


def test_geo_code_and_name_normalisation():
    assert hcp_silver.normalise_geo_code(61410101.0) == "61410101"
    assert hcp_silver.normalise_geo_code("06.141.01.01.", strip_leading_zero=True) == "61410101"
    assert hcp_silver.normalise_geo_code("Total") is None
    assert hcp_silver.leaf_key("Arrondissement d’Aïn-Chock") == "ain chock"
    assert hcp_silver.leaf_key("Arrondissement du Maârif") == "maarif"


def test_nightlight_trend_slope_per_year():
    months = pd.DataFrame([[1, 2, 3, 4], [5, 5, 5, np.nan]],
                          columns=["nl_2026_01", "nl_2026_02", "nl_2026_03", "nl_2026_04"])
    slope = gee_silver.nightlight_trend(months)
    assert abs(slope.iloc[0] - 12.0) < 1e-6  # +1 per month = +12 per year
    assert abs(slope.iloc[1]) < 1e-9
