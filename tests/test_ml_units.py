import numpy as np
import pandas as pd
import pytest

from geomlab.ml.evaluation import ndcg_at_k, precision_at_k, spatial_blocks, spatial_folds
from geomlab.ml.monitoring import psi
from geomlab.ml.scoring import saturation


def _rows():
    import h3
    from shapely.geometry import box

    from geomlab import geo

    cells = sorted(geo.polygon_to_cells(box(-7.66, 33.52, -7.52, 33.64), 9))
    return pd.DataFrame({"city": "c", "h3_id": cells, "business_type": "cafe",
                         "label": np.arange(len(cells)) % 4 == 0}).astype({"label": int}), h3


def test_spatial_folds_are_block_disjoint_with_buffer():
    rows, h3 = _rows()
    blocks = spatial_blocks(rows, 7)
    for tr, te in spatial_folds(rows, 4, 7, buffer_rings=1):
        assert not set(blocks.iloc[tr]) & set(blocks.iloc[te])
        test_cells = set(rows["h3_id"].iloc[te])
        for c in rows["h3_id"].iloc[tr]:
            assert not (set(h3.grid_disk(c, 1)) & test_cells)


def test_ranking_metrics():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_k(y, s, 2) == 0.5
    assert ndcg_at_k(y, s, 3) == pytest.approx((1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
    assert ndcg_at_k(np.array([1, 1, 0]), np.array([3, 2, 1]), 3) == pytest.approx(1.0)


def test_saturation_penalty_is_monotonic_and_bounded():
    pop = pd.Series([3500.0] * 4)
    s = saturation(pd.Series([0.0, 1.0, 2.0, 10.0]), pop, 3500.0)
    assert s.iloc[0] == 0 and s.is_monotonic_increasing and (s < 1).all()
    assert s.iloc[1] == pytest.approx(0.5)  # competitors == capacity


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.2


def test_competition_columns_are_never_features():
    from geomlab.features.dictionary import model_feature_columns

    df = pd.DataFrame({"pop_cell": [1.0], "ndvi_mean": [0.1], "comp_500m": [3], "h3_id": ["x"]})
    assert model_feature_columns(df) == ["ndvi_mean", "pop_cell"]
