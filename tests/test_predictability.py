"""Unit tests for core predictability maths."""

import numpy as np
import pandas as pd

from ssp.predictability import asymmetry_ratio, spearman_predictability
from ssp.descriptors import dynamic_storage_turnover, rainfall_fraction_meltseason
from ssp.variance_partition import variance_partition


def test_perfect_spearman():
    x = pd.Series(np.arange(50, dtype=float))
    y = pd.Series(np.arange(50, dtype=float))
    assert abs(spearman_predictability(x, y) - 1.0) < 1e-9


def test_asymmetry_symmetric_noise():
    rng = np.random.default_rng(0)
    swe = pd.Series(rng.normal(size=200))
    q = swe + 0.01 * rng.normal(size=200)
    a = asymmetry_ratio(swe, q, frac_best=0.1)
    assert abs(a) < 0.5  # nearly symmetric when q ≈ swe


def test_descriptors_and_vp():
    idx = pd.date_range("2001-10-01", periods=365, freq="D")
    melt = pd.Series(np.linspace(0, 1, 365), index=idx)
    rain = pd.Series(0.2, index=idx)
    storage = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 365)) + 2, index=idx)
    q = pd.Series(1.0, index=idx)
    fr = rainfall_fraction_meltseason(rain, melt, idx[100:200])
    assert 0 <= fr <= 1
    tau = dynamic_storage_turnover(storage, q)
    assert tau > 0

    df = pd.DataFrame(
        {
            "correlation": np.linspace(0.2, 0.9, 40) + 0.05 * np.random.default_rng(1).normal(size=40),
            "tau": np.linspace(0.8, 0.1, 40),
            "frain": np.linspace(0.1, 0.7, 40),
        }
    )
    vp = variance_partition(df, "correlation", ["tau", "frain"])
    assert set(vp.columns) >= {"predictor", "marginal_R2", "unique_delta_R2"}
    assert len(vp) == 2
