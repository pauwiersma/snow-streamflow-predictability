"""Catchment / hydroclimate descriptors (zetas) used as predictors of rho."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rainfall_fraction_meltseason(
    rainfall: pd.Series, snowmelt: pd.Series, melt_period: pd.DatetimeIndex
) -> float:
    """frain = P_liquid / (P_liquid + M) over the melt season."""
    p = rainfall.loc[melt_period].sum()
    m = snowmelt.loc[melt_period].sum()
    denom = p + m
    return float(p / denom) if denom > 0 else np.nan


def et_fraction_meltseason(
    et: pd.Series, rainfall: pd.Series, snowmelt: pd.Series, melt_period: pd.DatetimeIndex
) -> float:
    """ET / (P_liquid + M) over the melt season."""
    e = et.loc[melt_period].sum()
    water_in = rainfall.loc[melt_period].sum() + snowmelt.loc[melt_period].sum()
    return float(e / water_in) if water_in > 0 else np.nan


def dynamic_storage_turnover(
    total_storage: pd.Series, discharge: pd.Series
) -> float:
    """tau = Delta S_WY / sum(Q_WY).

    Delta S is max − min subsurface storage over the water year.
    """
    delta_s = float(total_storage.max() - total_storage.min())
    q_sum = float(discharge.sum())
    return delta_s / q_sum if q_sum > 0 else np.nan


def melt_half_flow_interval(snowmelt: pd.Series) -> float:
    """Days between 25th and 75th percentiles of cumulative melt."""
    cum = snowmelt.cumsum()
    if cum.iloc[-1] <= 0 or cum.isna().all():
        return np.nan
    total = cum.iloc[-1]
    t25 = cum.index[(cum >= 0.25 * total).argmax()]
    t75 = cum.index[(cum >= 0.75 * total).argmax()]
    return float((t75 - t25).days)


def melt_season_bounds(
    snowmelt: pd.Series, *, pre_days: int = 7, post_days: int = 30
) -> pd.DatetimeIndex:
    """Melt season: 1 week before 10% cumulative melt to 1 month after 90%."""
    cum = snowmelt.cumsum()
    total = cum.iloc[-1]
    if total <= 0 or np.isnan(total):
        return pd.DatetimeIndex([])
    t10 = cum.index[(cum >= 0.10 * total).argmax()]
    t90 = cum.index[(cum >= 0.90 * total).argmax()]
    start = t10 - pd.Timedelta(days=pre_days)
    end = t90 + pd.Timedelta(days=post_days)
    return snowmelt.loc[start:end].index
