"""Mutual SWE–Q predictability (Spearman rho) and asymmetry ratio A."""

from __future__ import annotations

import numpy as np
import pandas as pd


def spearman_predictability(swe_skill: pd.Series, q_skill: pd.Series) -> float:
    """Spearman rank correlation between SWE and Q skill across ensemble members.

    Higher skill must correspond to better performance for both series
    (e.g. NSE ↑ better, absolute bias ↓ better — invert bias metrics first).
    """
    aligned = pd.concat([swe_skill, q_skill], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def skill_to_rank(skill: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Convert skill scores to ranks (rank 1 = best)."""
    ascending = not higher_is_better
    return skill.rank(method="average", ascending=ascending)


def asymmetry_ratio(
    swe_skill: pd.Series,
    q_skill: pd.Series,
    *,
    higher_is_better_swe: bool = True,
    higher_is_better_q: bool = True,
    frac_best: float = 0.1,
) -> float:
    """Directional skill-transfer asymmetry.

    A = log2( mean(R_SWE | best Q) / mean(R_Q | best SWE) )

    A > 0: snow better constrains streamflow than vice versa.
    A < 0: streamflow better constrains snow than vice versa.
    """
    aligned = pd.concat(
        {"swe": swe_skill, "q": q_skill}, axis=1
    ).dropna()
    if len(aligned) < 10:
        return np.nan

    r_swe = skill_to_rank(aligned["swe"], higher_is_better=higher_is_better_swe)
    r_q = skill_to_rank(aligned["q"], higher_is_better=higher_is_better_q)
    n_best = max(1, int(round(frac_best * len(aligned))))

    best_q_idx = r_q.nsmallest(n_best).index
    best_swe_idx = r_swe.nsmallest(n_best).index
    mean_swe_given_best_q = float(r_swe.loc[best_q_idx].mean())
    mean_q_given_best_swe = float(r_q.loc[best_swe_idx].mean())
    if mean_q_given_best_swe <= 0:
        return np.nan
    return float(np.log2(mean_swe_given_best_q / mean_q_given_best_swe))


# Metrics where lower absolute error is better (invert before ranking/correlation
# if stored as positive error magnitudes).
LOWER_IS_BETTER = {
    "melt_sum_APE",
    "melt_sum_grid_MAPE",
    "Qmean_meltseason2_APE",
    "Qmean_meltseason_APE",
}


def prepare_skill_series(series: pd.Series, metric_name: str) -> pd.Series:
    """Flip sign of lower-is-better metrics so that higher = better skill."""
    if any(key in metric_name for key in LOWER_IS_BETTER) or metric_name.endswith(
        ("_APE", "_MAPE", "_ME", "_MAE", "_RMSE")
    ):
        # Absolute error metrics: negate so Spearman still rises with co-performance
        if metric_name.endswith(("_APE", "_MAPE", "_MAE", "_RMSE")) or "bias" in metric_name.lower():
            return -series.abs() if series.min() >= 0 else -series
    return series
