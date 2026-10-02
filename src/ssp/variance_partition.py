"""Variance partitioning of predictability via multiple linear regression."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score


def _fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if mask.sum() < X.shape[1] + 2:
        return np.nan
    model = LinearRegression().fit(X[mask], y[mask])
    return float(model.score(X[mask], y[mask]))


def _loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(y) < X.shape[1] + 3:
        return np.nan
    preds = np.empty_like(y)
    for train, test in LeaveOneOut().split(X):
        model = LinearRegression().fit(X[train], y[train])
        preds[test] = model.predict(X[test])
    return float(r2_score(y, preds))


def variance_partition(
    df: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
) -> pd.DataFrame:
    """Return marginal R2, unique Delta-R2, full-model R2, and LOO R2.

    Parameters
    ----------
    df :
        Rows = catchment-years; columns include ``target`` and ``predictors``.
    target :
        Column name of predictability rho (or similar).
    predictors :
        Descriptor column names (e.g. tau, frain, et_frac, area, melt_hfi, slope).
    """
    y = df[target].to_numpy(dtype=float)
    rows = []
    full_X = df[list(predictors)].to_numpy(dtype=float)
    full_r2 = _fit_r2(full_X, y)
    loo = _loo_r2(full_X, y)

    for pred in predictors:
        X_marg = df[[pred]].to_numpy(dtype=float)
        marginal = _fit_r2(X_marg, y)
        others = [p for p in predictors if p != pred]
        if others:
            X_reduced = df[others].to_numpy(dtype=float)
            reduced_r2 = _fit_r2(X_reduced, y)
            unique = full_r2 - reduced_r2 if np.isfinite(full_r2) and np.isfinite(reduced_r2) else np.nan
        else:
            unique = marginal
        rows.append(
            {
                "predictor": pred,
                "marginal_R2": marginal,
                "unique_delta_R2": unique,
                "full_model_R2": full_r2,
                "loo_R2": loo,
            }
        )
    return pd.DataFrame(rows)
