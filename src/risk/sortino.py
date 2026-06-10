"""Sortino Ratio.

Formula: (Rp - Rf) / sigma_downside * sqrt(N)

Critical: Downside deviation uses ALL observations in the denominator (n),
not just negative returns. Returns above target contribute 0 to the sum.
"""
from __future__ import annotations

import numpy as np


def sortino_ratio(
    returns: np.ndarray,
    rf: float = 0.0,
    target: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio.

    Parameters
    ----------
    returns : array-like
        Period returns.
    rf : float, default 0.0
        Annual risk-free rate.
    target : float or None
        Target return per period. If None, uses risk-free rate per period.
    periods_per_year : int, default 252
        Trading periods per year.

    Returns
    -------
    float
        Annualized Sortino ratio. inf if no downside risk.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return np.nan

    rf_per_period = (1 + rf) ** (1 / periods_per_year) - 1
    if target is None:
        target = rf_per_period

    downside = np.minimum(returns - target, 0.0)
    dd = np.sqrt(np.mean(np.square(downside)))

    if dd < 1e-12:
        return np.inf

    annualized_excess = np.mean(returns - rf_per_period) * periods_per_year
    annualized_dd = dd * np.sqrt(periods_per_year)
    return float(annualized_excess / annualized_dd)
