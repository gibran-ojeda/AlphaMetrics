"""Sharpe Ratio.

Formula: (Rp - Rf) / sigma_p * sqrt(N)
where N = 252 (daily), 52 (weekly), 12 (monthly).

Key pitfalls:
- np.std defaults to ddof=0, pandas to ddof=1. Use ddof=1 consistently.
- Risk-free rate must be converted to per-period: rf_daily = (1 + rf_annual)^(1/252) - 1.
Benchmarks: >1 good, >2 very good, >3 excellent.
"""
from __future__ import annotations

import numpy as np


def sharpe_ratio(
    returns: np.ndarray,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Period returns.
    rf : float, default 0.0
        Annual risk-free rate.
    periods_per_year : int, default 252
        Trading periods per year (252=daily, 52=weekly, 12=monthly).

    Returns
    -------
    float
        Annualized Sharpe ratio. NaN if std is near zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return np.nan

    rf_per_period = (1 + rf) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    std = np.std(excess, ddof=1)

    if std < 1e-12:
        return np.nan

    return float((np.mean(excess) / std) * np.sqrt(periods_per_year))
