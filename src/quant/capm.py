"""CAPM beta calculation.

Frequency matters: daily vs weekly returns yield different betas.
Bloomberg adjusted beta: (2/3) * raw_beta + (1/3) * 1.0 (mean-reversion toward 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass(frozen=True)
class BetaResult:
    beta: float
    alpha: float
    r_squared: float
    adjusted_beta: float


def capm_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> BetaResult:
    """Compute CAPM beta via OLS regression.

    Parameters
    ----------
    stock_returns : pd.Series
        Stock return series.
    market_returns : pd.Series
        Market (benchmark) return series.

    Returns
    -------
    BetaResult
        beta, alpha, r_squared, adjusted_beta (Bloomberg convention).
    """
    mask = ~(np.isnan(stock_returns) | np.isnan(market_returns))
    result = linregress(market_returns[mask], stock_returns[mask])

    raw_beta = result.slope
    adjusted = (2 / 3) * raw_beta + (1 / 3) * 1.0

    return BetaResult(
        beta=float(raw_beta),
        alpha=float(result.intercept),
        r_squared=float(result.rvalue**2),
        adjusted_beta=float(adjusted),
    )


def rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling CAPM beta (vectorized via covariance).

    Parameters
    ----------
    stock_returns, market_returns : pd.Series
        Return series.
    window : int, default 60
        Rolling window.

    Returns
    -------
    pd.Series
        Rolling beta values.
    """
    cov_roll = stock_returns.rolling(window).cov(market_returns)
    var_roll = market_returns.rolling(window).var()
    return cov_roll / var_roll
