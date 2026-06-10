"""Volatility models: rolling std dev, EWMA, and Parkinson estimator.

Annualization: daily sigma * sqrt(252), weekly * sqrt(52), monthly * sqrt(12).
Remember ddof=1 (pandas default) vs ddof=0 (numpy default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.validation import validate_series, validate_period

TRADING_DAYS = 252


def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Compute rolling historical volatility.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int, default 21
        Rolling window size.
    annualize : bool, default True
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
        Volatility values.
    """
    validate_period(window, "window")
    returns = validate_series(returns, "returns", min_length=window)
    vol = returns.rolling(window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def ewma_volatility(
    returns: pd.Series,
    span: int = 30,
    annualize: bool = True,
) -> pd.Series:
    """Compute EWMA volatility (RiskMetrics uses lambda=0.94 -> span ~ 32.33).

    Parameters
    ----------
    returns : pd.Series
        Return series.
    span : int, default 30
        EWM span parameter.
    annualize : bool, default True
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
        EWMA volatility.
    """
    validate_period(span, "span")
    returns = validate_series(returns, "returns")
    vol = returns.ewm(span=span).std(bias=False)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Compute Parkinson volatility estimator.

    5x more efficient than close-to-close for GBM processes.
    Captures intraday range but misses overnight gaps.

    Parameters
    ----------
    high, low : pd.Series
        High and low prices.
    window : int, default 21
        Rolling window.
    annualize : bool, default True
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
        Parkinson volatility.
    """
    validate_period(window, "window")
    high = validate_series(high, "high")
    low = validate_series(low, "low")

    log_hl = np.log(high / low)
    var = (1.0 / (4.0 * np.log(2))) * (log_hl**2).rolling(window).mean()
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol
