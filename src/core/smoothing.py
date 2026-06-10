"""Smoothing functions: Wilder vs standard EMA.

Wilder smoothing uses alpha = 1/N (used by RSI, ATR, ADX).
Standard EMA uses alpha = 2/(N+1) (used by MACD, Bollinger).
These are NOT interchangeable — for period 14:
  Wilder alpha = 0.0714, EMA alpha = 0.1333.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing method (alpha = 1/period).

    Seeds with SMA of first `period` values, then applies
    recursive smoothing: val_t = (val_{t-1} * (period-1) + x_t) / period.

    Parameters
    ----------
    series : pd.Series
        Input values (gains, losses, TR, etc.). First value should be at index 0+.
    period : int
        Smoothing period.

    Returns
    -------
    pd.Series
        Smoothed series. First `period - 1` values are NaN.
    """
    result = pd.Series(np.nan, index=series.index)

    first_valid = series.first_valid_index()
    if first_valid is None:
        return result

    start_pos = series.index.get_loc(first_valid)
    if start_pos + period > len(series):
        return result

    seed_slice = series.iloc[start_pos : start_pos + period]
    seed_pos = start_pos + period - 1
    result.iloc[seed_pos] = seed_slice.mean()

    for i in range(seed_pos + 1, len(series)):
        result.iloc[i] = (result.iloc[i - 1] * (period - 1) + series.iloc[i]) / period

    return result


def ema_smooth(series: pd.Series, period: int, sma_seed: bool = True) -> pd.Series:
    """Standard EMA (alpha = 2/(period+1)).

    Parameters
    ----------
    series : pd.Series
        Input values.
    period : int
        EMA span.
    sma_seed : bool
        If True, seed with SMA of first `period` values (TA-Lib compatible).
        If False, seed with first value (pandas ewm adjust=False behavior).

    Returns
    -------
    pd.Series
        EMA values. First `period - 1` values are NaN when sma_seed=True.
    """
    alpha = 2.0 / (period + 1)
    result = pd.Series(np.nan, index=series.index)

    if sma_seed:
        if len(series) < period:
            return result
        result.iloc[period - 1] = series.iloc[:period].mean()
        start = period
    else:
        result.iloc[0] = series.iloc[0]
        start = 1

    for i in range(start, len(series)):
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]

    return result
