"""True Range and Average True Range (ATR).

True Range captures overnight gaps: TR = max(H-L, |H-PrevClose|, |L-PrevClose|).
ATR uses Wilder smoothing (alpha = 1/period), NOT SMA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.validation import validate_series, validate_period


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute True Range.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series.

    Returns
    -------
    pd.Series
        True Range. First value is NaN (no previous close).
    """
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]  # First bar: no prev close
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range with Wilder smoothing.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series.
    period : int, default 14
        ATR smoothing period.

    Returns
    -------
    pd.Series
        ATR values. First ``period`` values are NaN.
    """
    validate_period(period)
    tr = true_range(high, low, close)

    atr_vals = pd.Series(np.nan, index=close.index)
    atr_vals.iloc[period] = tr.iloc[1 : period + 1].mean()

    for i in range(period + 1, len(close)):
        atr_vals.iloc[i] = (tr.iloc[i] + atr_vals.iloc[i - 1] * (period - 1)) / period

    return atr_vals
