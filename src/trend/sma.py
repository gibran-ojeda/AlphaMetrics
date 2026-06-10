"""Simple Moving Average (SMA)."""
from __future__ import annotations

import pandas as pd

from core.validation import validate_series, validate_period


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Simple Moving Average.

    Parameters
    ----------
    close : pd.Series
        Price series.
    period : int, default 20
        Lookback window. Common values: 20, 50, 100, 200.

    Returns
    -------
    pd.Series
        SMA values. First ``period - 1`` values are NaN.
    """
    validate_period(period)
    close = validate_series(close, "close", min_length=period)
    return close.rolling(window=period, min_periods=period).mean()
