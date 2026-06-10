"""Exponential Moving Average (EMA).

Smoothing factor: k = 2/(n+1). For period 12, k ~ 0.1538; period 26, k ~ 0.0741.
Recursive formula: EMA_t = k * close_t + (1-k) * EMA_{t-1}.

TA-Lib seeds EMA with SMA of first N values. pandas ewm(adjust=False) seeds with x_0.
For long series they converge; for short series the divergence is significant.
"""
from __future__ import annotations

import pandas as pd

from core.smoothing import ema_smooth
from core.validation import validate_series, validate_period


def ema(close: pd.Series, period: int = 20, sma_seed: bool = True) -> pd.Series:
    """Compute Exponential Moving Average.

    Parameters
    ----------
    close : pd.Series
        Price series.
    period : int, default 20
        EMA span.
    sma_seed : bool, default True
        If True, seed with SMA of first ``period`` values (TA-Lib compatible).

    Returns
    -------
    pd.Series
        EMA values.
    """
    validate_period(period)
    close = validate_series(close, "close", min_length=period)
    return ema_smooth(close, period, sma_seed=sma_seed)
