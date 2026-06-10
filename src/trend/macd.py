"""Moving Average Convergence Divergence (MACD).

MACD line = EMA(fast) - EMA(slow).
Signal line = EMA(MACD line, signal period).
Histogram = MACD - Signal.
Default periods: fast=12, slow=26, signal=9.
"""
from __future__ import annotations

import pandas as pd

from core.types import MACDResult
from core.validation import validate_series, validate_period


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """Compute MACD indicator.

    Parameters
    ----------
    close : pd.Series
        Price series.
    fast_period : int, default 12
        Fast EMA period.
    slow_period : int, default 26
        Slow EMA period.
    signal_period : int, default 9
        Signal line EMA period.

    Returns
    -------
    MACDResult
        Named result with .macd, .signal, .histogram attributes.
    """
    validate_period(fast_period, "fast_period")
    validate_period(slow_period, "slow_period")
    validate_period(signal_period, "signal_period")
    close = validate_series(close, "close", min_length=slow_period)

    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd_line - signal_line

    return MACDResult(macd=macd_line, signal=signal_line, histogram=histogram)
