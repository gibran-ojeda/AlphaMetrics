"""Relative Strength Index (RSI) using Wilder smoothing.

TA-Lib uses Wilder smoothing (alpha = 1/N), NOT standard EMA (alpha = 2/(N+1)).
For period 14: Wilder alpha = 1/14 ~ 0.0714, not 2/15 ~ 0.1333.
Output range: 0-100. Overbought >= 70, oversold <= 30.

Convergence warning: first ~10*N periods have precision errors up to ~5%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.validation import validate_series, validate_period


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI with Wilder smoothing (TA-Lib compatible).

    Parameters
    ----------
    close : pd.Series
        Price series.
    period : int, default 14
        RSI lookback period.

    Returns
    -------
    pd.Series
        RSI values (0-100). First ``period`` values are NaN.
    """
    validate_period(period)
    close = validate_series(close, "close", min_length=period + 1)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = pd.Series(np.nan, index=close.index)
    avg_loss = pd.Series(np.nan, index=close.index)

    # SMA seed over first `period` changes (indices 1..period)
    avg_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1 : period + 1].mean()

    # Wilder smoothing: alpha = 1/period
    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    # All gains -> RS=inf -> RSI=100; all losses -> RS=0 -> RSI=0
    return 100.0 - (100.0 / (1.0 + rs))
