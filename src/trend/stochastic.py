"""Stochastic Oscillator (%K, %D).

Fast %K = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) * 100
Slow %K = SMA(Fast %K, 3)
Slow %D = SMA(Slow %K, 3)
Output range: 0-100. Overbought >= 80, oversold <= 20.

Edge case: When High == Low (zero range), TA-Lib returns 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import StochasticResult
from core.validation import validate_series, validate_period


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> StochasticResult:
    """Compute Slow Stochastic Oscillator.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series.
    fastk_period : int, default 14
        Fast %K lookback.
    slowk_period : int, default 3
        Slow %K smoothing (SMA of Fast %K).
    slowd_period : int, default 3
        Slow %D smoothing (SMA of Slow %K).

    Returns
    -------
    StochasticResult
        Named result with .k (Slow %K) and .d (Slow %D).
    """
    validate_period(fastk_period, "fastk_period")
    validate_period(slowk_period, "slowk_period")
    validate_period(slowd_period, "slowd_period")
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")

    lowest = low.rolling(fastk_period).min()
    highest = high.rolling(fastk_period).max()
    hl_range = highest - lowest

    # Zero range -> 0 (TA-Lib convention)
    fast_k = np.where(hl_range == 0, 0.0, (close - lowest) / hl_range * 100.0)
    fast_k = pd.Series(fast_k, index=close.index)

    slow_k = fast_k.rolling(slowk_period).mean()
    slow_d = slow_k.rolling(slowd_period).mean()

    return StochasticResult(k=slow_k, d=slow_d)


def stochastic_fast(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 14,
    fastd_period: int = 3,
) -> StochasticResult:
    """Compute Fast Stochastic Oscillator.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series.
    fastk_period : int, default 14
        Fast %K lookback.
    fastd_period : int, default 3
        Fast %D smoothing (SMA of Fast %K).

    Returns
    -------
    StochasticResult
        Named result with .k (Fast %K) and .d (Fast %D).
    """
    validate_period(fastk_period, "fastk_period")
    validate_period(fastd_period, "fastd_period")
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")

    lowest = low.rolling(fastk_period).min()
    highest = high.rolling(fastk_period).max()
    hl_range = highest - lowest

    fast_k = np.where(hl_range == 0, 0.0, (close - lowest) / hl_range * 100.0)
    fast_k = pd.Series(fast_k, index=close.index)
    fast_d = fast_k.rolling(fastd_period).mean()

    return StochasticResult(k=fast_k, d=fast_d)
