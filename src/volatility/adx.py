"""Average Directional Index (ADX).

6-step calculation chain with Wilder smoothing throughout:
1. +DM / -DM from consecutive highs/lows
2. Wilder smooth TR, +DM, -DM
3. +DI / -DI = 100 * smoothed_DM / smoothed_TR
4. DX = 100 * |+DI - -DI| / (+DI + -DI)
5. ADX = Wilder smoothing of DX

ADX measures trend STRENGTH only, not direction.
ADX > 25 = strong trend, < 20 = weak/no trend.
Lookback is ~2*period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import ADXResult
from core.validation import validate_series, validate_period


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> ADXResult:
    """Compute ADX, +DI, and -DI.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series.
    period : int, default 14
        Smoothing period.

    Returns
    -------
    ADXResult
        Named result with .adx, .plus_di, .minus_di attributes.
    """
    validate_period(period)
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")
    n = len(close)

    # Step 1: Directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # True range
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Step 2: Wilder smoothing of TR, +DM, -DM
    smooth_tr = pd.Series(np.nan, index=close.index)
    smooth_plus = pd.Series(np.nan, index=close.index)
    smooth_minus = pd.Series(np.nan, index=close.index)

    smooth_tr.iloc[period] = tr.iloc[1 : period + 1].sum()
    smooth_plus.iloc[period] = plus_dm.iloc[1 : period + 1].sum()
    smooth_minus.iloc[period] = minus_dm.iloc[1 : period + 1].sum()

    for i in range(period + 1, n):
        smooth_tr.iloc[i] = smooth_tr.iloc[i - 1] - smooth_tr.iloc[i - 1] / period + tr.iloc[i]
        smooth_plus.iloc[i] = (
            smooth_plus.iloc[i - 1] - smooth_plus.iloc[i - 1] / period + plus_dm.iloc[i]
        )
        smooth_minus.iloc[i] = (
            smooth_minus.iloc[i - 1] - smooth_minus.iloc[i - 1] / period + minus_dm.iloc[i]
        )

    # Step 3: +DI / -DI
    plus_di = 100.0 * smooth_plus / smooth_tr.replace(0, np.nan)
    minus_di = 100.0 * smooth_minus / smooth_tr.replace(0, np.nan)

    # Step 4: DX
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)

    # Step 5: ADX = Wilder smooth of DX
    adx_vals = pd.Series(np.nan, index=close.index)
    # First ADX = mean of first `period` DX values
    first_dx_idx = 2 * period
    if first_dx_idx < n:
        adx_vals.iloc[first_dx_idx] = dx.iloc[period + 1 : first_dx_idx + 1].mean()
        for i in range(first_dx_idx + 1, n):
            adx_vals.iloc[i] = (adx_vals.iloc[i - 1] * (period - 1) + dx.iloc[i]) / period

    return ADXResult(adx=adx_vals, plus_di=plus_di, minus_di=minus_di)
