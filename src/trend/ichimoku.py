"""Ichimoku Kinko Hyo.

Not available in standard TA-Lib. Pure pandas implementation.

Components:
- Tenkan-sen: (9-period H+L) / 2
- Kijun-sen: (26-period H+L) / 2
- Senkou Span A: (Tenkan + Kijun) / 2, shifted +26
- Senkou Span B: (52-period H+L) / 2, shifted +26
- Chikou Span: Close, shifted -26

Minimum data requirement: 78 periods (52 + 26) before all lines produce values.
"""
from __future__ import annotations

import pandas as pd

from core.types import IchimokuResult
from core.validation import validate_series, validate_period


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> IchimokuResult:
    """Compute Ichimoku Kinko Hyo.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series.
    tenkan_period : int, default 9
        Tenkan-sen (conversion line) period.
    kijun_period : int, default 26
        Kijun-sen (base line) period.
    senkou_b_period : int, default 52
        Senkou Span B period.
    displacement : int, default 26
        Forward/backward shift for Senkou/Chikou.

    Returns
    -------
    IchimokuResult
        Named result with .tenkan, .kijun, .senkou_a, .senkou_b, .chikou.
    """
    validate_period(tenkan_period, "tenkan_period")
    validate_period(kijun_period, "kijun_period")
    validate_period(senkou_b_period, "senkou_b_period")
    validate_period(displacement, "displacement")
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")

    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = (
        (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    ).shift(displacement)
    chikou = close.shift(-displacement)

    return IchimokuResult(
        tenkan=tenkan, kijun=kijun, senkou_a=senkou_a, senkou_b=senkou_b, chikou=chikou
    )
