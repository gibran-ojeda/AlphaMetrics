"""Stop loss implementations.

Edge case: gaps through stop levels mean fill occurs at the open price,
not the stop price. ATR-based stops adapt to volatility automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StopResult:
    stop_level: pd.Series
    triggered_date: object | None
    triggered_price: float | None


def fixed_stop(
    entry_price: float,
    prices: pd.Series,
    stop_pct: float = 0.02,
) -> StopResult:
    """Fixed percentage stop loss.

    Parameters
    ----------
    entry_price : float
        Entry price.
    prices : pd.Series
        Subsequent price series.
    stop_pct : float, default 0.02
        Stop distance as fraction (e.g., 0.02 = 2%).

    Returns
    -------
    StopResult
        Stop level, trigger date, and trigger price.
    """
    stop_level = pd.Series(entry_price * (1 - stop_pct), index=prices.index)
    triggered = prices[prices <= stop_level.iloc[0]]

    if len(triggered) > 0:
        return StopResult(
            stop_level=stop_level,
            triggered_date=triggered.index[0],
            triggered_price=float(triggered.iloc[0]),
        )
    return StopResult(stop_level=stop_level, triggered_date=None, triggered_price=None)


def trailing_stop(
    prices: pd.Series,
    trail_pct: float = 0.03,
) -> StopResult:
    """Trailing stop loss.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    trail_pct : float, default 0.03
        Trail distance as fraction.

    Returns
    -------
    StopResult
        Dynamic stop level, trigger date, and trigger price.
    """
    running_max = prices.expanding().max()
    stop_level = running_max * (1 - trail_pct)
    triggered = prices[prices <= stop_level]

    if len(triggered) > 0:
        return StopResult(
            stop_level=stop_level,
            triggered_date=triggered.index[0],
            triggered_price=float(triggered.iloc[0]),
        )
    return StopResult(stop_level=stop_level, triggered_date=None, triggered_price=None)


def atr_stop(
    entry_price: float,
    atr_values: pd.Series,
    multiplier: float = 3.0,
) -> pd.Series:
    """ATR-based stop level.

    Stop = entry_price - multiplier * ATR. Adapts to volatility.

    Parameters
    ----------
    entry_price : float
        Entry price.
    atr_values : pd.Series
        ATR series.
    multiplier : float, default 3.0
        ATR multiplier (typical: 2-3).

    Returns
    -------
    pd.Series
        Stop levels.
    """
    return entry_price - multiplier * atr_values
