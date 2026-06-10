"""Volume Weighted Average Price (VWAP).

Not in standard TA-Lib. Resets at each session start.
Typical Price = (H + L + C) / 3, then cumulative sum weighted by volume.

Edge cases: zero volume bars, overnight futures sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.validation import validate_series


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute VWAP with automatic session reset (daily).

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data with DatetimeIndex.

    Returns
    -------
    pd.Series
        VWAP values. NaN where cumulative volume is zero.
    """
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")
    volume = validate_series(volume, "volume")

    tp = (high + low + close) / 3.0
    tp_vol = tp * volume

    if hasattr(close.index, "date"):
        date_groups = close.index.date
        cum_tp_vol = tp_vol.groupby(date_groups).cumsum()
        cum_vol = volume.groupby(date_groups).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = volume.cumsum()

    return cum_tp_vol / cum_vol.replace(0, np.nan)
