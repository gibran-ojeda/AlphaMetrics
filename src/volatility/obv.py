"""On Balance Volume (OBV).

Cumulative indicator: when close > prev_close, add volume; when close < prev_close,
subtract volume. When close == prev_close, volume is NOT added or subtracted.

Use float64 to avoid integer overflow with high-volume instruments over long periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.validation import validate_series


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On Balance Volume.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        OBV values. Starts from bar 0 (no lookback).
    """
    close = validate_series(close, "close")
    volume = validate_series(volume, "volume")

    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()
