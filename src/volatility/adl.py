"""Accumulation/Distribution Line (ADL).

Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
Ranges from -1 (close at low) to +1 (close at high).
Unlike OBV's binary approach, ADL weights volume proportionally by close position.

Edge case: High == Low -> MFM = 0 (no range information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.validation import validate_series


def adl(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute Accumulation/Distribution Line.

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data.

    Returns
    -------
    pd.Series
        ADL values. Starts from bar 0 (no lookback).
    """
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")
    volume = validate_series(volume, "volume")

    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfm = mfm.fillna(0)  # High == Low -> no range info
    return (mfm * volume).cumsum()
