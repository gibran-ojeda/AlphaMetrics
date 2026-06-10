"""Bollinger Bands.

Middle = SMA(period). Upper/Lower = Middle +/- num_std * std.
TA-Lib uses population std dev (ddof=0). Pandas defaults to ddof=1.

%B ranges 0-1 normally; values below 0 or above 1 = price outside bands.
Squeeze detection: Bandwidth < ~0.02 (2%) signals impending volatility expansion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import BollingerResult
from core.validation import validate_series, validate_period


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Compute Bollinger Bands (TA-Lib compatible with ddof=0).

    Parameters
    ----------
    close : pd.Series
        Price series.
    period : int, default 20
        SMA and std dev lookback.
    num_std : float, default 2.0
        Number of standard deviations for bands.

    Returns
    -------
    BollingerResult
        Named result with .upper, .middle, .lower, .pct_b, .bandwidth.
    """
    validate_period(period)
    close = validate_series(close, "close", min_length=period)

    middle = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)  # Population std dev (TA-Lib compat)

    upper = middle + num_std * std
    lower = middle - num_std * std

    band_width = upper - lower
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    bandwidth = band_width / middle.replace(0, np.nan)

    return BollingerResult(
        upper=upper, middle=middle, lower=lower, pct_b=pct_b, bandwidth=bandwidth
    )
