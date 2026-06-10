"""Log returns and simple returns calculations.

Log returns are time-additive (multi-period = sum of single-period).
Simple returns are asset-additive (portfolio return = weighted sum).
Use log returns for time-series analysis; simple returns for portfolio calculations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.precision import ensure_float64


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple (arithmetic) returns: (P1 - P0) / P0.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price series. Must be > 0.

    Returns
    -------
    pd.Series or pd.DataFrame
        Simple returns. First value is NaN.
    """
    prices = ensure_float64(prices)
    return prices.pct_change()


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute logarithmic returns: ln(P1 / P0).

    Uses np.log1p for numerical precision with small returns.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price series. Must be > 0.

    Returns
    -------
    pd.Series or pd.DataFrame
        Log returns. First value is NaN.
    """
    prices = ensure_float64(prices)
    return np.log(prices / prices.shift(1))
