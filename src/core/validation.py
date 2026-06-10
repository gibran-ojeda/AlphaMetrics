"""Input validation for OHLCV data and indicator parameters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.precision import ensure_float64


def validate_ohlcv(df: pd.DataFrame, require_volume: bool = True) -> pd.DataFrame:
    """Validate and clean OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close (case-insensitive).
        Optionally volume.
    require_volume : bool
        If True, volume column is required.

    Returns
    -------
    pd.DataFrame
        Cleaned copy with float64 dtypes and sorted index.

    Raises
    ------
    ValueError
        If required columns are missing or data integrity checks fail.
    """
    df = df.copy()

    col_map = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close"]
    if require_volume:
        required.append("volume")

    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename = {col_map[r]: r for r in required}
    df = df.rename(columns=rename)

    price_cols = ["open", "high", "low", "close"]
    if require_volume:
        price_cols.append("volume")
    df[price_cols] = ensure_float64(df[price_cols])

    if df[["high", "low", "close"]].isnull().any().any():
        raise ValueError("NaN values found in price columns")

    if not (df["high"] >= df["low"]).all():
        raise ValueError("High < Low detected")

    if not (df["high"] >= df["close"]).all():
        raise ValueError("High < Close detected")

    if not (df["high"] >= df["open"]).all():
        raise ValueError("High < Open detected")

    if require_volume and not (df["volume"] >= 0).all():
        raise ValueError("Negative volume detected")

    if hasattr(df.index, "is_monotonic_increasing") and not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df


def validate_series(series: pd.Series, name: str = "input", min_length: int = 1) -> pd.Series:
    """Validate a single Series input.

    Parameters
    ----------
    series : pd.Series
        Input price or return series.
    name : str
        Name for error messages.
    min_length : int
        Minimum required length.

    Returns
    -------
    pd.Series
        float64 copy of the series.

    Raises
    ------
    TypeError
        If input is not a Series.
    ValueError
        If series is too short.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(series).__name__}")
    if len(series) < min_length:
        raise ValueError(f"{name} length {len(series)} < minimum {min_length}")
    return ensure_float64(series.copy())


def validate_period(period: int, name: str = "period") -> None:
    """Validate that a period parameter is a positive integer."""
    if not isinstance(period, int) or period < 1:
        raise ValueError(f"{name} must be a positive integer, got {period}")
