"""Shared test fixtures: sample OHLCV data and known-answer series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 100 bars of synthetic OHLCV data with realistic price action."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    close = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)
    open_ = low + np.random.rand(n) * (high - low)
    volume = np.random.randint(100_000, 1_000_000, size=n).astype(np.float64)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_close(sample_ohlcv: pd.DataFrame) -> pd.Series:
    """Close prices from sample OHLCV."""
    return sample_ohlcv["close"]


@pytest.fixture
def sample_returns(sample_close: pd.Series) -> pd.Series:
    """Simple returns from sample close prices."""
    return sample_close.pct_change().dropna()
