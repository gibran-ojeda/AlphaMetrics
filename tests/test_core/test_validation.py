"""Tests for core.validation."""
import numpy as np
import pandas as pd
import pytest

from core.validation import validate_ohlcv, validate_series, validate_period


class TestValidateOHLCV:
    def test_valid_dataframe(self, sample_ohlcv):
        result = validate_ohlcv(sample_ohlcv)
        assert result.dtypes["close"] == np.float64
        assert result.dtypes["volume"] == np.float64
        assert list(result.columns[:5]) == ["open", "high", "low", "close", "volume"]

    def test_missing_columns(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv(df)

    def test_high_less_than_low(self):
        df = pd.DataFrame({
            "open": [10.0], "high": [5.0], "low": [12.0],
            "close": [4.0], "volume": [100.0],
        })
        with pytest.raises(ValueError, match="High < Low"):
            validate_ohlcv(df)

    def test_negative_volume(self):
        df = pd.DataFrame({
            "open": [10.0], "high": [12.0], "low": [9.0],
            "close": [11.0], "volume": [-100.0],
        })
        with pytest.raises(ValueError, match="Negative volume"):
            validate_ohlcv(df)

    def test_no_volume_required(self):
        df = pd.DataFrame({
            "open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0],
        })
        result = validate_ohlcv(df, require_volume=False)
        assert "close" in result.columns

    def test_unsorted_index_gets_sorted(self):
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame({
            "open": [10.0, 9.0, 10.5], "high": [12.0, 11.0, 11.5],
            "low": [9.0, 8.0, 9.5], "close": [11.0, 10.0, 11.0],
            "volume": [100.0, 200.0, 150.0],
        }, index=dates)
        result = validate_ohlcv(df)
        assert result.index.is_monotonic_increasing

    def test_does_not_mutate_input(self, sample_ohlcv):
        original_dtype = sample_ohlcv["close"].dtype
        _ = validate_ohlcv(sample_ohlcv)
        assert sample_ohlcv["close"].dtype == original_dtype


class TestValidateSeries:
    def test_not_a_series(self):
        with pytest.raises(TypeError, match="must be a pd.Series"):
            validate_series([1, 2, 3])

    def test_too_short(self):
        s = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="length 2 < minimum 10"):
            validate_series(s, min_length=10)

    def test_returns_float64_copy(self):
        s = pd.Series([1, 2, 3], dtype=np.int64)
        result = validate_series(s)
        assert result.dtype == np.float64


class TestValidatePeriod:
    def test_zero(self):
        with pytest.raises(ValueError):
            validate_period(0)

    def test_negative(self):
        with pytest.raises(ValueError):
            validate_period(-5)

    def test_valid(self):
        validate_period(14)  # should not raise
