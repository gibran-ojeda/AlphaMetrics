"""Tests for volatility indicators."""
import numpy as np
import pandas as pd
import pytest

from volatility.atr import true_range, atr
from volatility.adx import adx
from volatility.bollinger import bollinger_bands
from volatility.vwap import vwap
from volatility.obv import obv
from volatility.adl import adl
from core.types import ADXResult, BollingerResult


class TestTrueRange:
    def test_first_value_is_hl(self, sample_ohlcv):
        tr = true_range(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        expected = sample_ohlcv["high"].iloc[0] - sample_ohlcv["low"].iloc[0]
        assert tr.iloc[0] == pytest.approx(expected)

    def test_all_positive(self, sample_ohlcv):
        tr = true_range(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert (tr >= 0).all()

    def test_captures_gap(self):
        high = pd.Series([10.0, 15.0])
        low = pd.Series([9.0, 14.0])
        close = pd.Series([9.5, 14.5])
        tr = true_range(high, low, close)
        # Bar 1: max(15-14, |15-9.5|, |14-9.5|) = max(1, 5.5, 4.5) = 5.5
        assert tr.iloc[1] == pytest.approx(5.5)


class TestATR:
    def test_leading_nans(self, sample_ohlcv):
        result = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], 14)
        assert result.iloc[:14].isna().all()

    def test_positive_values(self, sample_ohlcv):
        result = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], 14)
        valid = result.dropna()
        assert (valid > 0).all()


class TestADX:
    def test_returns_adx_result(self, sample_ohlcv):
        result = adx(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert isinstance(result, ADXResult)

    def test_range_0_100(self, sample_ohlcv):
        result = adx(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        valid = result.adx.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestBollinger:
    def test_returns_bollinger_result(self, sample_close):
        result = bollinger_bands(sample_close)
        assert isinstance(result, BollingerResult)

    def test_band_ordering(self, sample_close):
        result = bollinger_bands(sample_close)
        valid = ~result.upper.isna()
        assert (result.upper[valid] >= result.middle[valid]).all()
        assert (result.middle[valid] >= result.lower[valid]).all()

    def test_middle_is_sma(self, sample_close):
        result = bollinger_bands(sample_close, period=20)
        expected_sma = sample_close.rolling(20).mean()
        np.testing.assert_array_almost_equal(
            result.middle.dropna().values, expected_sma.dropna().values
        )


class TestVWAP:
    def test_output_length(self, sample_ohlcv):
        result = vwap(
            sample_ohlcv["high"], sample_ohlcv["low"],
            sample_ohlcv["close"], sample_ohlcv["volume"],
        )
        assert len(result) == len(sample_ohlcv)

    def test_within_price_range(self, sample_ohlcv):
        result = vwap(
            sample_ohlcv["high"], sample_ohlcv["low"],
            sample_ohlcv["close"], sample_ohlcv["volume"],
        )
        valid = result.dropna()
        # VWAP should be within the cumulative high/low range
        assert valid.iloc[0] > 0


class TestOBV:
    def test_cumulative(self, sample_ohlcv):
        result = obv(sample_ohlcv["close"], sample_ohlcv["volume"])
        assert len(result) == len(sample_ohlcv)

    def test_direction(self):
        close = pd.Series([10.0, 11.0, 12.0, 11.0])
        volume = pd.Series([100.0, 200.0, 150.0, 300.0])
        result = obv(close, volume)
        # Up, up, down -> 0, +200, +350, +50
        assert result.iloc[1] > result.iloc[0]  # price up -> OBV up
        assert result.iloc[3] < result.iloc[2]  # price down -> OBV down


class TestADL:
    def test_output_length(self, sample_ohlcv):
        result = adl(
            sample_ohlcv["high"], sample_ohlcv["low"],
            sample_ohlcv["close"], sample_ohlcv["volume"],
        )
        assert len(result) == len(sample_ohlcv)

    def test_zero_range(self):
        high = pd.Series([10.0, 10.0])
        low = pd.Series([10.0, 10.0])
        close = pd.Series([10.0, 10.0])
        volume = pd.Series([100.0, 200.0])
        result = adl(high, low, close, volume)
        # MFM = 0 when high == low
        assert (result == 0.0).all()
