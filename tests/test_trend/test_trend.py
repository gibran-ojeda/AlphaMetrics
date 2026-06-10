"""Tests for trend indicators."""
import numpy as np
import pandas as pd
import pytest

from trend.sma import sma
from trend.ema import ema
from trend.rsi import rsi
from trend.macd import macd
from trend.stochastic import stochastic, stochastic_fast
from trend.ichimoku import ichimoku
from core.types import MACDResult, StochasticResult, IchimokuResult


class TestSMA:
    def test_output_length(self, sample_close):
        result = sma(sample_close, 20)
        assert len(result) == len(sample_close)

    def test_leading_nans(self, sample_close):
        result = sma(sample_close, 20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_known_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_invalid_period(self, sample_close):
        with pytest.raises(ValueError):
            sma(sample_close, 0)


class TestEMA:
    def test_output_length(self, sample_close):
        result = ema(sample_close, 20)
        assert len(result) == len(sample_close)

    def test_converges_toward_price(self):
        s = pd.Series([100.0] * 50)
        result = ema(s, 10)
        assert result.iloc[-1] == pytest.approx(100.0)


class TestRSI:
    def test_range_0_100(self, sample_close):
        result = rsi(sample_close, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_all_gains(self):
        s = pd.Series(range(1, 52), dtype=float)
        result = rsi(s, 14)
        assert result.dropna().iloc[-1] == pytest.approx(100.0)

    def test_all_losses(self):
        s = pd.Series(range(50, 0, -1), dtype=float)
        result = rsi(s, 14)
        assert result.dropna().iloc[-1] == pytest.approx(0.0, abs=0.1)

    def test_leading_nans(self, sample_close):
        result = rsi(sample_close, 14)
        assert result.iloc[:14].isna().all()


class TestMACD:
    def test_returns_macd_result(self, sample_close):
        result = macd(sample_close)
        assert isinstance(result, MACDResult)

    def test_histogram_equals_diff(self, sample_close):
        result = macd(sample_close)
        diff = result.macd - result.signal
        np.testing.assert_array_almost_equal(
            result.histogram.dropna().values, diff.dropna().values
        )

    def test_output_lengths(self, sample_close):
        result = macd(sample_close)
        assert len(result.macd) == len(sample_close)
        assert len(result.signal) == len(sample_close)
        assert len(result.histogram) == len(sample_close)


class TestStochastic:
    def test_range_0_100(self, sample_ohlcv):
        result = stochastic(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        assert isinstance(result, StochasticResult)
        valid_k = result.k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_fast_vs_slow(self, sample_ohlcv):
        slow = stochastic(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        fast = stochastic_fast(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        # Fast %K has more valid values (less smoothing)
        assert fast.k.notna().sum() >= slow.k.notna().sum()

    def test_zero_range(self):
        n = 20
        s = pd.Series([50.0] * n)
        result = stochastic(s, s, s, fastk_period=5)
        valid = result.k.dropna()
        assert (valid == 0.0).all()


class TestIchimoku:
    def test_returns_ichimoku_result(self, sample_ohlcv):
        result = ichimoku(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        assert isinstance(result, IchimokuResult)

    def test_five_components(self, sample_ohlcv):
        result = ichimoku(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        assert len(result.tenkan) == len(sample_ohlcv)
        assert len(result.kijun) == len(sample_ohlcv)
        assert len(result.senkou_a) == len(sample_ohlcv)
        assert len(result.senkou_b) == len(sample_ohlcv)
        assert len(result.chikou) == len(sample_ohlcv)

    def test_chikou_shifted_back(self, sample_ohlcv):
        result = ichimoku(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        # Chikou = close shifted -26, so last 26 are NaN
        assert result.chikou.iloc[-26:].isna().all()
