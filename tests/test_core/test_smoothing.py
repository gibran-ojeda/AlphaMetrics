"""Tests for core.smoothing."""
import numpy as np
import pandas as pd
import pytest

from core.smoothing import wilder_smooth, ema_smooth


class TestWilderSmooth:
    def test_nan_count(self, sample_close):
        result = wilder_smooth(sample_close, 14)
        assert result.iloc[:13].isna().all()
        assert result.iloc[13:].notna().all()

    def test_seed_is_sma(self, sample_close):
        period = 14
        result = wilder_smooth(sample_close, period)
        expected_seed = sample_close.iloc[:period].mean()
        assert result.iloc[period - 1] == pytest.approx(expected_seed)

    def test_insufficient_data(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = wilder_smooth(s, 10)
        assert result.isna().all()


class TestEMASmooth:
    def test_sma_seed_nan_count(self, sample_close):
        result = ema_smooth(sample_close, 20, sma_seed=True)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_sma_seed_vs_first_value(self, sample_close):
        r1 = ema_smooth(sample_close, 10, sma_seed=True)
        r2 = ema_smooth(sample_close, 10, sma_seed=False)
        # They should differ because of different seeding
        assert not np.allclose(r1.dropna().values, r2.dropna().values[:len(r1.dropna())])

    def test_insufficient_data(self):
        s = pd.Series([1.0, 2.0])
        result = ema_smooth(s, 10, sma_seed=True)
        assert result.isna().all()


class TestWilderVsEMA:
    def test_different_results(self, sample_close):
        w = wilder_smooth(sample_close, 14)
        e = ema_smooth(sample_close, 14, sma_seed=True)
        # Same seed but different alpha -> diverge after seed
        common = w.iloc[14:].values
        common_e = e.iloc[14:].values[:len(common)]
        assert not np.allclose(common, common_e)
