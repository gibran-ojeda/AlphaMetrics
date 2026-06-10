"""Tests for stats modules."""
import numpy as np
import pandas as pd
import pytest

from stats.returns import simple_returns, log_returns
from stats.volatility_models import rolling_volatility, ewma_volatility, parkinson_volatility
from stats.correlation import correlation_matrix
from stats.regression import simple_ols


class TestReturns:
    def test_first_value_nan(self, sample_close):
        sr = simple_returns(sample_close)
        lr = log_returns(sample_close)
        assert pd.isna(sr.iloc[0])
        assert pd.isna(lr.iloc[0])

    def test_log_approx_simple_for_small_returns(self):
        prices = pd.Series([100.0, 100.5, 101.0, 100.8])
        sr = simple_returns(prices).dropna()
        lr = log_returns(prices).dropna()
        np.testing.assert_array_almost_equal(sr.values, lr.values, decimal=3)

    def test_log_returns_additive(self, sample_close):
        lr = log_returns(sample_close).dropna()
        # Sum of log returns = total log return
        total_log = np.log(sample_close.iloc[-1] / sample_close.iloc[0])
        assert lr.sum() == pytest.approx(total_log, rel=1e-10)

    def test_dataframe_input(self):
        df = pd.DataFrame({"A": [100.0, 110.0, 105.0], "B": [50.0, 52.0, 51.0]})
        sr = simple_returns(df)
        assert sr.shape == df.shape
        assert list(sr.columns) == ["A", "B"]


class TestVolatilityModels:
    def test_rolling_positive(self, sample_returns):
        vol = rolling_volatility(sample_returns, window=21)
        valid = vol.dropna()
        assert (valid > 0).all()

    def test_ewma_vs_rolling_different(self, sample_returns):
        r = rolling_volatility(sample_returns, window=21)
        e = ewma_volatility(sample_returns, span=21)
        # They should differ (EWMA weights recent observations more)
        common = r.dropna().index.intersection(e.dropna().index)
        assert not np.allclose(r[common].values, e[common].values)

    def test_parkinson_positive(self, sample_ohlcv):
        vol = parkinson_volatility(sample_ohlcv["high"], sample_ohlcv["low"], window=21)
        valid = vol.dropna()
        assert (valid > 0).all()

    def test_annualization(self, sample_returns):
        ann = rolling_volatility(sample_returns, window=21, annualize=True)
        raw = rolling_volatility(sample_returns, window=21, annualize=False)
        valid = ann.dropna().index
        ratio = ann[valid].values / raw[valid].values
        np.testing.assert_array_almost_equal(ratio, np.sqrt(252), decimal=5)


class TestCorrelation:
    def test_symmetric(self):
        np.random.seed(42)
        df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
        corr = correlation_matrix(df)
        np.testing.assert_array_almost_equal(corr.values, corr.values.T)

    def test_diagonal_ones(self):
        np.random.seed(42)
        df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
        corr = correlation_matrix(df)
        np.testing.assert_array_almost_equal(np.diag(corr.values), [1.0, 1.0])

    def test_range(self):
        np.random.seed(42)
        df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
        corr = correlation_matrix(df)
        assert (corr.values >= -1).all()
        assert (corr.values <= 1).all()


class TestRegression:
    def test_perfect_linear(self):
        x = pd.Series(np.arange(50, dtype=float))
        y = 2.0 * x + 5.0
        result = simple_ols(x, y)
        assert result.slope == pytest.approx(2.0)
        assert result.intercept == pytest.approx(5.0)
        assert result.r_squared == pytest.approx(1.0)

    def test_r_squared_range(self, sample_returns):
        x = pd.Series(np.random.randn(len(sample_returns)))
        result = simple_ols(x, sample_returns)
        assert 0 <= result.r_squared <= 1
