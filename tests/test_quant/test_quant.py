"""Tests for quant modules."""
import numpy as np
import pytest

from quant.black_scholes import bs_price, bs_greeks, implied_vol, BSResult
from quant.portfolio import max_sharpe_portfolio, min_variance_portfolio
from quant.capm import capm_beta, rolling_beta
from quant.monte_carlo import gbm_paths, bootstrap_paths

import pandas as pd


class TestBlackScholes:
    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        call = bs_price(S, K, T, r, sigma, "call")
        put = bs_price(S, K, T, r, sigma, "put")
        # C - P = S - K*exp(-rT)
        parity = S - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, rel=1e-6)

    def test_at_expiry_intrinsic(self):
        # ITM call at expiry
        assert bs_price(110, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(10.0)
        # OTM call at expiry
        assert bs_price(90, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(0.0)
        # ITM put at expiry
        assert bs_price(90, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(10.0)

    def test_greeks_returns_result(self):
        result = bs_greeks(100, 100, 1.0, 0.05, 0.2)
        assert isinstance(result, BSResult)
        assert 0 < result.delta < 1  # ATM call delta ~ 0.5
        assert result.gamma > 0
        assert result.vega > 0

    def test_implied_vol_round_trip(self):
        S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.25
        price = bs_price(S, K, T, r, sigma, "call")
        iv = implied_vol(price, S, K, T, r, "call")
        assert iv == pytest.approx(sigma, rel=1e-6)

    def test_implied_vol_no_solution(self):
        # Price below intrinsic value -> no valid vol
        # Deep ITM put priced below intrinsic: impossible
        result = implied_vol(0.0, 100, 100, 0.05, 1.0, "call")
        assert np.isnan(result)


class TestPortfolio:
    def test_weights_sum_to_one(self):
        mu = np.array([0.10, 0.15, 0.12])
        cov = np.array([[0.04, 0.006, 0.002], [0.006, 0.09, 0.009], [0.002, 0.009, 0.01]])
        result = max_sharpe_portfolio(mu, cov)
        assert np.sum(result.weights) == pytest.approx(1.0)

    def test_weights_non_negative(self):
        mu = np.array([0.10, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = max_sharpe_portfolio(mu, cov)
        assert (result.weights >= -1e-6).all()

    def test_volatility_positive(self):
        mu = np.array([0.10, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = max_sharpe_portfolio(mu, cov)
        assert result.volatility > 0

    def test_min_variance(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = min_variance_portfolio(cov)
        assert np.sum(result.weights) == pytest.approx(1.0)
        assert result.volatility > 0


class TestCAPM:
    def test_beta_one_when_same(self):
        np.random.seed(42)
        market = pd.Series(np.random.randn(100) * 0.01)
        result = capm_beta(market, market)
        assert result.beta == pytest.approx(1.0)
        assert result.r_squared == pytest.approx(1.0)

    def test_adjusted_beta_formula(self):
        np.random.seed(42)
        stock = pd.Series(np.random.randn(100) * 0.015)
        market = pd.Series(np.random.randn(100) * 0.01)
        result = capm_beta(stock, market)
        expected_adj = (2 / 3) * result.beta + (1 / 3) * 1.0
        assert result.adjusted_beta == pytest.approx(expected_adj)

    def test_rolling_beta_length(self):
        np.random.seed(42)
        stock = pd.Series(np.random.randn(100) * 0.01)
        market = pd.Series(np.random.randn(100) * 0.01)
        result = rolling_beta(stock, market, window=20)
        assert len(result) == 100


class TestMonteCarlo:
    def test_gbm_shape(self):
        paths = gbm_paths(100, 0.05, 0.2, 1.0, 252, 1000, seed=42)
        assert paths.shape == (253, 1000)

    def test_gbm_starts_at_s0(self):
        paths = gbm_paths(100, 0.05, 0.2, 1.0, 10, 50, seed=42)
        np.testing.assert_array_equal(paths[0, :], 100.0)

    def test_gbm_reproducible(self):
        p1 = gbm_paths(100, 0.05, 0.2, 1.0, 10, 5, seed=99)
        p2 = gbm_paths(100, 0.05, 0.2, 1.0, 10, 5, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_gbm_positive_prices(self):
        paths = gbm_paths(100, -0.1, 0.5, 2.0, 500, 100, seed=42)
        assert (paths > 0).all()

    def test_bootstrap_shape(self):
        hist = np.random.randn(250) * 0.01
        paths = bootstrap_paths(100, hist, 60, 500, seed=42)
        assert paths.shape == (61, 500)

    def test_bootstrap_starts_at_s0(self):
        hist = np.random.randn(250) * 0.01
        paths = bootstrap_paths(50, hist, 30, 100, seed=42)
        np.testing.assert_array_equal(paths[0, :], 50.0)
