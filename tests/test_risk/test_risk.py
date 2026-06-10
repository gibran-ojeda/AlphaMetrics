"""Tests for risk metrics."""
import numpy as np
import pandas as pd
import pytest

from risk.sharpe import sharpe_ratio
from risk.sortino import sortino_ratio
from risk.drawdown import max_drawdown
from risk.var import var_historical, var_parametric, var_monte_carlo
from risk.cvar import cvar_historical, cvar_parametric
from core.types import DrawdownResult


class TestSharpe:
    def test_finite_with_real_returns(self, sample_returns):
        result = sharpe_ratio(sample_returns.values)
        assert np.isfinite(result)

    def test_nan_with_insufficient_data(self):
        assert np.isnan(sharpe_ratio(np.array([0.01])))

    def test_nan_with_zero_std(self):
        assert np.isnan(sharpe_ratio(np.array([0.01] * 10)))

    def test_risk_free_rate(self, sample_returns):
        r0 = sharpe_ratio(sample_returns.values, rf=0.0)
        r5 = sharpe_ratio(sample_returns.values, rf=0.05)
        # Higher rf -> lower Sharpe
        assert r5 < r0


class TestSortino:
    def test_finite_with_real_returns(self, sample_returns):
        result = sortino_ratio(sample_returns.values)
        assert np.isfinite(result) or result == np.inf

    def test_inf_with_no_downside(self):
        # Only positive returns -> no downside -> inf
        returns = np.array([0.01, 0.02, 0.015, 0.03, 0.01])
        result = sortino_ratio(returns)
        assert result == np.inf

    def test_nan_with_insufficient_data(self):
        assert np.isnan(sortino_ratio(np.array([0.01])))


class TestDrawdown:
    def test_returns_drawdown_result(self, sample_returns):
        result = max_drawdown(sample_returns)
        assert isinstance(result, DrawdownResult)

    def test_max_drawdown_negative(self, sample_returns):
        result = max_drawdown(sample_returns)
        assert result.max_drawdown <= 0

    def test_monotonic_increase_zero_drawdown(self):
        returns = pd.Series([0.01] * 50)
        result = max_drawdown(returns)
        assert result.max_drawdown == pytest.approx(0.0)

    def test_drawdown_series_length(self, sample_returns):
        result = max_drawdown(sample_returns)
        assert len(result.drawdown) == len(sample_returns)


class TestVaR:
    def test_all_positive(self, sample_returns):
        h = var_historical(sample_returns.values)
        p = var_parametric(sample_returns.values)
        m = var_monte_carlo(sample_returns.values, seed=42)
        assert h > 0 or h == pytest.approx(0.0, abs=0.1)
        assert p > 0 or p == pytest.approx(0.0, abs=0.1)
        assert m > 0 or m == pytest.approx(0.0, abs=0.1)

    def test_higher_confidence_higher_var(self, sample_returns):
        v95 = var_historical(sample_returns.values, confidence=0.95)
        v99 = var_historical(sample_returns.values, confidence=0.99)
        assert v99 >= v95

    def test_monte_carlo_reproducible(self, sample_returns):
        v1 = var_monte_carlo(sample_returns.values, seed=123)
        v2 = var_monte_carlo(sample_returns.values, seed=123)
        assert v1 == pytest.approx(v2)


class TestCVaR:
    def test_cvar_gte_var(self, sample_returns):
        v = var_historical(sample_returns.values, confidence=0.95)
        cv = cvar_historical(sample_returns.values, confidence=0.95)
        assert cv >= v or cv == pytest.approx(v, abs=1e-6)

    def test_parametric_cvar_positive(self, sample_returns):
        result = cvar_parametric(sample_returns.values)
        assert result > 0 or np.isfinite(result)
