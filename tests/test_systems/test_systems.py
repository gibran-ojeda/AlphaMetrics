"""Tests for systems modules."""
import numpy as np
import pandas as pd
import pytest

from systems.costs import TransactionCostModel, slippage_percentage, market_impact_sqroot
from systems.stop_loss import fixed_stop, trailing_stop, atr_stop
from systems.order_types import OrderSimulator, FillResult
from systems.position_sizing import kelly_discrete, kelly_continuous, position_size
from systems.walk_forward import rolling_walk_forward, expanding_walk_forward
from systems.backtest import backtest_signals


class TestCosts:
    def test_total_cost_positive(self):
        model = TransactionCostModel()
        cost = model.total_cost(price=100.0, shares=100)
        assert cost > 0

    def test_min_commission(self):
        model = TransactionCostModel(commission_per_share=0.001, min_commission=5.0)
        cost_comm = model.total_cost(price=100.0, shares=1)
        # 1 share * 0.001 = 0.001 < min 5.0
        assert cost_comm >= 5.0

    def test_slippage_buy_increases(self):
        fill = slippage_percentage(100.0, 0.001, "buy")
        assert fill > 100.0

    def test_slippage_sell_decreases(self):
        fill = slippage_percentage(100.0, 0.001, "sell")
        assert fill < 100.0

    def test_market_impact_positive(self):
        impact = market_impact_sqroot(sigma=0.02, order_qty=10000, market_volume=1_000_000)
        assert impact > 0


class TestStopLoss:
    def test_fixed_stop_constant_level(self):
        prices = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0])
        result = fixed_stop(100.0, prices, stop_pct=0.03)
        # Stop at 97.0; price hits 97.0 at index 3
        assert result.stop_level.iloc[0] == pytest.approx(97.0)
        assert result.triggered_date is not None

    def test_trailing_stop_rises_with_price(self):
        prices = pd.Series([100.0, 105.0, 110.0, 108.0, 106.0],
                           index=pd.date_range("2024-01-01", periods=5))
        result = trailing_stop(prices, trail_pct=0.05)
        # Stop level should rise as price rises
        assert result.stop_level.iloc[2] > result.stop_level.iloc[0]

    def test_atr_stop_level(self):
        atr_vals = pd.Series([2.0, 2.5, 3.0])
        stops = atr_stop(entry_price=100.0, atr_values=atr_vals, multiplier=2.0)
        assert stops.iloc[0] == pytest.approx(96.0)
        assert stops.iloc[2] == pytest.approx(94.0)


class TestOrderTypes:
    def setup_method(self):
        self.sim = OrderSimulator()
        self.bar = {"open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0}

    def test_market_fill_at_open(self):
        result = self.sim.fill_market(self.bar)
        assert result.filled is True
        assert result.fill_price == 100.0

    def test_limit_buy_fills_when_low_below_limit(self):
        result = self.sim.fill_limit(self.bar, limit_price=99.0, side="buy")
        assert result.filled is True
        assert result.fill_price <= 99.0

    def test_limit_buy_no_fill_when_low_above_limit(self):
        result = self.sim.fill_limit(self.bar, limit_price=97.0, side="buy")
        assert result.filled is False

    def test_stop_sell_triggers(self):
        result = self.sim.fill_stop(self.bar, stop_price=99.0, side="sell")
        assert result.filled is True


class TestPositionSizing:
    def test_kelly_negative_means_no_trade(self):
        # Low win rate, low payoff -> negative Kelly
        k = kelly_discrete(win_rate=0.3, avg_win=1.0, avg_loss=2.0)
        assert k < 0

    def test_position_size_zero_for_negative_kelly(self):
        assert position_size(-0.5) == 0.0

    def test_half_kelly(self):
        k = kelly_discrete(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        full = position_size(k, fraction=1.0, max_pct=1.0)
        half = position_size(k, fraction=0.5, max_pct=1.0)
        assert half == pytest.approx(full * 0.5)

    def test_cap_at_max_pct(self):
        assert position_size(kelly_fraction=10.0, max_pct=0.25) <= 0.25

    def test_kelly_continuous(self):
        returns = np.array([0.01, 0.02, -0.005, 0.015, -0.01])
        k = kelly_continuous(returns)
        assert np.isfinite(k)


class TestWalkForward:
    def test_rolling_fold_sizes(self):
        data = pd.Series(range(500), dtype=float)
        folds = list(rolling_walk_forward(data, train_size=252, test_size=63, step_size=21))
        assert len(folds) > 0
        for train, test in folds:
            assert len(train) == 252
            assert len(test) == 63

    def test_expanding_train_grows(self):
        data = pd.Series(range(500), dtype=float)
        folds = list(expanding_walk_forward(data, min_train_size=100, test_size=50, step_size=50))
        assert len(folds) >= 2
        assert len(folds[1][0]) > len(folds[0][0])

    def test_no_overlap_between_train_test(self):
        data = pd.Series(range(500), dtype=float)
        for train, test in rolling_walk_forward(data, train_size=100, test_size=50, step_size=50):
            train_indices = set(train.index)
            test_indices = set(test.index)
            assert train_indices.isdisjoint(test_indices)


class TestBacktest:
    def test_equity_starts_at_init_cash(self, sample_close):
        entries = pd.Series(False, index=sample_close.index)
        exits = pd.Series(False, index=sample_close.index)
        result = backtest_signals(sample_close, entries, exits, init_cash=10_000)
        assert result.equity_curve.iloc[0] == pytest.approx(10_000)

    def test_no_trades_when_no_signals(self, sample_close):
        entries = pd.Series(False, index=sample_close.index)
        exits = pd.Series(False, index=sample_close.index)
        result = backtest_signals(sample_close, entries, exits)
        assert result.n_trades == 0

    def test_costs_reduce_returns(self, sample_close):
        entries = pd.Series(False, index=sample_close.index)
        exits = pd.Series(False, index=sample_close.index)
        entries.iloc[5] = True
        exits.iloc[50] = True

        no_cost = backtest_signals(sample_close, entries, exits)
        with_cost = backtest_signals(
            sample_close, entries, exits, cost_model=TransactionCostModel()
        )
        assert with_cost.total_return <= no_cost.total_return
