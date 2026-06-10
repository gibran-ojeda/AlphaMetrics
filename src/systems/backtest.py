"""Backtesting engine — vectorized signal-based approach.

Critical pitfalls to avoid:
- Lookahead bias (using future data)
- Survivorship bias
- Overfitting via parameter sweeps without out-of-sample validation
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from systems.costs import TransactionCostModel


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    annual_return: float
    n_trades: int
    win_rate: float


def backtest_signals(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 10_000.0,
    cost_model: TransactionCostModel | None = None,
) -> BacktestResult:
    """Run a simple long-only signal-based backtest.

    Parameters
    ----------
    close : pd.Series
        Close price series.
    entries : pd.Series
        Boolean series. True on entry bars.
    exits : pd.Series
        Boolean series. True on exit bars.
    init_cash : float, default 10000
        Initial capital.
    cost_model : TransactionCostModel or None
        Transaction cost model. None = no costs.

    Returns
    -------
    BacktestResult
        Equity curve, returns, and performance metrics.
    """
    position = pd.Series(0.0, index=close.index)
    in_position = False

    for i in range(len(close)):
        if not in_position and entries.iloc[i]:
            position.iloc[i] = 1.0
            in_position = True
        elif in_position and exits.iloc[i]:
            position.iloc[i] = 0.0
            in_position = False
        elif in_position:
            position.iloc[i] = 1.0

    # Compute returns (position applied to next-bar returns to avoid lookahead)
    market_returns = close.pct_change().fillna(0)
    strategy_returns = position.shift(1).fillna(0) * market_returns

    # Apply transaction costs
    if cost_model is not None:
        trades = position.diff().abs()
        trade_bars = trades[trades > 0]
        for idx in trade_bars.index:
            cost_frac = cost_model.total_cost(close.loc[idx], 1) / close.loc[idx]
            strategy_returns.loc[idx] -= cost_frac

    equity = init_cash * (1 + strategy_returns).cumprod()

    # Trade stats
    position_changes = position.diff()
    n_entries = (position_changes == 1).sum()
    n_exits = (position_changes == -1).sum()
    n_trades = min(n_entries, n_exits)

    # Win rate
    trade_returns = []
    entry_idx = None
    for i in range(len(position_changes)):
        if position_changes.iloc[i] == 1:
            entry_idx = i
        elif position_changes.iloc[i] == -1 and entry_idx is not None:
            trade_ret = close.iloc[i] / close.iloc[entry_idx] - 1
            trade_returns.append(trade_ret)
            entry_idx = None

    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0.0

    # Annual return
    n_days = len(close)
    total_return = float(equity.iloc[-1] / init_cash - 1)
    annual_return = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)

    return BacktestResult(
        equity_curve=equity,
        returns=strategy_returns,
        total_return=total_return,
        annual_return=annual_return,
        n_trades=int(n_trades),
        win_rate=win_rate,
    )
