"""Position sizing with Kelly Criterion.

Half-Kelly is standard practice — reduces variance by 50% while giving up
only ~25% of growth. Cap position size at 25% maximum.
Negative Kelly means don't trade (negative expectancy).
"""
from __future__ import annotations

import numpy as np


def kelly_discrete(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Discrete Kelly fraction: f* = (b*p - q) / b.

    Parameters
    ----------
    win_rate : float
        Probability of winning (0-1).
    avg_win : float
        Average win amount (positive).
    avg_loss : float
        Average loss amount (positive or negative, abs is used).

    Returns
    -------
    float
        Kelly fraction. Negative means don't trade.
    """
    b = avg_win / abs(avg_loss)  # Payoff ratio
    q = 1 - win_rate
    return (b * win_rate - q) / b


def kelly_continuous(returns: np.ndarray) -> float:
    """Continuous Kelly fraction: f* = mu / sigma^2.

    Parameters
    ----------
    returns : array-like
        Historical returns.

    Returns
    -------
    float
        Kelly fraction.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    var = np.var(returns)
    if var < 1e-12:
        return 0.0
    return float(np.mean(returns) / var)


def position_size(
    kelly_fraction: float,
    fraction: float = 0.5,
    max_pct: float = 0.25,
) -> float:
    """Compute practical position size from Kelly.

    Parameters
    ----------
    kelly_fraction : float
        Raw Kelly fraction.
    fraction : float, default 0.5
        Fractional Kelly (0.5 = half-Kelly).
    max_pct : float, default 0.25
        Maximum position size cap.

    Returns
    -------
    float
        Position size as fraction of capital (0 if Kelly is negative).
    """
    if kelly_fraction <= 0:
        return 0.0
    return min(kelly_fraction * fraction, max_pct)
