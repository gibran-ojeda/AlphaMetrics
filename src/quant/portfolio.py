"""Portfolio optimization (Markowitz mean-variance).

Edge cases:
- Singular covariance -> use Ledoit-Wolf shrinkage or add 1e-6 * I.
- Corner solutions (100% in one asset) -> add L2 regularization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float


def max_sharpe_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
) -> PortfolioResult:
    """Find the maximum Sharpe ratio portfolio.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector (annualized).
    cov : np.ndarray
        Covariance matrix (annualized).
    rf : float, default 0.0
        Risk-free rate.

    Returns
    -------
    PortfolioResult
        Optimal weights and portfolio metrics.
    """
    n = len(mu)
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            return 1e10
        return -(ret - rf) / vol

    result = minimize(
        neg_sharpe,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
    )

    w = result.x
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0

    return PortfolioResult(weights=w, expected_return=ret, volatility=vol, sharpe=sharpe)


def min_variance_portfolio(
    cov: np.ndarray,
) -> PortfolioResult:
    """Find the minimum variance portfolio.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.

    Returns
    -------
    PortfolioResult
        Optimal weights and portfolio metrics.
    """
    n = cov.shape[0]
    cov = np.asarray(cov, dtype=np.float64)

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)

    result = minimize(
        portfolio_vol,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
    )

    w = result.x
    vol = float(np.sqrt(w @ cov @ w))

    return PortfolioResult(weights=w, expected_return=0.0, volatility=vol, sharpe=0.0)
