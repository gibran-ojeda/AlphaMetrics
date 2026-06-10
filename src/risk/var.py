"""Value at Risk (VaR) — three methods.

Sign convention: returns POSITIVE values representing loss magnitude.
95% confidence z-score = 1.645, 99% = 2.326, 99.9% = 3.090.
Time horizon scaling: VaR_T = VaR_1 * sqrt(T) (only under i.i.d. normal assumption).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR (empirical quantile).

    Parameters
    ----------
    returns : array-like
        Period returns.
    confidence : float, default 0.95
        Confidence level (0.95 = 95%).

    Returns
    -------
    float
        VaR as positive loss magnitude.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    return float(-np.percentile(returns, (1 - confidence) * 100))


def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR.

    Parameters
    ----------
    returns : array-like
        Period returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        VaR as positive loss magnitude.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=0)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def var_monte_carlo(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR.

    Parameters
    ----------
    returns : array-like
        Period returns (used to estimate mu and sigma).
    confidence : float, default 0.95
        Confidence level.
    n_sims : int, default 10000
        Number of simulations.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    float
        VaR as positive loss magnitude.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=0)
    rng = np.random.default_rng(seed)
    sim = rng.normal(mu, sigma, n_sims)
    return float(-np.percentile(sim, (1 - confidence) * 100))
