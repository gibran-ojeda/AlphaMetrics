"""Monte Carlo simulation: Geometric Brownian Motion and Bootstrap.

Always use the log scheme (exp(cumsum)) — Euler discretization can produce
negative prices. MC converges at O(1/sqrt(N)), so 4x paths gives 2x accuracy.

Memory: 1M paths x 252 steps x 8 bytes ~ 2GB. Process in batches for large sims.
Always use np.random.default_rng(seed), not legacy np.random.seed().
"""
from __future__ import annotations

import numpy as np


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate GBM price paths (vectorized).

    S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*sqrt(t)*Z)

    Parameters
    ----------
    S0 : float
        Initial price.
    mu : float
        Drift (annualized expected return).
    sigma : float
        Volatility (annualized).
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulation paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Price paths, shape (n_steps + 1, n_paths).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_steps, n_paths))
    log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(log_rets, axis=0)])
    return S0 * np.exp(log_paths)


def bootstrap_paths(
    S0: float,
    hist_log_returns: np.ndarray,
    n_days: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate bootstrap simulation paths (non-parametric, preserves fat tails).

    Parameters
    ----------
    S0 : float
        Initial price.
    hist_log_returns : np.ndarray
        Historical log returns to sample from.
    n_days : int
        Number of days to simulate.
    n_paths : int
        Number of simulation paths.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Price paths, shape (n_days + 1, n_paths).
    """
    rng = np.random.default_rng(seed)
    hist = np.asarray(hist_log_returns, dtype=np.float64)
    indices = rng.integers(0, len(hist), (n_days, n_paths))
    sampled = hist[indices]
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(sampled, axis=0)])
    return S0 * np.exp(log_paths)
