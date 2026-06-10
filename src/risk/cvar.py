"""Expected Shortfall (CVaR / Conditional VaR).

Mean of losses beyond VaR — a coherent risk measure (subadditive, unlike VaR).
Required by Basel III/IV for market risk since 2019.
ES should always >= VaR for the same confidence level.

Edge case: with 250 observations at 99% confidence, only ~2-3 observations land
in the tail. Bootstrap confidence intervals are essential for reliability.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall (CVaR).

    Parameters
    ----------
    returns : array-like
        Period returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        CVaR as positive loss magnitude. NaN if no tail observations.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return np.nan
    return float(-np.mean(tail))


def cvar_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) Expected Shortfall.

    Formula: ES = -mu + sigma * phi(Phi^{-1}(alpha)) / alpha

    Parameters
    ----------
    returns : array-like
        Period returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        CVaR as positive loss magnitude.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=0)
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    return float(-mu + sigma * norm.pdf(z) / alpha)
