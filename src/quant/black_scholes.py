"""Black-Scholes option pricing and implied volatility.

Greeks: Delta, Gamma (same for call/put), Theta (/365 for calendar day),
Vega (per 1% vol move), Rho.
At expiry (T->0): delta snaps to 0 or 1, gamma spikes then collapses, vega -> 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass(frozen=True)
class BSResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes option price.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility (annualized).
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Option price.
    """
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if sigma <= 0:
        df = np.exp(-r * T)
        return max(S - K * df, 0.0) if option_type == "call" else max(K * df - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> BSResult:
    """Compute Black-Scholes price and all Greeks.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    option_type : str
        "call" or "put".

    Returns
    -------
    BSResult
        price, delta, gamma, theta, vega, rho.
    """
    price = bs_price(S, K, T, r, sigma, option_type)

    if T <= 0 or sigma <= 0:
        return BSResult(price=price, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    gamma = float(norm.pdf(d1) / (S * sigma * sqrt_t))
    vega = float(S * norm.pdf(d1) * sqrt_t / 100)  # Per 1% vol move

    if option_type == "call":
        delta = float(norm.cdf(d1))
        theta = float(
            (-S * norm.pdf(d1) * sigma / (2 * sqrt_t) - r * K * np.exp(-r * T) * norm.cdf(d2))
            / 365
        )
        rho = float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100)
    else:
        delta = float(norm.cdf(d1) - 1)
        theta = float(
            (-S * norm.pdf(d1) * sigma / (2 * sqrt_t) + r * K * np.exp(-r * T) * norm.cdf(-d2))
            / 365
        )
        rho = float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)

    return BSResult(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Compute implied volatility via Brent's method.

    Parameters
    ----------
    market_price : float
        Observed market price.
    S, K, T, r : float
        Standard BS parameters.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Implied volatility. NaN if no solution found.
    """
    obj = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
    try:
        return float(brentq(obj, 1e-6, 10.0, xtol=1e-10))
    except ValueError:
        return np.nan
