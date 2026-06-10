"""Slippage and transaction cost models.

Typical costs:
- Liquid large-cap round-trip: 5-15 bps (commission + spread + impact)
- Small-cap: 50-100+ bps
- IB Fixed pricing: $0.005/share, min $1.00
- Zero-commission brokers still incur spread cost ~5 bps
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransactionCostModel:
    """Configurable transaction cost model.

    Parameters
    ----------
    commission_per_share : float
        Commission per share (e.g., 0.005 for IB Fixed).
    half_spread_bps : float
        Half bid-ask spread in basis points.
    min_commission : float
        Minimum commission per order.
    """

    commission_per_share: float = 0.005
    half_spread_bps: float = 2.5
    min_commission: float = 1.0

    def total_cost(self, price: float, shares: float) -> float:
        """Compute total transaction cost.

        Parameters
        ----------
        price : float
            Execution price per share.
        shares : float
            Number of shares (sign ignored).

        Returns
        -------
        float
            Total cost (commission + spread).
        """
        abs_shares = abs(shares)
        commission = max(abs_shares * self.commission_per_share, self.min_commission)
        spread_cost = abs_shares * price * (self.half_spread_bps / 10_000)
        return commission + spread_cost


def slippage_percentage(price: float, slippage_pct: float, side: str = "buy") -> float:
    """Apply percentage slippage to a price.

    Parameters
    ----------
    price : float
        Original price.
    slippage_pct : float
        Slippage as decimal (e.g., 0.001 = 10 bps).
    side : str
        "buy" (slippage increases price) or "sell" (decreases).

    Returns
    -------
    float
        Adjusted fill price.
    """
    sign = 1 if side == "buy" else -1
    return price * (1 + sign * slippage_pct)


def market_impact_sqroot(
    sigma: float,
    order_qty: float,
    market_volume: float,
    eta: float = 0.1,
) -> float:
    """Square-root market impact model (Almgren).

    MI = eta * sigma * sqrt(V_order / V_market)

    Parameters
    ----------
    sigma : float
        Daily volatility.
    order_qty : float
        Order quantity.
    market_volume : float
        Average daily market volume.
    eta : float, default 0.1
        Impact coefficient.

    Returns
    -------
    float
        Expected market impact as price fraction.
    """
    if market_volume <= 0:
        return np.nan
    return eta * sigma * np.sqrt(abs(order_qty) / market_volume)
