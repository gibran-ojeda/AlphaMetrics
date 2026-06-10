"""Order types simulation.

Stop-limit orders require two-stage logic: first check if stop triggers,
then check if limit is achievable. Gap openings beyond limit/stop prices
are the primary edge case.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillResult:
    filled: bool
    fill_price: float | None


class OrderSimulator:
    """Simulate order fills against OHLC bar data."""

    def fill_market(self, bar: dict) -> FillResult:
        """Market order: fill at next bar open.

        Parameters
        ----------
        bar : dict
            OHLC bar with keys: open, high, low, close.

        Returns
        -------
        FillResult
        """
        return FillResult(filled=True, fill_price=bar["open"])

    def fill_limit(
        self,
        bar: dict,
        limit_price: float,
        side: str = "buy",
    ) -> FillResult:
        """Limit order fill simulation.

        Parameters
        ----------
        bar : dict
            OHLC bar.
        limit_price : float
            Limit price.
        side : str
            "buy" or "sell".

        Returns
        -------
        FillResult
        """
        if side == "buy" and bar["low"] <= limit_price:
            return FillResult(filled=True, fill_price=min(bar["open"], limit_price))
        if side == "sell" and bar["high"] >= limit_price:
            return FillResult(filled=True, fill_price=max(bar["open"], limit_price))
        return FillResult(filled=False, fill_price=None)

    def fill_stop(
        self,
        bar: dict,
        stop_price: float,
        side: str = "sell",
    ) -> FillResult:
        """Stop order fill simulation.

        Parameters
        ----------
        bar : dict
            OHLC bar.
        stop_price : float
            Stop trigger price.
        side : str
            "sell" (stop-loss) or "buy" (stop-entry).

        Returns
        -------
        FillResult
        """
        if side == "sell" and bar["low"] <= stop_price:
            fill = min(bar["open"], stop_price) if bar["open"] > stop_price else bar["open"]
            return FillResult(filled=True, fill_price=fill)
        if side == "buy" and bar["high"] >= stop_price:
            fill = max(bar["open"], stop_price) if bar["open"] < stop_price else bar["open"]
            return FillResult(filled=True, fill_price=fill)
        return FillResult(filled=False, fill_price=None)

    def fill_stop_limit(
        self,
        bar: dict,
        stop_price: float,
        limit_price: float,
        side: str = "sell",
    ) -> FillResult:
        """Stop-limit order: two-stage logic.

        First checks if stop triggers, then checks if limit is achievable.

        Parameters
        ----------
        bar : dict
            OHLC bar.
        stop_price : float
            Stop trigger price.
        limit_price : float
            Limit price after stop triggers.
        side : str
            "sell" or "buy".

        Returns
        -------
        FillResult
        """
        stop_result = self.fill_stop(bar, stop_price, side)
        if not stop_result.filled:
            return FillResult(filled=False, fill_price=None)
        return self.fill_limit(bar, limit_price, side)
