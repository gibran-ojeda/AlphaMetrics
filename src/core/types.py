"""Shared result dataclasses for multi-output indicators."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MACDResult:
    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


@dataclass(frozen=True)
class BollingerResult:
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    pct_b: pd.Series
    bandwidth: pd.Series


@dataclass(frozen=True)
class StochasticResult:
    k: pd.Series
    d: pd.Series


@dataclass(frozen=True)
class IchimokuResult:
    tenkan: pd.Series
    kijun: pd.Series
    senkou_a: pd.Series
    senkou_b: pd.Series
    chikou: pd.Series


@dataclass(frozen=True)
class ADXResult:
    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


@dataclass(frozen=True)
class DrawdownResult:
    drawdown: pd.Series
    max_drawdown: float
    peak_date: object
    trough_date: object
    recovery_date: object | None
    duration: int | None
