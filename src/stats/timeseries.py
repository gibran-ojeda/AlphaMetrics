"""ARIMA and GARCH time series models.

ARIMA: never apply to raw prices (non-stationary). Use returns or differences.
GARCH(1,1): sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}.
Stationarity constraint: alpha + beta < 1.

arch library convention: p = GARCH lags (sigma^2 terms), q = ARCH lags (epsilon^2 terms)
— reversed from some textbooks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ARIMAResult:
    model: object
    aic: float
    bic: float
    params: dict


@dataclass(frozen=True)
class GARCHResult:
    model: object
    params: dict
    conditional_volatility: pd.Series
    converged: bool


def fit_arima(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
) -> ARIMAResult:
    """Fit ARIMA model.

    Parameters
    ----------
    series : pd.Series
        Stationary time series (returns, not prices).
    order : tuple
        (p, d, q) order.

    Returns
    -------
    ARIMAResult

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    model = ARIMA(series, order=order).fit()
    return ARIMAResult(
        model=model,
        aic=model.aic,
        bic=model.bic,
        params=dict(model.params),
    )


def auto_arima(
    series: pd.Series,
    max_p: int = 5,
    max_q: int = 5,
    seasonal: bool = False,
):
    """Auto ARIMA using Hyndman-Khandakar stepwise search.

    Parameters
    ----------
    series : pd.Series
        Stationary time series.
    max_p, max_q : int
        Maximum AR and MA orders.
    seasonal : bool
        Whether to fit seasonal component.

    Returns
    -------
    pmdarima AutoARIMA result.

    Raises
    ------
    ImportError
        If pmdarima is not installed.
    """
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError("pmdarima is required: pip install pmdarima")

    return pm.auto_arima(
        series,
        start_p=0, max_p=max_p,
        d=0,
        start_q=0, max_q=max_q,
        seasonal=seasonal,
        stepwise=True,
        information_criterion="bic",
    )


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
    mean: str = "Constant",
) -> GARCHResult:
    """Fit GARCH(p,q) model.

    Parameters
    ----------
    returns : pd.Series
        Return series. Percentage returns (*100) often converge better.
    p : int, default 1
        GARCH lags (sigma^2 terms).
    q : int, default 1
        ARCH lags (epsilon^2 terms).
    dist : str, default "t"
        Error distribution: "normal", "t", "skewt".
    mean : str, default "Constant"
        Mean model: "Constant", "Zero", "AR".

    Returns
    -------
    GARCHResult

    Raises
    ------
    ImportError
        If arch is not installed.
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch is required: pip install arch")

    am = arch_model(returns, mean=mean, vol="Garch", p=p, q=q, dist=dist)
    res = am.fit(disp="off")

    return GARCHResult(
        model=res,
        params=dict(res.params),
        conditional_volatility=res.conditional_volatility,
        converged=res.convergence_flag == 0,
    )
