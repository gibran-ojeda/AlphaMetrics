"""Linear regression (OLS) with statsmodels.

Common pitfall: forgetting sm.add_constant() forces regression through origin.
For time series: check heteroscedasticity, autocorrelation, multicollinearity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass(frozen=True)
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    stderr: float


def simple_ols(x: pd.Series, y: pd.Series) -> RegressionResult:
    """Simple bivariate OLS regression using scipy.

    Parameters
    ----------
    x : pd.Series
        Independent variable (e.g., market returns).
    y : pd.Series
        Dependent variable (e.g., stock returns).

    Returns
    -------
    RegressionResult
        slope, intercept, r_squared, p_value, stderr.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    result = linregress(x[mask], y[mask])
    return RegressionResult(
        slope=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue**2,
        p_value=result.pvalue,
        stderr=result.stderr,
    )


def ols_statsmodels(y: pd.Series, X: pd.DataFrame, hac_lags: int | None = None):
    """OLS regression with statsmodels (full diagnostics).

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    X : pd.DataFrame
        Independent variables (constant NOT added automatically).
    hac_lags : int or None
        If provided, use HAC (Newey-West) standard errors.

    Returns
    -------
    statsmodels RegressionResultsWrapper

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    X = sm.add_constant(X)
    if hac_lags is not None:
        return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return sm.OLS(y, X).fit()


def rolling_ols(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling OLS regression (vectorized via covariance).

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    x : pd.Series
        Independent variable.
    window : int, default 60
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Columns: beta, alpha, r_squared.
    """
    cov_xy = y.rolling(window).cov(x)
    var_x = x.rolling(window).var()
    beta = cov_xy / var_x

    mean_y = y.rolling(window).mean()
    mean_x = x.rolling(window).mean()
    alpha = mean_y - beta * mean_x

    ss_res = ((y - alpha - beta * x) ** 2).rolling(window).sum()
    ss_tot = ((y - mean_y) ** 2).rolling(window).sum()
    r_squared = 1 - ss_res / ss_tot.replace(0, np.nan)

    return pd.DataFrame({"beta": beta, "alpha": alpha, "r_squared": r_squared})
