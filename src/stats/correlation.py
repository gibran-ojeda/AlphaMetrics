"""Correlation matrix and covariance estimation.

Includes Ledoit-Wolf shrinkage for well-conditioned, always PSD covariance matrices.
Rolling/subsampled correlation matrices can lose positive semi-definiteness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series for multiple assets.
    method : str, default "pearson"
        Correlation method: "pearson", "spearman", or "kendall".

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    return returns.corr(method=method)


def rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Compute pairwise rolling correlation.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series for multiple assets.
    window : int, default 60
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Multi-indexed rolling correlation.
    """
    return returns.rolling(window).corr()


def covariance_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Compute Ledoit-Wolf shrinkage covariance matrix.

    Requires scikit-learn. Always PSD, well-conditioned.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series (rows=observations, columns=assets). NaN rows dropped.

    Returns
    -------
    tuple[np.ndarray, float]
        (covariance_matrix, shrinkage_coefficient).

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError("scikit-learn is required for Ledoit-Wolf shrinkage: pip install scikit-learn")

    clean = returns.dropna().values
    lw = LedoitWolf().fit(clean)
    return lw.covariance_, lw.shrinkage_


def is_psd(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is positive semi-definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigenvalues >= -tol))
