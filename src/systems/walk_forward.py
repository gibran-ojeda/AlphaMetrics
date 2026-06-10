"""Walk-forward validation.

Parameters: train_size=252 (1 year), test_size=63 (1 quarter), step_size=21 (1 month).
Gap parameter prevents data leakage from overlapping features/labels.
Rolling windows adapt faster to regime changes than expanding windows.
"""
from __future__ import annotations

from typing import Generator

import pandas as pd


def rolling_walk_forward(
    data: pd.DataFrame | pd.Series,
    train_size: int = 252,
    test_size: int = 63,
    step_size: int = 21,
    gap: int = 0,
) -> Generator[tuple, None, None]:
    """Rolling window walk-forward validation.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Time-indexed data.
    train_size : int, default 252
        Training window size (1 year).
    test_size : int, default 63
        Test window size (1 quarter).
    step_size : int, default 21
        Step between windows (1 month).
    gap : int, default 0
        Gap between train and test to prevent data leakage.

    Yields
    ------
    tuple[pd.DataFrame|pd.Series, pd.DataFrame|pd.Series]
        (train_data, test_data) for each fold.
    """
    total = len(data)
    for start in range(0, total - train_size - gap - test_size + 1, step_size):
        train_end = start + train_size
        test_start = train_end + gap
        test_end = test_start + test_size

        train = data.iloc[start:train_end]
        test = data.iloc[test_start:test_end]
        yield train, test


def expanding_walk_forward(
    data: pd.DataFrame | pd.Series,
    min_train_size: int = 252,
    test_size: int = 63,
    step_size: int = 21,
    gap: int = 0,
) -> Generator[tuple, None, None]:
    """Expanding window walk-forward validation.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Time-indexed data.
    min_train_size : int, default 252
        Minimum initial training window.
    test_size : int, default 63
        Test window size.
    step_size : int, default 21
        Step between windows.
    gap : int, default 0
        Gap between train and test.

    Yields
    ------
    tuple[pd.DataFrame|pd.Series, pd.DataFrame|pd.Series]
        (train_data, test_data) for each fold.
    """
    total = len(data)
    for train_end in range(min_train_size, total - gap - test_size + 1, step_size):
        test_start = train_end + gap
        test_end = test_start + test_size

        train = data.iloc[:train_end]
        test = data.iloc[test_start:test_end]
        yield train, test
