"""Numerical precision utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_float64(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Coerce numeric data to float64 for TA-Lib compatibility and precision."""
    if isinstance(data, pd.DataFrame):
        return data.astype(np.float64)
    return data.astype(np.float64)
