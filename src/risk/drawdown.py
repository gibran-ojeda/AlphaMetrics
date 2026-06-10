"""Maximum Drawdown with peak/trough/recovery tracking.

Calmar Ratio = Annual Return / |MaxDrawdown|.
Edge case: monotonically increasing series yields zero drawdown.
"""
from __future__ import annotations

import pandas as pd

from core.types import DrawdownResult


def max_drawdown(returns: pd.Series) -> DrawdownResult:
    """Compute maximum drawdown with full detail.

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    DrawdownResult
        drawdown series, max_drawdown value, peak/trough/recovery dates, duration.
    """
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak

    max_dd = dd.min()
    trough_idx = dd.idxmin()

    # Peak date: last time wealth was at its max before the trough
    peak_before_trough = peak.loc[:trough_idx]
    peak_idx = wealth.loc[:trough_idx].idxmax()

    # Recovery date: first time wealth returns to peak after trough
    recovery_idx = None
    duration = None
    peak_value = peak.loc[trough_idx]
    after_trough = wealth.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_value]
    if len(recovered) > 1:  # First match is trough itself if exact, skip
        recovery_idx = recovered.index[1] if recovered.index[0] == trough_idx else recovered.index[0]

    if recovery_idx is not None and hasattr(returns.index, 'get_loc'):
        duration = returns.index.get_loc(recovery_idx) - returns.index.get_loc(peak_idx)

    return DrawdownResult(
        drawdown=dd,
        max_drawdown=float(max_dd),
        peak_date=peak_idx,
        trough_date=trough_idx,
        recovery_date=recovery_idx,
        duration=duration,
    )
