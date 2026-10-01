"""
Targets. The signal is computed after the close of day t and traded at the
OPEN of t+1, so the label is the return a position actually earns:

    fwd_ret_H(t) = adj_open(t+1+H) / adj_open(t+1) - 1      (per entity)

`label_end` records the date of the last price the label touches
(t+1+H); purging in v4.validation uses it, so no training label overlaps a
test period. Labels at the end of an entity's history are NaN (not
traded far enough ahead) and are never used for training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """panel sorted by entity, date. Returns fwd_ret, label_end per row."""
    g = panel.groupby("entity", sort=False)
    o_in = g["adj_open"].shift(-1)
    o_out = g["adj_open"].shift(-(1 + horizon))
    end = g["date"].shift(-(1 + horizon))
    return pd.DataFrame({"fwd_ret": o_out / o_in - 1, "label_end": end}, index=panel.index)


def cross_sectional_targets(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Per-date targets over the rows given (the universe): excess return vs
    the day's mean, its percentile rank, and integer relevance bins for
    LambdaRank (0 = worst ... n_bins-1 = best)."""
    by = df.groupby("date")["fwd_ret"]
    out = pd.DataFrame(index=df.index)
    out["fwd_excess"] = df["fwd_ret"] - by.transform("mean")
    out["fwd_rank"] = by.rank(pct=True)
    out["relevance"] = np.floor(out["fwd_rank"].clip(upper=0.9999) * n_bins).astype("Int64")
    return out
