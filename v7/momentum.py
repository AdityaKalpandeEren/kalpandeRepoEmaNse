"""
V7 momentum score - NSE "Nifty200 Momentum 30" style, shared by the research
(v7.swing_v7) and the live paper runner (v7.live) so both rank identically.

  score = 0.5 * z(6-month return / 1-year volatility)
        + 0.5 * z(12-month return / 1-year volatility)

Returns use adjusted closes (splits / bonuses don't move them); volatility is
the annualised std of daily log returns over 252 sessions; z-scores are
cross-sectional over the stocks scored on that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def m30_score(panel: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """panel: date, entity, adj_close, ret_cc (full history). members: the
    (date, entity) rows to score. Returns date, entity, score."""
    p = panel[["date", "entity", "adj_close", "ret_cc"]].sort_values(["entity", "date"])
    g = p.groupby("entity")
    r6 = p["adj_close"] / g["adj_close"].shift(126) - 1
    r12 = p["adj_close"] / g["adj_close"].shift(252) - 1
    lr = np.log1p(p["ret_cc"].clip(lower=-0.95))
    sig = lr.groupby(p["entity"]).transform(lambda s: s.rolling(252, min_periods=200).std()) * np.sqrt(252)
    p = p.assign(m6=r6 / sig, m12=r12 / sig)[["date", "entity", "m6", "m12"]]
    m = members.merge(p, on=["date", "entity"], how="left").dropna(subset=["m6", "m12"])
    for c in ("m6", "m12"):
        grp = m.groupby("date")[c]
        m[f"z_{c}"] = (m[c] - grp.transform("mean")) / grp.transform("std")
    m["score"] = 0.5 * m["z_m6"] + 0.5 * m["z_m12"]
    return m[["date", "entity", "score"]]
