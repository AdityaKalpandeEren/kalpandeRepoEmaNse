"""
V7 scores, shared by the research (v7.swing_v7, v7.swing_v7c) and the live
paper runner (v7.live) so both rank identically.

M30 - NSE "Nifty200 Momentum 30" style momentum:
  score = 0.5 * z(6-month return / 1-year volatility)
        + 0.5 * z(12-month return / 1-year volatility)
Returns use adjusted closes (splits / bonuses don't move them); volatility is
the annualised std of daily log returns over 252 sessions; z-scores are
cross-sectional over the stocks scored on that date.

D1 - the live V7 score since 2026-10-13 (user decision; research: v7/README.md 3f):
  score = 0.8 * z(M30) + 0.2 * z(delivery trend)
delivery trend = the stock's mean delivery % over its last 20 sessions minus
its mean over its last 120. Both z-scores are per date across the stocks with
an M30 score; a stock without a delivery value gets a delivery z of 0 (its
momentum alone decides).
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


DELIVERY_WEIGHT = 0.2


def delivery_trend(panel: pd.DataFrame, entities=None) -> pd.DataFrame:
    """date, entity, deliv_trend: 20-session mean delivery % minus the 120-session
    mean, per stock (NaN without enough history or without a deliv_pct column)."""
    p = panel if entities is None else panel[panel["entity"].isin(entities)]
    if "deliv_pct" not in p:
        return p[["date", "entity"]].assign(deliv_trend=np.nan)
    p = p[["date", "entity", "deliv_pct"]].sort_values(["entity", "date"])
    g = p.groupby("entity")["deliv_pct"]
    trend = g.transform(lambda s: s.rolling(20, min_periods=10).mean()) - \
        g.transform(lambda s: s.rolling(120, min_periods=60).mean())
    return p[["date", "entity"]].assign(deliv_trend=trend)


def d1_score(panel: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """The live V7 score: 0.8 z(M30) + 0.2 z(delivery trend). Same arguments and
    output as m30_score; the panel also needs deliv_pct."""
    m = m30_score(panel, members)
    m = m.merge(delivery_trend(panel, m["entity"].unique()), on=["date", "entity"], how="left")
    z = {}
    for c in ("score", "deliv_trend"):
        grp = m.groupby("date")[c]
        z[c] = (m[c] - grp.transform("mean")) / grp.transform("std")
    m["score"] = (1 - DELIVERY_WEIGHT) * z["score"] + DELIVERY_WEIGHT * z["deliv_trend"].fillna(0.0)
    return m.dropna(subset=["score"])[["date", "entity", "score"]]
