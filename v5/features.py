"""
V5.0 additions on top of the V4 feature set.

Lessons applied (from V1-V4 + literature):
  * V4's 5-day model had real ranking skill (IC 0.047, t 5.1) that 21x
    turnover destroyed -> V5 learns a SLOWER, cleaner target and its
    scores are smoothed through time before trading (see v5.portfolio).
  * Raw forward returns are mostly market + sector noise -> the target is
    beta- and industry-neutral (residual) return.
  * One horizon is fragile (V4 H=20 relied on 5 trades) -> the target is
    the average per-day rank of residual returns over 5, 10 and 20 days.
  * Robust Indian anomalies (52-week-high proximity, 12-1 momentum;
    volume / delivery) are given explicitly as factor scores and features.

Every feature uses data up to the close of t only (same rules as V4, see
swing/tests). Targets use t+1 open .. t+1+H open and carry label_end for
purging (the longest horizon, 20, drives the purge).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing.features import labels as v4_labels

HORIZONS = (5, 10, 20)


def extra_stock_features(panel: pd.DataFrame, entities: set[str]) -> pd.DataFrame:
    """Causal per-entity features not in V4: delivery accumulation, volume
    shocks with delivery, 52-week-high proximity on closes, up-volume share."""
    p = panel[panel["entity"].isin(entities)].sort_values(["entity", "date"])
    out = []
    for ent, g in p.groupby("entity", sort=False):
        f = pd.DataFrame(index=g.index)
        val, dp, r = g["value"], g["deliv_pct"] if "deliv_pct" in g else pd.Series(np.nan, index=g.index), g["ret_cc"]
        deliv_val = val * dp / 100.0                                     # Rs value actually delivered
        f["deliv_val_ratio_20_120"] = deliv_val.rolling(20, min_periods=10).mean() / deliv_val.rolling(120, min_periods=60).mean()
        f["deliv_pct_20_minus_120"] = dp.rolling(20, min_periods=10).mean() - dp.rolling(120, min_periods=60).mean()
        vz = (np.log1p(val) - np.log1p(val).rolling(60, min_periods=40).mean()) / np.log1p(val).rolling(60, min_periods=40).std()
        shock = (vz > 2).astype(float)
        f["vol_shock_up_20"] = (shock * (r > 0)).rolling(20, min_periods=10).sum()
        f["vol_shock_down_20"] = (shock * (r < 0)).rolling(20, min_periods=10).sum()
        f["hi_deliv_up_days_20"] = ((dp > dp.rolling(60, min_periods=40).mean()) & (r > 0)).astype(float).rolling(20, min_periods=10).sum()
        up_val = val.where(r > 0, 0.0)
        f["up_value_share_20"] = up_val.rolling(20, min_periods=10).sum() / val.rolling(20, min_periods=10).sum()
        c = g["adj_close"]
        f["close_to_252_high"] = c / c.rolling(252, min_periods=200).max()
        f["days_since_252_high"] = c.rolling(252, min_periods=200).apply(lambda a: len(a) - 1 - int(np.argmax(a)), raw=True)
        out.append(f)
    return pd.concat(out)


def residual_targets(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Blended multi-horizon residual-return rank target per (date, entity).

    For each H: fwd_H (next open -> open H later); per day remove the part
    explained by beta_126 (cross-sectional OLS) and the industry mean; rank
    the residual. Target = mean of the H ranks. label_end = the 20-day end."""
    cand = panel[panel["entity"].isin(set(df["entity"]))].sort_values(["entity", "date"]).reset_index(drop=True)
    key = cand[["date", "entity"]]
    out = df[["date", "entity", "beta_126", "industry"]].copy()
    ranks = []
    for h in HORIZONS:
        fr = v4_labels.forward_returns(cand, h)
        m = pd.concat([key, fr], axis=1).rename(columns={"fwd_ret": f"fwd_{h}", "label_end": f"end_{h}"})
        out = out.merge(m, on=["date", "entity"], how="left")
        def _resid(g):
            gy, gb = g[f"fwd_{h}"], g["beta_126"].fillna(1.0)
            ok = gy.notna()
            res = pd.Series(np.nan, index=g.index)
            if ok.sum() < 10:
                return res
            b = np.polyfit(gb[ok], gy[ok], 1) if gb[ok].std() > 0 else (0.0, gy[ok].mean())
            r = gy - (b[0] * gb + b[1])
            r = r - r.groupby(g["industry"]).transform("mean")
            return r
        out[f"resid_{h}"] = out.groupby("date", group_keys=False).apply(_resid, include_groups=False)
        ranks.append(out.groupby("date")[f"resid_{h}"].rank(pct=True))
    out["target"] = pd.concat(ranks, axis=1).mean(axis=1, skipna=False)
    out["label_end"] = out[f"end_{max(HORIZONS)}"]
    keep = ["date", "entity", "target", "label_end"] + [f"fwd_{h}" for h in HORIZONS]
    return out[keep]


def factor_score(df: pd.DataFrame) -> pd.Series:
    """Literature-robust Indian anomalies, rank-averaged per day:
    52-week-high proximity (George-Hwang; robust in India 2004-2023) and
    12-1 month momentum."""
    r1 = df.groupby("date")["close_to_252_high"].rank(pct=True)
    r2 = df.groupby("date")["mom_12_1"].rank(pct=True)
    return (r1 + r2) / 2
