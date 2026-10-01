"""
Per-stock technical features. Every function is causal: the value on row t
uses only rows <= t of the same entity (rolling / ewm / shift(+k) only).
Inputs are the adjusted, scale-free prices from swing.data.panel, so absolute
price levels never enter a feature - only ratios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def per_entity(g: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """g: one entity's rows sorted by date. Returns a frame of features (same index)."""
    c, o, h, lo = g["adj_close"], g["adj_open"], g["adj_high"], g["adj_low"]
    r = g["ret_cc"]
    f = pd.DataFrame(index=g.index)
    for w in windows:
        f[f"ret_{w}"] = c / c.shift(w) - 1
    f["ret_1"] = r
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["mom_6_1"] = c.shift(21) / c.shift(126) - 1
    for w in (21, 63):
        f[f"vol_{w}"] = r.rolling(w, min_periods=w // 2).std()
    f["downvol_63"] = r.where(r < 0, 0).rolling(63, min_periods=30).std()
    f["vol_ratio_21_63"] = f["vol_21"] / f["vol_63"]
    f["skew_63"] = r.rolling(63, min_periods=40).skew()
    f["max_ret_21"] = r.rolling(21, min_periods=10).max()
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_2"] = _rsi(c, 2)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    ma20, sd20 = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std()
    f["bb_pos_20"] = (c - ma20) / (2 * sd20)
    for w in (50, 200):
        f[f"dist_ma{w}"] = c / c.rolling(w, min_periods=int(w * 0.8)).mean() - 1
    f["dist_52w_high"] = c / h.rolling(252, min_periods=200).max() - 1
    f["dist_52w_low"] = c / lo.rolling(252, min_periods=200).min() - 1
    f["gap_1"] = o / c.shift() - 1
    f["range_1"] = (h - lo) / c
    f["close_loc_1"] = (c - lo) / (h - lo).replace(0, np.nan)
    lv = np.log1p(g["value"])
    f["value_z_63"] = (lv - lv.rolling(63, min_periods=40).mean()) / lv.rolling(63, min_periods=40).std()
    f["value_ratio_5_63"] = g["value"].rolling(5).mean() / g["value"].rolling(63, min_periods=40).mean()
    f["log_value_63"] = lv.rolling(63, min_periods=40).mean()
    if "deliv_pct" in g:
        dp = g["deliv_pct"]
        f["deliv_pct"] = dp
        f["deliv_pct_5"] = dp.rolling(5, min_periods=3).mean()
        f["deliv_z_63"] = (dp - dp.rolling(63, min_periods=40).mean()) / dp.rolling(63, min_periods=40).std()
    f["ret_ma_slope_20"] = ma20 / ma20.shift(5) - 1
    return f


def compute(panel: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Technical features for every row of `panel` (sorted by entity, date)."""
    parts = [per_entity(g, windows) for _, g in panel.groupby("entity", sort=False)]
    return pd.concat(parts).loc[panel.index]


def market_relative(panel: pd.DataFrame, feats: pd.DataFrame, nifty_ret: pd.Series) -> pd.DataFrame:
    """Beta to NIFTY 50 (126d) and beta-adjusted residual returns."""
    out = pd.DataFrame(index=panel.index)
    m = panel["date"].map(nifty_ret)
    for ent, idx in panel.groupby("entity", sort=False).groups.items():
        r = panel.loc[idx, "ret_cc"]
        mr = m.loc[idx]
        cov = r.rolling(126, min_periods=80).cov(mr)
        var = mr.rolling(126, min_periods=80).var()
        beta = cov / var
        out.loc[idx, "beta_126"] = beta
        lr, lmr = np.log1p(r), np.log1p(mr)
        for w in (5, 21, 63):
            sr = np.expm1(lr.rolling(w).sum())
            smr = np.expm1(lmr.rolling(w).sum())
            out.loc[idx, f"resid_ret_{w}"] = sr - beta * smr
            out.loc[idx, f"rel_ret_{w}"] = sr - smr
        out.loc[idx, "idio_vol_63"] = (r - beta * mr).rolling(63, min_periods=40).std()
    return out
