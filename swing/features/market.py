"""
Date-level (market / regime / macro / positioning) features. One row per
NSE trading date, each built only from data published by the close of that
date (see swing.data.macro for the timing of non-NSE series).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing.data import macro as macro_mod


def index_close(indices: pd.DataFrame, name: str) -> pd.Series:
    s = indices[indices["index"] == name].drop_duplicates("date").set_index("date")["close"]
    return s.sort_index()


def build(dates: pd.DatetimeIndex, indices: pd.DataFrame, participant: pd.DataFrame,
          macro: pd.DataFrame, universe_rows: pd.DataFrame) -> pd.DataFrame:
    """universe_rows: panel rows of universe members (date, ret_cc, dist_ma50/200)."""
    f = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))

    nifty = index_close(indices, "NIFTY 50").reindex(f.index).ffill()
    f["nifty_ret_1"] = nifty.pct_change()
    for w in (5, 21, 63):
        f[f"nifty_ret_{w}"] = nifty / nifty.shift(w) - 1
    f["nifty_dist_ma200"] = nifty / nifty.rolling(200, min_periods=150).mean() - 1
    f["nifty_vol_21"] = f["nifty_ret_1"].rolling(21, min_periods=15).std()
    for name, col in (("NIFTY MIDCAP 50", "mid_vs_nifty_21"), ("NIFTY BANK", "bank_vs_nifty_21")):
        s = index_close(indices, name).reindex(f.index).ffill()
        f[col] = (s / s.shift(21)) / (nifty / nifty.shift(21)) - 1

    # breadth across the universe
    u = universe_rows
    f["breadth_ma50"] = (u["dist_ma50"] > 0).groupby(u["date"]).mean().reindex(f.index)
    f["breadth_ma200"] = (u["dist_ma200"] > 0).groupby(u["date"]).mean().reindex(f.index)
    adv = (u["ret_cc"] > 0).groupby(u["date"]).mean().reindex(f.index)
    f["adv_ratio_5"] = adv.rolling(5, min_periods=3).mean()
    f["xs_dispersion_21"] = u.groupby("date")["ret_21"].std().reindex(f.index)

    # positioning: FII / client index futures, index option put-call ratio
    if not participant.empty:
        p = participant.drop_duplicates(["date", "participant"])
        piv = p.pivot(index="date", columns="participant")
        for who in ("FII", "Client", "Pro", "DII"):
            try:
                lng, sht = piv[("fut_idx_long", who)], piv[("fut_idx_short", who)]
            except KeyError:
                continue
            ratio = (lng / (lng + sht)).reindex(f.index)
            f[f"{who.lower()}_fut_long_ratio"] = ratio
            f[f"{who.lower()}_fut_long_chg5"] = ratio - ratio.shift(5)
        puts = piv["opt_idx_put_long"].sum(axis=1)
        calls = piv["opt_idx_call_long"].sum(axis=1)
        f["idx_pcr_oi"] = (puts / calls.replace(0, np.nan)).reindex(f.index)

    # macro (timing-aligned)
    m = macro_mod.align(macro, f.index)
    if "india_vix" in m:
        vix = m["india_vix"]
        f["vix"] = vix
        f["vix_pct_1y"] = vix.rolling(252, min_periods=150).rank(pct=True)
        f["vix_chg_5"] = vix / vix.shift(5) - 1
    for name in ("usdinr", "brent", "spx"):
        if name in m:
            for w in (1, 5, 21):
                f[f"{name}_ret_{w}"] = m[name] / m[name].shift(w) - 1
    if "us10y" in m:
        f["us10y_chg_5"] = m["us10y"] - m["us10y"].shift(5)
        f["us10y_chg_21"] = m["us10y"] - m["us10y"].shift(21)
    return f
