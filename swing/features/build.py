"""
Assemble the V4 modelling dataset: one row per (universe member, date) with
stock features, their cross-sectional ranks, sector- and market-relative
returns, date-level regime/macro/positioning features, and labels.

Rule enforced by swing/tests/test_no_lookahead.py: the feature row for date t
is identical whether or not data after t exists.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from swing.features import labels as labels_mod
from swing.features import market as market_mod
from swing.features import technical

log = logging.getLogger(__name__)

META = ["date", "entity", "symbol", "industry", "label_end", "fwd_ret", "fwd_excess", "fwd_rank",
        "relevance", "in_universe"]
# stock-level features that also get a per-day percentile rank
RANKED_PREFIXES = ("ret_", "mom_", "vol_", "downvol", "skew", "max_ret", "rsi", "macd", "atr", "bb_",
                   "dist_", "value_", "log_value", "deliv", "beta", "resid_", "rel_", "idio", "sect_rel")


def build(panel: pd.DataFrame, industry: pd.Series, indices: pd.DataFrame, participant: pd.DataFrame,
          macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """panel: output of swing.data.panel.build (all EQ stocks). Returns universe rows only."""
    cand = panel[panel["entity"].isin(panel.loc[panel["in_universe"], "entity"].unique())]
    cand = cand.sort_values(["entity", "date"]).reset_index(drop=True)
    log.info("features: %d candidate entities, %d rows", cand["entity"].nunique(), len(cand))

    tech = technical.compute(cand, cfg["features"]["windows"])
    nifty = market_mod.index_close(indices, "NIFTY 50")
    nifty_ret = nifty.pct_change()
    rel = technical.market_relative(cand, tech, nifty_ret)
    lab = labels_mod.forward_returns(cand, cfg["label"]["horizon"])
    df = pd.concat([cand[["date", "entity", "symbol", "in_universe", "ret_cc"]], tech, rel, lab], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df["industry"] = df["entity"].map(industry).fillna("UNKNOWN")

    u = df[df["in_universe"]].copy()
    # sector-relative returns: vs the median of same-industry universe peers that day
    for w in (5, 21, 63):
        med = u.groupby(["date", "industry"])[f"ret_{w}"].transform("median")
        u[f"sect_rel_{w}"] = u[f"ret_{w}"] - med
    u["industry_size"] = u.groupby(["date", "industry"])["entity"].transform("size")

    feat_cols = [c for c in u.columns if c not in META and c not in ("ret_cc",)]
    if cfg["features"].get("cross_sectional", True):
        ranked = [c for c in feat_cols if c.startswith(RANKED_PREFIXES)]
        ranks = u.groupby("date")[ranked].rank(pct=True)
        ranks.columns = [f"xs_{c}" for c in ranked]
        u = pd.concat([u, ranks], axis=1)

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    mkt = market_mod.build(dates, indices, participant, macro, u)
    u = u.merge(mkt, left_on="date", right_index=True, how="left")

    u = pd.concat([u, labels_mod.cross_sectional_targets(u)], axis=1)
    drop = set(cfg["features"].get("drop") or [])
    u = u.drop(columns=[c for c in drop if c in u.columns])
    u = u.replace([np.inf, -np.inf], np.nan)
    return u.sort_values(["date", "entity"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = set(META) | {"ret_cc"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
