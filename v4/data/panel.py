"""
Daily stock panel for V4: renames linked, corporate actions adjusted, and a
point-in-time large-cap universe.

ENTITIES. A company keeps one `entity` id across symbol changes (e.g.
MCDOWELL-N -> UNITDSPR, ZOMATO -> ETERNAL): when a symbol first appears on
the same day the old one disappears and both share an ISIN, it continues
the old entity. A symbol is reused for a new entity only after a 30-day gap
with a different ISIN.

ADJUSTMENT. The bhavcopy PREVCLOSE is NOT adjusted on ex-dates, so splits,
bonuses and consolidations come from NSE's corporate-actions feed
(v4.data.corporate_actions): ret = close / (prevclose * factor) - 1, and on
a demerger ex-date the opening gap is dropped (ret = close/open - 1).
Adjusted closes are the cumulative product of those returns (a scale-free
index per entity); adjusted O/H/L use the same per-day factor. Traded
VALUE (Rs) is used instead of share volume, so splits don't distort it.
Ordinary cash dividends are not added back (price returns, like the NIFTY
50 price index used as the benchmark).

UNIVERSE (point in time). On the first trading day of each month, using
data up to the previous trading day only: EQ stocks with >= min_history
days traded and close >= min_price, ranked by median traded value over the
last value_window days; the top `size` are members for that month. Stocks
later delisted or merged are members while they qualified.
"""
from __future__ import annotations

import logging
import os
from datetime import date

import numpy as np
import pandas as pd

from v4.data import corporate_actions, nse_daily

log = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close", "prevclose"]


def link_entities(eq: pd.DataFrame, max_gap_days: int = 30) -> pd.Series:
    """eq: long frame sorted by date with symbol, isin, date. Returns entity id per row."""
    sym_entity: dict[str, str] = {}
    sym_last: dict[str, pd.Timestamp] = {}
    sym_isin: dict[str, str] = {}
    isin_entity: dict[str, str] = {}
    out = np.empty(len(eq), dtype=object)
    prev_day_syms: set[str] = set()
    pos = 0
    for day, g in eq.groupby("date", sort=True):
        today_syms = set(g["symbol"])
        vanished = prev_day_syms - today_syms
        for i, (sym, isin) in enumerate(zip(g["symbol"].values, g["isin"].values)):
            ent = None
            if sym in sym_entity:
                gap = (day - sym_last[sym]).days
                if gap <= max_gap_days or sym_isin.get(sym) == isin:
                    ent = sym_entity[sym]
            if ent is None and isin in isin_entity:
                old = isin_entity[isin]
                # rename: the old symbol of that entity vanished today (or recently)
                old_syms = [s for s, e in sym_entity.items() if e == old]
                if any(s in vanished or (day - sym_last[s]).days <= 5 for s in old_syms):
                    ent = old
            if ent is None:
                ent = sym if sym not in sym_entity else f"{sym}~{day:%Y%m%d}"
            sym_entity[sym], sym_last[sym], sym_isin[sym] = ent, day, isin
            isin_entity[isin] = ent
            out[pos + i] = ent
        pos += len(g)
        prev_day_syms = today_syms
    return pd.Series(out, index=eq.index, name="entity")


def build(cfg: dict, cache_dir: str, start: date, end: date, rebuild: bool = False) -> pd.DataFrame:
    """Adjusted long panel of every EQ stock, with entity, delivery %, universe flag."""
    f = os.path.join(cache_dir, f"panel_{start:%Y%m%d}_{end:%Y%m%d}.parquet")
    if os.path.exists(f) and not rebuild:
        return pd.read_parquet(f)

    eq = nse_daily.load_kind(cache_dir, "equity", start, end)
    eq = eq.dropna(subset=["close", "prevclose"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    log.info("panel: %d equity rows, %d days", len(eq), eq["date"].nunique())
    eq["entity"] = link_entities(eq)
    eq = eq.drop_duplicates(["entity", "date"], keep="last")

    dl = nse_daily.load_kind(cache_dir, "delivery", start, end)
    adj = corporate_actions.adjustments(corporate_actions.fetch(cache_dir, start, end))
    eq = assemble(eq, dl, cfg, adj)
    eq.to_parquet(f, index=False)
    return eq


def assemble(eq: pd.DataFrame, dl: pd.DataFrame | None, cfg: dict,
             adj: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pure part of build(): eq has entity, symbol, isin, date, OHLC, prevclose,
    volume, value, trades. Adds delivery, adjusted prices, n_days, universe."""
    if dl is not None and not dl.empty:
        eq = eq.merge(dl.drop_duplicates(["symbol", "date"]), on=["symbol", "date"], how="left")
    eq = eq.sort_values(["entity", "date"]).reset_index(drop=True)
    for c in PRICE_COLS + ["volume", "value", "trades"]:
        eq[c] = eq[c].astype("float64")
    # corporate actions: PREVCLOSE in the bhavcopy is NOT adjusted on ex-dates
    eq["ca_factor"], eq["demerger"] = 1.0, False
    if adj is not None and not adj.empty:
        # match on (symbol, date) first - a split often brings a NEW ISIN on the
        # ex-date while the feed lists the old one - then on (isin, date)
        by_sym = adj.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])
        by_isin = adj.drop_duplicates(["isin", "date"]).set_index(["isin", "date"])
        k_sym = pd.MultiIndex.from_arrays([eq["symbol"], eq["date"]])
        k_isin = pd.MultiIndex.from_arrays([eq["isin"], eq["date"]])
        f = pd.Series(by_sym["factor"].reindex(k_sym).values, index=eq.index)
        d = pd.Series(by_sym["demerger"].reindex(k_sym).values, index=eq.index)
        f = f.fillna(pd.Series(by_isin["factor"].reindex(k_isin).values, index=eq.index))
        d = d.fillna(pd.Series(by_isin["demerger"].reindex(k_isin).values, index=eq.index))
        eq["ca_factor"] = f.fillna(1.0).astype(float)
        eq["demerger"] = d.astype("boolean").fillna(False).astype(bool)
    eq["prevclose_adj"] = eq["prevclose"] * eq["ca_factor"]
    eq["ret_cc"] = eq["close"] / eq["prevclose_adj"] - 1.0
    # demerger ex-date: the spun-off value isn't known -> drop the opening gap
    eq.loc[eq["demerger"], "ret_cc"] = eq.loc[eq["demerger"], "close"] / eq.loc[eq["demerger"], "open"] - 1.0
    first = eq["entity"] != eq["entity"].shift()
    eq.loc[first, "ret_cc"] = 0.0
    # Bad prints (e.g. a PREVCLOSE of 0 after a relisting) -> neutral day.
    bad = ~np.isfinite(eq["ret_cc"]) | (eq["ret_cc"].abs() > 0.6)
    eq.loc[bad, "ret_cc"] = 0.0
    eq["adj_close"] = (1.0 + eq["ret_cc"]).groupby(eq["entity"]).cumprod()
    k = eq["adj_close"] / eq["close"]
    for c in ("open", "high", "low"):
        eq[f"adj_{c}"] = eq[c] * k
    # Days whose return was neutralised (bad print / demerger / rights): the
    # raw open can belong to the pre-event price, so keep O/H/L consistent
    # with the neutral close instead of scaling a stale open.
    neutral = bad | (eq["demerger"] & ((eq["open"] / eq["prevclose_adj"] - 1).abs() > 0.15))
    prev_adj = eq.groupby("entity")["adj_close"].shift().fillna(eq["adj_close"])
    eq.loc[neutral, "adj_open"] = prev_adj[neutral]
    eq.loc[neutral, "adj_high"] = np.maximum(eq.loc[neutral, "adj_close"], prev_adj[neutral])
    eq.loc[neutral, "adj_low"] = np.minimum(eq.loc[neutral, "adj_close"], prev_adj[neutral])
    eq["neutral_day"] = neutral
    eq["n_days"] = eq.groupby("entity").cumcount() + 1

    return add_universe(eq, cfg)


def add_universe(eq: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adds `in_universe` (bool) using monthly point-in-time membership."""
    u = cfg["universe"]
    eq = eq.sort_values(["entity", "date"])
    med_val = eq.groupby("entity")["value"].transform(
        lambda s: s.rolling(u["value_window"], min_periods=u["value_window"] // 2).median())
    eq["_med_val"] = med_val
    days = pd.DatetimeIndex(sorted(eq["date"].unique()))
    month_first = days[~days.to_period("M").duplicated()]
    members = []
    for d in month_first:
        i = days.get_loc(d)
        if i == 0:
            continue
        prev = days[i - 1]                                   # data up to the prior session only
        snap = eq[eq["date"] == prev]
        snap = snap[(snap["n_days"] >= u["min_history"]) & (snap["close"] >= u["min_price"])]
        top = snap.nlargest(u["size"], "_med_val")["entity"]
        members.append(pd.DataFrame({"month": d.to_period("M"), "entity": top.values}))
    mem = pd.concat(members, ignore_index=True)
    mem["in_universe"] = True
    eq["month"] = eq["date"].dt.to_period("M")
    eq = eq.merge(mem, on=["month", "entity"], how="left")
    eq["in_universe"] = eq["in_universe"].astype("boolean").fillna(False).astype(bool)
    return eq.drop(columns=["month"]).sort_values(["entity", "date"]).reset_index(drop=True)


def entity_table(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per entity: latest symbol, all symbols, all ISINs, first/last date."""
    g = panel.groupby("entity")
    return pd.DataFrame({
        "entity": list(g.groups),
        "symbol": g["symbol"].last().values,
        "symbols": g["symbol"].agg(lambda s: sorted(set(s))).values,
        "isins": g["isin"].agg(lambda s: sorted(set(s))).values,
        "first": g["date"].min().values, "last": g["date"].max().values,
    })
