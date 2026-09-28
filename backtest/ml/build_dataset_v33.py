"""
V3.3 (formerly V3.2+): adds NSE market data, corporate events and multi-day holding
labels to the V3.2 dataset (backtest/ml/build_dataset_v32.py output).

    python -m data.nse_archives --from 2021-11-01 --to <today>   # first
    python -m backtest.ml.build_dataset_v33

New inputs (all known before 9:45 on day d):
  DELIVERY / VOLUME (CM bhavcopy of the PREVIOUS session)
    nse_deliv_pct, nse_deliv_pct_vs20, nse_deliv_qty_vs20 (unusual
    delivery - genuine buying/selling vs intraday churn), nse_qty_vs20
    (unusual volume), nse_trades_vs20, nse_trade_size_vs20 (bigger
    tickets = institutions), nse_turnover_log, nse_deliv_pct_chg5
  F&O (F&O bhavcopy of the PREVIOUS session; NaN for non-F&O stocks)
    fo_stock, fut_oi_chg_pct, fut_oi_chg5_pct, oi_buildup (+1 long
    build-up, +0.5 short covering, -0.5 long unwinding, -1 short build-up
    from price x OI change), pcr_oi, pcr_chg5, fut_vol_vs20,
    opt_vol_vs20, call_oi_chg_pct, put_oi_chg_pct
  EVENTS (only items whose PUBLIC timestamp is before 9:45 on day d)
    ann_count_1d / ann_count_5d; flags in the last 24h for order/contract
    wins, M&A, results, fund-raising, rating actions, resignations,
    dividends/bonus/buyback; board meeting: days to the next announced
    meeting, whether it is for results, days since the last one;
    insider: promoter-group open-market net buy value disclosed in the
    last 5 / 20 days, number of promoter buys in 20 days
New labels (entry at the 9:45 open, NSE DELIVERY costs 0.20%/side for
multi-day holds):
    net_h3, net_h5 = hold to the close of the 3rd / 5th trading day
    (entry day counts as day 1); excess_h3 / excess_h5 = minus that
    day's universe average. The same-day label net_ret / excess (9:45 ->
    15:15, intraday costs) is kept as horizon 1.

Output: config.ML_V33_DATASET_PATH.
"""
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import config
from backtest.candle_cache import load_daily_cached
from backtest.ml.common import access_token
from data import nse_archives as N

HORIZONS = (3, 5)
DELIVERY_COST = config.SWING_COST_PCT_PER_SIDE

_FLAGS = {
    "ann_order": r"\b(order|contract|award|letter of (award|intent)|bagged|secures?)\b",
    "ann_ma": r"\b(acquisition|acquire|merger|amalgamation|takeover|stake)\b",
    "ann_results": r"\b(financial results?|outcome of board meeting|quarterly results?)\b",
    "ann_fundraise": r"\b(fund ?raising|qip|preferential|rights issue|ncd|debentures?)\b",
    "ann_rating": r"\bcredit rating\b",
    "ann_resign": r"\b(resignation|resigns?|cessation)\b",
    "ann_payout": r"\b(dividend|bonus|split|buy ?back)\b",
}


def _ratio_vs_prior(s: pd.Series, n: int = 20) -> pd.Series:
    """value / mean of the previous n values (the value itself excluded)."""
    return s / s.rolling(n, min_periods=max(5, n // 2)).mean().shift(1)


def cm_features(cm: pd.DataFrame) -> pd.DataFrame:
    cm = cm.sort_values(["symbol", "date"]).copy()
    g = cm.groupby("symbol", group_keys=False)
    cm["trade_size"] = cm["qty"] / cm["trades"].replace(0, np.nan)
    out = pd.DataFrame({"symbol": cm["symbol"], "date": cm["date"]})
    out["nse_deliv_pct"] = cm["deliv_pct"]
    out["nse_deliv_pct_vs20"] = g["deliv_pct"].apply(_ratio_vs_prior)
    out["nse_deliv_qty_vs20"] = g["deliv_qty"].apply(_ratio_vs_prior)
    out["nse_qty_vs20"] = g["qty"].apply(_ratio_vs_prior)
    out["nse_trades_vs20"] = g["trades"].apply(_ratio_vs_prior)
    out["nse_trade_size_vs20"] = g["trade_size"].apply(_ratio_vs_prior)
    out["nse_turnover_log"] = np.log1p(cm["turnover_lacs"])
    out["nse_deliv_pct_chg5"] = cm["deliv_pct"] - g["deliv_pct"].apply(
        lambda s: s.rolling(5, min_periods=3).mean().shift(1))
    return out


def fo_features(fo: pd.DataFrame) -> pd.DataFrame:
    fo = fo.sort_values(["symbol", "date"]).copy()
    g = fo.groupby("symbol", group_keys=False)
    out = pd.DataFrame({"symbol": fo["symbol"], "date": fo["date"]})
    out["fo_stock"] = 1.0
    prev_oi = (fo["fut_oi"] - fo["fut_oi_chg"]).replace(0, np.nan)
    out["fut_oi_chg_pct"] = fo["fut_oi_chg"] / prev_oi
    out["fut_oi_chg5_pct"] = g["fut_oi"].apply(lambda s: s / s.shift(5) - 1)
    pcr = fo["put_oi"] / fo["call_oi"].replace(0, np.nan)
    out["pcr_oi"] = pcr
    out["pcr_chg5"] = pcr - pcr.groupby(fo["symbol"]).shift(5)
    out["fut_vol_vs20"] = g["fut_vol"].apply(_ratio_vs_prior)
    fo["opt_vol"] = fo["call_vol"] + fo["put_vol"]
    out["opt_vol_vs20"] = fo.groupby("symbol", group_keys=False)["opt_vol"].apply(_ratio_vs_prior)
    out["call_oi_chg_pct"] = fo["call_oi_chg"] / (fo["call_oi"] - fo["call_oi_chg"]).replace(0, np.nan)
    out["put_oi_chg_pct"] = fo["put_oi_chg"] / (fo["put_oi"] - fo["put_oi_chg"]).replace(0, np.nan)
    return out


def shift_to_next_session(panel: pd.DataFrame, cols) -> pd.DataFrame:
    """Values published after session t become usable on session t+1."""
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel[cols] = panel.groupby("symbol")[cols].shift(1)
    return panel


def event_features(ds: pd.DataFrame, ann, bm, pit) -> pd.DataFrame:
    """Per (symbol, date) event features using only items public before
    9:45 on that date."""
    out = {}
    cut = pd.to_datetime(ds["date"]) + pd.Timedelta(hours=9, minutes=45)
    ds = ds.assign(cut=cut)
    flag_rx = {k: re.compile(v, re.I) for k, v in _FLAGS.items()}
    if ann is not None and not ann.empty:
        ann = ann.dropna(subset=["time", "symbol"]).copy()
        blob = (ann["desc"].fillna("") + " " + ann["text"].fillna(""))
        for k, rx in flag_rx.items():
            ann[k] = blob.str.contains(rx.pattern.replace("(", "(?:").replace("(?:?", "(?"), case=False, regex=True).astype(float)
        ann = ann.sort_values("time")
        by_sym = {s: g for s, g in ann.groupby("symbol")}
    else:
        by_sym = {}
    bm_by = {s: g.sort_values("announced") for s, g in bm.dropna(subset=["announced"]).groupby("symbol")} \
        if bm is not None and not bm.empty else {}
    if pit is not None and not pit.empty:
        p = pit.dropna(subset=["time"]).copy()
        p = p[p["category"].astype(str).str.contains("Promoter", case=False, na=False)
              & p["mode"].astype(str).str.contains("Market", case=False, na=False)]
        p["signed"] = np.where(p["type"].astype(str).str.contains("Buy", case=False), 1, -1) * p["value"].fillna(0)
        pit_by = {s: g.sort_values("time") for s, g in p.groupby("symbol")}
    else:
        pit_by = {}

    rows = []
    for sym, g in ds.groupby("symbol"):
        a = by_sym.get(sym)
        b = bm_by.get(sym)
        pi = pit_by.get(sym)
        for idx, c in zip(g.index, g["cut"]):
            r = {}
            if a is not None:
                w1 = a[(a["time"] <= c) & (a["time"] > c - pd.Timedelta(days=1))]
                w5 = a[(a["time"] <= c) & (a["time"] > c - pd.Timedelta(days=5))]
                r["ann_count_1d"], r["ann_count_5d"] = float(len(w1)), float(len(w5))
                for k in _FLAGS:
                    r[k] = float(w1[k].max()) if len(w1) else 0.0
            else:
                r["ann_count_1d"] = r["ann_count_5d"] = 0.0
                for k in _FLAGS:
                    r[k] = 0.0
            r["bm_days_to_next"], r["bm_next_results"], r["bm_days_since"] = 30.0, 0.0, 60.0
            if b is not None:
                known = b[b["announced"] <= c]
                day = c.normalize()
                nxt = known[known["meeting"] >= day].sort_values("meeting")
                if len(nxt):
                    r["bm_days_to_next"] = float(min(30, (nxt["meeting"].iloc[0] - day).days))
                    r["bm_next_results"] = float(bool(re.search("result", str(nxt["purpose"].iloc[0]), re.I)))
                past = known[known["meeting"] < day]
                if len(past):
                    r["bm_days_since"] = float(min(60, (day - past["meeting"].max()).days))
            r["ins_promoter_net_5d"] = r["ins_promoter_net_20d"] = r["ins_promoter_buys_20d"] = 0.0
            if pi is not None:
                w5 = pi[(pi["time"] <= c) & (pi["time"] > c - pd.Timedelta(days=5))]
                w20 = pi[(pi["time"] <= c) & (pi["time"] > c - pd.Timedelta(days=20))]
                r["ins_promoter_net_5d"] = float(np.sign(w5["signed"].sum()) * np.log1p(abs(w5["signed"].sum())))
                r["ins_promoter_net_20d"] = float(np.sign(w20["signed"].sum()) * np.log1p(abs(w20["signed"].sum())))
                r["ins_promoter_buys_20d"] = float((w20["signed"] > 0).sum())
            out[idx] = r
    return pd.DataFrame.from_dict(out, orient="index")


def multi_day_labels(ds: pd.DataFrame, tok: str) -> pd.DataFrame:
    labels = {f"net_h{h}": pd.Series(np.nan, index=ds.index) for h in HORIZONS}
    to = str(pd.to_datetime(ds["date"]).max().date() + timedelta(days=15))
    for sym, g in ds.groupby("symbol"):
        daily = load_daily_cached(sym, "2021-11-01", to, tok)
        daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date.astype(str)
        daily = daily.drop_duplicates("date", keep="last").reset_index(drop=True)
        pos = {d: i for i, d in enumerate(daily["date"])}
        closes = daily["close"].to_numpy(float)
        for idx, d, entry in zip(g.index, g["date"], g["entry"]):
            i = pos.get(d)
            if i is None:
                continue
            for h in HORIZONS:
                j = i + h - 1
                if j < len(closes):
                    labels[f"net_h{h}"].at[idx] = (closes[j] * (1 - DELIVERY_COST)) / (entry * (1 + DELIVERY_COST)) - 1
    out = pd.DataFrame(labels)
    for h in HORIZONS:
        out[f"excess_h{h}"] = out[f"net_h{h}"] - out[f"net_h{h}"].groupby(ds["date"]).transform("mean")
    return out


FO_COLS = ("fo_stock", "fut_oi_chg_pct", "fut_oi_chg5_pct", "pcr_oi", "pcr_chg5", "fut_vol_vs20",
           "opt_vol_vs20", "call_oi_chg_pct", "put_oi_chg_pct")


def _with_placeholder(panel: pd.DataFrame, symbols, as_of) -> pd.DataFrame:
    """Live: today's bhavcopy doesn't exist yet. Add an empty row per symbol
    for `as_of` so shift_to_next_session hands it the PREVIOUS session's
    values - exactly what the dataset rows for past days received."""
    if as_of is None:
        return panel
    ph = pd.DataFrame({"symbol": list(symbols), "date": as_of})
    return pd.concat([panel[panel["date"] != as_of], ph], ignore_index=True)


def add_nse_features(ds: pd.DataFrame, a: date, b: date, as_of=None, verbose=True) -> pd.DataFrame:
    """Adds every V3.3 NSE feature to V3.2 rows `ds` (columns symbol, date
    as 'YYYY-MM-DD', d_ret1, ...). Shared by the dataset build and the live
    runner (as_of = today's date)."""
    syms = ds["symbol"].unique()
    cm = N.panel(N.cm_day, a, b)
    if len(cm):
        cmf = cm_features(_with_placeholder(cm, syms, as_of))
        cmf = shift_to_next_session(cmf, [c for c in cmf.columns if c not in ("symbol", "date")])
        cmf["date"] = cmf["date"].astype(str)
        ds = ds.merge(cmf, on=["symbol", "date"], how="left")
    fo = N.panel(N.fo_day, a, b)
    if len(fo):
        fof = fo_features(fo)
        if as_of is not None:
            fof = pd.concat([fof[fof["date"] != as_of],
                             pd.DataFrame({"symbol": fof["symbol"].unique(), "date": as_of})], ignore_index=True)
        fof = shift_to_next_session(fof, list(FO_COLS))
        fof["date"] = fof["date"].astype(str)
        ds = ds.merge(fof, on=["symbol", "date"], how="left")
        ds["fo_stock"] = ds["fo_stock"].fillna(0.0)
        # price x OI interpretation, both from the previous session
        up, oi_up = ds["d_ret1"] > 0, ds["fut_oi_chg_pct"] > 0
        ds["oi_buildup"] = np.select([up & oi_up, up & ~oi_up, ~up & ~oi_up, ~up & oi_up],
                                     [1.0, 0.5, -0.5, -1.0], default=np.nan)
        ds.loc[ds["fut_oi_chg_pct"].isna(), "oi_buildup"] = np.nan
    if verbose:
        print(f"  CM rows {len(cm)}, F&O rows {len(fo)}", flush=True)
    # Events need their own look-back (bm_days_since is capped at 60 days,
    # insider sums cover 20) and board meetings a month AHEAD (a meeting
    # announced today may be held next month - that's what bm_days_to_next
    # measures). Only items public before 9:45 on each day are used.
    ev_a = a if as_of is None else min(a, as_of - timedelta(days=75))
    ann = N.event_panel(N.announcements_month, ev_a, b)
    bm = N.event_panel(N.board_meetings_month, ev_a, b + timedelta(days=35))
    pit = N.event_panel(N.insider_month, ev_a, b)
    if verbose:
        print(f"  events: announcements {len(ann)}, board meetings {len(bm)}, insider {len(pit)}", flush=True)
    return ds.join(event_features(ds, ann, bm, pit))


def main():
    tok = access_token()
    ds = pd.read_csv(config.ML_V32_DATASET_PATH)
    a = date(2021, 11, 1)
    b = pd.to_datetime(ds["date"]).max().date()
    print(f"V3.2 dataset: {len(ds)} rows. Loading NSE panels {a}..{b} ...", flush=True)
    ds = add_nse_features(ds, a, b)
    print("  multi-day labels ...", flush=True)
    ds = ds.join(multi_day_labels(ds, tok))
    ds.to_csv(config.ML_V33_DATASET_PATH, index=False)
    new = [c for c in ds.columns if c.startswith(("nse_", "fo_", "fut_", "pcr", "opt_", "call_", "put_", "oi_",
                                                   "ann_", "bm_", "ins_"))]
    print(f"\nSaved {config.ML_V33_DATASET_PATH}: {len(ds)} rows, {len(new)} new NSE features")
    print("Coverage (share of rows with a value):")
    for c in new:
        print(f"  {c:<24}{ds[c].notna().mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
