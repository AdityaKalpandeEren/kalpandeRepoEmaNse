"""
V8 "leaders" research on NSE: do the published setups of the most successful
momentum traders work here, after delivery costs?  (setups: v8/setups.py)

Data: every NSE EQ stock 2013-01 .. 2026-09 incl. later delisted ones (bhavcopy
panel, corporate-action adjusted, swing/cache) - no survivorship. Liquid =
20-day median traded value >= Rs 5 crore and price >= Rs 30, on that day.
Costs: delivery charges (swing/config.yaml costs: STT 0.1% each side, stamp,
exchange, SEBI, GST, DP) + slippage 12 bps at the open/close, 25 bps for
stop orders (buy-stop entries and stop-loss exits).  Rs 10 lakh book.

Variants, fixed before the first run (all reported, all counted as trials):
  BO_10       breakout, 1/3 sold after 3 sessions, rest trails the 10-day average, market filter on
  BO_20       same, 20-day average
  BO_10_nf    BO_10 without the market filter
  EP_10       episodic pivot, 10-day trail, market filter on
  EP_20       same, 20-day trail
  EP_fresh_10 EP_10, only stocks that had not rallied > 30% in the 6 months before the gap
  BOEP_10     BO + EP in one book (how Qullamaggie trades), 10-day trail, market filter on
  ATH_10ATR   all-time-high trend following, 10 x ATR(42) trailing stop, no market filter
  ATH504_10ATR  (added after the first run) same with a 2-year high - what the live job can see
BO/EP: risk 1% of equity per trade, max 20% in one stock, max 10 positions.
ATH:   risk 1%, max 10% in one stock, max 30 positions.
Market filter (Qullamaggie): NIFTY 50 10-day average above its 20-day average.

Windows: one continuous run 2014-01 .. 2026-09, reported as
  WF  2017-01 .. 2025-06 (V7's walk-forward window) and its halves A / B
  PV  2025-07 .. 2026-09 (previously viewed by V4/V5/V7 research)
Keep rule (fixed in advance): beat NIFTY in both WF halves AND beat live V7
(D1) on WF Sharpe. PV is shown, not used to choose.

    python -m v8.research            # all variants
    python -m v8.research --v7       # also (re)build live V7's daily returns for the comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from swing import config as cfgmod
from swing.backtest.costs import order_charges
from swing.data import nse_daily
from swing.research import record_trials
from swing.validation import metrics
from v8 import engine, setups

log = logging.getLogger("v8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "v8", "results")
PANEL = os.path.join(ROOT, "swing", "cache", "panel_20130101_20260930.parquet")
START, END = "2014-01-01", "2026-09-30"
WINDOWS = {"EARLY": ("2014-01-01", "2016-12-31"), "WF": ("2017-01-01", "2025-06-30"),
           "WF_A": ("2017-01-01", "2021-06-30"), "WF_B": ("2021-07-01", "2025-06-30"),
           "PV": ("2025-07-01", "2026-09-30")}
RF = 0.065

PARAMS = {"min_value": 5e7, "min_price": 30.0, "ep_gap": 0.08, "ep_vol": 3.0, "ep_fresh_max": 0.30,
          "bo_rs_min": 0.85, "bo_tight_adr": 2.5}
BOOK = {"start_equity": 1_000_000, "risk_pct": 0.01, "max_pos": 10, "max_weight": 0.20, "partial_day": 3,
        "partial_frac": 1 / 3, "bo_stop_cap": 0.08, "ep_stop_cap": 0.12, "ep_skip_close_at_high": True,
        "fill_mode": "pessimistic", "regime": True, "trail_ma": 10}
VARIANTS = {
    "BO_10": {"setups": ["BO"]},
    "BO_20": {"setups": ["BO"], "trail_ma": 20},
    "BO_10_nf": {"setups": ["BO"], "regime": False},
    "EP_10": {"setups": ["EP"]},
    "EP_20": {"setups": ["EP"], "trail_ma": 20},
    "EP_fresh_10": {"setups": ["EP"], "ep_signal": "ep_fresh"},
    "BOEP_10": {"setups": ["BO", "EP"]},
    "ATH_10ATR": {"setups": ["ATH"], "regime": False, "max_pos": 30, "max_weight": 0.10},
    # added after the first run: the version a live job can run (it keeps ~2 years of
    # daily data), i.e. a 2-year-high breakout with the same 10 x ATR(42) trailing stop
    "ATH504_10ATR": {"setups": ["ATH"], "regime": False, "max_pos": 30, "max_weight": 0.10, "ath_signal": "ath_504"},
}


class NseCosts:
    def __init__(self, c: dict, slip_open=0.0012, slip_stop=0.0025):
        self.c, self.s = c, {"open": slip_open, "close": slip_open, "stop": slip_stop}
        self.buy_rate = order_charges(1e6, "buy", c) / 1e6

    def slip(self, kind):
        return self.s[kind]

    def buy(self, money):
        return order_charges(money, "buy", self.c)

    def sell(self, money):
        return order_charges(money, "sell", self.c)


def load():
    t = time.time()
    p = pd.read_parquet(PANEL, columns=["date", "entity", "symbol", "close", "adj_open", "adj_high", "adj_low",
                                        "adj_close", "value"])
    p = p.sort_values(["entity", "date"])
    med = p.groupby("entity")["value"].transform(lambda s: s.rolling(20, min_periods=15).median())
    keep = p.loc[(med >= PARAMS["min_value"]) & (p["close"] >= PARAMS["min_price"]), "entity"].unique()
    p = p[p["entity"].isin(keep)]
    cash = p.groupby("entity")["symbol"].agg(lambda s: any(setups.cash_like(s.unique())))
    p = p[~p["entity"].map(cash)]                                    # no money-market / bond ETFs
    wide = {k: p.pivot(index="date", columns="entity", values=v)
            for k, v in (("o", "adj_open"), ("h", "adj_high"), ("l", "adj_low"), ("c", "adj_close"),
                         ("val", "value"), ("raw", "close"))}
    names = p.groupby("entity")["symbol"].last()
    log.info("panel: %d entities ever liquid, %d sessions (%.0fs)", len(keep), len(wide["c"]), time.time() - t)
    return wide, names


def nifty(dates) -> pd.Series:
    cfg = cfgmod.load({})
    ix = nse_daily.load_kind(cfgmod.path(cfg, "cache"), "indices", pd.Timestamp("2013-01-01").date(),
                             pd.Timestamp(END).date())
    s = ix[ix["index"] == "NIFTY 50"].drop_duplicates("date").set_index("date")["close"].sort_index()
    return s.reindex(dates).ffill()


def arrays(wide, ind):
    c = wide["c"]
    px = {"dates": c.index, "entities": list(c.columns), "O": wide["o"].to_numpy(float), "H": wide["h"].to_numpy(float),
          "L": wide["l"].to_numpy(float), "C": c.to_numpy(float), "CF": c.ffill().to_numpy(float)}
    lv = c.notna().to_numpy()
    px["last_valid"] = np.where(lv.any(0), len(c) - 1 - np.argmax(lv[::-1], axis=0), -1)
    na = {}
    for k in ("bo", "ep", "ep_fresh", "ath", "ath_504"):
        na[k] = ind[k].fillna(False).to_numpy(bool)
    for k in ("pivot", "adr", "rs", "vratio", "atr42", "sma10", "sma20"):
        na[k] = ind[k].to_numpy(float)
    return px, na


def window_stats(ret, bench, w):
    a, b = WINDOWS[w]
    r = ret[(ret.index >= a) & (ret.index <= b)]
    return engine.perf(r, RF, bench)


def v7_reference(rebuild: bool) -> pd.Series | None:
    """Live V7 (D1 score, top 30 of the top-200, monthly, 10% stop, regime) daily returns,
    one continuous run 2017-01 .. 2026-09 with V7's own engine and costs."""
    f = os.path.join(RES, "v7_d1_daily.parquet")
    if os.path.exists(f) and not rebuild:
        return pd.read_parquet(f)["ret"]
    if not rebuild:
        return None
    from v7.momentum import d1_score
    from v7.swing_v7 import M30_PORT, V5_OVERRIDES, prepare, run_bt
    x = prepare()
    live = {**M30_PORT, "portfolio.top_n": 30, "risk.stop_loss": 0.10, "risk.regime_filter": True}
    cfg = cfgmod.load({**V5_OVERRIDES, **live})
    res = run_bt(cfg, d1_score(x["panel"], x["u200"]), x["prices"], x["info200"], x["ctx"], ("2017-01-01", END))
    r = res.daily["ret"].iloc[1:]
    r.to_frame("ret").to_parquet(f)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7", action="store_true", help="rebuild live V7's daily returns for the comparison")
    ap.add_argument("--only", help="comma list of variants")
    a = ap.parse_args()
    cfgmod.setup_logging()
    os.makedirs(RES, exist_ok=True)
    wide, names = load()
    t = time.time()
    ind = setups.indicators(wide["o"], wide["h"], wide["l"], wide["c"], wide["val"], wide["raw"], PARAMS)
    px, na = arrays(wide, ind)
    idx = nifty(px["dates"])
    reg = setups.regime(idx, px["dates"])
    bench = idx.pct_change()
    log.info("indicators %.0fs | signals per year: BO %.0f EP %.0f EP_fresh %.0f ATH %.0f | market filter on %.0f%% of days",
             time.time() - t, na["bo"].sum() / 12.75, na["ep"].sum() / 12.75, na["ep_fresh"].sum() / 12.75,
             na["ath"].sum() / 12.75, 100 * reg.mean())
    costs = NseCosts(cfgmod.load({})["costs"])
    v7 = v7_reference(a.v7)

    rows, rets, trade_rows = [], {}, []
    todo = {k: v for k, v in VARIANTS.items() if not a.only or k in a.only.split(",")}
    for name, over in todo.items():
        cfg = {**BOOK, **over}
        t = time.time()
        res = engine.run(px, na, cfg, costs, START, END, reg)
        r = res.daily["ret"]
        rets[name] = r
        res.trades.assign(variant=name, symbol=res.trades["entity"].map(names) if len(res.trades) else None) \
            .to_csv(os.path.join(RES, f"trades_{name}.csv"), index=False)
        res.daily.to_csv(os.path.join(RES, f"daily_{name}.csv"))
        row = {"variant": name}
        for w in WINDOWS:
            s = window_stats(r, bench, w)
            row.update({f"{w}_CAGR": s.get("CAGR"), f"{w}_Sharpe": s.get("Sharpe"), f"{w}_MaxDD": s.get("MaxDD")})
        tr = res.trades
        wf_tr = tr[(tr["entry_date"] >= WINDOWS["WF"][0]) & (tr["entry_date"] <= WINDOWS["WF"][1])] if len(tr) else tr
        row.update({f"WF_{k}": v for k, v in engine.trade_stats(wf_tr).items()})
        row["avg_exposure"] = (res.daily["invested"] / res.daily["equity"]).mean()
        row["cost_drag"] = res.daily["costs"].iloc[-1] / res.daily["equity"].mean() / (len(r) / 252)
        rows.append(row)
        log.info("%-12s WF CAGR %+.1f%% Sharpe %.2f MaxDD %.1f%% | PV %+.1f%% | trades %d (%.0fs)", name,
                 100 * row["WF_CAGR"], row["WF_Sharpe"], 100 * row["WF_MaxDD"], 100 * (row["PV_CAGR"] or 0),
                 len(tr), time.time() - t)

    t = pd.DataFrame(rows).set_index("variant")
    ref = {"NIFTY 50": bench.reindex(px["dates"]).fillna(0)}
    if v7 is not None:
        ref["V7 live (D1)"] = v7
    for nm, r in ref.items():
        row = {}
        for w in WINDOWS:
            s = window_stats(r, None, w)
            row.update({f"{w}_CAGR": s.get("CAGR"), f"{w}_Sharpe": s.get("Sharpe"), f"{w}_MaxDD": s.get("MaxDD")})
        t.loc[nm, list(row)] = list(row.values())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = [f"{w}_{m}" for w in WINDOWS for m in ("CAGR", "Sharpe", "MaxDD")]
    print("\nPERFORMANCE after all costs (CAGR / Sharpe above 6.5% rf / MaxDD):")
    print(t[cols].astype(float).round(3).to_string())
    tcols = [c for c in t.columns if c.startswith("WF_") and c.split("_", 1)[1] in
             ("Trades", "Win%", "AvgWin%", "AvgLoss%", "AvgR", "PF", "Hold", "Top7%share")] + ["avg_exposure", "cost_drag"]
    print("\nTRADES inside WF (entries 2017-01 .. 2025-06):")
    print(t.loc[list(todo), tcols].astype(float).round(3).to_string())

    # keep rule + deflated Sharpe over every trial ever recorded (V4/V5/V7 + V8)
    c = cfgmod.load({})
    c["v8_run"] = "v8_leaders_nse"
    wf_rets = {f"v8: {k}": v[(v.index >= WINDOWS['WF'][0]) & (v.index <= WINDOWS['WF'][1])] for k, v in rets.items()}
    trials = record_trials(cfgmod.path(c, "results", "v7_swing"), c, wf_rets)
    nb = {w: window_stats(ref["NIFTY 50"], None, w) for w in ("WF_A", "WF_B")}
    v7_wf = window_stats(v7, None, "WF")["Sharpe"] if v7 is not None else None
    print("\nKEEP RULE (beat NIFTY CAGR in WF_A and WF_B, and live V7's WF Sharpe):")
    for k in todo:
        ra, rb = t.loc[k, "WF_A_CAGR"], t.loc[k, "WF_B_CAGR"]
        ok = ra > nb["WF_A"]["CAGR"] and rb > nb["WF_B"]["CAGR"] and (v7_wf is None or t.loc[k, "WF_Sharpe"] > v7_wf)
        dsr = metrics.deflated_sharpe(wf_rets[f"v8: {k}"], trials)
        print(f"  {k:12s} {'KEEP' if ok else 'no  '}  DSR {dsr['DSR']:.3f} over {dsr['N_trials']} trials (PSR vs 0 {dsr['PSR_vs_0']:.3f})")
    t.to_csv(os.path.join(RES, "summary.csv"))
    by_year = pd.DataFrame({k: v.groupby(v.index.year).apply(lambda x: (1 + x).prod() - 1) for k, v in {**rets, **ref}.items()})
    print("\nCALENDAR-YEAR RETURNS:")
    print((100 * by_year).round(1).to_string())
    by_year.to_csv(os.path.join(RES, "by_year.csv"))


if __name__ == "__main__":
    main()
