"""
V5.0 paper signals: the target portfolio for the NEXT session's open.

    python -m v5.paper                                  # after ~19:00 IST (bhavcopy published)
    python -m v5.paper --holdings live_state/v5/holdings.csv

Same data + features as research, the live model saved by `python -m
v5.run_v5` (swing/results/v5_final_model.joblib), and the same smoothing: the
last 30 sessions are scored so the EWMA has history. Writes
live_state/v5/signals_<YYYYMMDD>.csv (rank, symbol, smoothed score, target
weight, action vs holdings). Rebalance on the weekly schedule (every 5th
session); in between, only act on SELLs. Nothing is sent to a broker - see
SEBI rules (broker-registered API, algo ID) before automating.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date, timedelta

import joblib
import pandas as pd

from swing import config as cfgmod
from swing.data import nse_daily
from swing.risk import sizing
from swing.research import load_data
from v5 import features as v5f
from v5.run_v5 import blend, build_v5_dataset, predict_ml, smooth

log = logging.getLogger("v5.paper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", help="CSV with a 'symbol' column of current positions")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    cfgmod.setup_logging()
    mpath = os.path.join(cfgmod.path(cfgmod.load(), "results"), "v5_final_model.joblib")
    if not os.path.exists(mpath):
        raise SystemExit("No V5 model yet - run `python -m v5.run_v5` first.")
    b = joblib.load(mpath)
    cfg, v5, feats = b["config"], b["v5"], b["features"]
    cfg["data"]["end"] = None
    cache = cfgmod.path(cfg, "cache")
    if not a.no_download:
        nse_daily.download(cache, date.today() - timedelta(days=20), date.today(), workers=2)
    data = load_data(cfg, download=False)
    df = build_v5_dataset(cfg, data)
    days = sorted(df["date"].unique())[-30:]
    recent = df[df["date"].isin(days)]
    ml = predict_ml(b["models"], recent, feats)
    fac = pd.DataFrame({"date": recent["date"], "entity": recent["entity"], "score": v5f.factor_score(recent).values})
    raw = {"v5_ml": ml, "v5_factor": fac, "v5_combo": blend(ml, fac, v5["combo_weight_ml"])}[b["selected"]]
    s = smooth(raw, v5["halflife"])
    last = s["date"].max()
    today = s[s["date"] == last].set_index("entity")["score"]
    info = recent[recent["date"] == last].set_index("entity")
    held = set()
    if a.holdings and os.path.exists(a.holdings):
        hs = set(pd.read_csv(a.holdings)["symbol"].astype(str).str.upper())
        held = set(info.index[info["symbol"].isin(hs)])
    p = cfg["portfolio"]
    names = sizing.select(today, held, p["top_n"], p["hold_buffer"])
    w = sizing.weights(names, info["vol_63"], info["industry"], cfg)
    out = info[["symbol", "industry"]].copy()
    out["score"] = today
    out["rank"] = out["score"].rank(ascending=False, method="first").astype(int)
    out["target_weight"] = w.reindex(out.index).fillna(0.0).round(4)
    out["action"] = ["HOLD" if (e in held and e in names) else "BUY" if e in names else
                     "SELL" if e in held else "" for e in out.index]
    out = out.sort_values("rank")
    out = out[(out["rank"] <= p["top_n"] * 2) | (out["action"] != "")]
    od = os.path.join(cfgmod.ROOT, "live_state", "v5")
    os.makedirs(od, exist_ok=True)
    f = os.path.join(od, f"signals_{pd.Timestamp(last):%Y%m%d}.csv")
    out.reset_index().to_csv(f, index=False)
    log.info("V5 (%s) signals from the %s close -> %s", b["selected"], pd.Timestamp(last).date(), f)
    print(out.head(p["top_n"]).to_string())


if __name__ == "__main__":
    main()
