"""
V4.0 paper trading: tomorrow's target portfolio as a CSV (no orders sent).

Run after the NSE close (bhavcopy is published ~18:00-19:00 IST):

    python -m v4.paper                       # signals for the next session
    python -m v4.paper --holdings live_state/v4/holdings.csv   # mark which names to buy / sell / keep

Uses the final model saved by `python -m v4.run_research --holdout`
(v4/results/v4_final_model.joblib) and the same data + feature pipeline as
the research run, so live features match the backtest exactly. Output:
live_state/v4/signals_<YYYYMMDD>.csv with rank, symbol, score, target
weight, industry, and action (BUY / HOLD / SELL) vs the given holdings.
Trades are meant for the NEXT session's open, as in the backtest.

SEBI: this only writes a file. Any automated order placement through a
broker API must follow SEBI's retail algo framework (broker-registered
API, algo ID) - check with your broker first.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date, timedelta

import joblib
import pandas as pd

from v4 import config as cfgmod
from v4.data import nse_daily
from v4.features import build as fbuild
from v4.models import baselines, gbm
from v4.risk import sizing
from v4.run_research import dataset, ensemble, load_data

log = logging.getLogger("v4.paper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", help="CSV with a 'symbol' column of current positions")
    ap.add_argument("--model", default=None, help="path to v4_final_model.joblib")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    cfgmod.setup_logging()
    mpath = a.model or os.path.join(cfgmod.path(cfgmod.load(), "results"), "v4_final_model.joblib")
    if not os.path.exists(mpath):
        raise SystemExit("No final model yet - run `python -m v4.run_research --holdout` first.")
    bundle = joblib.load(mpath)
    cfg = bundle["config"]
    cfg["data"]["end"] = None                                 # up to yesterday / today's file if published
    cache = cfgmod.path(cfg, "cache")
    end = date.today()
    if not a.no_download:
        nse_daily.download(cache, end - timedelta(days=20), end, workers=2)
    data = load_data(cfg, download=False)
    df = dataset(cfg, data, rebuild=True)
    last = df["date"].max()
    today = df[df["date"] == last].copy()
    feats = bundle["features"]
    sel = bundle["selected"]
    if sel in baselines.NAMES:
        today["score"] = baselines.score(sel, today).values
    elif sel == "ensemble":
        parts = {n: pd.DataFrame({"date": today["date"], "entity": today["entity"],
                                  "score": gbm.predict(m, today, feats)}) for n, m in bundle["models"].items()}
        today = today.merge(ensemble(parts), on=["date", "entity"])
    else:
        today["score"] = gbm.predict(bundle["models"][sel], today, feats)

    held = set()
    if a.holdings and os.path.exists(a.holdings):
        held_syms = set(pd.read_csv(a.holdings)["symbol"].astype(str).str.upper())
        held = set(today.loc[today["symbol"].isin(held_syms), "entity"])
    p = cfg["portfolio"]
    names = sizing.select(today.set_index("entity")["score"], held, p["top_n"], p["hold_buffer"])
    t = today.set_index("entity")
    w = sizing.weights(names, t["vol_63"], t["industry"], cfg)
    out = t.loc[:, ["symbol", "industry", "score"]].copy()
    out["rank"] = out["score"].rank(ascending=False, method="first").astype(int)
    out["target_weight"] = w.reindex(out.index).fillna(0.0).round(4)
    out["action"] = ["HOLD" if (e in held and e in names) else "BUY" if e in names else
                     "SELL" if e in held else "" for e in out.index]
    out = out.sort_values("rank")
    out = out[(out["rank"] <= p["top_n"] * 2) | (out["action"] != "")]
    od = os.path.join(cfgmod.ROOT, "live_state", "v4")
    os.makedirs(od, exist_ok=True)
    f = os.path.join(od, f"signals_{last:%Y%m%d}.csv")
    out.reset_index().to_csv(f, index=False)
    log.info("signals from the %s close (model %s, trained through %s) -> %s", last.date(), sel,
             bundle.get("trained_through"), f)
    print(out.head(p["top_n"] * 2).to_string())


if __name__ == "__main__":
    main()
