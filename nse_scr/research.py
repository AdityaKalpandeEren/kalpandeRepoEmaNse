"""
NSE SCR research: every setup on every NSE volume-shocker day (bhavcopy:
full-day volume >= 3x the 20-day average, intraday high >= +3%, turnover
>= Rs 5 cr, price >= Rs 20, last ~6 months), simulated with the live rules
on Upstox 5-min bars, exact NSE intraday charges included.

    python -m nse_scr.research

No lookahead from the day selection: the live trigger needs >= 3x the
average DAILY volume traded BY the signal bar and >= +3%, so it can only
fire on days that end with >= 3x volume and a >= +3% high - the selection
only drops days that could never trigger.
Variants (setup x stop cap) are chosen on the first half of the days and
checked on the second half.
"""
from __future__ import annotations

import itertools
import logging
import math
import os

import numpy as np
import pandas as pd

from backtest.candle_cache import load_symbol_history_cached
from backtest.ml.common import access_token
from nse_scr import strategy as S

log = logging.getLogger("nse_scr")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "nse_scr", "cache")


def load_days():
    d = pd.read_parquet(os.path.join(CACHE, "daily.parquet"))
    ev = pd.read_parquet(os.path.join(CACHE, "events.parquet"))
    ctx = d.set_index(["symbol", "date"])[["prevclose", "avgvol20", "value"]]
    tok = access_token()
    out = []
    for sym, g in ev.groupby("symbol"):
        try:
            bars = load_symbol_history_cached(sym, 5, g.date.min().strftime("%Y-%m-%d"), g.date.max().strftime("%Y-%m-%d"), tok)
        except Exception:
            continue
        if bars.empty:
            continue
        bars = bars.set_index("timestamp").sort_index()
        for day in g.date:
            x = bars[bars.index.date == day.date()]
            x = x[(x.index.time >= S.dtime(9, 15)) & (x.index.time <= S.dtime(15, 25))]
            if len(x) < 20:
                continue
            c = ctx.loc[(sym, day)]
            out.append((sym, day, x, float(c["prevclose"]), float(c["avgvol20"])))
    return out


def run(days, mode, stop_pct):
    S.STOP_PCT = stop_pct
    rows = []
    for sym, day, x, pc, av in days:
        f = S.bar_features(x, pc, av)
        busy, n = None, 0
        for ts in S.setups(x, f, mode):
            if n >= S.MAX_PER_SYMBOL_DAY or (busy is not None and ts < busy):
                continue
            tr = S.simulate(sym, x, ts, mode=mode)
            if tr is None:
                continue
            n += 1
            busy = tr.exit_ts
            r = f.loc[ts]
            rows.append({"symbol": sym, "date": day, "entry_ts": tr.entry_ts, "R": tr.R, "ret_pct": tr.ret_pct,
                         "outcome": tr.outcome, "pct": r["pct"], "rvol": r["rvol"], "turnover": r["turnover"],
                         "hour": tr.entry_ts.hour})
    return pd.DataFrame(rows)


def stats(t):
    if t.empty:
        return {"n": 0}
    day = t.groupby("date")["ret_pct"].mean()
    return {"n": len(t), "win%": round((t.R > 0).mean() * 100, 1), "avg_R": round(t.R.mean(), 3),
            "net%": round(t.ret_pct.mean(), 3), "day_t": round(day.mean() / day.std() * math.sqrt(len(day)), 2) if len(day) > 2 else np.nan}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    days = load_days()
    alld = sorted({d for _, d, *_ in days})
    half = alld[len(alld) // 2]
    log.info("shocker stock-days with 5-min bars: %d (%s .. %s), halves split %s", len(days), alld[0].date(), alld[-1].date(), half.date())
    rows = []
    for mode, sp in itertools.product(("pullback", "hod"), (0.02, 0.03, 0.04)):
        t = run(days, mode, sp)
        a, b = t[t.date < half], t[t.date >= half]
        rows.append({"setup": mode, "stop_cap": sp, **{f"A_{k}": v for k, v in stats(a).items()}, **{f"B_{k}": v for k, v in stats(b).items()}})
        t.to_parquet(os.path.join(CACHE, f"trades_{mode}_{int(sp * 100)}.parquet"), index=False)
        log.info("%s stop %.0f%% done", mode, sp * 100)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    log.info("RESULTS (net of 10 bps slippage/side + exact NSE charges):\n%s", r.to_string(index=False))
    r.to_csv(os.path.join(CACHE, "grid.csv"), index=False)


if __name__ == "__main__":
    main()
