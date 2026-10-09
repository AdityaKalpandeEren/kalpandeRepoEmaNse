"""
NSE LCR research: pullback-continuation setups on NSE MEGA / LARGE / MID
volume-shocker days (bhavcopy: full-day volume >= 1x the 20-day average and
intraday high >= +1.5%, last ~6 months), Upstox 5-min candles, live rules,
exact NSE charges + 5 bps slippage per side.

    python -m nse_lcr.research

Daily EMAs use Upstox daily closes up to the PREVIOUS session. Variants
(EMA rule x stop cap) are judged on the first half of the days, checked on
the second half; the PERFECT subset (above all six EMAs) is reported too.
"""
from __future__ import annotations

import itertools
import logging
import math
import os

import numpy as np
import pandas as pd

from backtest.candle_cache import load_daily_cached, load_symbol_history_cached
from backtest.ml.common import access_token
from nse_lcr import strategy as S

log = logging.getLogger("nse_lcr")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "nse_lcr", "cache")


def load_days():
    ev = pd.read_parquet(os.path.join(CACHE, "events.parquet"))
    uni = pd.read_parquet(os.path.join(CACHE, "universe.parquet")).set_index("symbol")["bucket"]
    daily = pd.read_parquet(os.path.join(ROOT, "nse_scr", "cache", "daily.parquet")).set_index(["symbol", "date"])
    tok = access_token()
    out = []
    for sym, g in ev.groupby("symbol"):
        try:
            bars = load_symbol_history_cached(sym, 5, g.date.min().strftime("%Y-%m-%d"), g.date.max().strftime("%Y-%m-%d"), tok)
            dly = load_daily_cached(sym, "2025-01-01", "2026-10-08", tok)
        except Exception:
            continue
        if bars.empty or dly.empty:
            continue
        bars = bars.set_index("timestamp").sort_index()
        dly["d"] = pd.to_datetime(dly["timestamp"]).dt.tz_localize(None).dt.normalize()
        for day in g.date:
            hist = dly[dly["d"] < day]
            if len(hist) < 60 or (sym, day) not in daily.index:
                continue
            x = bars[bars.index.date == day.date()]
            x = x[(x.index.time >= S.dtime(9, 15)) & (x.index.time <= S.dtime(15, 25))]
            if len(x) < 40:
                continue
            c = daily.loc[(sym, day)]
            out.append({"symbol": sym, "date": day, "x": x, "prev_close": float(c["prevclose"]),
                        "avg_vol20": float(c["avgvol20"]), "emas": S.daily_emas(hist["close"]), "bucket": uni.get(sym, "?")})
    return out


def run(days, ema_mode, stop_pct):
    S.STOP_PCT = stop_pct
    rows = []
    for d in days:
        x = d["x"]
        f = S.bar_features(x, d["prev_close"], d["avg_vol20"])
        busy, n = None, 0
        for ts in S.setups(x, f, d["emas"], ema_mode):
            if n >= S.MAX_PER_SYMBOL_DAY or (busy is not None and ts < busy):
                continue
            tr = S.simulate(d["symbol"], x, ts)
            if tr is None:
                continue
            n += 1
            busy = tr.exit_ts
            rows.append({"symbol": d["symbol"], "date": d["date"], "bucket": d["bucket"], "entry_ts": tr.entry_ts,
                         "R": tr.R, "ret_pct": tr.ret_pct, "outcome": tr.outcome,
                         "perfect": S.is_perfect(float(x.loc[ts, "close"]), d["emas"])})
    return pd.DataFrame(rows)


def stats(t):
    if t.empty:
        return {"n": 0}
    day = t.groupby("date")["ret_pct"].mean()
    return {"n": len(t), "win%": round((t.R > 0).mean() * 100, 1), "avg_R": round(t.R.mean(), 3),
            "net%": round(t.ret_pct.mean(), 3),
            "day_t": round(day.mean() / day.std() * math.sqrt(len(day)), 2) if len(day) > 2 and day.std() > 0 else np.nan}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    days = load_days()
    alld = sorted({d["date"] for d in days})
    half = alld[len(alld) // 2]
    log.info("stock-days: %d (%s .. %s), split %s", len(days), alld[0].date(), alld[-1].date(), half.date())
    rows = []
    for em, sp in itertools.product(("none", "min", "all"), (0.015, 0.02)):
        t = run(days, em, sp)
        groups = {f"{em} {sp:.3f}": t}
        if em == "min":
            groups[f"  min {sp:.3f} PERFECT"] = t[t.perfect]
            groups[f"  min {sp:.3f} not perfect"] = t[~t.perfect]
        for name, x in groups.items():
            a, b = x[x.date < half], x[x.date >= half]
            rows.append({"variant": name, **{f"A_{k}": v for k, v in stats(a).items()},
                         **{f"B_{k}": v for k, v in stats(b).items()}, "per_day": round(len(x) / len(alld), 1)})
        t.to_parquet(os.path.join(CACHE, f"trades_{em}_{int(sp * 1000)}.parquet"), index=False)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    log.info("RESULTS (net of 5 bps slippage/side + exact NSE charges):\n%s", r.to_string(index=False))
    r.to_csv(os.path.join(CACHE, "grid.csv"), index=False)


if __name__ == "__main__":
    main()
