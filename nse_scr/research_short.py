"""
NSE SCR short-side research (intraday short, MIS), live rules from
nse_scr/strategy.py (short_setups / simulate_short), Upstox 5-min candles,
exact NSE charges (STT on the sell = the entry) + 10 bps slippage per side.

    python -m nse_scr.research_short

  lod   DOWN volume-shocker days (bhavcopy: volume >= 3x the 20-day average,
        low <= -3%, turnover >= Rs 5 cr, price >= Rs 20) - same window as the
        long research (Apr-Oct 2026); events_down.parquet
  fade  the long research's UP shocker days (events.parquet)

No lookahead from the day selection (the live trigger itself needs >= 3x
volume by the signal candle and the -3% / +FADE_MIN_UP move). Variants are
chosen on the first half of the days and checked on the second half.
"""
from __future__ import annotations

import itertools
import logging
import os

import pandas as pd

from nse_scr import research as R
from nse_scr import strategy as S

log = logging.getLogger("nse_scr")
CACHE = R.CACHE


def down_events():
    f = os.path.join(CACHE, "events_down.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    up = pd.read_parquet(os.path.join(CACHE, "events.parquet"))
    d = pd.read_parquet(os.path.join(CACHE, "daily.parquet"))
    d = d[(d.date >= up.date.min()) & (d.date <= up.date.max())]
    ev = d[(d.volume >= 3 * d.avgvol20) & (d.low <= d.prevclose * 0.97) & (d.value >= 5e7) & (d.close >= 20)]
    ev = ev[["symbol", "date"]].reset_index(drop=True)
    ev.to_parquet(f, index=False)
    return ev


def prefetch(ev):
    """Download the missing 5-min months in parallel, ONE attempt each (a few
    symbols map to a wrong Upstox instrument and would sit in 6 slow
    retries); returns the events whose months all loaded."""
    from concurrent.futures import ThreadPoolExecutor
    from backtest import candle_cache as C
    from backtest.ml.common import access_token
    tok, need = access_token(), set()
    for sym, g in ev.groupby("symbol"):
        for ms, me in C._month_ranges(g.date.min().date(), g.date.max().date()):
            if not os.path.exists(C._path(sym, 5, ms)):
                need.add((sym, ms, me))

    def one(t):
        sym, ms, me = t
        try:
            df = C._fetch_month(sym, 5, ms, me, tok, attempts=1)
        except Exception:
            return sym
        path = C._path(sym, 5, ms)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_pickle(path)
        return None

    with ThreadPoolExecutor(6) as ex:
        bad = {b for b in ex.map(one, sorted(need)) if b}
    log.info("prefetch: %d months downloaded, %d symbols skipped (no Upstox data)", len(need), len(bad))
    return ev[~ev.symbol.isin(bad)]


def run(days, mode, stop_pct, min_up=None):
    S.SHORT_STOP_PCT = stop_pct
    if min_up is not None:
        S.FADE_MIN_UP = min_up
    rows = []
    for sym, day, x, pc, av in days:
        f = S.bar_features(x, pc, av)
        busy, n = None, 0
        for ts in S.short_setups(x, f, mode):
            if n >= S.MAX_PER_SYMBOL_DAY or (busy is not None and ts < busy):
                continue
            tr = S.simulate_short(sym, x, ts, mode=mode)
            if tr is None:
                continue
            n += 1
            busy = tr.exit_ts
            r = f.loc[ts]
            rows.append({"symbol": sym, "date": day, "entry_ts": tr.entry_ts, "R": tr.R, "ret_pct": tr.ret_pct,
                         "gross_pct": (1 - tr.exit / tr.entry) * 100, "outcome": tr.outcome, "pct": r["pct"],
                         "hod_pct": r["hod_pct"], "rvol": r["rvol"], "hour": tr.entry_ts.hour})
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    rows = []
    ev = prefetch(down_events())
    ev.to_parquet(os.path.join(CACHE, "events_down.parquet"), index=False)
    for mode in ("lod", "fade"):
        days = _load(mode)
        alld = sorted({d for _, d, *_ in days})
        half = alld[len(alld) // 2]
        log.info("%s: %d stock-days (%s .. %s), split %s", mode, len(days), alld[0].date(), alld[-1].date(), half.date())
        grid = itertools.product((0.015, 0.02, 0.03), (None,)) if mode == "lod" else \
            itertools.product((0.02, 0.03), (0.03, 0.05, 0.08))
        for sp, mu in grid:
            t = run(days, mode, sp, mu)
            name = f"{mode} stop {sp:.3f}" + (f" up>={mu:.0%}" if mu else "")
            a, b = t[t.date < half], t[t.date >= half]
            rows.append({"variant": name, **{f"A_{k}": v for k, v in R.stats(a).items()},
                         **{f"B_{k}": v for k, v in R.stats(b).items()},
                         "gross%": round(t.gross_pct.mean(), 3) if len(t) else None,
                         "per_day": round(len(t) / len(alld), 1)})
            t.to_parquet(os.path.join(CACHE, f"short_{mode}_{int(sp * 1000)}_{int((mu or 0) * 100)}.parquet"), index=False)
            log.info("%s done", name)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    log.info("SHORT RESULTS (net of 10 bps slippage/side + exact NSE charges):\n%s", r.to_string(index=False))
    r.to_csv(os.path.join(CACHE, "grid_short.csv"), index=False)


def _load(mode):
    if mode == "fade":
        return R.load_days()
    real = R.pd.read_parquet
    def rp(path, *a, **k):
        if path.endswith(os.sep + "events.parquet"):
            path = os.path.join(CACHE, "events_down.parquet")
        return real(path, *a, **k)
    R.pd.read_parquet = rp
    try:
        return R.load_days()
    finally:
        R.pd.read_parquet = real


if __name__ == "__main__":
    main()
