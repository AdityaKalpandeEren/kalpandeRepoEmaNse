"""
NSE LCR (Large-Cap Runners) - MEGA (NIFTY 50) / LARGE (rest of NIFTY 100) /
MID (NIFTY Midcap 150) stocks with unusual volume. Port of the US LCR rules
(kalpandeRepoEmaUs us_lcr/strategy.py) with NSE specifics from nse_scr.

Everything on candle t uses data up to the CLOSE of candle t only.

WATCH  TIME-ADJUSTED volume (by candle t) >= RVOL_MIN x what a normal day
       has traded by that time (NSE intraday profile, VOL_CURVE), up >=
       WATCH_PCT vs the previous close, turnover today >= MIN_TURNOVER,
       above VWAP, not circuit-locked.
TREND  minimum: price above the 10- and 20-day EMAs of the daily close
       (as of the PREVIOUS session); above ALL six (10/20/30/40/60/180) is
       flagged PERFECT TRADE.
SETUP  pullback continuation (recent high of day, pullback on lighter
       volume, reclaim of the previous candle's high).
ENTRY  next 5-min candle's open + SLIP (no fill on a locked candle); stop
       under the pullback low, capped at STOP_PCT; breakeven after +1R then
       trail under candle lows; square-off 15:15 IST.
COSTS  exact NSE intraday charges (strategy/v33_costs) + SLIP per side.

RESEARCH (2026-10-09, nse_lcr/research.py; 5,256 MEGA/LARGE/MID shocker days
Apr-Oct 2026, EMA none / 10&20 / all six x 1.5% / 2% stop): EVERY variant
loses in both halves, -0.09..-0.18% per trade net (~0% before costs);
PERFECT (all six EMAs) is WORSE on NSE (-0.14..-0.18%). Live at the user's
request (10&20 EMA rule, 2% stop), labelled UNPROVEN - unlike US LCR.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

RVOL_MIN = 2.0
WATCH_PCT = 0.015
MIN_TURNOVER = 2e8                # Rs 20 crore traded today
STOP_PCT = 0.02
SLIP = 0.0005                     # 5 bps per side - liquid large / mid caps
PB_LOOKBACK, PB_MIN_PCT = 6, 0.008
FIRST_ENTRY, LAST_ENTRY, SQUARE_OFF = dtime(9, 30), dtime(14, 45), dtime(15, 15)
MAX_PER_SYMBOL_DAY = 2
NOTIONAL = 200_000                # Rs per paper trade (for the exact charge model)
EMA_SPANS = (10, 20, 30, 40, 60, 180)
MIN_SPANS = (10, 20)
EMA_MODE = "min"                  # "none" | "min" (10 & 20) | "all" (all six)

# Typical cumulative share of an NSE stock's daily volume by each time (IST,
# 09:15-15:30; heavy open and close, incl. the closing half hour).
VOL_CURVE = [(9 * 60 + 15, 0.0), (9 * 60 + 20, 0.05), (9 * 60 + 30, 0.10), (10 * 60, 0.20), (10 * 60 + 30, 0.28),
             (11 * 60, 0.35), (12 * 60, 0.47), (13 * 60, 0.57), (14 * 60, 0.68), (14 * 60 + 30, 0.75),
             (15 * 60, 0.86), (15 * 60 + 30, 1.0)]


@dataclass
class Trade:
    symbol: str
    day: object
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry: float
    stop0: float
    exit_ts: pd.Timestamp | None = None
    exit: float | None = None
    outcome: str = "OPEN"
    R: float = 0.0
    ret_pct: float = 0.0          # net of slippage AND charges


def expected_share(ts) -> float:
    m = ts.hour * 60 + ts.minute + 5
    xs, ys = zip(*VOL_CURVE)
    return max(float(np.interp(m, xs, ys)), 0.02)


def daily_emas(closes: pd.Series) -> dict:
    return {f"ema{n}": float(closes.ewm(span=n, adjust=False).mean().iloc[-1]) for n in EMA_SPANS}


def ema_ok(price: float, emas: dict, mode: str | None = None) -> bool:
    mode = mode or EMA_MODE
    if mode == "none" or not emas:
        return True
    spans = MIN_SPANS if mode == "min" else EMA_SPANS
    return all(price > emas[f"ema{n}"] for n in spans)


def is_perfect(price: float, emas: dict) -> bool:
    return bool(emas) and all(price > emas[f"ema{n}"] for n in EMA_SPANS)


def ema_tags(price: float, emas: dict) -> str:
    return " ".join(f"{'✅' if price > emas[f'ema{n}'] else '❌'}{n}d" for n in EMA_SPANS)


def bar_features(x: pd.DataFrame, prev_close: float, avg_vol20: float) -> pd.DataFrame:
    f = pd.DataFrame(index=x.index)
    c, h, lo, v = x["close"], x["high"], x["low"], x["volume"].fillna(0)
    f["pct"] = c / prev_close - 1
    f["cum_vol"] = v.cumsum()
    tp = (h + lo + c) / 3
    f["turnover"] = (v * tp).cumsum()
    share = pd.Series([expected_share(ts) for ts in x.index], index=x.index)
    f["rvol"] = f["cum_vol"] / (avg_vol20 * share) if avg_vol20 and avg_vol20 > 0 else np.nan
    f["vwap"] = (tp * v).cumsum() / f["cum_vol"].replace(0, np.nan)
    f["dist_vwap"] = c / f["vwap"] - 1
    f["locked"] = h == lo
    return f


def pullback_low(h, lo, v, c, i):
    j0 = max(0, i - PB_LOOKBACK)
    if i - j0 < 2:
        return None
    k = j0 + int(np.argmax(h[j0:i]))
    hod = h[:i].max()
    if h[k] < hod * 0.999 or k >= i - 1:
        return None
    pl = lo[k + 1:i].min()
    if pl <= hod * (1 - PB_MIN_PCT) and v[k + 1:i].mean() < v[k] and c[i] > h[i - 1] and c[i] >= hod * 0.99:
        return float(pl)
    return None


def setups(x: pd.DataFrame, f: pd.DataFrame, emas: dict | None = None, ema_mode: str | None = None) -> pd.Index:
    t = x.index.time
    watch = ((t >= FIRST_ENTRY) & (t <= LAST_ENTRY) & (f["rvol"] >= RVOL_MIN) & (f["pct"] >= WATCH_PCT)
             & (f["turnover"] >= MIN_TURNOVER) & (f["dist_vwap"] > 0) & ~f["locked"]).fillna(False).values
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    return x.index[[i for i in range(2, len(x)) if watch[i] and ema_ok(c[i], emas or {}, ema_mode)
                    and pullback_low(h, lo, v, c, i) is not None]]


def initial_stop(x: pd.DataFrame, i: int, entry: float) -> float:
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    s = pullback_low(h, lo, v, c, i)
    s = entry * (1 - STOP_PCT) if s is None else s
    return max(min(s, entry * 0.998), entry * (1 - STOP_PCT))


def charges_pct(entry: float, exit_: float) -> float:
    from strategy.v33_costs import round_trip
    qty = max(int(NOTIONAL // entry), 1)
    return round_trip(entry, exit_, qty, slippage_bps=0)["charges"] / (qty * entry)


def simulate(sym, x: pd.DataFrame, sig_ts, final: bool = True) -> Trade | None:
    i = x.index.get_loc(sig_ts)
    if i + 1 >= len(x):
        return None
    eb, ets = x.iloc[i + 1], x.index[i + 1]
    if ets.time() >= SQUARE_OFF or float(eb["high"]) == float(eb["low"]):
        return None
    entry = float(eb["open"]) * (1 + SLIP)
    stop = initial_stop(x, i, entry)
    risk = entry - stop
    tr = Trade(sym, ets.date(), sig_ts, ets, entry, stop)
    best = entry
    for j in range(i + 1, len(x)):
        ts, b = x.index[j], x.iloc[j]
        if j > i + 1 and float(b["open"]) <= stop:
            px, out = float(b["open"]), "STOP_GAP"
        elif float(b["low"]) <= stop:
            px, out = stop, ("STOP" if stop < entry else "TRAIL")
        elif ts.time() >= SQUARE_OFF:
            px, out = float(b["open"]), "SQUARE_OFF"
        else:
            best = max(best, float(b["close"]))
            if best - entry >= risk:
                stop = max(stop, entry, float(b["low"]))
            continue
        e = px * (1 - SLIP)
        net = e / entry - 1 - charges_pct(entry, e)
        tr.exit_ts, tr.exit, tr.outcome = ts, e, out
        tr.ret_pct, tr.R = net * 100, net * entry / risk
        return tr
    if not final:
        tr.stop0 = stop
        tr.R = (float(x.iloc[-1]["close"]) - entry) / risk
        return tr
    e = float(x.iloc[-1]["close"]) * (1 - SLIP)
    net = e / entry - 1 - charges_pct(entry, e)
    tr.exit_ts, tr.exit, tr.outcome = x.index[-1], e, "SQUARE_OFF"
    tr.ret_pct, tr.R = net * 100, net * entry / risk
    return tr
