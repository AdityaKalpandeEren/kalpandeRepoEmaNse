"""
NSE SCR (Volume Shockers & Runners) - the rules, shared by research and the
live job (port of the US SCR rules in kalpandeRepoEmaUs us_scr/strategy.py).

Everything on bar t uses data up to the CLOSE of bar t only.

WATCH  (dynamic list, rebuilt every run from NSE's live screeners):
       volume traded today >= RVOL_MIN x the stock's 20-day average DAILY
       volume (a "volume shocker"), up >= WATCH_PCT vs the previous close,
       turnover today >= MIN_TURNOVER, price >= MIN_PRICE.
SETUP  pullback continuation (default) or high-of-day breakout - see
       SETUP_MODE; chosen by research (us_scr lesson: pullback lost least).
ENTRY  the NEXT 5-min bar's open + slippage; no fill on a circuit-LOCKED
       bar (high == low) - you can't buy a stock frozen at its upper band.
STOP   structural (pullback low / breakout-bar low), capped at STOP_PCT.
       A bar that opens below the stop exits at that open.
EXIT   after +1R the stop moves to breakeven, then trails under each closed
       bar's low; square-off 15:15 IST (the NSE research session close).
COSTS  exact NSE intraday charges (strategy/v33_costs: Rs 20/order + GST,
       STT on sells, exchange, SEBI, stamp) + SLIP per side.

RESEARCH (2026-10-07, nse_scr/research.py): 7,628 shocker stock-days
(Apr-Oct 2026), 6 variants (pullback / HOD x 2 / 3 / 4% stop), halves A/B:
EVERY variant loses in BOTH halves, -0.36..-0.54% per trade net, day-level
t -7 to -9 (about -0.12% even before costs). The bigger the shock the worse
(volume > 25x: -0.8..-0.9%; already up > 12%: -0.8%) - by the time NSE's
screeners show a shocker the move is mostly done. Live paper trades run
anyway at the user's request (least-bad: HOD, 2% stop), labelled UNPROVEN.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

RVOL_MIN = 3.0
WATCH_PCT = 0.03
MIN_TURNOVER = 5e7            # Rs 5 crore traded today
MIN_PRICE = 20.0
VOL_SURGE = 2.0
STOP_PCT = 0.02              # least-bad cap in the 2026-10-07 test (2 / 3 / 4% tried)
SLIP = 0.0010                 # 10 bps per side
SETUP_MODE = "hod"            # "pullback" | "hod" - hod lost least (-0.38% / -0.36% per trade)
PB_LOOKBACK, PB_MIN_PCT = 6, 0.01
FIRST_ENTRY, LAST_ENTRY, SQUARE_OFF = dtime(9, 30), dtime(14, 45), dtime(15, 15)
MAX_PER_SYMBOL_DAY = 2
NOTIONAL = 100_000            # Rs per paper trade (for the exact charge model)


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


def bar_features(x: pd.DataFrame, prev_close: float, avg_vol20: float) -> pd.DataFrame:
    f = pd.DataFrame(index=x.index)
    c, h, lo, v = x["close"], x["high"], x["low"], x["volume"].fillna(0)
    f["pct"] = c / prev_close - 1
    f["cum_vol"] = v.cumsum()
    tp = (h + lo + c) / 3
    f["turnover"] = (v * tp).cumsum()
    f["rvol"] = f["cum_vol"] / avg_vol20 if avg_vol20 and avg_vol20 > 0 else np.nan
    f["hod_prior"] = h.cummax().shift()
    f["new_hod"] = c > f["hod_prior"]
    f["hod_pct"] = h.cummax() / prev_close - 1
    f["vol_ratio"] = v / v.where(v > 0).shift().rolling(12, min_periods=3).median()
    f["vwap"] = (tp * v).cumsum() / f["cum_vol"].replace(0, np.nan)
    f["dist_vwap"] = c / f["vwap"] - 1
    f["locked"] = (h == lo)
    f["minute"] = x.index.hour * 60 + x.index.minute
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
    if pl <= hod * (1 - PB_MIN_PCT) and v[k + 1:i].mean() < v[k] and c[i] > h[i - 1] and c[i] >= hod * 0.985:
        return float(pl)
    return None


def setups(x: pd.DataFrame, f: pd.DataFrame, mode: str | None = None) -> pd.Index:
    mode = mode or SETUP_MODE
    t = x.index.time
    watch = ((t >= FIRST_ENTRY) & (t <= LAST_ENTRY) & (f["rvol"] >= RVOL_MIN) & (f["pct"] >= WATCH_PCT)
             & (f["turnover"] >= MIN_TURNOVER) & (x["close"] >= MIN_PRICE) & (f["dist_vwap"] > 0)
             & ~f["locked"]).fillna(False).values
    if mode == "hod":
        m = watch & (f["new_hod"] & (f["vol_ratio"] >= VOL_SURGE)).fillna(False).values
        return x.index[m]
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    return x.index[[i for i in range(2, len(x)) if watch[i] and pullback_low(h, lo, v, c, i) is not None]]


def initial_stop(x: pd.DataFrame, i: int, entry: float, mode: str | None = None) -> float:
    mode = mode or SETUP_MODE
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    s = pullback_low(h, lo, v, c, i) if mode == "pullback" else float(lo[i])
    s = entry * (1 - STOP_PCT) if s is None else s
    return max(min(s, entry * 0.997), entry * (1 - STOP_PCT))


def charges_pct(entry: float, exit_: float) -> float:
    from strategy.v33_costs import round_trip
    qty = max(int(NOTIONAL // entry), 1)
    return round_trip(entry, exit_, qty, slippage_bps=0)["charges"] / (qty * entry)


def simulate(sym, x: pd.DataFrame, sig_ts, final: bool = True, mode: str | None = None) -> Trade | None:
    i = x.index.get_loc(sig_ts)
    if i + 1 >= len(x):
        return None
    eb, ets = x.iloc[i + 1], x.index[i + 1]
    if ets.time() >= SQUARE_OFF or float(eb["high"]) == float(eb["low"]):
        return None                                           # square-off reached / circuit-locked: no fill
    entry = float(eb["open"]) * (1 + SLIP)
    stop = initial_stop(x, i, entry, mode)
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
        exitp = px * (1 - SLIP)
        net = exitp / entry - 1 - charges_pct(entry, exitp)
        tr.exit_ts, tr.exit, tr.outcome = ts, exitp, out
        tr.ret_pct, tr.R = net * 100, net * entry / risk
        return tr
    if not final:
        tr.stop0 = stop
        tr.R = (float(x.iloc[-1]["close"]) - entry) / risk
        return tr
    exitp = float(x.iloc[-1]["close"]) * (1 - SLIP)
    net = exitp / entry - 1 - charges_pct(entry, exitp)
    tr.exit_ts, tr.exit, tr.outcome = x.index[-1], exitp, "SQUARE_OFF"
    tr.ret_pct, tr.R = net * 100, net * entry / risk
    return tr


# ---------------------------------------------------------------- SHORT side
# Intraday short (MIS: sell first, buy back the same day; EQ series only -
# BE / T2T stocks can't be traded intraday). Two setups:
#   "lod"   volume shocker DOWN >= WATCH_PCT, below VWAP, a new LOW of day
#           on >= VOL_SURGE x candle volume (mirror of the long HOD rule)
#   "fade"  volume shocker that spiked >= FADE_MIN_UP (day high vs previous
#           close) and now closes back BELOW VWAP for the first time (the
#           long-side research: big shocks reverse - fade the failed spike)
# Entry = the next candle's open - slippage; no fill on a circuit-locked
# candle (can't sell a stock frozen at its lower band). Stop above the
# signal candle (lod) / the last FADE_LOOKBACK candles' high (fade), capped
# at SHORT_STOP_PCT; breakeven after +1R, then trail above each closed
# candle's high; square-off 15:15.
#
# RESEARCH (2026-10-09, nse_scr/research_short.py, Apr-Oct 2026, halves A/B,
# net of exact charges + 10 bps/side): "lod" loses in both halves (-0.23 to
# -0.47%/trade, 3/day). "fade" is near break-even: with a 3% stop and spike
# >= 8%, -0.11% (A) / +0.02% (B) per trade, ~21 setups/day, day-level t
# -1.5 / 0.0 - better than the longs (-0.4%) but NOT a proven edge.
SHORT_MODE = "fade"          # 2026-10-09 research_short: lod lost in both halves (-0.23..-0.47%)
SHORT_STOP_PCT = 0.03        # least-bad on half A: fade, 3% stop, spike >= 8%
FADE_MIN_UP = 0.08
FADE_LOOKBACK = 6


def short_setups(x: pd.DataFrame, f: pd.DataFrame, mode: str | None = None) -> pd.Index:
    mode = mode or SHORT_MODE
    t = x.index.time
    base = ((t >= FIRST_ENTRY) & (t <= LAST_ENTRY) & (f["rvol"] >= RVOL_MIN) & (f["turnover"] >= MIN_TURNOVER)
            & (x["close"] >= MIN_PRICE) & ~f["locked"]).fillna(False).values
    below = (f["dist_vwap"] < 0).fillna(False).values
    if mode == "lod":
        lod_prior = x["low"].cummin().shift()
        m = base & below & ((f["pct"] <= -WATCH_PCT) & (x["close"] < lod_prior)
                            & (f["vol_ratio"] >= VOL_SURGE)).fillna(False).values
        return x.index[m]
    first_loss = below & ~pd.Series(below, index=x.index).shift(fill_value=True).values
    m = base & first_loss & (f["hod_pct"] >= FADE_MIN_UP).fillna(False).values
    return x.index[m]


def initial_stop_short(x: pd.DataFrame, i: int, entry: float, mode: str | None = None) -> float:
    mode = mode or SHORT_MODE
    h = x["high"].to_numpy(float)
    s = float(h[i]) if mode == "lod" else float(h[max(0, i - FADE_LOOKBACK + 1):i + 1].max())
    return min(max(s, entry * 1.003), entry * (1 + SHORT_STOP_PCT))


def charges_pct_short(entry: float, exit_: float) -> float:
    from strategy.v33_costs import round_trip
    qty = max(int(NOTIONAL // entry), 1)
    return round_trip(exit_, entry, qty, slippage_bps=0)["charges"] / (qty * entry)   # buy = cover, sell = entry (STT on it)


def simulate_short(sym, x: pd.DataFrame, sig_ts, final: bool = True, mode: str | None = None) -> Trade | None:
    i = x.index.get_loc(sig_ts)
    if i + 1 >= len(x):
        return None
    eb, ets = x.iloc[i + 1], x.index[i + 1]
    if ets.time() >= SQUARE_OFF or float(eb["high"]) == float(eb["low"]):
        return None                                           # square-off reached / circuit-locked: no fill
    entry = float(eb["open"]) * (1 - SLIP)
    stop = initial_stop_short(x, i, entry, mode)
    risk = stop - entry
    tr = Trade(sym, ets.date(), sig_ts, ets, entry, stop)
    best = entry
    for j in range(i + 1, len(x)):
        ts, b = x.index[j], x.iloc[j]
        if j > i + 1 and float(b["open"]) >= stop:
            px, out = float(b["open"]), "STOP_GAP"
        elif float(b["high"]) >= stop:
            px, out = stop, ("STOP" if stop > entry else "TRAIL")
        elif ts.time() >= SQUARE_OFF:
            px, out = float(b["open"]), "SQUARE_OFF"
        else:
            best = min(best, float(b["close"]))
            if entry - best >= risk:
                stop = min(stop, entry, float(b["high"]))
            continue
        exitp = px * (1 + SLIP)
        net = 1 - exitp / entry - charges_pct_short(entry, exitp)
        tr.exit_ts, tr.exit, tr.outcome = ts, exitp, out
        tr.ret_pct, tr.R = net * 100, net * entry / risk
        return tr
    if not final:
        tr.stop0 = stop
        tr.R = (entry - float(x.iloc[-1]["close"])) / risk
        return tr
    exitp = float(x.iloc[-1]["close"]) * (1 + SLIP)
    net = 1 - exitp / entry - charges_pct_short(entry, exitp)
    tr.exit_ts, tr.exit, tr.outcome = x.index[-1], exitp, "SQUARE_OFF"
    tr.ret_pct, tr.R = net * 100, net * entry / risk
    return tr
