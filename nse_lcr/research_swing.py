"""
NSE LCR intraday + swing research (max hold 5 sessions).

    python -m nse_lcr.research_swing

Same entries as nse_lcr.research (pullback continuation on large / mid-cap
volume shockers, above the 10 & 20-day EMAs, 2% stop cap). Day 0 is
simulated on 5-min candles exactly as live (stop, breakeven + trail after
+1R). Trades still open at 15:15 are then handled by a CARRY rule:

  intraday   square off at 15:15 (intraday charges)          - current live
  all        carry every open trade
  profit     carry if the price at 15:15 is above the entry, else square off
  plus1R     carry only if >= +1R at 15:15, else square off

A carried trade keeps its stop (raised to breakeven), then on each next
session (Upstox daily candles): open <= stop -> exit at the open (gap);
low <= stop -> exit at the stop; else the stop trails up to that day's low;
exit at the close of the 5th session at the latest. Carried trades pay
DELIVERY charges (swing.backtest.costs, V5's model: STT 0.1% both sides,
stamp, DP, 12 bps slippage) instead of intraday charges.
Variants are judged on the first half of the days, checked on the second.
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import pandas as pd

from backtest.candle_cache import load_daily_cached
from backtest.ml.common import access_token
from nse_lcr import research as R
from nse_lcr import strategy as S

log = logging.getLogger("nse_lcr_swing")
MAX_DAYS = 5


def delivery_net(entry_raw: float, exit_raw: float) -> float:
    """Net return of a delivery round trip (V5's exact cost model incl. slippage)."""
    from swing import config as cfgmod
    from swing.backtest import costs as C
    c = cfgmod.load()["costs"]
    buy, sell = C.fill_price(entry_raw, "buy", c), C.fill_price(exit_raw, "sell", c)
    qty = max(int(S.NOTIONAL // buy), 1)
    paid = qty * buy + C.order_charges(qty * buy, "buy", c)
    got = qty * sell - C.order_charges(qty * sell, "sell", c)
    return got / paid - 1


def day0(x, i):
    """Walk day 0 like the live job. Returns (status, price, stop, entry_raw, ts)
    status 'closed' (stop hit intraday) or 'open' (alive at 15:15, price = 15:15 open)."""
    eb, ets = x.iloc[i + 1], x.index[i + 1]
    if ets.time() >= S.SQUARE_OFF or float(eb["high"]) == float(eb["low"]):
        return None
    entry_raw = float(eb["open"])
    entry = entry_raw * (1 + S.SLIP)
    stop = S.initial_stop(x, i, entry)
    risk, best = entry - stop, entry
    for j in range(i + 1, len(x)):
        ts, b = x.index[j], x.iloc[j]
        if j > i + 1 and float(b["open"]) <= stop:
            return "closed", float(b["open"]), stop, entry_raw, risk
        if float(b["low"]) <= stop:
            return "closed", stop, stop, entry_raw, risk
        if ts.time() >= S.SQUARE_OFF:
            return "open", float(b["open"]), stop, entry_raw, risk
        best = max(best, float(b["close"]))
        if best - entry >= risk:
            stop = max(stop, entry, float(b["low"]))
    return "open", float(x.iloc[-1]["close"]), stop, entry_raw, risk


def carry(daily: pd.DataFrame, day, stop: float, entry_raw: float):
    """Swing exit over the next MAX_DAYS sessions. Returns (exit_price, outcome, days_held)."""
    nxt = daily[daily["d"] > day].head(MAX_DAYS)
    if nxt.empty:
        return None
    stop = max(stop, entry_raw)                               # breakeven once carried
    for k, (_, b) in enumerate(nxt.iterrows(), 1):
        if b["open"] <= stop:
            return float(b["open"]), "SWING_GAP", k
        if b["low"] <= stop:
            return stop, "SWING_STOP", k
        if k == MAX_DAYS:
            return float(b["close"]), "SWING_DAY5", k
        stop = max(stop, float(b["low"]))                     # trail to the day's low
    return float(nxt.iloc[-1]["close"]), "SWING_DATA_END", len(nxt)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    S.STOP_PCT = 0.02
    days = R.load_days()
    alld = sorted({d["date"] for d in days})
    half = alld[len(alld) // 2]
    tok = access_token()
    dly = {}
    rows = []
    for d in days:
        x = d["x"]
        f = S.bar_features(x, d["prev_close"], d["avg_vol20"])
        busy, n = None, 0
        for ts in S.setups(x, f, d["emas"]):
            if n >= S.MAX_PER_SYMBOL_DAY or (busy is not None and ts < busy):
                continue
            i = x.index.get_loc(ts)
            r0 = day0(x, i)
            if r0 is None:
                continue
            n += 1
            status, px, stop, entry_raw, risk = r0
            entry = entry_raw * (1 + S.SLIP)
            busy = x.index[-1] if status == "open" else ts + pd.Timedelta(minutes=10)
            intraday_net = px * (1 - S.SLIP) / entry - 1 - S.charges_pct(entry, px * (1 - S.SLIP))
            row = {"symbol": d["symbol"], "date": d["date"], "perfect": S.is_perfect(float(x.loc[ts, "close"]), d["emas"]),
                   "status0": status, "R_at_1515": (px - entry) / risk if status == "open" else np.nan,
                   "intraday": intraday_net}
            if status == "open":
                if d["symbol"] not in dly:
                    try:
                        z = load_daily_cached(d["symbol"], "2026-01-01", "2026-10-08", tok)
                        z["d"] = pd.to_datetime(z["timestamp"]).dt.tz_localize(None).dt.normalize()
                        dly[d["symbol"]] = z
                    except Exception:
                        dly[d["symbol"]] = pd.DataFrame()
                c = carry(dly[d["symbol"]], d["date"], stop, entry_raw) if not dly[d["symbol"]].empty else None
                if c is not None:
                    row.update(swing=delivery_net(entry_raw, c[0]), swing_outcome=c[1], swing_days=c[2])
            rows.append(row)
    t = pd.DataFrame(rows)
    t["half"] = np.where(t.date < half, "A", "B")
    alive = t.status0 == "open"
    rule = {"intraday": pd.Series(False, index=t.index), "all": alive & t.swing.notna(),
            "profit": alive & t.swing.notna() & (t.R_at_1515 > 0), "plus1R": alive & t.swing.notna() & (t.R_at_1515 >= 1)}
    out = []
    for name, carried in rule.items():
        t["ret"] = np.where(carried, t.swing, t.intraday) * 100
        for h in ("A", "B"):
            x = t[t.half == h]
            day = x.groupby("date").ret.mean()
            out.append({"carry rule": name, "half": h, "trades": len(x), "carried": int(carried[x.index].sum()),
                        "win%": round((x.ret > 0).mean() * 100, 1), "net%/trade": round(x.ret.mean(), 3),
                        "day_t": round(day.mean() / day.std() * math.sqrt(len(day)), 2)})
    r = pd.DataFrame(out).pivot(index="carry rule", columns="half")
    pd.set_option("display.width", 220)
    log.info("NSE LCR intraday + swing (max %d sessions), net of charges:\n%s", MAX_DAYS, r.to_string())
    sw = t[t.swing.notna()]
    log.info("carried trades by outcome:\n%s", sw.groupby("swing_outcome").agg(n=("swing", "size"), net=("swing", lambda s: round(s.mean() * 100, 2)),
                                                                                days=("swing_days", "mean")).to_string())
    log.info("swing (carry all) by PERFECT: %s", sw.groupby("perfect").swing.agg(lambda s: round(s.mean() * 100, 3)).to_dict())
    t.to_parquet(os.path.join(R.CACHE, "swing_trades.parquet"), index=False)



def swing_from_entry(x, i, daily, day, stop_cap, trail_from_day2):
    """Hold from the entry: wide structural stop (cap stop_cap), no intraday trail;
    day 0 on 5-min candles, then up to MAX_DAYS daily sessions; exit day-5 close.
    Returns (net_return, outcome, sessions_held, carried)."""
    eb = x.iloc[i + 1]
    if x.index[i + 1].time() >= S.SQUARE_OFF or float(eb["high"]) == float(eb["low"]):
        return None
    entry_raw = float(eb["open"])
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    pl = S.pullback_low(h, lo, v, c, i)
    stop = max(min(pl if pl else entry_raw * (1 - stop_cap), entry_raw * 0.995), entry_raw * (1 - stop_cap))
    for j in range(i + 1, len(x)):                                       # day 0: stop only
        b = x.iloc[j]
        if j > i + 1 and float(b["open"]) <= stop:
            e = float(b["open"])
            return e * (1 - S.SLIP) / (entry_raw * (1 + S.SLIP)) - 1 - S.charges_pct(entry_raw, e), "DAY0_GAP", 0, False
        if float(b["low"]) <= stop:
            return stop * (1 - S.SLIP) / (entry_raw * (1 + S.SLIP)) - 1 - S.charges_pct(entry_raw, stop), "DAY0_STOP", 0, False
    nxt = daily[daily["d"] > day].head(MAX_DAYS)
    if nxt.empty:
        return None
    prev_low = None
    for k, (_, b) in enumerate(nxt.iterrows(), 1):
        if trail_from_day2 and prev_low is not None:
            stop = max(stop, prev_low)
        if b["open"] <= stop:
            return delivery_net(entry_raw, float(b["open"])), "GAP", k, True
        if b["low"] <= stop:
            return delivery_net(entry_raw, stop), "STOP", k, True
        if k == MAX_DAYS:
            return delivery_net(entry_raw, float(b["close"])), "DAY5", k, True
        prev_low = float(b["low"])
    return delivery_net(entry_raw, float(nxt.iloc[-1]["close"])), "DATA_END", len(nxt), True


def main_swing():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    days = R.load_days()
    alld = sorted({d["date"] for d in days})
    half = alld[len(alld) // 2]
    tok = access_token()
    dly = {}
    variants = {"S1 hold, stop<=4%": (0.04, False), "S2 hold, stop<=6%": (0.06, False), "S3 hold, stop<=4%, trail day lows": (0.04, True)}
    rows, keep = [], []
    for d in days:
        if d["symbol"] not in dly:
            try:
                z = load_daily_cached(d["symbol"], "2026-01-01", "2026-10-08", tok)
                z["d"] = pd.to_datetime(z["timestamp"]).dt.tz_localize(None).dt.normalize()
                dly[d["symbol"]] = z
            except Exception:
                dly[d["symbol"]] = pd.DataFrame()
        if dly[d["symbol"]].empty:
            continue
        x = d["x"]
        f = S.bar_features(x, d["prev_close"], d["avg_vol20"])
        sigs = S.setups(x, f, d["emas"])
        if not len(sigs):
            continue
        ts = sigs[0]                                                    # one swing position per stock per day
        i = x.index.get_loc(ts)
        for name, (cap, trail) in variants.items():
            r = swing_from_entry(x, i, dly[d["symbol"]], d["date"], cap, trail)
            if r is not None:
                keep.append({"variant": name, "symbol": d["symbol"], "date": d["date"], "net": r[0] * 100, "outcome": r[1],
                             "days": r[2], "carried": r[3], "perfect": S.is_perfect(float(x.loc[ts, "close"]), d["emas"])})
    t = pd.DataFrame(keep)
    t["half"] = np.where(t.date < half, "A", "B")
    for name, g in t.groupby("variant"):
        for h in ("A", "B"):
            x = g[g.half == h]
            day = x.groupby("date").net.mean()
            rows.append({"variant": name, "half": h, "trades": len(x), "carried%": round(x.carried.mean() * 100),
                         "win%": round((x.net > 0).mean() * 100, 1), "net%/trade": round(x.net.mean(), 3),
                         "avg_days": round(x.days.mean(), 1), "day_t": round(day.mean() / day.std() * math.sqrt(len(day)), 2)})
    pd.set_option("display.width", 220)
    log.info("NSE LCR SWING from entry (max %d sessions), net of charges:\n%s", MAX_DAYS, pd.DataFrame(rows).pivot(index="variant", columns="half").to_string())
    for name, g in t.groupby("variant"):
        log.info("%s outcomes: %s | PERFECT %s", name, g.groupby("outcome").net.agg(["size", "mean"]).round(2).to_dict("index"),
                 g.groupby("perfect").net.mean().round(3).to_dict())
    t.to_parquet(os.path.join(R.CACHE, "swing_from_entry.parquet"), index=False)


if __name__ == "__main__":
    import sys
    main_swing() if "--hold" in sys.argv else main()
