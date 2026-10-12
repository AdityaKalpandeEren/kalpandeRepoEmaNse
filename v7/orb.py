"""
V7 research: Opening Range Breakout on NSE "stocks in play" - Zarattini,
Barbon & Aziz (2024), "A Profitable Day Trading Strategy For The U.S. Equity
Market", adapted to NSE intraday (MIS), long AND short.

Every session, universe = that month's point-in-time top-200 by traded value
(same rule as swing/data/panel.add_universe):
  09:20  first 5-minute bar closed. RVOL = its volume / the mean first-bar
         volume of the previous 14 sessions. Direction = colour of that bar
         (doji: no trade). Eligible: price >= Rs 50, RVOL >= 1. Rank by RVOL,
         keep the top N.
  entry  stop order at the opening-range high (long) / low (short), live
         until the 14:45 bar; filled at the level, or at a bar's open when
         it gaps through, plus slippage
  stop   k x ATR(14) from the fill;  exit at the stop or the 15:15 square-off
5-minute bars can't tell whether the stop was hit after the entry inside the
entry bar, so both bounds are reported: OPTIMISTIC (it wasn't) and
PESSIMISTIC (a low/high through the stop in the entry bar = stopped).
Costs: exact MIS charges (strategy.v33_costs) at the trade's notional
(Rs 5,000 risk per trade, notional capped at Rs 10 lakh) + slippage per side.

Pre-registered grid, all reported: N in {10, 20} x k in {0.10, 0.25, 0.50}.
Splits: DISCOVERY 2022-2024 | VALIDATION 2025-01..06 | HOLDOUT 2025-07..
(holdout only with --holdout).

    python -m v7.orb [--holdout] [--slip-bps 2]
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

from strategy import v33_costs

CACHE = "backtest/cache/candles/5m"
PANEL = "swing/cache/v6_panel_20140101_20261002_n200.parquet"
LOOKBACK = 14
ENTRY_LAST = 66            # 14:45 bar
EXIT_SLOT = 71             # 15:10 bar -> its close = 15:15 square-off
KS = (0.10, 0.25, 0.50)
NS = (10, 20)
RISK_RS, MAX_NOTIONAL = 5_000.0, 1_000_000.0
MIN_PRICE = 50.0
SPLITS = [("DISCOVERY", "2022-01-01", "2024-12-31"), ("VALIDATION", "2025-01-01", "2025-06-30"),
          ("HOLDOUT", "2025-07-01", "2026-12-31")]
OUT = "v7/results"


def membership() -> pd.DataFrame:
    """(month, symbol) rows of the point-in-time top-200 (latest symbol per entity)."""
    d = pd.read_parquet(PANEL, columns=["date", "entity", "symbol", "in_universe"])
    d = d[d["date"] >= "2021-12-01"]
    d["month"] = d["date"].dt.to_period("M")
    first = d.groupby("month")["date"].transform("min")
    u = d[(d["date"] == first) & d["in_universe"]][["month", "entity"]]
    latest = d.sort_values("date").groupby("entity")["symbol"].last()
    u["symbol"] = u["entity"].map(latest)
    return u


def load_symbol(sym: str):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sym)
    files = sorted(f for f in glob.glob(f"{CACHE}/{safe}/*.pkl") if os.path.basename(f)[:7] >= "2021-11")
    if not files:
        return None
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True).drop_duplicates("timestamp")
    if df.empty:
        return None
    ts = df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz is not None else df["timestamp"]
    df["day"] = ts.dt.normalize()
    df["slot"] = ((ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 15)) // 5
    df = df[(df["slot"] >= 0) & (df["slot"] < 75)]
    days = np.array(sorted(df["day"].unique()))
    di = {d: i for i, d in enumerate(days)}
    mats = {c: np.full((len(days), 75), np.nan) for c in ("open", "high", "low", "close", "volume")}
    r, s = df["day"].map(di).to_numpy(), df["slot"].to_numpy()
    for c in mats:
        mats[c][r, s] = df[c].to_numpy(dtype=float)
    return pd.DatetimeIndex(days), mats


def symbol_trades(sym: str, months: set, slip: float) -> pd.DataFrame:
    got = load_symbol(sym)
    if got is None:
        return pd.DataFrame()
    days, m = got
    O, H, L, C, V = m["open"], m["high"], m["low"], m["close"], m["volume"]
    nbars = np.sum(~np.isnan(C), axis=1)
    dh, dl = np.nanmax(H, axis=1), np.nanmin(L, axis=1)
    last = np.array([row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan for row in C])
    pc = np.r_[np.nan, last[:-1]]
    tr = np.nanmax(np.vstack([dh - dl, np.abs(dh - pc), np.abs(dl - pc)]), axis=0)
    v0 = V[:, 0]
    s_tr, s_v0 = pd.Series(tr), pd.Series(v0)
    atr = s_tr.rolling(LOOKBACK).mean().shift(1).to_numpy()
    v0avg = s_v0.rolling(LOOKBACK).mean().shift(1).to_numpy()
    span = (pd.Series(days) - pd.Series(days).shift(LOOKBACK)).dt.days.to_numpy()   # contiguous history only
    rows = []
    for i, day in enumerate(days):
        if pd.Period(day, "M") not in months or nbars[i] < 70 or np.isnan(O[i, 0]) or not (span[i] <= 25):
            continue
        if not (atr[i] > 0) or not (v0avg[i] > 0):
            continue
        o0, c0, h0, l0 = O[i, 0], C[i, 0], H[i, 0], L[i, 0]
        if o0 < MIN_PRICE or c0 == o0:
            continue
        dirn = 1 if c0 > o0 else -1
        rvol = v0[i] / v0avg[i]
        level = h0 if dirn == 1 else l0
        hit = H[i, 1:ENTRY_LAST + 1] >= level if dirn == 1 else L[i, 1:ENTRY_LAST + 1] <= level
        hit = np.nan_to_num(hit, nan=0).astype(bool)
        base = {"day": day, "symbol": sym, "dir": dirn, "rvol": rvol, "atr_pct": atr[i] / o0, "price": o0}
        if not hit.any():
            rows.append({**base, "triggered": False})
            continue
        j = 1 + int(np.argmax(hit))
        raw_fill = max(level, O[i, j]) if dirn == 1 else min(level, O[i, j])
        fill = raw_fill * (1 + dirn * slip)
        rec = {**base, "triggered": True, "entry_slot": j}
        for k in KS:
            dist = k * atr[i]
            stop = fill - dirn * dist
            for bound in ("opt", "pes"):
                ex, why = np.nan, "EOD"
                start = j if bound == "pes" else j + 1
                for b in range(start, EXIT_SLOT + 1):
                    lo, hi, op = L[i, b], H[i, b], O[i, b]
                    if np.isnan(lo):
                        continue
                    if (dirn == 1 and lo <= stop) or (dirn == -1 and hi >= stop):
                        gap = b > j and ((dirn == 1 and op < stop) or (dirn == -1 and op > stop))
                        ex, why = (op if gap else stop), "STOP"
                        break
                if why == "EOD":
                    ex = C[i, EXIT_SLOT] if not np.isnan(C[i, EXIT_SLOT]) else last[i]
                ex = ex * (1 - dirn * slip)
                gross = dirn * (ex / fill - 1)
                notional = min(RISK_RS / (dist / fill), MAX_NOTIONAL)
                qty = max(int(notional // fill), 1)
                buy, sell = (fill, ex) if dirn == 1 else (ex, fill)
                ch = v33_costs.round_trip(buy, sell, qty, slippage_bps=0.0)["charges"] / (qty * fill)
                net = gross - ch
                rec[f"R_{k}_{bound}"] = net / (dist / fill)
                rec[f"Rg_{k}_{bound}"] = gross / (dist / fill)
                rec[f"why_{k}_{bound}"] = why
        rows.append(rec)
    return pd.DataFrame(rows)


def build(slip_bps: float) -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    f = f"{OUT}/orb_candidates_slip{slip_bps:g}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    mem = membership()
    by_sym = mem.groupby("symbol")["month"].apply(set)
    parts = []
    for n, (sym, months) in enumerate(by_sym.items(), 1):
        t = symbol_trades(sym, months, slip_bps / 1e4)
        if not t.empty:
            parts.append(t)
        if n % 50 == 0:
            print(f"[orb] {n}/{len(by_sym)} symbols", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(f, index=False)
    return out


def day_stats(tr: pd.DataFrame, col: str) -> dict:
    if tr.empty:
        return {}
    dr = tr.groupby("day")[col].sum()
    sd = dr.std()
    return {"trades": len(tr), "days": len(dr), "avgR": tr[col].mean(), "win%": 100 * (tr[col] > 0).mean(),
            "sumR": tr[col].sum(), "dayR": dr.mean(), "day_t": dr.mean() / (sd / math.sqrt(len(dr))) if sd > 0 else np.nan,
            "up_days%": 100 * (dr > 0).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--slip-bps", type=float, default=2.0)
    a = ap.parse_args()
    cand = build(a.slip_bps)
    cand = cand[cand["rvol"] >= 1]
    cand["rank"] = cand.groupby("day")["rvol"].rank(ascending=False, method="first")
    print(f"candidates: {len(cand)} symbol-days, {cand['symbol'].nunique()} symbols, "
          f"{cand['day'].min():%Y-%m-%d} .. {cand['day'].max():%Y-%m-%d}; trigger rate {cand['triggered'].mean():.0%}")
    splits = SPLITS if a.holdout else SPLITS[:2]
    for n in NS:
        top = cand[(cand["rank"] <= n) & cand["triggered"]]
        for k in KS:
            for bound in ("opt", "pes"):
                col = f"R_{k}_{bound}"
                line = [f"N{n:<2d} k={k:.2f} {bound}"]
                for split, lo, hi in splits:
                    s = top[(top["day"] >= lo) & (top["day"] <= hi)]
                    st = day_stats(s, col)
                    g = s[f"Rg_{k}_{bound}"].mean() if len(s) else np.nan
                    line.append(f"{split[:4]}: n {st.get('trades', 0):5d} avgR {st.get('avgR', np.nan):+.3f} "
                                f"(gross {g:+.3f}) win {st.get('win%', np.nan):4.1f}% day-t {st.get('day_t', np.nan):+5.2f}")
                print("  ".join(line))
    # per-year detail for the paper's own setting (N20, k 0.10) and the widest stop
    for n, k in ((20, 0.10), (20, 0.50)):
        top = cand[(cand["rank"] <= n) & cand["triggered"] & (cand["day"] < (SPLITS[2][1] if not a.holdout else "2100"))]
        for bound in ("opt", "pes"):
            col = f"R_{k}_{bound}"
            yr = top.groupby(top["day"].dt.year)[col].agg(["count", "mean", "sum"])
            print(f"  by year N{n} k={k} {bound}: " + "  ".join(f"{y}: {r['mean']:+.3f}R x{int(r['count'])}" for y, r in yr.iterrows()))


if __name__ == "__main__":
    main()
