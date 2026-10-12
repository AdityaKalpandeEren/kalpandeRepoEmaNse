"""
V7 research: intraday index momentum on NIFTY 50 / Nifty Bank, adapted from
Zarattini, Aziz & Barbon (2024) "Beat the Market: An Effective Intraday
Momentum Strategy for S&P500 ETF (SPY)".

Noise area: for each 5-minute slot t, sigma_t = mean over the previous 14
sessions of |close_t / open - 1|. Bands: upper = max(open, prev close) *
(1 + VM * sigma_t), lower = min(open, prev close) * (1 - VM * sigma_t).
Every 30 minutes (09:45 .. 14:45 IST) a flat book goes long above the upper
band and short below the lower band; an open position is closed (or reversed)
when price is back on the wrong side of its stop band; everything is closed at
the 15:15 bar's close (15:20). Index spot candles stand in for the futures.

Variants are fixed in advance (VARIANTS) and all of them are reported:
  stop "upper" = exit a long once price is back inside the noise area (paper)
  stop "lower" = hold a long until price breaks the opposite band
Costs: index-futures round trip COST_RT (STT 0.02% sell + exchange, stamp,
brokerage, GST, ~1 point slippage each side), stress-tested at 2x.

Splits: DISCOVERY 2022-2024 | VALIDATION 2025-01..06 | HOLDOUT 2025-07..
(the holdout is printed only with --holdout).

    python -m v7.index_momentum [--holdout]
"""
import argparse
import glob
import math

import numpy as np
import pandas as pd

CACHE = "backtest/cache/candles/5m"
SERIES = {"NIFTY 50": "NSE_INDEX_Nifty_50", "NIFTY BANK": "NSE_INDEX_Nifty_Bank"}
LOOKBACK = 14
DECISION_SLOTS = list(range(6, 67, 6))       # bar index 6 = 09:45 close ... 66 = 14:45 close
EXIT_SLOT = 72                               # 15:15 bar -> closes 15:20
COST_RT = 0.00035
VARIANTS = [("upper", 1.0), ("lower", 1.0)]  # (stop band, volatility multiplier)
SPLITS = [("DISCOVERY", "2022-01-01", "2024-12-31"), ("VALIDATION", "2025-01-01", "2025-06-30"),
          ("HOLDOUT", "2025-07-01", "2026-12-31")]


def load_days(series: str) -> pd.DataFrame:
    """5-minute bars -> one row per session with 75-slot arrays (09:15..15:25)."""
    frames = [pd.read_pickle(f) for f in sorted(glob.glob(f"{CACHE}/{series}/*.pkl"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    df["day"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
    df["slot"] = ((df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) - (9 * 60 + 15)) // 5
    df = df[(df["slot"] >= 0) & (df["slot"] < 75)]
    days = []
    for day, g in df.groupby("day"):
        if len(g) < 70:                          # special / truncated sessions
            continue
        c = np.full(75, np.nan)
        c[g["slot"].to_numpy()] = g["close"].to_numpy()
        c = pd.Series(c).ffill().to_numpy()
        days.append((day, float(g["open"].iloc[0]), c))
    out = pd.DataFrame(days, columns=["day", "open", "close"])
    out["prev_close"] = out["close"].map(lambda a: a[-1]).shift(1)
    return out.dropna(subset=["prev_close"]).reset_index(drop=True)


def simulate(days: pd.DataFrame, stop_band: str, vm: float, cost_rt: float) -> pd.DataFrame:
    moves = np.vstack([np.abs(c / o - 1) for o, c in zip(days["open"], days["close"])])
    rows = []
    for i in range(LOOKBACK, len(days)):
        sigma = moves[i - LOOKBACK:i].mean(axis=0)
        o, pc, c = days.at[i, "open"], days.at[i, "prev_close"], days.at[i, "close"]
        ub, lb = max(o, pc) * (1 + vm * sigma), min(o, pc) * (1 - vm * sigma)
        pos, entry, ret, trades = 0, 0.0, 0.0, 0
        for s in DECISION_SLOTS + [EXIT_SLOT]:
            px = c[s]
            want = pos
            if s == EXIT_SLOT:
                want = 0
            elif pos == 0:
                want = 1 if px > ub[s] else -1 if px < lb[s] else 0
            elif pos == 1:
                stop = ub[s] if stop_band == "upper" else lb[s]
                want = -1 if px < lb[s] else 0 if px < stop else 1
            else:
                stop = lb[s] if stop_band == "upper" else ub[s]
                want = 1 if px > ub[s] else 0 if px > stop else -1
            if want != pos:
                if pos != 0:                     # close the open position
                    ret += pos * (px / entry - 1) - cost_rt
                    trades += 1
                if want != 0:
                    entry = px
                pos = want
        rows.append((days.at[i, "day"], ret, trades, c[EXIT_SLOT] / o - 1))
    return pd.DataFrame(rows, columns=["day", "ret", "trades", "index_ret"])


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {}
    eq = (1 + r).cumprod()
    sd = r.std()
    return {"days": len(r), "ann_ret%": 100 * ((eq.iloc[-1]) ** (252 / len(r)) - 1),
            "ann_vol%": 100 * sd * math.sqrt(252), "sharpe": r.mean() / sd * math.sqrt(252) if sd > 0 else float("nan"),
            "t": r.mean() / (sd / math.sqrt(len(r))) if sd > 0 else float("nan"),
            "maxDD%": 100 * (eq / eq.cummax() - 1).min(), "hit%": 100 * (r > 0).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="also print the untouched holdout")
    a = ap.parse_args()
    splits = SPLITS if a.holdout else SPLITS[:2]
    pd.set_option("display.width", 200)
    for name, series in SERIES.items():
        days = load_days(series)
        print(f"\n=== {name}: {len(days)} sessions {days['day'].min():%Y-%m-%d} .. {days['day'].max():%Y-%m-%d}")
        for stop_band, vm in VARIANTS:
            for cost_mult in (1, 2):
                res = simulate(days, stop_band, vm, COST_RT * cost_mult)
                for split, lo, hi in splits:
                    s = res[(res["day"] >= lo) & (res["day"] <= hi)]
                    st = stats(s["ret"])
                    if not st:
                        continue
                    print(f"  stop={stop_band:5s} vm={vm} cost x{cost_mult} {split:10s} "
                          + " ".join(f"{k} {v:7.2f}" for k, v in st.items())
                          + f"  trades/day {s['trades'].mean():.2f}  | buy&hold intraday sharpe {stats(s['index_ret']).get('sharpe', float('nan')):.2f}")
                if cost_mult == 1:
                    yr = res[res["day"] < (splits[-1][2] if a.holdout else "2025-07-01")].copy()
                    yr["year"] = yr["day"].dt.year
                    print("      by year (cost x1): " + "  ".join(
                        f"{y}: {100 * g['ret'].sum():+.1f}% (sh {stats(g['ret']).get('sharpe', float('nan')):.2f})"
                        for y, g in yr.groupby("year")))


if __name__ == "__main__":
    main()
