"""
Option B: SWING_TREND_BREAKOUT filtered by the V3.2 morning ranking (NSE, testing only).

    python -m backtest.run_swing_v32_filter
    python -m backtest.run_swing_v32_filter --pct 0.7          # top 30% (default, fixed before testing)

How a trade happens:
  day D (close)   SWING_TREND_BREAKOUT signal (strategy/swing_nse.py)
  day D+1 9:45    V3.2 scores the whole universe from the opening 30 min,
                  overnight macro, sector, catalysts. Take the trade ONLY
                  if the breakout stock ranks in the top (1 - pct) of the
                  universe that morning; buy at the 9:45 candle's open.
  holding         SHORT SWING (max --max-hold trading days, default 20):
                  stop = entry - 2 x ATR(14); target = entry + --target-r x
                  the stop distance (default 2R); exit at the next open
                  after a close below the prior --trail-days lows (default
                  5); time exit at the close of day --max-hold. Stop/target
                  are checked on the entry day from 9:45 on, then on each
                  day's range (stop first if both; gaps fill at the open).
                  Delivery costs config.SWING_COST_PCT_PER_SIDE each side.

WALK-FORWARD V3.2 - no lookahead: the model used on any day of year Y is
trained only on dataset rows dated before Jan 1 of year Y (2023 <- 2022,
2024 <- 2022-23, ...). The saved model_v32.joblib, trained on everything,
is NOT used here.

Compared on the same period:
  UNFILTERED  every breakout, same 9:45 entry and exits (isolates the filter)
  FILTERED    the V3.2 top-ranked breakouts only
  NIFTY buy-and-hold
per trade, per year, and as a 10-position portfolio.
"""
import argparse
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from backtest.candle_cache import load_daily_cached, load_symbol_history_cached
from backtest.ml.common import access_token, load_symbol_list
from backtest.run_swing_nse import _curve_stats, portfolio
from strategy import sector_map
from strategy.swing_nse import enrich_daily, swing_trend_breakout


def walk_forward_scores(ds: pd.DataFrame, feats: list, first_year: int) -> pd.Series:
    from backtest.ml.train_v32 import models
    ds = ds.copy()
    ds["year"] = pd.to_datetime(ds["date"]).dt.year
    lo, hi = ds["excess"].quantile([0.005, 0.995])
    ds["target"] = ds["excess"].clip(lo, hi)
    out = pd.Series(np.nan, index=ds.index)
    for y in sorted(ds["year"].unique()):
        if y < first_year:
            continue
        train = ds[ds["year"] < y]
        test = ds[ds["year"] == y]
        m = models()["hgb_small"]().fit(train[feats], train["target"])
        out.loc[test.index] = m.predict(test[feats])
        print(f"  V3.2 walk-forward: trained on {train['date'].min()}..{train['date'].max()} "
              f"({len(train)} rows) -> scored {y} ({len(test)} rows)", flush=True)
    return out


def simulate(symbol, df, entry_info: dict, regime_ok, take, max_hold, trail_days, target_r) -> list:
    """Trend-breakout trades entered at 9:45 on the day after the signal.
    entry_info[date] = (entry_price_945, low_after_945, high_after_945, pct_rank)."""
    cost = config.SWING_COST_PCT_PER_SIDE
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    trail = df["low"].rolling(trail_days, min_periods=trail_days).min().shift(1).to_numpy(float)
    dates = df["date"].to_numpy()
    n, trades, i = len(df), [], config.SWING_WARMUP_DAYS
    while i < n - 1:
        plan = swing_trend_breakout(df, i)
        if plan is None:
            i += 1
            continue
        fi = i + 1
        info = entry_info.get(dates[fi])
        if info is None or not take(info[3]):
            i += 1
            continue
        fill, low_after, high_after, pr = info
        stop = fill - plan.stop_value * float(df["atr"].iat[i])
        if not (0 < stop < fill):
            i += 1
            continue
        target = fill + target_r * (fill - stop)
        exit_idx, exit_px, outcome = None, None, None
        # entry day, 9:45 onward: stop first if both levels were touched
        if low_after <= stop:
            exit_idx, exit_px, outcome = fi, stop, "STOP"
        elif high_after >= target:
            exit_idx, exit_px, outcome = fi, target, "TARGET"
        j = fi
        while exit_idx is None and j < n:
            if j > fi:
                if l[j] <= stop:
                    exit_idx, exit_px, outcome = j, (o[j] if o[j] < stop else stop), "STOP"
                    break
                if h[j] >= target:
                    exit_idx, exit_px, outcome = j, (o[j] if o[j] > target else target), "TARGET"
                    break
            if j - fi >= max_hold - 1:
                exit_idx, exit_px, outcome = j, c[j], "TIME"
                break
            if trail[j] == trail[j] and c[j] < trail[j] and j + 1 < n:
                exit_idx, exit_px, outcome = j + 1, o[j + 1], "TRAIL_EXIT"
                break
            j += 1
        if exit_idx is None:
            exit_idx, exit_px, outcome = n - 1, c[n - 1], "OPEN_AT_END"
        buy, sell = fill * (1 + cost), exit_px * (1 - cost)
        trades.append({"symbol": symbol, "signal_date": dates[i], "entry_date": dates[fi],
                       "exit_date": dates[exit_idx], "entry": fill, "stop": stop, "exit": exit_px,
                       "outcome": outcome, "ret": sell / buy - 1, "r": (sell - buy) / (fill - stop),
                       "hold_days": int(exit_idx - fi), "v32_pct": pr,
                       "rs_63": float(df["rs_63"].iat[i]) if df["rs_63"].iat[i] == df["rs_63"].iat[i] else 0.0,
                       "regime_ok": bool(regime_ok.get(dates[i], False))})
        i = max(exit_idx, i + 1)
    return trades


def trade_stats(t: pd.DataFrame) -> str:
    if t.empty:
        return "no trades"
    m = t.groupby(pd.to_datetime(t.entry_date).dt.to_period("M"))["ret"].mean()
    tt = t.ret.mean() / (t.ret.std(ddof=1) / math.sqrt(len(t))) if len(t) > 2 else 0
    tm = m.mean() / (m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 2 else 0
    g, lz = t.ret[t.ret > 0].sum(), -t.ret[t.ret < 0].sum()
    return (f"n={len(t):4d}  win={100*(t.ret>0).mean():5.1f}%  avg={100*t.ret.mean():+6.2f}%/trade  "
            f"PF={g/lz if lz else float('inf'):5.2f}  hold avg={t.hold_days.mean()+1:4.1f}d "
            f"median={t.hold_days.median()+1:3.0f}d max={t.hold_days.max()+1:3.0f}d  "
            f"t_trade={tt:5.2f}  t_month={tm:5.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pct", type=float, default=0.70, help="take if V3.2 percentile >= this (default 0.70)")
    ap.add_argument("--first-year", type=int, default=2023)
    ap.add_argument("--max-hold", type=int, default=20, help="max trading days held (default 20)")
    ap.add_argument("--trail-days", type=int, default=5, help="exit on a close below the prior N-day low (5)")
    ap.add_argument("--target-r", type=float, default=2.0, help="profit target in R (default 2)")
    args = ap.parse_args()
    tok = access_token()

    ds = pd.read_csv(config.ML_V32_DATASET_PATH)
    ds = ds[ds["n_day"] >= 20].dropna(subset=["net_ret", "excess"]).reset_index(drop=True)
    from backtest.ml.train_v32 import BOOKKEEPING
    feats = [c for c in ds.columns if c not in BOOKKEEPING]
    print("Walk-forward V3.2 scores:")
    ds["score"] = walk_forward_scores(ds, feats, args.first_year)
    ds = ds.dropna(subset=["score"])
    ds["pct"] = ds.groupby("date")["score"].rank(pct=True)
    start = ds["date"].min()
    print(f"Scored days: {ds['date'].nunique()} ({start} .. {ds['date'].max()})\n")

    # per (symbol, date): 9:45 entry, low after 9:45, V3.2 percentile
    info = {}
    for sym, g in ds.groupby("symbol"):
        intr = load_symbol_history_cached(sym, 5, start, ds["date"].max(), tok)
        intr["date"] = pd.to_datetime(intr["timestamp"]).dt.date
        intr = intr[intr["timestamp"].dt.hour * 60 + intr["timestamp"].dt.minute >= 9 * 60 + 45]
        low_after = intr.groupby("date")["low"].min()
        high_after = intr.groupby("date")["high"].max()
        info[sym] = {}
        for d, e, p in zip(g["date"], g["entry"], g["pct"]):
            dd = pd.Timestamp(d).date()
            info[sym][dd] = (e, float(low_after.get(dd, e)), float(high_after.get(dd, e)), p)

    nifty = load_daily_cached("NSE_INDEX|Nifty 50", config.SWING_FROM, ds["date"].max(), tok, is_key=True)
    nifty["date"] = pd.to_datetime(nifty["timestamp"]).dt.date
    nclose = nifty.drop_duplicates("date", keep="last").set_index("date")["close"]
    regime_ok = (nclose > nclose.rolling(200, min_periods=200).mean()).to_dict()
    sector_close = {}
    for code, key in sector_map.SECTORS.items():
        if code != "OTHER":
            s = load_daily_cached(key, config.SWING_FROM, ds["date"].max(), tok, is_key=True)
            sc = s.set_index(pd.to_datetime(s["timestamp"]).dt.date)["close"]
            sector_close[code] = sc[~sc.index.duplicated(keep="last")]

    unf, fil, closes = [], [], {}
    for sym in sorted(info):
        raw = load_daily_cached(sym, config.SWING_FROM, ds["date"].max(), tok)
        if len(raw) < config.SWING_WARMUP_DAYS + 20:
            continue
        df = enrich_daily(raw, nclose, sector_close.get(sector_map.lookup(sym)["sector"]))
        closes[sym] = dict(zip(df["date"], df["close"]))
        kw = dict(max_hold=args.max_hold, trail_days=args.trail_days, target_r=args.target_r)
        unf += simulate(sym, df, info[sym], regime_ok, lambda p: True, **kw)
        fil += simulate(sym, df, info[sym], regime_ok, lambda p: p >= args.pct, **kw)
    tu, tf = pd.DataFrame(unf), pd.DataFrame(fil)
    first_day = pd.Timestamp(start).date()
    tu = tu[tu.entry_date >= first_day]
    tf = tf[tf.entry_date >= first_day]

    print("=" * 110)
    print(f"PER TRADE ({start} .. {ds['date'].max()}), 9:45 entry, delivery costs | exits: 2 ATR stop, "
          f"{args.target_r}R target, close < prior {args.trail_days}-day low, max {args.max_hold} days")
    print("=" * 110)
    print(f"  UNFILTERED breakouts          {trade_stats(tu)}")
    print(f"  V3.2 top {100*(1-args.pct):.0f}% filter          {trade_stats(tf)}")
    print(f"  rejected by the filter        {trade_stats(tu[tu.v32_pct < args.pct])}")
    print("\n  Exit mix (filtered): " + str(tf.outcome.value_counts().to_dict()))
    print("\n  Average return per trade by year (%) [trades]:")
    for label, t in (("unfiltered", tu), ("filtered", tf)):
        y = t.assign(y=pd.to_datetime(t.entry_date).dt.year).groupby("y")["ret"].agg(["mean", "size"])
        print(f"    {label:<12}" + "  ".join(f"{yy}: {r['mean']*100:+.2f} [{int(r['size'])}]" for yy, r in y.iterrows()))

    cal = [d for d in nclose.index if d >= first_day]
    bench = _curve_stats(pd.Series([nclose[d] for d in cal], index=pd.to_datetime(cal)))
    print("\n" + "=" * 110)
    print(f"PORTFOLIO (max {config.SWING_MAX_POSITIONS} positions, equal weight) vs NIFTY, same period")
    print("=" * 110)
    print(f"  {'NIFTY buy & hold':<30} CAGR {bench['CAGR%']:>6}%  maxDD {bench['maxDD%']:>7}%  x{bench['final_x']}  {bench['yearly']}")
    for label, t in (("UNFILTERED breakouts", tu), (f"V3.2-filtered (top {100*(1-args.pct):.0f}%)", tf)):
        if t.empty:
            continue
        p = portfolio(t.assign(strategy="X"), closes, cal)
        print(f"  {label:<30} CAGR {p['CAGR%']:>6}%  maxDD {p['maxDD%']:>7}%  x{p['final_x']}  "
              f"exposure {p['avg_exposure%']}%  {p['yearly']}")

    os.makedirs("backtest/results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tf.to_csv(f"backtest/results/swing_v32_filtered_{stamp}.csv", index=False)
    tu.to_csv(f"backtest/results/swing_v32_unfiltered_{stamp}.csv", index=False)
    print(f"\nSaved trades: backtest/results/swing_v32_*_{stamp}.csv")


if __name__ == "__main__":
    main()
