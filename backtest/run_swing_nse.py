"""
NSE swing backtest - daily candles, multi-day holds (strategy/swing_nse.py).

    python -m backtest.run_swing_nse                       # all 4 strategies, 2015 -> today
    python -m backtest.run_swing_nse --strategies SWING_TREND_BREAKOUT --from 2018-01-01
    python -m backtest.run_swing_nse --symbols RELIANCE,TCS,INFY

Trade mechanics (all strategies):
  - signal on day D's close; entry at day D+1's OPEN
  - stop hit on a later day: exit at the stop, or at that day's OPEN if the
    stock gapped below it (the realistic, worse fill); same-day stop+target
    -> stop first (conservative)
  - close-based exits (trailing / mean-reversion): decided on day D's
    close, executed at day D+1's open
  - NSE delivery costs config.SWING_COST_PCT_PER_SIDE on entry AND exit
  - one open trade per (strategy, symbol) at a time

Reported:
  1. per strategy: trades, win %, avg R, avg net return, profit factor, hold
     days, t-stat per trade AND per month (trades in one month are not
     independent), and every YEAR separately
  2. the same with a market filter: new entries only while NIFTY closes
     above its 200-day average
  3. a PORTFOLIO simulation per strategy (max config.SWING_MAX_POSITIONS
     positions, equal weight, strongest relative strength first on busy
     days) - CAGR, max drawdown, yearly returns - against simply holding
     NIFTY over the same period. A swing system is only worth running if
     it beats buy-and-hold on return AND/OR drawdown.

SURVIVORSHIP BIAS: the symbol list is today's watchlist + universe + NIFTY
50. Companies that fell out of favour or the index since 2015 are missing,
which makes every long strategy (and "buy the list") look better than it
would have been live. Compare strategies with each other and with NIFTY,
and treat absolute returns as optimistic.
"""
import argparse
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from backtest.candle_cache import load_daily_cached
from backtest.ml.common import access_token, load_symbol_list
from strategy import sector_map
from strategy.swing_nse import SWING_STRATEGIES, enrich_daily, stop_and_target


def simulate_symbol(symbol, df, name, entry_fn, exit_fn, regime_ok) -> list:
    cost = config.SWING_COST_PCT_PER_SIDE
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    dates = df["date"].to_numpy()
    n = len(df)
    trades = []
    i = config.SWING_WARMUP_DAYS
    while i < n - 1:
        plan = entry_fn(df, i)
        if plan is None:
            i += 1
            continue
        fill_idx = i + 1
        fill = o[fill_idx]
        stop, target = stop_and_target(plan, fill, df.iloc[i])
        if not (0 < stop < fill):
            i += 1
            continue
        exit_idx, exit_px, outcome = None, None, None
        j = fill_idx
        while j < n:
            if l[j] <= stop:
                exit_px = o[j] if (j > fill_idx and o[j] < stop) else stop
                exit_idx, outcome = j, "STOP"
                break
            if target is not None and h[j] >= target:
                exit_px = o[j] if (j > fill_idx and o[j] > target) else target
                exit_idx, outcome = j, "TARGET"
                break
            if j - fill_idx >= plan.max_hold:
                exit_idx, exit_px, outcome = j, c[j], "TIME"
                break
            if exit_fn(df, j) and j + 1 < n:
                exit_idx, exit_px, outcome = j + 1, o[j + 1], "SIGNAL_EXIT"
                break
            j += 1
        if exit_idx is None:
            exit_idx, exit_px, outcome = n - 1, c[n - 1], "OPEN_AT_END"
        buy, sell = fill * (1 + cost), exit_px * (1 - cost)
        trades.append({
            "strategy": name, "symbol": symbol, "signal_date": dates[i], "entry_date": dates[fill_idx],
            "exit_date": dates[exit_idx], "entry": round(fill, 2), "stop": round(stop, 2),
            "target": round(target, 2) if target else None, "exit": round(exit_px, 2), "outcome": outcome,
            "ret": sell / buy - 1, "r": (sell - buy) / (fill - stop), "hold_days": int(exit_idx - fill_idx),
            "regime_ok": bool(regime_ok.get(dates[i], False)), "rs_63": float(df["rs_63"].iat[i])
            if df["rs_63"].iat[i] == df["rs_63"].iat[i] else 0.0, "reason": plan.reason,
        })
        i = max(exit_idx, i + 1)
    return trades


# ═══════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════

def _t(x) -> float:
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    gains, losses = t.loc[t.ret > 0, "ret"].sum(), -t.loc[t.ret < 0, "ret"].sum()
    monthly = t.groupby(pd.to_datetime(t.entry_date).dt.to_period("M"))["ret"].mean()
    return {
        "n": len(t), "win%": round((t.ret > 0).mean() * 100, 1), "avg_R": round(t.r.mean(), 3),
        "avg_ret%": round(t.ret.mean() * 100, 2), "PF": round(gains / losses, 2) if losses else float("inf"),
        "hold_d": round(t.hold_days.mean(), 1), "t_trade": round(_t(t.ret), 2),
        "t_month": round(_t(monthly), 2), "months+": f"{(monthly > 0).sum()}/{len(monthly)}",
    }


def portfolio(trades: pd.DataFrame, closes: dict, calendar: list) -> dict:
    """Equal-weight, max N positions, strongest RS first on busy days;
    daily mark-to-market for the drawdown."""
    N = config.SWING_MAX_POSITIONS
    cash = equity = config.SWING_START_EQUITY
    t = trades.sort_values(["entry_date", "rs_63"], ascending=[True, False])
    by_entry = {d: g for d, g in t.groupby("entry_date")}
    open_pos = []            # dicts: symbol, alloc, fill, exit_date, ret
    curve, taken = [], 0
    for d in calendar:
        # exits first (money is free again the same day)
        still = []
        for p in open_pos:
            if p["exit_date"] <= d:
                cash += p["alloc"] * (1 + p["ret"])
            else:
                still.append(p)
        open_pos = still
        equity = cash + sum(p["alloc"] * closes[p["symbol"]].get(d, p["last"]) / p["fill"] for p in open_pos)
        for p in open_pos:
            p["last"] = closes[p["symbol"]].get(d, p["last"])
        if d in by_entry:
            held = {p["symbol"] for p in open_pos}
            for _, row in by_entry[d].iterrows():
                if len(open_pos) >= N:
                    break
                if row.symbol in held:
                    continue
                alloc = min(cash, equity / N)
                if alloc <= 0:
                    break
                cash -= alloc
                open_pos.append({"symbol": row.symbol, "alloc": alloc, "fill": row.entry,
                                 "exit_date": row.exit_date, "ret": row.ret, "last": row.entry})
                held.add(row.symbol)
                taken += 1
        curve.append((d, equity, len(open_pos)))
    eq = pd.Series([e for _, e, _ in curve], index=pd.to_datetime([d for d, _, _ in curve]))
    pos = pd.Series([k for _, _, k in curve], index=eq.index)
    return _curve_stats(eq) | {"trades_taken": taken, "avg_exposure%": round(pos.mean() / N * 100, 1)}


def _curve_stats(eq: pd.Series) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    yearly = eq.groupby(eq.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    return {"CAGR%": round(cagr * 100, 1), "maxDD%": round(dd * 100, 1),
            "final_x": round(eq.iloc[-1] / eq.iloc[0], 2),
            "yearly": {int(y): round(v * 100, 1) for y, v in yearly.items()}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols")
    p.add_argument("--from", dest="from_date", default=config.SWING_FROM)
    p.add_argument("--to", dest="to_date", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--strategies", default=",".join(SWING_STRATEGIES))
    p.add_argument("--out", default="backtest/results")
    return p.parse_args()


def main():
    args = parse_args()
    tok = access_token()
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bad = [w for w in wanted if w not in SWING_STRATEGIES]
    if bad:
        raise SystemExit(f"Unknown strategies {bad}. Valid: {', '.join(SWING_STRATEGIES)}")
    symbols = ([s.strip().upper() for s in args.symbols.split(",")] if args.symbols else
               list(dict.fromkeys(load_symbol_list("watchlist.txt") + load_symbol_list("universe.txt")
                                  + list(config.ML_V2_BREADTH_UNIVERSE))))

    print(f"NSE swing backtest {args.from_date}..{args.to_date} | {len(symbols)} symbols | "
          f"{wanted}\nCosts {config.SWING_COST_PCT_PER_SIDE*100:.2f}% per side | portfolio "
          f"max {config.SWING_MAX_POSITIONS} positions\n")
    nifty = load_daily_cached("NSE_INDEX|Nifty 50", args.from_date, args.to_date, tok, is_key=True)
    nifty["date"] = pd.to_datetime(nifty["timestamp"]).dt.date
    nclose = nifty.drop_duplicates("date", keep="last").set_index("date")["close"]
    regime_ok = (nclose > nclose.rolling(200, min_periods=200).mean()).to_dict()
    calendar = list(nclose.index)

    sector_close = {}
    for code, key in sector_map.SECTORS.items():
        if code == "OTHER":
            continue
        try:
            s = load_daily_cached(key, args.from_date, args.to_date, tok, is_key=True)
            sc = s.set_index(pd.to_datetime(s["timestamp"]).dt.date)["close"]
            sector_close[code] = sc[~sc.index.duplicated(keep="last")]
        except Exception as e:
            print(f"  sector {code} unavailable: {e}")

    all_trades, closes = [], {}
    for k, sym in enumerate(symbols, 1):
        try:
            raw = load_daily_cached(sym, args.from_date, args.to_date, tok)
        except Exception as e:
            print(f"[{k}/{len(symbols)}] {sym}: skipped ({str(e)[:60]})")
            continue
        if len(raw) < config.SWING_WARMUP_DAYS + 20:
            print(f"[{k}/{len(symbols)}] {sym}: too little history ({len(raw)} days)")
            continue
        sec = sector_map.lookup(sym)["sector"]
        df = enrich_daily(raw, nclose, sector_close.get(sec))
        closes[sym] = dict(zip(df["date"], df["close"]))
        n_before = len(all_trades)
        for name in wanted:
            entry_fn, exit_fn = SWING_STRATEGIES[name]
            all_trades += simulate_symbol(sym, df, name, entry_fn, exit_fn, regime_ok)
        print(f"[{k}/{len(symbols)}] {sym}: {len(df)} days, {len(all_trades) - n_before} trades", flush=True)

    t = pd.DataFrame(all_trades)
    if t.empty:
        print("No trades.")
        return
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out, f"swing_trades_{stamp}.csv")
    t.to_csv(csv_path, index=False)

    bench = _curve_stats(pd.Series(nclose.values, index=pd.to_datetime(list(nclose.index))))
    lines = [f"# NSE swing backtest {args.from_date} .. {args.to_date}", "",
             f"{len(symbols)} symbols | costs {config.SWING_COST_PCT_PER_SIDE*100:.2f}%/side | "
             f"portfolio max {config.SWING_MAX_POSITIONS} positions, equal weight", "",
             "**Survivorship bias:** today's stock list - absolute returns are optimistic.", ""]

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("PER-TRADE RESULTS (net of costs)")
    out("=" * 100)
    cols = ["n", "win%", "avg_R", "avg_ret%", "PF", "hold_d", "t_trade", "t_month", "months+"]
    out(f"{'STRATEGY':<24}{'FILTER':<9}" + "".join(f"{c:>10}" for c in cols))
    for name in wanted:
        for label, sub in (("none", t[t.strategy == name]),
                           ("NIFTY>200", t[(t.strategy == name) & t.regime_ok])):
            s = stats(sub)
            out(f"{name:<24}{label:<9}" + "".join(f"{str(s.get(c, '-')):>10}" for c in cols))

    out("\n" + "=" * 100)
    out("AVERAGE NET RETURN PER TRADE BY YEAR (%)  [trades]   - consistency check")
    out("=" * 100)
    t["year"] = pd.to_datetime(t.entry_date).dt.year
    years = sorted(t.year.unique())
    out(f"{'STRATEGY':<24}" + "".join(f"{y:>12}" for y in years))
    for name in wanted:
        sub = t[t.strategy == name]
        cells = []
        for y in years:
            yy = sub[sub.year == y]
            cells.append(f"{yy.ret.mean()*100:+.2f} [{len(yy)}]" if len(yy) else "-")
        out(f"{name:<24}" + "".join(f"{c:>12}" for c in cells))

    out("\n" + "=" * 100)
    out(f"PORTFOLIO (start Rs {config.SWING_START_EQUITY:,.0f}, max {config.SWING_MAX_POSITIONS} "
        f"positions) vs NIFTY buy-and-hold")
    out("=" * 100)
    out(f"{'NIFTY buy & hold':<34} CAGR {bench['CAGR%']:>6}%  maxDD {bench['maxDD%']:>7}%  x{bench['final_x']}")
    results = {}
    for name in wanted:
        for label, sub in (("", t[t.strategy == name]), (" +NIFTY>200", t[(t.strategy == name) & t.regime_ok])):
            if sub.empty:
                continue
            p = portfolio(sub, closes, calendar)
            results[name + label] = p
            out(f"{name + label:<34} CAGR {p['CAGR%']:>6}%  maxDD {p['maxDD%']:>7}%  x{p['final_x']}  "
                f"trades {p['trades_taken']}  exposure {p['avg_exposure%']}%")
    out("\nYearly return % (portfolio):")
    out(f"{'':<34}" + "".join(f"{y:>8}" for y in bench["yearly"]))
    out(f"{'NIFTY':<34}" + "".join(f"{bench['yearly'].get(y, '-'):>8}" for y in bench["yearly"]))
    for k, p in results.items():
        out(f"{k:<34}" + "".join(f"{p['yearly'].get(y, '-'):>8}" for y in bench["yearly"]))

    md = os.path.join(args.out, f"swing_report_{stamp}.md")
    with open(md, "w") as f:
        f.write("\n".join(lines).replace("=" * 100, ""))
    print(f"\nSaved trades: {csv_path}\nSaved report: {md}")


if __name__ == "__main__":
    main()
