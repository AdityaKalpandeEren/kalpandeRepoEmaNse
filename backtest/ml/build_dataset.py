"""
Builds the labeled training set for the V1 ML meta-label filter
(model L_ML_META) from NSE Upstox candles - port of us_alert_bot's
backtest/ml/build_dataset.py.

Every base-model candidate (A-K + SCORE_ENGINE) becomes one row: its
features (strategy/ml_features.py) + whether it won, using the SAME
forward simulation, fill rule, costs and 15:15 square-off as
backtest/run_research.py. Symbols run in parallel worker processes.

    python -m backtest.ml.build_dataset --from 2025-09-01 --to 2026-09-25 --workers 6
    python -m backtest.ml.build_dataset --days 120 --symbols RELIANCE,TCS,INFY

Output: backtest/ml/data/dataset.csv, chronological (the trainer's
walk-forward split depends on time order).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from backtest.ml.common import (access_token, load_history_with_retry, resolve_dates,
                                resolve_symbols, write_rows_atomic)
from strategy.ml_features import FEATURE_COLUMNS

BOOKKEEPING_COLUMNS = ["symbol", "entry_time", "source_strategy", "direction",
                       "regime", "outcome", "r_multiple", "label"]
CSV_COLUMNS = BOOKKEEPING_COLUMNS + FEATURE_COLUMNS


def _enrich_day(day_df, candle_minutes):
    from strategy.indicators import enrich
    return enrich(
        day_df.copy(),
        ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW,
        atr_period=config.ATR_PERIOD, vol_period=config.VOLUME_AVG_PERIOD,
        struct_lookback=config.STRUCT_LOOKBACK, or_minutes=config.OPENING_RANGE_MINUTES,
        candle_minutes=candle_minutes,
    )


def build_rows_for_day(symbol, day_df, candle_minutes, directions, apply_costs=True):
    from strategy.ml_features import extract_features
    from strategy.regime import classify, is_tradeable
    from strategy.session import trim_to_research_session
    from strategy.strategies import BASE_MODELS, _component_scores
    from strategy.trade_engine import simulate_forward_directional, build_research_trade

    rows = []
    day_df = trim_to_research_session(day_df.reset_index(drop=True))
    if len(day_df) < config.MIN_WARMUP_CANDLES + 2:
        return rows
    enriched = _enrich_day(day_df, candle_minutes)

    last_fired, counts = {}, {}
    for i in range(config.MIN_WARMUP_CANDLES, len(enriched) - 1):
        sub = enriched.iloc[: i + 1]
        row = sub.iloc[-1]
        regime = classify(row)
        for direction in directions:
            if not is_tradeable(regime, direction):
                continue
            if direction == "short" and len(sub) < config.SHORT_MIN_SESSION_CANDLES:
                continue
            for name, fn in BASE_MODELS.items():
                key = (name, direction)
                if counts.get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
                    continue
                if i - last_fired.get(key, -10**9) < config.SIGNAL_COOLDOWN_CANDLES:
                    continue
                try:
                    sig = fn(symbol, sub, direction, regime)
                except Exception:
                    continue
                if sig is None:
                    continue
                fill_idx = i + 1
                entry_price = float(day_df.iloc[fill_idx]["open"])
                if direction == "long" and entry_price <= sig.stop_loss:
                    continue
                if direction == "short" and entry_price >= sig.stop_loss:
                    continue
                exit_time, exit_price, outcome, exit_idx = simulate_forward_directional(
                    day_df, fill_idx, direction, entry_price, sig.stop_loss, sig.target)
                trade = build_research_trade(
                    signal=sig, entry_time=day_df.iloc[fill_idx]["timestamp"], entry=entry_price,
                    exit_time=exit_time, exit_price=exit_price, outcome=outcome,
                    candles_held=exit_idx - fill_idx, apply_costs=apply_costs)
                comps = _component_scores(row, sub, direction)
                feats = extract_features(row, sub, direction, regime, comps, name)
                record = {
                    "symbol": symbol, "entry_time": str(trade.entry_time),
                    "source_strategy": name, "direction": direction, "regime": regime.label,
                    "outcome": outcome, "r_multiple": trade.r_multiple,
                    "label": 1 if trade.r_multiple > 0 else 0,
                }
                record.update(feats)
                rows.append(record)
                last_fired[key] = i
                counts[key] = counts.get(key, 0) + 1
    return rows


def process_symbol(symbol, interval, from_date, to_date, directions, apply_costs):
    from backtest.data_loader import load_symbol_history, group_by_day
    try:
        hist = load_history_with_retry(symbol, interval, from_date, to_date)
    except Exception as e:
        return symbol, [], f"FAILED: {e}"
    days = group_by_day(hist)
    if not days:
        return symbol, [], "no data"
    rows = []
    for _, day_df in days.items():
        rows.extend(build_rows_for_day(symbol, day_df, interval, directions, apply_costs))
    return symbol, rows, f"{len(days)} days, {len(rows)} candidates"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols")
    p.add_argument("--from", dest="from_date")
    p.add_argument("--to", dest="to_date")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES)
    p.add_argument("--directions", default="long,short")
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--max-symbols", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="backtest/ml/data/dataset.csv")
    return p.parse_args()


def main():
    args = parse_args()
    from_date, to_date = resolve_dates(args)
    symbols = resolve_symbols(args)
    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    print(f"Building V1 dataset: {len(symbols)} symbols | {from_date}..{to_date} | "
          f"directions={list(directions)} | workers={args.workers}")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_symbol, s, args.interval, from_date, to_date,
                            directions, not args.no_costs): s for s in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                sym, sym_rows, status = fut.result()
            except Exception as e:
                sym, sym_rows, status = futs[fut], [], f"CRASHED: {e!r}"
            rows.extend(sym_rows)
            print(f"[{n}/{len(symbols)}] {sym}: {status}", flush=True)
    rows.sort(key=lambda r: (r["entry_time"], r["symbol"], r["source_strategy"], r["direction"]))
    write_rows_atomic(args.out, CSV_COLUMNS, rows)
    print(f"\nTotal candidate trades written: {len(rows)}\nSaved: {args.out}")
    if len(rows) < 500:
        print("Warning: under 500 rows is thin for training - widen the window or symbols.")
        sys.exit(0)


if __name__ == "__main__":
    main()
