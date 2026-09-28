"""
Builds the labeled training set for model L_ML_META_V3 (NSE).

Same scheme as build_dataset_v2.py, with V3's candidate set and trade:
  - candidates come from strategies.ml_v3_candidates() - the 12 base
    models PLUS the two relative-strength breakout models, all put on the
    V3 cost-aware stop (config.ML_V3_MIN_STOP_PCT) - the same function the
    live model calls (train/serve parity);
  - features are the V3 set (sector tape, overnight macro, dynamic
    sector/stock sensitivities, breakout structure);
  - labels use V3's exits (= V2's: stop/target, NIFTY/India VIX shock,
    15:15 flat) on the V3 stop/target.

    python -m backtest.ml.build_dataset_v3 --from 2025-09-26 --to 2026-09-25 --workers 7
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from backtest.data_loader import load_symbol_history, group_by_day
from backtest.ml.common import (access_token, load_history_with_retry, resolve_dates,
                                resolve_symbols, write_rows_atomic)
from strategy.indicators import enrich
from strategy.session import trim_to_research_session
from strategy.ml_features_v3 import FEATURE_COLUMNS_V3
from strategy.regime import classify, is_tradeable
from strategy.trade_engine import simulate_forward_v2, build_research_trade

BOOKKEEPING_COLUMNS = ["symbol", "date", "entry_time", "source_strategy", "direction",
                       "regime", "outcome", "r_multiple", "label"]
CSV_COLUMNS = BOOKKEEPING_COLUMNS + FEATURE_COLUMNS_V3


def _enrich_day(day_df, candle_minutes):
    return enrich(
        day_df.copy(),
        ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW,
        atr_period=config.ATR_PERIOD, vol_period=config.VOLUME_AVG_PERIOD,
        struct_lookback=config.STRUCT_LOOKBACK, or_minutes=config.OPENING_RANGE_MINUTES,
        candle_minutes=candle_minutes,
    )


def build_rows_for_day(symbol, day_df, candle_minutes, directions, ctx, apply_costs=True):
    from strategy.strategies import ml_v3_candidates, ml_v2_in_session

    rows = []
    day_df = trim_to_research_session(day_df.reset_index(drop=True))
    if len(day_df) < config.MIN_WARMUP_CANDLES + 2:
        return rows
    enriched = _enrich_day(day_df, candle_minutes)

    last_fired, counts = {}, {}
    for i in range(config.MIN_WARMUP_CANDLES, len(enriched) - 1):
        sub = enriched.iloc[: i + 1]
        if not ml_v2_in_session(sub):
            continue
        row = sub.iloc[-1]
        regime = classify(row)

        for direction in directions:
            # Same gates evaluate_all() applies before any model runs.
            if not is_tradeable(regime, direction):
                continue
            if direction == "short" and len(sub) < config.SHORT_MIN_SESSION_CANDLES:
                continue

            try:
                cands = ml_v3_candidates(symbol, sub, direction, regime, ctx)
            except Exception as e:
                print(f"  [{symbol}] candidate error at {row['timestamp']}: {e!r}")
                continue

            for name, sig, feats in cands:
                key = (name, direction)
                if counts.get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
                    continue
                if i - last_fired.get(key, -10**9) < config.SIGNAL_COOLDOWN_CANDLES:
                    continue

                fill_idx = i + 1
                entry_price = float(day_df.iloc[fill_idx]["open"])
                if direction == "long" and entry_price <= sig.stop_loss:
                    continue
                if direction == "short" and entry_price >= sig.stop_loss:
                    continue

                exit_time, exit_price, outcome, exit_idx = simulate_forward_v2(
                    day_df, fill_idx, direction, entry_price, sig.stop_loss, sig.target, ctx)
                trade = build_research_trade(
                    signal=sig, entry_time=day_df.iloc[fill_idx]["timestamp"], entry=entry_price,
                    exit_time=exit_time, exit_price=exit_price, outcome=outcome,
                    candles_held=exit_idx - fill_idx, apply_costs=apply_costs)

                record = {
                    "symbol": symbol,
                    "date": str(trade.entry_time.date()),
                    "entry_time": str(trade.entry_time),
                    "source_strategy": name,
                    "direction": direction,
                    "regime": regime.label,
                    "outcome": outcome,
                    "r_multiple": trade.r_multiple,
                    "label": 1 if trade.r_multiple > 0 else 0,
                }
                record.update(feats)
                rows.append(record)
                last_fired[key] = i
                counts[key] = counts.get(key, 0) + 1
    return rows


def process_symbol(symbol, interval, from_date, to_date, directions, apply_costs):
    """Worker entry point: all rows for one symbol, or an error string."""
    from strategy.market_context import set_history_start
    from strategy.market_context_v3 import get_context_v3
    set_history_start(from_date)
    ctx = get_context_v3(live=False)
    try:
        hist = load_history_with_retry(symbol, interval, from_date, to_date)
    except Exception as e:
        return symbol, [], f"FAILED: {e}"
    days = group_by_day(hist)
    if not days:
        return symbol, [], "no data"
    try:
        ctx.ensure_v3()
        ctx.ensure_symbol(symbol)
    except Exception as e:
        return symbol, [], f"context FAILED: {e}"
    rows = []
    for _, day_df in days.items():
        rows.extend(build_rows_for_day(symbol, day_df, interval, directions, ctx, apply_costs))
    return symbol, rows, f"{len(days)} days, {len(rows)} candidates"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated NSE symbols. Default: watchlist + universe + NIFTY 50")
    p.add_argument("--from", dest="from_date")
    p.add_argument("--to", dest="to_date")
    p.add_argument("--days", type=int, default=365, help="Last N calendar days (default 365)")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES)
    p.add_argument("--directions", default="long",
                   help="Default long (the live bot is long-only; the RS models are long-only)")
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--max-symbols", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=config.ML_V3_DATASET_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    from_date, to_date = resolve_dates(args)
    symbols = resolve_symbols(args)
    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    apply_costs = not args.no_costs
    print(f"Building V3 dataset: {len(symbols)} symbols | {from_date}..{to_date} "
          f"| {args.interval}-min | directions={list(directions)} | workers={args.workers}")

    # Warm the shared India market context ONCE (index history back to
    # --from, breadth, daily bars) so workers read it from the disk cache.
    from strategy.market_context import set_history_start
    from strategy.market_context_v3 import get_context_v3
    print("Loading V3 context (India VIX, NIFTY/BANK, 13 sector indices, breadth, macro)...", flush=True)
    set_history_start(from_date)
    get_context_v3(live=False).ensure_v3()

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_symbol, s, args.interval, from_date, to_date,
                            directions, apply_costs): s for s in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            sym = futs[fut]
            try:
                _, rows, status = fut.result()
            except Exception as e:
                rows, status = [], f"CRASHED: {e!r}"
            all_rows.extend(rows)
            print(f"[{n}/{len(symbols)}] {sym}: {status}", flush=True)

    all_rows.sort(key=lambda r: (r["entry_time"], r["symbol"], r["source_strategy"], r["direction"]))
    write_rows_atomic(args.out, CSV_COLUMNS, all_rows)
    print(f"\nTotal candidate trades written: {len(all_rows)}")
    print(f"Saved: {args.out}")
    if len(all_rows) < 1000:
        print("\nWarning: under ~1000 rows is thin for ~120 features. Widen the window/symbols.")


if __name__ == "__main__":
    main()
