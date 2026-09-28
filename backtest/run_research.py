"""
Multi-strategy research backtest for the NSE bot (ported from
us_alert_bot): runs the research entry models, long and/or short, over
the same historical Upstox candles, and ranks them by expectancy.

    # the five live research strategies, long only, last ~6 months
    python -m backtest.run_research --from 2026-04-01 --to 2026-09-25 \\
        --strategies K_RSI2_REVERSION,SCORE_ENGINE,L_ML_META,L_ML_META_V2,J_VWAP_BAND_REVERSION \\
        --directions long --risk-sweep

    # a focused subset of symbols
    python -m backtest.run_research --symbols RELIANCE,TCS,INFY,SBIN --days 90

    # measure what the regime filter / costs are worth
    python -m backtest.run_research --days 90 --no-regime-filter
    python -m backtest.run_research --days 90 --no-costs

Models: A_BREAKOUT, B_BREAKOUT_CLOSE, C_BREAKOUT_RETEST, D_VWAP_RECLAIM,
E_VWAP_REJECTION, F_EMA_PULLBACK, G_CONFLUENCE, H_ORB_VWAP,
I_EMA_STACK_BREAKOUT, J_VWAP_BAND_REVERSION, K_RSI2_REVERSION,
SCORE_ENGINE, L_ML_META (needs backtest/ml/train_meta_model.py first),
L_ML_META_V2 (needs backtest/ml/train_meta_model_v2.py first),
L_ML_META_V3 (needs backtest/ml/train_meta_model_v3.py first).

NSE specifics: IST session 9:15-15:30, research trades squared off at
15:15 (config.RESEARCH_SQUAREOFF_*), NSE intraday costs
(config.SLIPPAGE_BPS + COMMISSION_BPS). Upstox serves years of 5-min
history, so unlike the US bot you can test many months at once - do.

Read the caveats printed at the end before acting on any ranking.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import config
from main import load_access_token
from backtest.candle_cache import load_symbol_history_cached
from backtest.data_loader import group_by_day
from backtest.research_simulator import simulate_day_research
from backtest.research_report import write_research_report, print_research_summary
from strategy.strategies import ENTRY_MODELS


def load_symbol_list(filename: str) -> list:
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated NSE symbols. Defaults to watchlist.txt.")
    p.add_argument("--universe", action="store_true",
                   help="Use watchlist.txt + universe.txt (deduped) instead of watchlist only")
    p.add_argument("--from", dest="from_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--to", dest="to_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--days", type=int, help="Last N calendar days ending today")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES)
    p.add_argument("--strategies", default=",".join(ENTRY_MODELS),
                   help="Comma-separated subset of the model names above")
    p.add_argument("--directions", default="long",
                   help="long | short | long,short (default: long)")
    p.add_argument("--no-regime-filter", action="store_true")
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--risk-sweep", action="store_true",
                   help="Report P&L at 0.25/0.5/0.75/1.0%% risk per trade")
    p.add_argument("--max-symbols", type=int, default=200)
    p.add_argument("--out", default="backtest/results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.days:
        today = datetime.now()
        args.to_date = today.strftime("%Y-%m-%d")
        args.from_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
    if not args.from_date or not args.to_date:
        print("Provide either --from and --to, or --days N.")
        sys.exit(1)

    try:
        access_token = load_access_token()
    except Exception as e:
        print(f"Could not load an Upstox access token: {e}")
        sys.exit(1)

    if args.no_regime_filter:
        config.REGIME_FILTER_ENABLED = False
    apply_costs = not args.no_costs

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbol_list("watchlist.txt")
        if args.universe:
            symbols += load_symbol_list("universe.txt")
    symbols = list(dict.fromkeys(symbols))[: args.max_symbols]

    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [w for w in wanted if w not in ENTRY_MODELS]
    if unknown:
        print(f"Unknown strategy name(s) {unknown} - check the spelling. "
              f"Valid: {', '.join(ENTRY_MODELS)}")
        sys.exit(1)
    models = {k: v for k, v in ENTRY_MODELS.items() if k in wanted}
    # The ML filters return no signals at all when their model file is
    # missing - say so up front instead of silently reporting zero trades.
    for name, path, how in (
            ("L_ML_META", config.ML_META_MODEL_PATH,
             "python -m backtest.ml.build_dataset && python -m backtest.ml.train_meta_model"),
            ("L_ML_META_V2", config.ML_V2_MODEL_PATH,
             "python -m backtest.ml.build_dataset_v2 && "
             "python -m backtest.ml.train_meta_model_v2 --directions long"),
            ("L_ML_META_V3", config.ML_V3_MODEL_PATH,
             "python -m backtest.ml.build_dataset_v3 && python -m backtest.ml.train_meta_model_v3")):
        if name in models and not os.path.exists(path):
            print(f"WARNING: {name} has no trained model ({path}) - it will produce NO trades.\n"
                  f"         Train it first: {how}\n")
    if "L_ML_META_V2" in models or "L_ML_META_V3" in models:
        # Index history (India VIX / NIFTY / BANK / sectors) must reach back to --from.
        from strategy.market_context import set_history_start
        set_history_start(args.from_date)
    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())

    print(f"NSE research backtest: {len(symbols)} symbols | {args.from_date}..{args.to_date} "
          f"| {args.interval}-min candles")
    print(f"Models    : {list(models)}")
    print(f"Directions: {list(directions)}")
    print(f"Costs     : {'ON' if apply_costs else 'OFF'} "
          f"({config.SLIPPAGE_BPS}bps slip + {config.COMMISSION_BPS}bps charges per side)")
    print(f"Square-off: {config.RESEARCH_SQUAREOFF_HOUR}:{config.RESEARCH_SQUAREOFF_MINUTE:02d} IST | "
          f"warm-up {config.MIN_WARMUP_CANDLES} candles")
    print(f"Regime gate: {'ON' if config.REGIME_FILTER_ENABLED else 'OFF'}\n")

    all_trades = []
    total_days = 0
    for n, symbol in enumerate(symbols, 1):
        print(f"[{n}/{len(symbols)}] {symbol} ...", end=" ", flush=True)
        try:
            hist = load_symbol_history_cached(symbol, args.interval, args.from_date, args.to_date, access_token)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        days = group_by_day(hist)
        if not days:
            print("no data")
            continue
        sym_trades = []
        for _, day_df in days.items():
            sym_trades.extend(simulate_day_research(
                symbol, day_df, args.interval, models, directions, apply_costs))
        total_days = max(total_days, len(days))
        all_trades.extend(sym_trades)
        print(f"{len(days)} days, {len(sym_trades)} trades")

    print(f"\nTotal trades simulated: {len(all_trades)}")
    if not all_trades:
        print("Nothing to report. Try more symbols, a longer window, or --no-regime-filter.")
        return

    meta = {
        "from": args.from_date, "to": args.to_date, "interval": args.interval,
        "symbols": len(symbols), "days": total_days,
        "costs": (f"{config.SLIPPAGE_BPS}+{config.COMMISSION_BPS}bps/side" if apply_costs else "NONE"),
        "regime_filter": "ON" if config.REGIME_FILTER_ENABLED else "OFF",
    }
    result = write_research_report(all_trades, args.out, meta)
    print_research_summary(result["summary"])

    if args.risk_sweep:
        print("\n" + "=" * 78)
        print("POSITION-SIZING SWEEP")
        print("=" * 78)
        total_r = result["summary"]["overall"]["total_r"]
        dd_r = result["summary"]["overall"]["max_drawdown_r"]
        print(f"Account equity: Rs {config.ACCOUNT_EQUITY:,.0f}")
        print(f"{'RISK/TRADE':>12}{'TOTAL P&L':>16}{'MAX DD':>16}{'DD %':>10}")
        for risk in (0.0025, 0.005, 0.0075, 0.01):
            budget = config.ACCOUNT_EQUITY * risk
            print(f"{risk*100:>11.2f}%{total_r*budget:>16,.0f}{dd_r*budget:>16,.0f}"
                  f"{abs(dd_r*budget)/config.ACCOUNT_EQUITY*100:>9.1f}%")

    print(f"\nSaved trade log: {result['csv']}")
    print(f"Saved report   : {result['report']}")
    print("\n" + "=" * 78)
    print("BEFORE YOU ACT ON THIS")
    print("=" * 78)
    print("1. Many models on one dataset: the winner is partly selection luck.")
    print("   Re-run the leaders on a DIFFERENT date range before believing them.")
    print("2. Check the t-stat. Below 2.0 the result is inside the noise band.")
    print("3. ML models (L_ML_META / V2) are IN-SAMPLE on any dates they were trained on.")


if __name__ == "__main__":
    main()
