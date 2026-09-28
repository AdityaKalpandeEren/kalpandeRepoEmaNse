"""
Historical backtest for the NSE bot's EMA-cross / VWAP-retest /
VWAP-broad-test signals - answers "how often would these alerts
actually have won?" using real Upstox historical candles, walked
forward exactly the way the live bot sees them (see backtest/simulator.py
and strategy/trade_engine.py docstrings for the no-lookahead / fill
rules).

Usage examples (run from the nse_alert_bot/ directory):

    python -m backtest.run_backtest --from 2025-08-01 --to 2025-09-10

    python -m backtest.run_backtest --symbols RELIANCE,TCS,INFY \\
        --from 2025-08-01 --to 2025-09-10

    python -m backtest.run_backtest --from 2025-08-01 --to 2025-09-10 \\
        --strategies EMA_CROSS,VWAP_RETEST

Notes:
- Needs a working UPSTOX_ACCESS_TOKEN in .env (same one main.py uses).
- 1-minute/5-minute historical data on Upstox only goes back so far
  (roughly the last month for 1-min bars) - if a --from date is too
  old for your chosen --interval, Upstox will just return fewer/no
  candles for the earliest chunk; widen --interval or narrow the range.
- Results are written to backtest/results/ as a CSV (every simulated
  trade) and a Markdown summary report.
"""
import argparse
import sys

import config
from main import load_access_token
from backtest.data_loader import load_symbol_history, group_by_day
from backtest.simulator import simulate_day, STRATEGIES
from backtest.report import write_report, print_summary


def load_symbol_list(filename: str) -> list:
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated NSE trading symbols. Defaults to watchlist.txt.")
    p.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    p.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES,
                    help=f"Candle size in minutes (default: {config.CANDLE_INTERVAL_MINUTES}, matches config.py)")
    p.add_argument("--strategies", default="EMA_CROSS,VWAP_RETEST,VWAP_BROAD_TEST",
                    help="Comma-separated subset of EMA_CROSS,VWAP_RETEST,VWAP_BROAD_TEST")
    p.add_argument("--out", default="backtest/results", help="Output directory for the CSV + report")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        access_token = load_access_token()
    except Exception as e:
        print(f"Could not load an Upstox access token: {e}")
        print("Set UPSTOX_ACCESS_TOKEN in .env (same as main.py needs) and try again.")
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else load_symbol_list("watchlist.txt")
    strategy_names = [s.strip() for s in args.strategies.split(",")]
    strategies = {k: v for k, v in STRATEGIES.items() if k in strategy_names}
    if not strategies:
        print(f"No valid strategies in '{args.strategies}'. Choose from: {list(STRATEGIES)}")
        sys.exit(1)

    print(f"Backtesting {len(symbols)} symbol(s) from {args.from_date} to {args.to_date}, "
          f"{args.interval}-min candles, strategies: {list(strategies)}\n")

    all_trades = []
    for symbol in symbols:
        print(f"Fetching {symbol} ...")
        try:
            hist = load_symbol_history(symbol, args.interval, args.from_date, args.to_date, access_token)
        except Exception as e:
            print(f"  skip {symbol}: {e}")
            continue

        days = group_by_day(hist)
        if not days:
            print("  no historical candles returned for this range")
            continue
        print(f"  {len(days)} trading day(s) of data")

        for day, day_df in days.items():
            trades = simulate_day(symbol, day_df, strategies)
            all_trades.extend(trades)
            if trades:
                detail = ", ".join(f"{t.strategy}:{t.outcome}({t.r_multiple}R)" for t in trades)
                print(f"    {day}: {detail}")

    print(f"\nTotal trades simulated: {len(all_trades)}")
    if not all_trades:
        print("Nothing to report - try a wider date range, more symbols, or looser strategies.")
        return

    result = write_report(all_trades, args.out, label="backtest")
    print_summary(result["summary"])
    print(f"\nSaved trade log:   {result['csv']}")
    print(f"Saved summary:     {result['report']}")


if __name__ == "__main__":
    main()
