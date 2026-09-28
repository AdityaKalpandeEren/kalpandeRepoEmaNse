"""Shared helpers for the NSE ML dataset builders (V1 and V2)."""
import json
import os
from datetime import datetime, timedelta

import config


def load_symbol_list(filename: str) -> list:
    try:
        with open(filename) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []


def access_token() -> str:
    if config.UPSTOX_ACCESS_TOKEN:
        return config.UPSTOX_ACCESS_TOKEN
    with open(config.TOKEN_FILE) as f:
        return json.load(f)["access_token"]


def resolve_dates(args):
    if args.days and not (args.from_date and args.to_date):
        today = datetime.now()
        args.to_date = today.strftime("%Y-%m-%d")
        args.from_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
    return args.from_date, args.to_date


def resolve_symbols(args) -> list:
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # Default training universe: watchlist + universe + the NIFTY 50
        # (more symbols = more rows; ML needs breadth, not just history).
        syms = (load_symbol_list("watchlist.txt") + load_symbol_list("universe.txt")
                + list(config.ML_V2_BREADTH_UNIVERSE))
    syms = list(dict.fromkeys(s for s in syms if s))
    return syms[: args.max_symbols]


def write_rows_atomic(path: str, columns: list, rows: list):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_history_with_retry(symbol, interval, from_date, to_date, attempts: int = 4):
    """backtest.data_loader.load_symbol_history with backoff - a multi-hour
    dataset build should not silently lose symbols to a brief network or
    DNS blip."""
    import time
    from backtest.candle_cache import load_symbol_history_cached
    last = None
    for k in range(attempts):
        try:
            return load_symbol_history_cached(symbol, interval, from_date, to_date, access_token())
        except ValueError:
            raise                      # unknown symbol - retrying won't help
        except Exception as e:
            last = e
            time.sleep(5 * (k + 1))
    raise last
