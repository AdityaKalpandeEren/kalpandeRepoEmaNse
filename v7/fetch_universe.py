"""
Fill the 5-minute candle cache for the V7 research universe.

v7/universe_months.csv lists every (symbol, month) the point-in-time top-200
universe needs for 2022-01..2026-09 (the month a stock is in the universe plus
the month before, for the 14-day RVOL / ATR warm-up). This downloads the
months that are not cached yet into backtest/cache/candles/5m/ - the same
monthly pickles backtest.candle_cache writes, so every other tool reuses them.

Upstox's historical-candle API is public; it is paced at ~1 request/s to stay
inside its ~2,000 requests / 30 min limit. Resume-safe: re-run to continue.

    python -m v7.fetch_universe
"""
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

from backtest import candle_cache

PACE_S = 0.95


def main():
    tok = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    todo = pd.read_csv("v7/universe_months.csv")
    todo = todo[~todo["cached"] & todo["has_key"]].sort_values(["symbol", "month"])
    done = failed = 0
    t0 = time.time()
    for i, r in enumerate(todo.itertuples(), 1):
        y, m = map(int, r.month.split("-"))
        m_start = date(y, m, 1)
        m_end = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
        path = candle_cache._path(r.symbol, 5, m_start)
        if os.path.exists(path):
            continue
        t = time.time()
        try:
            df = candle_cache._fetch_month(r.symbol, 5, m_start, m_end, tok, attempts=4)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            df.to_pickle(tmp)
            os.replace(tmp, path)
            done += 1
        except Exception as e:                       # keep going; a re-run retries it
            failed += 1
            print(f"[fetch] FAIL {r.symbol} {r.month}: {e!r}"[:200], flush=True)
        if i % 100 == 0:
            rate = i / (time.time() - t0)
            print(f"[fetch] {i}/{len(todo)} done {done} failed {failed} "
                  f"({rate:.2f}/s, ~{(len(todo) - i) / max(rate, 1e-9) / 60:.0f} min left)", flush=True)
        time.sleep(max(0.0, PACE_S - (time.time() - t)))
    print(f"[fetch] finished: {done} fetched, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
