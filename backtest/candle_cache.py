"""
On-disk cache of historical Upstox candles for research backtests and ML
dataset builds.

Past months never change, so each (symbol, month) is fetched once and
kept under backtest/cache/candles/. A long build therefore survives
network drops (re-run it and it resumes where it stopped) and every
later backtest/rebuild over the same dates reads from disk. The current
month is always re-fetched, since it is still being written.

    python -m backtest.candle_cache --from 2025-09-26 --to 2026-09-25   # pre-download

The production scanner and backtest/run_backtest.py don't use this.
"""
import argparse
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd

CACHE_DIR = os.path.join("backtest", "cache", "candles")

_session = None


def _get_candles(key, unit, interval, to_date, from_date, access_token):
    """Same request/shape as data.upstox_client.get_historical_candles, but
    over ONE reused HTTP session: bulk downloads make thousands of calls,
    and opening a fresh connection (DNS lookup + TLS) for each one was
    measured at up to ~14 s per call on a slow resolver vs ~0.1 s reused."""
    global _session
    import requests
    if _session is None:
        _session = requests.Session()
    url = f"https://api.upstox.com/v3/historical-candle/{key}/{unit}/{interval}/{to_date}/{from_date}"
    resp = _session.get(url, headers={"Accept": "application/json",
                                      "Authorization": f"Bearer {access_token}"}, timeout=30)
    resp.raise_for_status()
    candles = resp.json().get("data", {}).get("candles", [])
    cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    if not candles:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(candles, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df


def _month_ranges(from_d: date, to_d: date):
    cur = date(from_d.year, from_d.month, 1)
    while cur <= to_d:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, nxt - timedelta(days=1)
        cur = nxt


def _path(symbol: str, interval: int, month_start: date) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol)
    return os.path.join(CACHE_DIR, f"{interval}m", safe, f"{month_start:%Y-%m}.pkl")


def _fetch_month(symbol, interval, m_start, m_end, access_token, attempts=6):
    from data.instruments import get_instrument_key
    # An Upstox instrument key ("NSE_INDEX|Nifty 50") is used as-is; a
    # trading symbol is looked up (ValueError for unknown symbols - not retried).
    key = symbol if "|" in symbol else get_instrument_key(symbol)
    last = None
    for k in range(attempts):
        try:
            df = _get_candles(key, "minutes", interval, m_end.strftime("%Y-%m-%d"),
                              m_start.strftime("%Y-%m-%d"), access_token)
            return df[["timestamp", "open", "high", "low", "close", "volume"]] if not df.empty else df
        except Exception as e:
            last = e
            time.sleep(min(60, 5 * (k + 1)))
    raise last


def load_symbol_history_cached(symbol: str, interval: int, from_date: str, to_date: str,
                               access_token: str) -> pd.DataFrame:
    """Same result shape as backtest.data_loader.load_symbol_history."""
    from_d = datetime.strptime(from_date, "%Y-%m-%d").date()
    to_d = datetime.strptime(to_date, "%Y-%m-%d").date()
    today = date.today()
    frames = []
    for m_start, m_end in _month_ranges(from_d, to_d):
        path = _path(symbol, interval, m_start)
        complete_month = m_end < today.replace(day=1)
        if complete_month and os.path.exists(path):
            frames.append(pd.read_pickle(path))
            continue
        snap = None if complete_month else _read_today_snapshot(path, to_d)
        if snap is not None:
            frames.append(snap)
            continue
        df = _fetch_month(symbol, interval, m_start, min(m_end, today), access_token)
        if not complete_month:
            _write_today_snapshot(path, df, to_d)
        if complete_month:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            df.to_pickle(tmp)
            os.replace(tmp, path)
        frames.append(df)
        time.sleep(0.05)
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    tz = out["timestamp"].dt.tz
    lo, hi = pd.Timestamp(from_d), pd.Timestamp(to_d + timedelta(days=1))
    if tz is not None:
        lo, hi = lo.tz_localize(tz), hi.tz_localize(tz)
    return out[(out["timestamp"] >= lo) & (out["timestamp"] < hi)].reset_index(drop=True)


def main():
    import config
    from backtest.ml.common import access_token, load_symbol_list
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--symbols")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES)
    p.add_argument("--with-indices", action="store_true",
                   help="Also cache NIFTY 50, India VIX and the sector indices")
    args = p.parse_args()
    syms = ([s.strip() for s in args.symbols.split(",")] if args.symbols else
            list(dict.fromkeys(load_symbol_list("watchlist.txt") + load_symbol_list("universe.txt")
                               + list(config.ML_V2_BREADTH_UNIVERSE))))
    if args.with_indices:
        from strategy.sector_map import SECTORS
        syms = ["NSE_INDEX|Nifty 50", "NSE_INDEX|India VIX"] + \
               [k for c, k in SECTORS.items() if c != "OTHER"] + syms
    tok = access_token()
    failed = []
    for n, s in enumerate(syms, 1):
        try:
            df = load_symbol_history_cached(s, args.interval, args.from_date, args.to_date, tok)
            print(f"[{n}/{len(syms)}] {s}: {len(df)} candles", flush=True)
        except Exception as e:
            failed.append(s)
            print(f"[{n}/{len(syms)}] {s}: FAILED {str(e)[:120]}", flush=True)
    print(f"\nDone. Failed: {failed or 'none'}  (re-run to resume; cached months are kept)")




# ═══════════════════════════════════════════════════════════════════
# DAILY candles (swing research). Upstox serves ~3 years of daily bars
# per request, so history is cached per (symbol, calendar year); past
# years never change, the current year is always re-fetched.
# ═══════════════════════════════════════════════════════════════════

def _today_snapshot(path: str) -> str:
    """Same-day copy of a still-open month / year. Once fetched, data up to
    YESTERDAY can't change during today, so a request that ends before today
    reuses it (V3.3 pre-warms these at 09:20 so the 09:45 pick doesn't
    re-download ~268 symbols). Older day-copies are deleted."""
    return f"{path}.day{date.today():%Y%m%d}"


def _read_today_snapshot(path: str, to_d: date):
    p = _today_snapshot(path)
    if to_d < date.today() and os.path.exists(p):
        return pd.read_pickle(p)
    return None


def _write_today_snapshot(path: str, df: pd.DataFrame, to_d: date) -> None:
    if to_d >= date.today() or df is None:
        return
    import glob
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for old in glob.glob(f"{path}.day*"):
        if not old.endswith(f".day{date.today():%Y%m%d}"):
            try:
                os.remove(old)
            except OSError:
                pass
    tmp = f"{_today_snapshot(path)}.tmp.{os.getpid()}"
    df.to_pickle(tmp)
    os.replace(tmp, _today_snapshot(path))


def _daily_path(key_name: str, year: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key_name)
    return os.path.join(CACHE_DIR, "1d", safe, f"{year}.pkl")


def load_daily_cached(symbol_or_key: str, from_date: str, to_date: str, access_token: str,
                      is_key: bool = False, attempts: int = 6) -> pd.DataFrame:
    """Daily candles for a symbol (or an Upstox instrument key such as
    'NSE_INDEX|Nifty 50' with is_key=True), [from_date, to_date]."""
    key = symbol_or_key
    if not is_key:
        from data.instruments import get_instrument_key
        key = get_instrument_key(symbol_or_key)
    from_d = datetime.strptime(from_date, "%Y-%m-%d").date()
    to_d = datetime.strptime(to_date, "%Y-%m-%d").date()
    frames = []
    for year in range(from_d.year, to_d.year + 1):
        path = _daily_path(symbol_or_key, year)
        past_year = year < date.today().year
        if past_year and os.path.exists(path):
            frames.append(pd.read_pickle(path))
            continue
        snap = None if past_year else _read_today_snapshot(path, to_d)
        if snap is not None:
            frames.append(snap)
            continue
        y_from, y_to = date(year, 1, 1), min(date(year, 12, 31), date.today())
        last = None
        df = None
        for k in range(attempts):
            try:
                df = _get_candles(key, "days", 1, y_to.strftime("%Y-%m-%d"),
                                  y_from.strftime("%Y-%m-%d"), access_token)
                last = None
                break
            except Exception as e:
                last = e
                time.sleep(min(60, 5 * (k + 1)))
        if last is not None:
            raise last
        df = df[["timestamp", "open", "high", "low", "close", "volume"]] if not df.empty else df
        if not past_year:
            _write_today_snapshot(path, df, to_d)
        if past_year:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            df.to_pickle(tmp)
            os.replace(tmp, path)
        frames.append(df)
        time.sleep(0.1)
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    d = out["timestamp"].dt.date
    return out[(d >= from_d) & (d <= to_d)].reset_index(drop=True)


if __name__ == "__main__":
    main()
