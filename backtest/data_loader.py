"""
Fetches historical intraday candles across a date range and splits them
back into per-trading-day sessions, since the bot's indicators (VWAP,
EMA, avg-volume) all reset every session - exactly like the live bot
only ever seeing "today's" candles.
"""
import time
from datetime import datetime, timedelta

import pandas as pd

from data.instruments import get_instrument_key
from data.upstox_client import get_historical_candles

# Upstox caps how much minute-level data one call can return (roughly a
# month for 1-minute bars, more for coarser intervals). 25 days is a
# safe chunk size regardless of the interval you pick.
MAX_CHUNK_DAYS = 25


def _daterange_chunks(from_dt: datetime, to_dt: datetime, max_days: int = MAX_CHUNK_DAYS):
    cur_to = to_dt
    while cur_to >= from_dt:
        cur_from = max(from_dt, cur_to - timedelta(days=max_days))
        yield cur_from, cur_to
        cur_to = cur_from - timedelta(days=1)


def load_symbol_history(symbol: str, interval_minutes: int, from_date: str, to_date: str,
                         access_token: str, pause_seconds: float = 0.25) -> pd.DataFrame:
    """
    Returns one combined DataFrame (columns: timestamp, open, high, low,
    close, volume) spanning every trading day in [from_date, to_date].
    Use group_by_day() to split it before running strategy checks.
    """
    instrument_key = get_instrument_key(symbol)
    from_dt = datetime.strptime(from_date, "%Y-%m-%d")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d")

    frames = []
    for chunk_from, chunk_to in _daterange_chunks(from_dt, to_dt):
        df = get_historical_candles(
            instrument_key,
            unit="minutes",
            interval=interval_minutes,
            to_date=chunk_to.strftime("%Y-%m-%d"),
            from_date=chunk_from.strftime("%Y-%m-%d"),
            access_token=access_token,
        )
        if not df.empty:
            frames.append(df)
        time.sleep(pause_seconds)  # be polite across many chunked calls

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return combined


def group_by_day(df: pd.DataFrame) -> dict:
    """{date: day_df} - each day_df is one trading session's candles,
    in the same shape the live bot's per-symbol df is."""
    if df.empty:
        return {}
    df = df.copy()
    df["_date"] = df["timestamp"].dt.date
    return {d: g.drop(columns="_date").reset_index(drop=True) for d, g in df.groupby("_date")}
