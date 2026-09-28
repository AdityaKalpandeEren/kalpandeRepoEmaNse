"""
NSE session helpers for the research engine.

The research engine (backtest AND live paper trading) only ever sees
candles that CLOSE at or before the intraday square-off time
(config.RESEARCH_SQUAREOFF_*, 15:15 IST): MIS positions are auto-squared-
off by the broker around then, so a simulated trade that lived until the
15:30 close would be one you couldn't actually have held. Trimming the
day in one place means the backtest, the dataset builders and the live
paper trader all agree on where the day ends - and the "day's last
candle" square-off rule inherited from the US engine lands at 15:15.

The production screener (strategy/screener.py) does not use this.
"""
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import config

IST = ZoneInfo(config.MARKET_TIMEZONE)


def _as_ist(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def session_open(ts) -> pd.Timestamp:
    ts = _as_ist(ts)
    return ts.normalize() + pd.Timedelta(hours=config.MARKET_OPEN_HOUR, minutes=config.MARKET_OPEN_MINUTE)


def minutes_since_open(ts) -> float:
    """Minutes from the 9:15 IST open to candle timestamp `ts` (a candle's
    timestamp is its START time)."""
    ts = _as_ist(ts)
    return (ts - session_open(ts)).total_seconds() / 60.0


def squareoff_time(ts) -> pd.Timestamp:
    ts = _as_ist(ts)
    return ts.normalize() + pd.Timedelta(hours=config.RESEARCH_SQUAREOFF_HOUR,
                                         minutes=config.RESEARCH_SQUAREOFF_MINUTE)


def trim_to_research_session(day_df: pd.DataFrame) -> pd.DataFrame:
    """Candles of one day that close at or before the square-off time."""
    if day_df is None or day_df.empty:
        return day_df
    bar = timedelta(minutes=config.CANDLE_INTERVAL_MINUTES)
    cutoff = squareoff_time(day_df["timestamp"].iloc[0])
    return day_df[day_df["timestamp"] + bar <= cutoff].reset_index(drop=True)
