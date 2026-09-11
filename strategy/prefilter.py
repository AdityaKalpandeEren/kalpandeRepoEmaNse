"""
Stage 1 of the dynamic scanner: a cheap, single-bulk-API-call filter
that narrows a broad universe of symbols down to ones trading near
today's high with a green day - i.e. genuinely "near breakout" right
now. Only symbols that pass this go on to the expensive per-symbol
5-min-candle fetch and the full EMA/VWAP checks.

This is intentionally simple (day-level OHLC only, not multi-day
average volume) - a fast, cheap gate, not a signal in itself.
"""
import config


def find_near_breakout(ohlc_map: dict) -> list:
    """ohlc_map: {instrument_key: {"open","high","low","close","volume"}}
    (the output of data.upstox_client.get_bulk_ohlc). Returns the list
    of instrument_keys that pass the prefilter."""
    shortlist = []

    for instrument_key, bar in ohlc_map.items():
        o, h, l, c = bar.get("open"), bar.get("high"), bar.get("low"), bar.get("close")
        if not all(isinstance(x, (int, float)) and x > 0 for x in [o, h, l, c]):
            continue

        pct_from_high = (h - c) / h
        pct_change_from_open = (c - o) / o

        near_high = pct_from_high <= config.BREAKOUT_MAX_PCT_FROM_HIGH
        green_day = pct_change_from_open >= config.BREAKOUT_MIN_PCT_FROM_OPEN

        if near_high and green_day:
            shortlist.append(instrument_key)

    return shortlist
