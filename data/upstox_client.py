import requests
import pandas as pd

BASE_URL = "https://api.upstox.com/v3/historical-candle/intraday"
BULK_OHLC_URL = "https://api.upstox.com/v3/market-quote/ohlc"


def get_intraday_candles(instrument_key: str, interval_minutes: int, access_token: str) -> pd.DataFrame:
    """Fetch today's intraday candles for one instrument."""
    url = f"{BASE_URL}/{instrument_key}/minutes/{interval_minutes}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    payload = resp.json()

    candles = payload.get("data", {}).get("candles", [])
    columns = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    if not candles:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(candles, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df


def get_bulk_ohlc(instrument_keys: list, access_token: str) -> dict:
    """Fetch today's OHLC+volume for up to 500 instruments in as few calls as
    possible (chunked at 500 per Upstox's limit) - one cheap way to screen a
    broad watchlist before running the expensive per-symbol candle fetch on
    just the shortlist that passes a prefilter.

    Returns {instrument_key: {"open":.., "high":.., "low":.., "close":.., "volume":..}}
    - only instruments Upstox actually returned data for are present.
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    results = {}

    for i in range(0, len(instrument_keys), 500):
        chunk = instrument_keys[i:i + 500]
        params = {"instrument_key": ",".join(chunk), "interval": "1d"}
        resp = requests.get(BULK_OHLC_URL, headers=headers, params=params)
        resp.raise_for_status()
        payload = resp.json()

        for entry in payload.get("data", {}).values():
            live = entry.get("live_ohlc") or {}
            key = entry.get("instrument_token")
            if key and live:
                results[key] = {
                    "open": live.get("open"),
                    "high": live.get("high"),
                    "low": live.get("low"),
                    "close": live.get("close"),
                    "volume": live.get("volume"),
                }

    return results