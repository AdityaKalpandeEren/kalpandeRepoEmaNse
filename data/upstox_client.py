import requests
import pandas as pd

BASE_URL = "https://api.upstox.com/v3/historical-candle/intraday"


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
    # Upstox returns newest-first; put it in chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df
