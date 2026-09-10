"""
Runs ONE scan pass across the watchlist and exits - designed to be
triggered on a schedule (e.g. every 5 minutes during NSE market hours)
by GitHub Actions, mirroring the BTC bot's scan_once.py.

Auth: prefers a long-lived (1-year) read-only Analytics Access Token
(account.upstox.com/developer/apps -> Analytics tab -> Generate Token),
set as config.UPSTOX_ACCESS_TOKEN / the UPSTOX_ACCESS_TOKEN GitHub
secret - no daily refresh needed. Falls back to the daily-refresh OAuth
flow's access_token.json for local runs if that token isn't set.
"""
import config
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data.instruments import get_instrument_key
from data.upstox_client import get_intraday_candles
from strategy.screener import check_signal, check_vwap_retest, evaluate
from alerts.telegram_bot import send_alert, format_signal_message, format_retest_message

IST = ZoneInfo("Asia/Kolkata")


def load_access_token() -> str:
    if config.UPSTOX_ACCESS_TOKEN:
        return config.UPSTOX_ACCESS_TOKEN
    import json
    with open(config.TOKEN_FILE) as f:
        return json.load(f)["access_token"]


def load_watchlist() -> list:
    with open("watchlist.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def is_market_hours() -> bool:
    now = datetime.now(IST)
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    active_from = open_t + timedelta(minutes=config.SKIP_FIRST_MINUTES)
    return active_from <= now <= close_t


def main():
    now = datetime.now(IST)
    if not is_market_hours():
        print(f"[{now.strftime('%H:%M:%S')} IST] Outside market hours - skipping this pass.")
        return

    access_token = load_access_token()
    watchlist = load_watchlist()

    for symbol in watchlist:
        try:
            instrument_key = get_instrument_key(symbol)
            df = get_intraday_candles(instrument_key, config.CANDLE_INTERVAL_MINUTES, access_token)

            status = evaluate(symbol, df)
            print(f"{symbol}: {status}")

            signal = check_signal(symbol, df)
            if signal:
                message = format_signal_message(signal)
                send_alert(message)
                print(f">>> EMA-CROSS ALERT SENT: {symbol}")

            retest = check_vwap_retest(symbol, df)
            if retest:
                message = format_retest_message(retest)
                send_alert(message)
                print(f">>> VWAP-RETEST ALERT SENT: {symbol} ({retest.aggressor})")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    main()
