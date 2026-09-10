import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from data.instruments import get_instrument_key
from data.upstox_client import get_intraday_candles
from strategy.screener import check_signal, check_vwap_retest, evaluate
from alerts.telegram_bot import send_alert, format_signal_message, format_retest_message
from alerts.logger import init_db, log_alert

IST = ZoneInfo("Asia/Kolkata")


def load_access_token() -> str:
    # Prefer the long-lived Analytics token - no daily refresh needed.
    if config.UPSTOX_ACCESS_TOKEN:
        return config.UPSTOX_ACCESS_TOKEN
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
    init_db()
    access_token = load_access_token()
    watchlist = load_watchlist()
    # Two independent "already alerted today" sets, so an EMA-cross alert
    # for a symbol doesn't block a later VWAP-retest alert for the same
    # symbol (and vice versa) - same one-alert-per-symbol-per-signal-per-day
    # rule the BTC bot's cooldown achieves via timestamps.
    ema_alerted_today = set()
    retest_alerted_today = set()
    today = datetime.now(IST).date()

    print(f"Watching {len(watchlist)} symbols: {watchlist}")

    while True:
        now = datetime.now(IST)

        if now.date() != today:
            ema_alerted_today.clear()
            retest_alerted_today.clear()
            today = now.date()
            access_token = load_access_token()  # re-read for the daily-refresh flow; a no-op if using a static Analytics token

        if not is_market_hours():
            print(f"[{now.strftime('%H:%M:%S')}] Outside market hours, sleeping...")
            time.sleep(60)
            continue

        for symbol in watchlist:
            try:
                instrument_key = get_instrument_key(symbol)
                df = get_intraday_candles(instrument_key, config.CANDLE_INTERVAL_MINUTES, access_token)

                status = evaluate(symbol, df)
                if status["status"] == "OK":
                    print(
                        f"[{now.strftime('%H:%M:%S')}] {symbol}: price={status['price']} "
                        f"ema={status['ema']} vwap={status['vwap']} "
                        f"cross={status['ema_cross']} above_vwap={status['above_vwap']} "
                        f"vol_ratio={status['vol_ratio']}x (need >{status['vol_needed']}x)"
                    )
                else:
                    print(f"[{now.strftime('%H:%M:%S')}] {symbol}: {status['status']}")

                if symbol not in ema_alerted_today:
                    signal = check_signal(symbol, df)
                    if signal:
                        message = format_signal_message(signal)
                        send_alert(message)
                        log_alert(signal)
                        ema_alerted_today.add(symbol)
                        print(f"[{now.strftime('%H:%M:%S')}] >>> EMA-CROSS ALERT SENT: {symbol}")

                if symbol not in retest_alerted_today:
                    retest = check_vwap_retest(symbol, df)
                    if retest:
                        message = format_retest_message(retest)
                        send_alert(message)
                        retest_alerted_today.add(symbol)
                        print(f"[{now.strftime('%H:%M:%S')}] >>> VWAP-RETEST ALERT SENT: {symbol} ({retest.aggressor})")
            except Exception as e:
                print(f"[{now.strftime('%H:%M:%S')}] Error processing {symbol}: {e}")

        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
