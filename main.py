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
from paper_trading import tracker as paper_tracker

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
    close_t = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    active_from = open_t + timedelta(minutes=config.SKIP_FIRST_MINUTES)
    return active_from <= now <= close_t


def is_at_or_past_close(now: datetime) -> bool:
    close_t = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return now >= close_t


def main():
    init_db()
    if config.PAPER_TRADING_ENABLED:
        paper_tracker.init_db()
        print("Paper trading: ON - every signal will also open/track a virtual trade "
              "in paper_trades.db (run `python -m paper_trading.generate_report` anytime for accuracy).")

    access_token = load_access_token()
    watchlist = load_watchlist()
    # Two independent "already alerted today" sets, so an EMA-cross alert
    # for a symbol doesn't block a later VWAP-retest alert for the same
    # symbol (and vice versa) - same one-alert-per-symbol-per-signal-per-day
    # rule the BTC bot's cooldown achieves via timestamps.
    ema_alerted_today = set()
    retest_alerted_today = set()
    today = datetime.now(IST).date()
    eod_squared_off_today = False

    print(f"Watching {len(watchlist)} symbols: {watchlist}")

    while True:
        now = datetime.now(IST)

        if now.date() != today:
            ema_alerted_today.clear()
            retest_alerted_today.clear()
            eod_squared_off_today = False
            today = now.date()
            access_token = load_access_token()  # re-read for the daily-refresh flow; a no-op if using a static Analytics token

        if not is_market_hours():
            # Right after market close, square off any paper trades still
            # OPEN from today before we start sleeping through the close.
            if config.PAPER_TRADING_ENABLED and not eod_squared_off_today and is_at_or_past_close(now):
                for symbol in watchlist:
                    try:
                        instrument_key = get_instrument_key(symbol)
                        df = get_intraday_candles(instrument_key, config.CANDLE_INTERVAL_MINUTES, access_token)
                        if not df.empty:
                            paper_tracker.square_off_eod(symbol, df.iloc[-1])
                    except Exception as e:
                        print(f"[{now.strftime('%H:%M:%S')}] EOD square-off error for {symbol}: {e}")
                eod_squared_off_today = True
                print(f"[{now.strftime('%H:%M:%S')}] Paper trading: squared off any open positions for today.")

            print(f"[{now.strftime('%H:%M:%S')}] Outside market hours, sleeping...")
            time.sleep(60)
            continue

        trade_date = now.date().isoformat()

        for symbol in watchlist:
            try:
                instrument_key = get_instrument_key(symbol)
                df = get_intraday_candles(instrument_key, config.CANDLE_INTERVAL_MINUTES, access_token)
                if df.empty:
                    continue

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

                # Evaluated once per poll regardless of alert-dedup state, so
                # paper trading (its own independent dedup, per trade_date)
                # never misses a signal just because a Telegram alert for it
                # already went out earlier today.
                signal = check_signal(symbol, df)
                retest = check_vwap_retest(symbol, df)
                last_row = df.iloc[-1]

                if signal and symbol not in ema_alerted_today:
                    message = format_signal_message(signal)
                    send_alert(message)
                    log_alert(signal)
                    ema_alerted_today.add(symbol)
                    print(f"[{now.strftime('%H:%M:%S')}] >>> EMA-CROSS ALERT SENT: {symbol}")

                if retest and symbol not in retest_alerted_today:
                    message = format_retest_message(retest)
                    send_alert(message)
                    retest_alerted_today.add(symbol)
                    print(f"[{now.strftime('%H:%M:%S')}] >>> VWAP-RETEST ALERT SENT: {symbol} ({retest.aggressor})")

                if config.PAPER_TRADING_ENABLED:
                    # 1) fill anything still pending from an earlier candle
                    paper_tracker.fill_pending(symbol, last_row)
                    # 2) check anything currently open against this candle
                    paper_tracker.check_open_trades(symbol, last_row)
                    # 3) register any new signal as a pending virtual trade
                    if signal and not paper_tracker.has_open_or_pending(symbol, "EMA_CROSS", trade_date):
                        paper_tracker.open_pending(symbol, "EMA_CROSS", signal, last_row["timestamp"], trade_date)
                    if retest and not paper_tracker.has_open_or_pending(symbol, "VWAP_RETEST", trade_date):
                        paper_tracker.open_pending(symbol, "VWAP_RETEST", retest, last_row["timestamp"], trade_date)

            except Exception as e:
                print(f"[{now.strftime('%H:%M:%S')}] Error processing {symbol}: {e}")

        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
