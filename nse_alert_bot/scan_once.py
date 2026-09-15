"""
Runs ONE scan pass and exits - designed to be triggered on a schedule
by GitHub Actions instead of looping forever on your own machine.

Two-stage scan:
1. Cheap prefilter: one bulk OHLC API call across universe.txt (a
   broad candidate list) to find symbols trading near today's high
   on a green day - i.e. "near breakout" right now.
2. Full check: the existing EMA-cross and VWAP-retest logic, run only
   on watchlist.txt (always) plus whatever passed stage 1 (dynamic).

Unlike the BTC bot, this needs a fresh Upstox access token each day -
supply it via the UPSTOX_ACCESS_TOKEN GitHub secret (paste in the
token from `python auth/get_token.py` each trading morning). The
workflow's cron is scoped to NSE market hours, but this script also
checks the time itself as a safety net (e.g. if you trigger it
manually outside market hours).
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from data.instruments import get_instrument_key
from data.upstox_client import get_intraday_candles, get_bulk_ohlc
from strategy.prefilter import find_near_breakout
from strategy.screener import check_signal, check_vwap_retest, check_vwap_broad_TEST, evaluate
from alerts.telegram_bot import send_alert, format_signal_message, format_retest_message

IST = ZoneInfo("Asia/Kolkata")


def load_symbol_list(filename: str) -> list:
    try:
        with open(filename) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []


def is_market_hours() -> bool:
    now = datetime.now(IST)
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    active_from = open_t + timedelta(minutes=config.SKIP_FIRST_MINUTES)
    return active_from <= now <= close_t


def build_dynamic_shortlist(access_token: str) -> list:
    """Stage 1: cheap bulk-OHLC prefilter over universe.txt. Returns a
    list of trading symbols (not instrument_keys) that are near breakout."""
    universe = load_symbol_list("universe.txt")
    if not universe:
        return []

    symbol_to_key = {}
    for symbol in universe:
        try:
            symbol_to_key[symbol] = get_instrument_key(symbol)
        except ValueError as e:
            print(f"Skipping '{symbol}' in universe.txt: {e}")

    if not symbol_to_key:
        return []

    ohlc_map = get_bulk_ohlc(list(symbol_to_key.values()), access_token)
    passed_keys = set(find_near_breakout(ohlc_map))

    key_to_symbol = {v: k for k, v in symbol_to_key.items()}
    shortlist = [key_to_symbol[k] for k in passed_keys if k in key_to_symbol]
    print(f"Stage 1 prefilter: {len(shortlist)}/{len(universe)} near breakout: {shortlist}")
    return shortlist


def main():
    if not is_market_hours():
        print(f"Outside NSE market hours ({datetime.now(IST).strftime('%H:%M:%S')} IST) - skipping.")
        return

    if not config.UPSTOX_ACCESS_TOKEN:
        print("No UPSTOX_ACCESS_TOKEN env var set - add today's token as a GitHub secret.")
        return

    access_token = config.UPSTOX_ACCESS_TOKEN

    watchlist = load_symbol_list("watchlist.txt")
    dynamic_shortlist = build_dynamic_shortlist(access_token)

    # watchlist is always scanned; the dynamic shortlist adds today's
    # near-breakout candidates on top, deduped, watchlist order first.
    seen = set(watchlist)
    combined = list(watchlist) + [s for s in dynamic_shortlist if not (s in seen or seen.add(s))]

    for symbol in combined:
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

            # retest = check_vwap_retest(symbol, df)
            # if retest:
            #     message = format_retest_message(retest)
            #     send_alert(message)
            #     print(f">>> VWAP-RETEST ALERT SENT: {symbol} ({retest.aggressor})")

            retest = check_vwap_broad_TEST(symbol, df)
            if retest:
                message = format_retest_message(retest)
                send_alert(message)
                print(f">>> [TEST] BROAD VWAP ALERT SENT: {symbol} ({retest.aggressor})")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    main()
