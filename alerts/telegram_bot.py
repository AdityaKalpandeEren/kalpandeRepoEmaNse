import requests
import config


def send_alert(message: str):
    """Production alert to the main chat and every TELEGRAM_EXTRA_CHAT_IDS receiver."""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    chats = list(dict.fromkeys([c for c in [config.TELEGRAM_CHAT_ID, *config.TELEGRAM_EXTRA_CHAT_IDS] if c]))
    for chat in chats:
        try:
            resp = requests.post(url, data={"chat_id": chat, "text": message, "parse_mode": "Markdown"}, timeout=20)
            if resp.status_code != 200:
                print(f"Telegram send to {chat} failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Telegram send to {chat} error: {e!r}")


def format_signal_message(signal) -> str:
    risk = signal.entry - signal.stop_loss
    reward = signal.target - signal.entry
    rr = reward / risk if risk else 0
    return (
        f"🟢 *{signal.symbol}* — [EMA CROSS] Long setup\n"
        f"Entry: ₹{signal.entry}\n"
        f"Stop Loss: ₹{signal.stop_loss}\n"
        f"Target: ₹{signal.target}\n"
        f"R:R ≈ 1:{rr:.1f}\n"
        f"Reason: {signal.reason}\n"
        f"Candle time (IST): {signal.candle_time}\n\n"
        f"_Manual trade only — bot does not place orders._"
    )


def format_retest_message(signal, label: str = "VWAP RETEST") -> str:
    risk = signal.entry - signal.stop_loss
    reward = signal.target - signal.entry
    rr = reward / risk if risk else 0
    aggressor_emoji = "🟢" if signal.aggressor == "BUY" else "🔴" if signal.aggressor == "SELL" else "⚪"
    return (
        f"🔵 *{signal.symbol}* — [{label}] VWAP ({signal.aggressor} aggressor)\n"
        f"Entry: ₹{signal.entry}\n"
        f"Stop Loss: ₹{signal.stop_loss}\n"
        f"Target: ₹{signal.target}\n"
        f"VWAP: ₹{signal.vwap}\n"
        f"R:R ≈ 1:{rr:.1f}\n"
        f"{aggressor_emoji} Last candle aggressor: *{signal.aggressor}* ({signal.buy_share_pct}% buy)\n"
        f"Buy vol: {signal.buy_volume:.2f} | Sell vol: {signal.sell_volume:.2f}\n"
        f"Candle time (IST): {signal.candle_time}\n\n"
        f"_Manual trade only — bot does not place orders._"
    )
