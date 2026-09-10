import os
from dotenv import load_dotenv

load_dotenv()

# --- Upstox credentials ---
# Preferred: a long-lived (1-year) read-only Analytics Access Token,
# generated from account.upstox.com/developer/apps -> Analytics tab ->
# Generate Token. No daily login needed - set it once here and in the
# UPSTOX_ACCESS_TOKEN GitHub secret and forget about it until it expires.
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

# Fallback: the daily-refresh OAuth flow (Algo Trading app type), only
# used by auth/get_token.py if you're not using an Analytics token.
UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "https://127.0.0.1:5000/")
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "access_token.json")

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Strategy parameters ---
EMA_PERIOD = 20
VOLUME_AVG_PERIOD = 20
VOLUME_MULTIPLIER = 2.0          # signal candle volume must be > 2x the avg
CANDLE_INTERVAL_MINUTES = 5
RISK_REWARD_RATIO = 1.5
MAX_RISK_PCT = 0.015             # skip signal if stop-loss implies >1.5% risk

# --- VWAP retest-for-long parameters ---
RETEST_TREND_LOOKBACK = 5        # how many prior candles to check for an established above-VWAP trend
RETEST_MIN_CANDLES_ABOVE = 3     # at least this many of those prior candles must close above VWAP
RETEST_TOUCH_BUFFER_PCT = 0.001  # 0.1% buffer - counts as "touching" VWAP even if it doesn't hit exactly

# --- Runtime ---
POLL_SECONDS = 60                # how often the loop checks for new candles
SKIP_FIRST_MINUTES = 15          # ignore signals in first 15 min after market open
