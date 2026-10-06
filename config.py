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

# --- Dynamic near-breakout prefilter (stage 1 of the two-stage scanner) ---
BREAKOUT_MAX_PCT_FROM_HIGH = 0.005   # within 0.5% of today's high counts as "near breakout"
BREAKOUT_MIN_PCT_FROM_OPEN = 0.0     # must be a green day (close >= open) to qualify

# --- Production alerts (scan_once.py EMA-cross + broad VWAP test) ---
# Paused for now so only the research strategies (LIVE_RESEARCH_*) alert.
# Set PRODUCTION_ALERTS_ENABLED=true (env / GitHub repo variable) to resume.
PRODUCTION_ALERTS_ENABLED = os.getenv("PRODUCTION_ALERTS_ENABLED", "false").lower() == "true"

# --- Runtime ---
POLL_SECONDS = 60               # how often the loop checks for new candles
SKIP_FIRST_MINUTES = 15          # ignore signals in first 15 min after market open

# --- Paper trading (main.py only - see paper_trading/tracker.py) ---
# When on, every signal main.py generates also opens a virtual position
# that gets tracked forward (filled on the next candle's open, closed on
# target/stop/EOD) into paper_trades.db, so you can run
# `python -m paper_trading.generate_report` to see real accuracy without
# risking money. Needs main.py running continuously - see that module's
# docstring for why.
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"

MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

MARKET_TIMEZONE = "Asia/Kolkata"
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15


# ═══════════════════════════════════════════════════════════════════
# RESEARCH ENGINE (strategy/strategies.py, strategy/regime.py,
# backtest/run_research.py) - ported from us_alert_bot.
# Used by the research backtest and by the live research alerts/paper
# trading block further down. The production EMA-cross / VWAP alerts
# (strategy/screener.py) read none of this.
# ═══════════════════════════════════════════════════════════════════

# --- EMA structure ---
EMA_FAST = 9
EMA_SLOW = 21
EMA_STACK_FAST = 10
EMA_STACK_MID = 20
EMA_STACK_SLOW = 50
EMA_STACK_MIN_RVOL = 1.2
EMA_STACK_MAX_EXTENSION_ATR = 3.0

# --- J: VWAP standard-deviation band reversion (long only) ---
VWAP_BAND_Z_THRESHOLD = 2.0
VWAP_BAND_MIN_RVOL = 1.0

# --- K: Connors RSI(2) mean reversion (long only) ---
RSI2_PERIOD = 2
RSI2_OVERSOLD = 10.0
RSI2_TREND_FILTER_EMA = EMA_STACK_SLOW

# --- Volatility ---
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
STRUCT_STOP_BUFFER_ATR = 0.25
ATR_PERCENTILE_LOOKBACK = 50

# --- Structure / retest ---
STRUCT_LOOKBACK = 10
RETEST_WINDOW = 5
RETEST_TOUCH_ATR = 0.20
OPENING_RANGE_MINUTES = 30

# --- VWAP reclaim / confluence / aggressor / rejection candle quality ---
VWAP_RECLAIM_LOOKBACK = 6
VWAP_RECLAIM_MIN_BARS = 4
CONFLUENCE_MAX_ATR = 0.5
VWAP_MAX_AGGRESSOR = 0.95
VWAP_REJECTION_MIN_RVOL = 1.0
VWAP_REJECTION_BODY_MIN = 0.3
VWAP_REJECTION_BODY_MAX = 0.7

# --- Regime classifier ---
REGIME_FILTER_ENABLED = os.getenv("REGIME_FILTER_ENABLED", "true").lower() == "true"
REGIME_BLOCK_HIGH_VOL = True
REGIME_ATR_HIGH_PCT = 0.85
REGIME_ATR_LOW_PCT = 0.20
REGIME_SLOPE_MIN = 0.0002
REGIME_EMA_SEPARATION_MIN = 0.001

# --- Short asymmetry ---
SHORT_STOP_ATR_MULT_EXTRA = 0.25
SHORT_RVOL_EXTRA = 0.5
SHORT_MIN_SESSION_CANDLES = 12

# --- Scoring engine ---
SCORE_WEIGHTS = {
    "trend": 0.25, "momentum": 0.15, "volume": 0.15,
    "vwap": 0.20, "volatility": 0.10, "price_action": 0.15,
}
SCORE_MIN_TOTAL = 55.0
SCORE_MIN_COMPONENT = 0.10
SCORE_FLOOR_COMPONENTS = ("trend", "vwap")
SCORE_SEPARATION_FULL = 0.004
SCORE_SLOPE_FULL = 0.0008
SCORE_VWAP_IDEAL_Z = 1.0
SCORE_VWAP_FADE_Z = 2.5
SCORE_ATR_IDEAL = 0.5

# --- Research backtest mechanics ---
# NSE DIFFERENCE: the US bot scans from 4:00 AM pre-market, so its 25-
# candle warm-up is long over by the 9:30 open. NSE has no extended
# session - 25 candles from 9:15 would mean nothing fires before 11:20.
# 12 candles = first possible signal on the 10:15 candle.
MIN_WARMUP_CANDLES = int(os.getenv("MIN_WARMUP_CANDLES", "12"))
MAX_TRADES_PER_SYMBOL_DAY = 3
SIGNAL_COOLDOWN_CANDLES = 6

# NSE DIFFERENCE: intraday (MIS) positions are auto-squared-off by the
# broker around 15:15-15:20, so research trades must be out before then,
# not at the 15:30 close. The research engine only sees candles that
# close at or before this time; the last one is the square-off candle.
RESEARCH_SQUAREOFF_HOUR = 15
RESEARCH_SQUAREOFF_MINUTE = 15

# --- Costs (applied to every simulated research trade) ---
# NSE DIFFERENCE: much higher than the US bot's 2+1 bps. Intraday equity
# per side, for a ~Rs 1 lakh position: brokerage ~Rs 20/order (~2 bps),
# STT 0.025% on the SELL side only (2.5 bps -> ~1.25 bps averaged per
# side), exchange txn ~0.3 bps, stamp duty 0.3 bps on the buy side, GST
# 18% on brokerage+txn. ~4 bps per side all-in, plus slippage. Smaller
# positions pay proportionally more brokerage - raise COMMISSION_BPS if
# you trade small size.
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "2.0"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "4.0"))

# --- Position sizing (reporting only; R-multiples are size-agnostic) ---
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "100000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.005"))


# ═══════════════════════════════════════════════════════════════════
# ML META-LABEL FILTER V1 (model L_ML_META) - see us_alert_bot for the
# design. Trained on THIS bot's NSE data; never reuse the US model file.
# ═══════════════════════════════════════════════════════════════════
ML_META_MODEL_PATH = "backtest/ml/model/meta_model.joblib"
ML_META_MIN_PROB = float(os.getenv("ML_META_MIN_PROB", "0.55"))
ML_META_TEST_FRAC = 0.3


# ═══════════════════════════════════════════════════════════════════
# ML META-LABEL FILTER V2 (model L_ML_META_V2) - NSE version.
# Same design as the US V2 (strategy/market_context.py,
# strategy/news_catalyst.py, strategy/ml_features_v2.py), with India's
# market inputs instead of the US ones:
#   VIX/VXN/VIX3M -> India VIX (no term-structure equivalent, so no
#                    VIX/VIX3M feature or veto)
#   QQQ/SPY       -> NIFTY 50 and NIFTY BANK (index candles have no
#                    volume, so "index vs VWAP" becomes "index vs its own
#                    session open")
#   76 US names   -> NIFTY 50 constituents for breadth
# News is still LIVE-ONLY (no headline history), exactly as in the US bot.
# ═══════════════════════════════════════════════════════════════════
ML_V2_MODEL_PATH = "backtest/ml/model/meta_model_v2.joblib"
ML_V2_DATASET_PATH = "backtest/ml/data/dataset_v2.csv"
ML_V2_CACHE_DIR = "backtest/ml/cache"
ML_V2_MIN_PROB = os.getenv("ML_V2_MIN_PROB", "auto")

# --- Session gate (minutes after the 9:15 IST open) ---
ML_V2_SESSION_START_MIN = 15      # 9:30 - skip the opening volatility
ML_V2_SESSION_END_MIN = 330       # 14:45 - last entry, leaves 30 min before square-off
ML_V2_FLAT_MIN = 355              # 15:10 candle -> out at its 15:15 close (MIS square-off)

# --- Hard market-risk vetoes for longs ---
# India VIX normally sits ~10-20; above 25 is a stressed market.
ML_V2_VETO_ENABLED = os.getenv("ML_V2_VETO_ENABLED", "true").lower() == "true"
ML_V2_IVIX_MAX_LONG = 25.0
ML_V2_IVIX_DAY_SPIKE_MAX_LONG = 0.15   # India VIX up >15% on the day

# --- Dynamic "bad catalyst during the trade" exit ---
# NIFTY moves less intraday than QQQ, so the index threshold is lower
# than the US bot's 0.6%.
ML_V2_SHOCK_EXIT_ENABLED = os.getenv("ML_V2_SHOCK_EXIT_ENABLED", "true").lower() == "true"
ML_V2_SHOCK_NIFTY_PCT = 0.005     # NIFTY 0.5% against the trade since entry
ML_V2_SHOCK_IVIX_PCT = 0.08       # India VIX +8% since entry (longs) / -8% (shorts)

# --- Breadth universe: NIFTY 50 constituents (edit when the index changes) ---
ML_V2_BREADTH_UNIVERSE = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TMPV", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# --- Live news catalyst overlay (strategy/news_catalyst.py) ---
ML_V2_NEWS_ENABLED = os.getenv("ML_V2_NEWS_ENABLED", "true").lower() == "true"
ML_V2_NEWS_SYMBOL_LOOKBACK_HOURS = 24
ML_V2_NEWS_MARKET_LOOKBACK_HOURS = 8
ML_V2_NEWS_MARKET_TICKERS = ["^NSEI", "^NSEBANK"]
ML_V2_NEWS_VETO = 0.35
ML_V2_NEWS_MARKET_VETO = 0.45
ML_V2_NEWS_BOOST = 0.35
ML_V2_NEWS_BOOST_PROB = 0.03
ML_V2_NEWS_EXIT = 0.45
ML_V2_NEWS_CACHE_SECONDS = 600
# LLM scoring of V2's headlines (strategy/llm_client.py, provider below).
# Falls back to the keyword lexicon with no key / on any error.
ML_V2_NEWS_LLM_ENABLED = os.getenv("ML_V2_NEWS_LLM_ENABLED", "true").lower() == "true"

# --- Where NSE news comes from (strategy/india_news.py) ---
# "rss" = Indian RSS feeds + Google News (fresh); "yahoo" = the old Yahoo
# .NS / ^NSEI feeds, which were found days-to-weeks stale for NSE.
INDIA_NEWS_SOURCE = os.getenv("INDIA_NEWS_SOURCE", "rss").lower()
INDIA_NEWS_MARKET_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",           # ET Markets
    "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms",  # ET Stocks
    "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",      # ET Economy
    "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",            # ET Industry
    "https://www.business-standard.com/rss/markets-106.rss",                          # BS Markets
    "https://www.business-standard.com/rss/companies-101.rss",                        # BS Companies
    "https://www.livemint.com/rss/markets",                                           # Mint Markets
    "https://www.livemint.com/rss/companies",                                         # Mint Companies
]
INDIA_NEWS_GOOGLE_ENABLED = os.getenv("INDIA_NEWS_GOOGLE_ENABLED", "true").lower() == "true"
# How to find a company in the news when its registered (instrument
# master) name is abbreviated or ambiguous: symbol -> (Google News search
# phrase, word that must appear in a headline).
INDIA_NEWS_NAMES = {
    "DRREDDY": ("Dr Reddy's", "Reddy"), "DIVISLAB": ("Divi's Laboratories", "Divi"),
    "SUNPHARMA": ("Sun Pharma", "Sun Pharma"), "APOLLOHOSP": ("Apollo Hospitals", "Apollo Hospital"),
    # Names shared with foreign companies / people / words ("Trent Williams",
    # "Titan International", Siemens AG, "Willmott Dixon"): Indian names only.
    "HYUNDAI": ("Hyundai Motor India", "Hyundai Motor India"),
    "TITAN": ("Titan Company", "Titan Company", "Titan shares", "Titan share price"),
    "TRENT": ("Trent Ltd", "Trent Ltd", "Trent shares", "Trent share price", "Zudio"),
    "SIEMENS": ("Siemens Ltd", "Siemens Ltd", "Siemens India", "Siemens shares"),
    "DIXON": ("Dixon Technologies", "Dixon Tech"),
    "FORTIS": ("Fortis Healthcare", "Fortis Healthcare", "Fortis Hospital"),
    "ETERNAL": ("Eternal Ltd", "Eternal Ltd", "Eternal shares", "Eternal share price", "Zomato", "Blinkit"),
    "MEDANTA": ("Medanta", "Medanta"), "KIMS": ("KIMS Hospitals", "KIMS"),
    "LALPATHLAB": ("Dr Lal PathLabs", "Lal Path"), "MAXHEALTH": ("Max Healthcare", "Max Healthcare"),
    "ASTERDM": ("Aster DM Healthcare", "Aster DM"), "RAINBOW": ("Rainbow Children's Medicare", "Rainbow Children"),
    "ABBOTINDIA": ("Abbott India", "Abbott India"), "POLYMED": ("Poly Medicure", "Poly Medicure"),
    "TORNTPHARM": ("Torrent Pharma", "Torrent Pharma"), "ZYDUSLIFE": ("Zydus Lifesciences", "Zydus"),
    "KCPSUGIND": ("KCP Sugar", "KCP Sugar"), "AVADHSUGAR": ("Avadh Sugar", "Avadh Sugar"),
    "TRIVENI": ("Triveni Engineering", "Triveni"), "DALMIASUG": ("Dalmia Bharat Sugar", "Dalmia Bharat Sugar"),
    "EIDPARRY": ("EID Parry", "Parry"), "RENUKA": ("Shree Renuka Sugars", "Renuka"),
    "BAJAJHIND": ("Bajaj Hindusthan Sugar", "Bajaj Hindusthan"), "DWARKESH": ("Dwarikesh Sugar", "Dwarikesh"),
    "BANARISUG": ("Bannari Amman Sugars", "Bannari Amman"), "NH": ("Narayana Health", "Narayana"),
    "UTTAMSUGAR": ("Uttam Sugar", "Uttam Sugar"), "DHAMPURSUG": ("Dhampur Sugar", "Dhampur Sugar"),
    # Registered names that are too generic to match on ("One 97" -> "one",
    # "Union" -> Union Budget, "Solar", "Coal", "Info", "Multi", "APL" -> Apple...)
    "PAYTM": ("Paytm", "Paytm"), "UNIONBANK": ("Union Bank of India", "Union Bank"),
    "SOLARINDS": ("Solar Industries", "Solar Industries"), "COALINDIA": ("Coal India", "Coal India"),
    "NAUKRI": ("Info Edge", "Info Edge"), "MCX": ("MCX India", "Multi Commodity Exchange"),
    "APLAPOLLO": ("APL Apollo", "APL Apollo"), "NYKAA": ("Nykaa", "Nykaa"),
    "ONGC": ("ONGC", "Oil and Natural Gas"), "MOTHERSON": ("Samvardhana Motherson", "Motherson"),
    "CGPOWER": ("CG Power", "CG Power"), "LTF": ("L&T Finance", "L&T Finance"),
    "LT": ("Larsen & Toubro", "Larsen"), "TMPV": ("Tata Motors", "Tata Motors"),
    # Hot / near-breakout additions (2026-09-30) with generic first words
    "SSWL": ("Steel Strips Wheels", "Steel Strips"), "ENGINERSIN": ("Engineers India", "Engineers India"),
    "MAHSEAMLES": ("Maharashtra Seamless", "Maharashtra Seamless"),
    "AARTIPHARM": ("Aarti Pharmalabs", "Aarti Pharmalabs"), "FINCABLES": ("Finolex Cables", "Finolex Cables"),
    "RML": ("Rane Madras", "Rane"), "ABDL": ("Allied Blenders", "Allied Blenders"),
    "GCSL": ("Gretex Corporate", "Gretex"), "NRBBEARING": ("NRB Bearings", "NRB Bearing"),
    "WELCORP": ("Welspun Corp", "Welspun Corp"), "SOLARA": ("Solara Active Pharma", "Solara"),
    "ARTEMISMED": ("Artemis Hospitals", "Artemis"), "QUADFUTURE": ("Quadrant Future Tek", "Quadrant Future"),
    "SMSPHARMA": ("SMS Pharmaceuticals", "SMS Pharma"), "CUPID": ("Cupid Ltd", "Cupid"),
    # Healthcare momentum additions (2026-09-30)
    "STAR": ("Strides Pharma", "Strides"), "IOLCP": ("IOL Chemicals", "IOL Chemicals"),
    "INDGN": ("Indegene", "Indegene"), "JUBLPHARMA": ("Jubilant Pharmova", "Jubilant Pharmova"),
    "SAILIFE": ("Sai Life Sciences", "Sai Life"), "WOCKPHARMA": ("Wockhardt", "Wockhardt"),
    "CAPLIPOINT": ("Caplin Point", "Caplin"), "AKUMS": ("Akums Drugs", "Akums"),
    "MOREPENLAB": ("Morepen Laboratories", "Morepen"), "YATHARTH": ("Yatharth Hospital", "Yatharth"),
    "SHILPAMED": ("Shilpa Medicare", "Shilpa Medicare"), "MARKSANS": ("Marksans Pharma", "Marksans"),
    "EMCURE": ("Emcure Pharmaceuticals", "Emcure"), "LENSKART": ("Lenskart", "Lenskart"),
}

# --- LLM for the news / catalyst readers (V2 and V3) ---
# "gemini" (default): Google AI Studio free tier, GEMINI_API_KEY.
# "anthropic": Claude (paid), ANTHROPIC_API_KEY + a Claude model id below.
NEWS_LLM_PROVIDER = os.getenv("NEWS_LLM_PROVIDER", "gemini").lower()
# flash-lite: higher free-tier limits, and in live testing (2026-09-29) it
# answered while gemini-flash-latest returned 503 "high demand" on every
# call. On a 500/503 the fallback model is tried once before giving up.
NEWS_LLM_MODEL = os.getenv("NEWS_LLM_MODEL", "gemini-flash-lite-latest")
NEWS_LLM_FALLBACK_MODEL = os.getenv("NEWS_LLM_FALLBACK_MODEL", "gemini-flash-latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Training ---
ML_V2_VAL_FRAC = 0.2
ML_V2_TEST_FRAC = 0.2
ML_V2_MIN_TRADES_FOR_THRESHOLD = 30
ML_V2_MIN_DAYS_FOR_THRESHOLD = 4
ML_V2_MAX_TRADES_PER_DAY = int(os.getenv("ML_V2_MAX_TRADES_PER_DAY", "10"))
# Day-level regime values kept OUT of the classifier (see us_alert_bot
# config for why: on short histories they act as date labels). The NSE
# bot has far more history (Upstox serves years of 5-min candles), so this
# list can be revisited once a long dataset is built.
ML_V2_EXCLUDE_FEATURES = [
    "ivix", "ivix_vs_sma50", "breadth_pct50", "breadth_chg5",
    "nifty_vs_sma50_d", "nifty_vs_sma200_d",
]


# ═══════════════════════════════════════════════════════════════════
# LIVE RESEARCH ALERTS + PAPER TRADING (paper_trading/research_live.py,
# hooked into scan_once.py after the production alerts)
# ═══════════════════════════════════════════════════════════════════
LIVE_RESEARCH_ENABLED = os.getenv("LIVE_RESEARCH_ENABLED", "true").lower() == "true"
LIVE_RESEARCH_STRATEGIES = [s.strip() for s in os.getenv(
    "LIVE_RESEARCH_STRATEGIES",
    "K_RSI2_REVERSION,SCORE_ENGINE,L_ML_META,L_ML_META_V2,L_ML_META_V3,J_VWAP_BAND_REVERSION",
).split(",") if s.strip()]
# Strategies still paper-traded and shown in the EOD report, but with no
# Telegram entry/exit alerts. SCORE_ENGINE muted 2026-10-01 (monitor only);
# remove it from this default to alert again.
LIVE_RESEARCH_SILENT_STRATEGIES = [s.strip() for s in os.getenv(
    "LIVE_RESEARCH_SILENT_STRATEGIES", "SCORE_ENGINE").split(",") if s.strip()]
LIVE_RESEARCH_DIRECTIONS = [d.strip() for d in os.getenv(
    "LIVE_RESEARCH_DIRECTIONS", "long").split(",") if d.strip()]
LIVE_RESEARCH_TELEGRAM = os.getenv("LIVE_RESEARCH_TELEGRAM", "true").lower() == "true"
LIVE_RESEARCH_CHAT_ID = os.getenv("LIVE_RESEARCH_CHAT_ID", "")
LIVE_STATE_DIR = os.getenv("LIVE_STATE_DIR", "live_state")
LIVE_ALERT_MAX_AGE_MIN = 15
LIVE_RESEARCH_V2_LIVE_CONTEXT = os.getenv("LIVE_RESEARCH_V2_LIVE_CONTEXT", "true").lower() == "true"
# EOD report goes out on the first run at/after this time (IST). Research
# trades are all out by the 15:15 square-off (the last research candle
# closes then), so 15:20 is final - and it lands inside the existing
# 9:15-15:30 workflow schedule (the 15:20 / 15:25 / 15:30 runs), no extra
# cron trigger needed.
LIVE_EOD_REPORT_HOUR = 15
LIVE_EOD_REPORT_MINUTE = 20


# ═══════════════════════════════════════════════════════════════════
# ML MODEL V3 (model L_ML_META_V3) - NSE only.
# strategy/sector_map.py, strategy/market_context_v3.py,
# strategy/v3_candidates.py, strategy/ml_features_v3.py,
# strategy/llm_catalyst.py, backtest/ml/*_v3.py
#
# V2 plus four things, each aimed at a gap the 12-month V2 run showed:
#  1. SECTOR tape: the stock's own sector index (14 NSE sector indices),
#     stock-vs-sector and sector-vs-NIFTY strength.
#  2. DYNAMIC MACRO LINKS: overnight Brent crude / USD-INR / S&P 500 /
#     US 10y / gold moves, multiplied by each sector's OWN recent
#     sensitivity to them (a rolling 120-day beta, recomputed daily,
#     point-in-time) - "crude up" means something different for ONGC,
#     BPCL and INFY, and that difference is learned, not hard-coded.
#  3. RELATIVE-STRENGTH BREAKOUT candidates that fire even on a red
#     market day (new day high / 20-day-high break while beating NIFTY
#     and the sector), on top of the 12 base models' candidates.
#  4. COST-AWARE GEOMETRY: every V3 trade's stop is at least
#     ML_V3_MIN_STOP_PCT away. Pure arithmetic from the NSE cost model -
#     at ~12 bps round trip, a 0.2% stop pays ~0.6R in charges.
# Plus a LIVE-ONLY Claude catalyst reader (policy/regulation/commodity
# news -> affected industries and stocks). Like V2's news, it cannot be
# backtested (no headline archive), so it is an overlay, not a feature.
# ═══════════════════════════════════════════════════════════════════
ML_V3_MODEL_PATH = "backtest/ml/model/meta_model_v3.joblib"
ML_V3_DATASET_PATH = "backtest/ml/data/dataset_v3.csv"
# Bar lowered from the model's own 0.675 ("auto") to 0.60 on 2026-10-06 (user):
# at 0.675 V3 never traded live (max live p ~0.58-0.62). On the 4 sessions the
# model never saw (Sep 28 - Oct 1): 0.60 -> 5 trades, ~1/day, -0.18R avg vs
# -0.36R for every candidate. Paper only; "auto" restores the model's bar.
ML_V3_MIN_PROB = os.getenv("ML_V3_MIN_PROB", "0.60")
ML_V3_MIN_STOP_PCT = float(os.getenv("ML_V3_MIN_STOP_PCT", "0.005"))   # 0.5%
ML_V3_BETA_WINDOW = 120            # trading days for the rolling sector betas
ML_V3_BETA_MIN_OBS = 60

# Relative-strength breakout candidates (strategy/v3_candidates.py)
ML_V3_RS_MIN_RVOL = 1.2
ML_V3_RS_20D_MIN_RVOL = 1.5

# Macro series (Yahoo daily; used as of the PRIOR US session only)
ML_V3_MACRO_TICKERS = {"BRENT": "BZ=F", "USDINR": "USDINR=X", "SPX": "^GSPC",
                       "US10Y": "^TNX", "GOLD": "GC=F"}

# Manual sector overrides (symbol -> sector code) for names the automatic
# industry mapping gets wrong. Codes: see strategy/sector_map.py SECTORS.
ML_V3_SECTOR_OVERRIDES = {
    "SBIN": "PSUBANK", "BANKBARODA": "PSUBANK", "PNB": "PSUBANK", "CANBK": "PSUBANK",
    "UNIONBANK": "PSUBANK", "BANKINDIA": "PSUBANK", "INDIANB": "PSUBANK",
    "IOB": "PSUBANK", "CENTRALBK": "PSUBANK", "UCOBANK": "PSUBANK", "MAHABANK": "PSUBANK",
}

# --- Live catalyst reader (strategy/llm_catalyst.py) ---
# "auto" = use the news LLM (NEWS_LLM_PROVIDER above - Gemini by default)
# when its key is set, the built-in sector rule table otherwise. "false"
# forces rules only.
ML_V3_LLM = os.getenv("ML_V3_LLM", "auto").lower()
ML_V3_NEWS_ENABLED = os.getenv("ML_V3_NEWS_ENABLED", "true").lower() == "true"
ML_V3_NEWS_CACHE_SECONDS = 900
ML_V3_STOCK_VETO = 0.35            # combined stock catalyst <= -this -> no long
ML_V3_SECTOR_VETO = 0.40           # industry/sector impact <= -this -> no long
ML_V3_BOOST_AT = 0.35              # combined catalyst >= this ...
ML_V3_BOOST_PROB = 0.05            # ... lowers the probability bar by up to this


# ═══════════════════════════════════════════════════════════════════
# NSE SWING / POSITIONAL RESEARCH (strategy/swing_nse.py,
# backtest/run_swing_nse.py) - DAILY candles, multi-day holds.
#
# Why this exists: the 12-month intraday research (see
# docs/NSE_RESEARCH_ML.md) found every 5-min long setup loses 0.15-0.30R
# per trade after NSE intraday charges, and no ML filter (V1/V2/V3) could
# fix that. A multi-day swing trade carries a larger move per rupee of
# charges, and the crude/sector/policy catalysts V3 tried to read play
# out over days, not 5-minute bars.
#
# All strategy parameters below are standard textbook values fixed BEFORE
# any backtest - not tuned on the data - so the multi-year result is an
# honest test rather than a curve fit.
# ═══════════════════════════════════════════════════════════════════
SWING_FROM = os.getenv("SWING_FROM", "2015-01-01")

# --- SWING_DAYS_STR (identical to us_alert_bot's) ---
SWING_EMA_FAST = 30
SWING_EMA_MID = 50
SWING_EMA_SLOW = 60
SWING_STRUCT_LOOKBACK = 50
SWING_VOLUME_AVG_PERIOD = 50
SWING_MIN_VOLUME_RATIO = 1.5
SWING_MIN_ABOVE_52W_LOW_PCT = 0.25
SWING_MAX_BELOW_52W_HIGH_PCT = 0.25
SWING_TARGET_PCT = 0.08
SWING_STOP_PCT = 0.02
SWING_MAX_HOLD_DAYS = 120

# --- shared swing mechanics ---
SWING_ATR_PERIOD = 14
SWING_RS_LOOKBACK = 63             # ~3 months of sessions for relative strength
SWING_WARMUP_DAYS = 210            # SMA200 + slack before any signal
# NSE DELIVERY costs, per side: STT 0.1% (buy AND sell), stamp 0.015%
# (buy), exchange+SEBI ~0.003%, brokerage ~Rs 20/order, DP ~Rs 15-20 per
# sell, GST, plus ~5 bps slippage on an open fill. ~0.20% per side,
# ~0.40% round trip.
SWING_COST_PCT_PER_SIDE = float(os.getenv("SWING_COST_PCT_PER_SIDE", "0.002"))

# --- portfolio simulation ---
SWING_MAX_POSITIONS = int(os.getenv("SWING_MAX_POSITIONS", "10"))
SWING_START_EQUITY = float(os.getenv("SWING_START_EQUITY", "1000000"))


# ═══════════════════════════════════════════════════════════════════
# L_ML_V32 - daily 9:45 cross-sectional ranking model (NSE, TESTING ONLY)
# strategy/v32_features.py, backtest/ml/build_dataset_v32.py,
# backtest/ml/train_v32.py. Not wired into live alerts/paper trading.
# ═══════════════════════════════════════════════════════════════════
ML_V32_DATASET_PATH = "backtest/ml/data/dataset_v32.csv"
ML_V32_MODEL_PATH = "backtest/ml/model/model_v32.joblib"
ML_V32_TOP_K_OPTIONS = (3, 5)       # stocks bought per day; chosen on validation

# V3.3 = V3.2 + NSE delivery / F&O OI-PCR / corporate events (formerly "V3.2+")
ML_V33_DATASET_PATH = "backtest/ml/data/dataset_v33.csv"
ML_V33_MODEL_PATH = "backtest/ml/model/model_v33.joblib"

# --- V3.3 live paper trading (live_v33.py) ---
V33_NOTIONAL_PER_STOCK = float(os.getenv("V33_NOTIONAL_PER_STOCK", "500000"))   # Rs per pick (paper)
V33_BROKERAGE_PER_ORDER = float(os.getenv("V33_BROKERAGE_PER_ORDER", "20"))     # Rs, flat
V33_SLIPPAGE_BPS = float(os.getenv("V33_SLIPPAGE_BPS", "2"))                     # per side, assumed
V33_TOP_K = int(os.getenv("V33_TOP_K", "0"))       # 0 = use the K chosen on validation (model bundle)
V33_STATE_DIR = os.getenv("V33_STATE_DIR", "live_state/v33")
