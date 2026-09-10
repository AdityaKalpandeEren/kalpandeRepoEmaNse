# NSE Trade Alert Bot (Telegram)

Scans a watchlist of NSE stocks every 5-min candle for two independent
long setups and sends you a Telegram message with entry / stop-loss /
target for each. **It never places trades — you execute manually.**

1. **EMA-cross** — EMA(20) crossover + price above VWAP + volume > 2x average.
2. **VWAP retest** — price has been trading above session VWAP (established
   uptrend), the candle dips to touch VWAP as support and closes back above
   it on a bullish candle. The alert also reports the candle's buy/sell
   volume split and which side dominated ("aggressor"), using the same
   geometric close-position-in-range method as the "Geometric" engine in
   the companion Volume Footprint TradingView indicator.

Each setup fires **at most once per symbol per day**, independently of
the other.

Everything used here is free (no subscriptions): your existing Upstox
trading account gives free API market-data access, and Telegram bots
are free.

---

## 1. Get an Upstox access token

Two ways to authenticate — pick one:

**Option A: Analytics Access Token (recommended — do this once, done for a year)**
1. Go to https://account.upstox.com/developer/apps → **Analytics** tab.
2. Click **Generate Token**. This is a long-lived (1-year), read-only
   token that covers Market Data APIs (candles) — exactly what this bot
   needs, with no daily login.
3. Copy it immediately — it's shown in full only once.

**Option B: Algo Trading app + daily login (only if you also need order-placement APIs later)**
1. Go to https://developer.upstox.com and log in with your Upstox account.
2. Create a new app of type "Algo Trading". You'll be asked for a
   **redirect URI** — just use `https://127.0.0.1:5000/` (it doesn't
   need to be a real running server; you just need the URL to redirect
   to so you can copy the `code` param from it).
3. Note down your **API Key** (client_id) and **API Secret** (client_secret).
4. Every trading morning, run `python auth/get_token.py` to mint a
   fresh token (Upstox tokens from this flow expire daily at 3:30 AM —
   there's no way around the manual login for this option).

## 2. Create your Telegram bot

1. Open Telegram, search for **@BotFather**, send `/newbot`.
2. Follow the prompts (name + username) — you'll get a **bot token**
   like `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`.
3. Send your new bot any message (e.g. "hi") so it has a chat to reply to.
4. Get your **chat_id**: open this URL in a browser
   (replace `<TOKEN>` with your bot token):
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
   Look for `"chat":{"id": ...}` in the response — that number is your chat_id.

## 3. Install and configure

```bash
cd nse_alert_bot
pip install -r requirements.txt

cp .env.example .env
# now edit .env and fill in:
#   UPSTOX_ACCESS_TOKEN (Option A above)
#     - or - UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_REDIRECT_URI (Option B)
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

Edit `watchlist.txt` to the NSE trading symbols you want scanned
(one per line, e.g. `RELIANCE`, `TCS`).

## 4. (Option B only) Generate today's access token

Upstox tokens expire daily at **3:30 AM**, so this step needs to run
once every trading morning before market open:

```bash
python auth/get_token.py
```

It prints a login URL — open it, log in to Upstox, and when you land
on the redirect page (it may show an error page, that's fine), copy
the full URL from your browser's address bar and paste it back into
the terminal. This saves `access_token.json`.

## 5. Run the bot

```bash
python main.py
```

It will:
- Sleep outside market hours (9:15 AM–3:30 PM IST, skipping the first
  15 min after open to avoid opening-bell noise)
- Poll every 60 seconds, evaluate the latest 5-min candle per symbol
- Send **one alert per symbol per day** (so you're not spammed if a
  stock keeps meeting the condition)
- Log every alert to `alerts_log.db` (SQLite) so you can review hit
  rate later

Leave it running in a terminal (or see step 6 for always-on hosting).
Stop with `Ctrl+C`.

## 6. Running it continuously for free

If you used **Option A (Analytics token)**, there's nothing to babysit:
`main.py` reads `UPSTOX_ACCESS_TOKEN` from `.env` and it's valid for a
year, so just leave it running:
- **Your own laptop/PC**: `python main.py` during market hours.
- **Oracle Cloud Free Tier** (genuinely free forever VM): SSH in, clone
  your code, run `main.py` under `systemd` or inside `tmux`/`screen` so
  it survives disconnects — no daily maintenance needed.
- **GitHub Actions**: set the `UPSTOX_ACCESS_TOKEN` repo secret once
  (repo → Settings → Secrets and variables → Actions), and
  `.github/workflows/nse-alert-scan.yml` handles the rest on its
  schedule — nothing to refresh.

If you used **Option B (daily-login OAuth flow)** instead, all of the
above still works, but you additionally need to re-run
`python auth/get_token.py` every trading morning before 9:15 AM IST,
and for GitHub Actions, push the fresh token into the secret each
morning too:

```bash
python auth/get_token.py                 # prints/saves the new token locally
TOKEN=$(python -c "import json; print(json.load(open('access_token.json'))['access_token'])")
gh secret set UPSTOX_ACCESS_TOKEN --body "$TOKEN"   # requires GitHub CLI: gh auth login
```

There's no way to fully automate that daily login without storing your
PIN/TOTP in the app (some open-source packages like `upstox-auto-login`
do this via Selenium) — a security trade-off worth avoiding, which is
the main reason Option A is worth using if all you need is market data.

Either way, outside market hours `scan_once.py`'s `is_market_hours()`
check makes the GitHub Actions run exit immediately, so there's no
cost to it firing on schedule.

## Tuning the strategy

All the knobs are in `config.py`:
- `EMA_PERIOD`, `CANDLE_INTERVAL_MINUTES` — timeframe and EMA length
- `VOLUME_MULTIPLIER` — how strong the volume spike must be
- `RISK_REWARD_RATIO` — target distance as a multiple of stop distance
- `MAX_RISK_PCT` — EMA-cross signals with a wider stop than this are skipped
  (avoids chasing extended breakouts)
- `RETEST_TREND_LOOKBACK` / `RETEST_MIN_CANDLES_ABOVE` — how established
  the above-VWAP uptrend must be before a retest counts
- `RETEST_TOUCH_BUFFER_PCT` — how close the candle's low must come to VWAP
  to count as a "touch"

## Project structure

```
nse_alert_bot/
├── config.py              # all strategy + credential settings
├── auth/get_token.py      # run daily to refresh Upstox access token
├── data/
│   ├── instruments.py     # symbol -> Upstox instrument_key lookup
│   └── upstox_client.py   # fetches intraday candles
├── strategy/
│   ├── indicators.py      # EMA, session VWAP, avg volume
│   └── screener.py        # EMA-cross + VWAP-retest trigger logic, SL/target calc
├── alerts/
│   ├── telegram_bot.py    # sends both alert message types
│   └── logger.py          # SQLite alert history (EMA-cross signals)
├── watchlist.txt
├── main.py                # always-on loop (market-hours aware)
├── scan_once.py           # single-pass scan for GitHub Actions
├── .github/workflows/nse-alert-scan.yml
└── .env / requirements.txt
```

## Note on the instrument lookup

`data/instruments.py` downloads Upstox's NSE instrument master CSV and
matches on the `tradingsymbol` / `instrument_type` columns. If Upstox
changes that file's column names and lookups start failing, open
`data/nse_instruments.csv` and check the actual header row — adjust
the column names in `instruments.py` to match.
