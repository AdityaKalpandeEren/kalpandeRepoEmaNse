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

## Testing accuracy: backtest & paper trading

Two independent ways to answer "how good are these alerts, really?" —
both use the exact same entry/stop/target math as the live signals
(`strategy/screener.py`), and the exact same win/loss accounting
(`strategy/trade_engine.py` + `backtest/report.py`), so the numbers
from each are directly comparable.

**Fill rule (applies to both):** a signal is generated from a candle
that just closed, so the earliest you could realistically act on the
Telegram alert is the *open of the next candle* — not that candle's
own close. Both methods below fill (and re-derive the target) at that
next-candle open, never at the signal candle's close, so the numbers
aren't inflated by lookahead.

**Same-candle stop/target rule:** if one candle's range touches both
levels, the STOP is counted as hit first (plain OHLC data can't tell
you the real intrabar order) — this makes the win rate a conservative
lower bound, not an optimistic one.

### 1. Historical backtest (do this first)

Runs the strategies over real past candles from Upstox and tells you,
trade by trade, whether each alert would have hit its target or its
stop:

```bash
python -m backtest.run_backtest --from 2025-08-01 --to 2025-09-10
```

Useful flags:
```bash
# just a couple of symbols, just one strategy
python -m backtest.run_backtest --symbols RELIANCE,TCS \
    --from 2025-08-15 --to 2025-09-10 --strategies EMA_CROSS

# a coarser candle size (also widens how far back Upstox will give you data)
python -m backtest.run_backtest --from 2024-09-01 --to 2025-09-10 --interval 15
```

It prints a live trade-by-trade log as it runs, then a summary
(overall, per-strategy, per-symbol: win rate, avg R per trade,
total R), and writes:
- `backtest/results/backtest_trades_<timestamp>.csv` — every simulated trade
- `backtest/results/backtest_report_<timestamp>.md` — the summary report

Notes:
- Needs `UPSTOX_ACCESS_TOKEN` set (same one `main.py` uses).
- Upstox limits how far back minute-level candles go (roughly the last
  month for 1-minute bars) — if a date range returns few/no candles,
  either narrow it or use a coarser `--interval`.
- `VWAP_BROAD_TEST` is the loose "above VWAP or touching it" version
  currently wired into `scan_once.py` (see below) — it will generate
  far more trades than `EMA_CROSS`/`VWAP_RETEST`, by design, since it's
  meant to show you raw VWAP behavior, not a refined setup.

### 2. Live paper trading (forward-test with `main.py`)

While `main.py` runs continuously, it now also opens a *virtual*
position for every signal, tracks it forward exactly like the
backtester (fills on the next candle's open, exits on target/stop/EOD
square-off), and logs the result to `paper_trades.db` — no real orders,
no capital at risk.

It's on by default. To turn it off, set in `.env`:
```
PAPER_TRADING_ENABLED=false
```

**This needs `main.py` running continuously** (your own machine, or a
free-tier VM like Oracle Cloud — see step 6 above), not the
GitHub-Actions `scan_once.py` path, because a pending paper trade needs
to still be there on the *next* poll to get filled and tracked; an
ephemeral CI runner won't persist that state between runs.

Generate an accuracy report from the paper-trading log anytime:

```bash
python -m paper_trading.generate_report
# or a specific window
python -m paper_trading.generate_report --from 2025-09-01 --to 2025-09-10
```

This prints the same overall/per-strategy/per-symbol breakdown as the
backtester and writes `paper_trading/results/paper_trading_trades_<timestamp>.csv`
and `..._report_<timestamp>.md`.

### Reading the report

- **Win rate (target vs stop only)** — of trades that actually
  resolved one way or the other, what % hit target. The purest
  "was this alert right" number.
- **Win rate (incl. EOD as loss)** — same, but treats trades still
  open at session close (squared off, not stopped/targeted) as losses
  — a stricter, more conservative number.
- **Avg R per trade (expectancy)** — the number that actually matters
  for whether the strategy makes money over time: average result per
  trade, in multiples of what you risked. Positive = profitable on
  average even with a sub-50% win rate, as long as winners (capped at
  `config.RISK_REWARD_RATIO`, currently 1.5R) outweigh losers (-1R)
  often enough.

### Same VWAP test logic as the US bot

`strategy/screener.py` now also has `check_vwap_broad_TEST` — the
loose "price is above VWAP or touching it" signal, ported over
unchanged from the US bot. `scan_once.py` is wired to use it live
(the stricter `check_vwap_retest` call is commented out right above
it, matching the US bot's current setup exactly) so you can compare
live VWAP behavior across both markets on equal footing. Swap the
comment back to go back to the stricter retest-only logic — nothing
else changes.

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
│   ├── instruments.py         # symbol -> Upstox instrument_key lookup
│   └── upstox_client.py       # intraday candles (live) + historical candles (backtest)
├── strategy/
│   ├── indicators.py      # EMA, session VWAP, avg volume
│   ├── screener.py        # EMA-cross + VWAP-retest + VWAP-broad-test trigger logic, SL/target calc
│   └── trade_engine.py    # shared trade-outcome simulation (backtest AND live paper trading)
├── backtest/
│   ├── data_loader.py     # fetches + chunks historical candles, splits into per-day sessions
│   ├── simulator.py       # walks each day candle-by-candle, converts signals to simulated trades
│   ├── report.py          # win rate / avg-R / expectancy report (shared with paper_trading/)
│   └── run_backtest.py    # CLI: python -m backtest.run_backtest --from ... --to ...
├── paper_trading/
│   ├── tracker.py             # live virtual-position tracking, driven by main.py's poll loop
│   └── generate_report.py     # CLI: python -m paper_trading.generate_report
├── alerts/
│   ├── telegram_bot.py    # sends both alert message types
│   └── logger.py          # SQLite alert history (EMA-cross signals)
├── watchlist.txt
├── main.py                # always-on loop (market-hours aware, also drives paper trading)
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
