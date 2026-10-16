# V8 — the published setups of the most successful momentum traders, tested on NSE

Started 2026-10-16 (user: "think like the most profitable traders in history … check online how their
models work … add as new"). Research first; one setup passed and runs as a new **paper** book, V8 TREND.

## 1. The setups (rules copied from the traders, fixed before the first run)

| Code | Trader / source | Rules (daily bars) |
|---|---|---|
| **BO** | Kristjan Kullamägi (Qullamaggie), "My 3 timeless setups" + Mark Minervini's trend template | Stage-2 uptrend (close > SMA50 > SMA150 > SMA200, SMA200 rising, ≥ 30% above the 52-week low, within 25% of the high), relative-strength top 15% (IBD-style 3/6/9/12-month), tight 10-day range (≤ 2.5 × ADR) near its high, surfing the rising 10/20-day averages. Buy-stop at the 10-day high next session (skip if it opens > 1 ADR above), stop = 1 ADR (max 8%). Sell 1/3 after 3 sessions and move the stop to breakeven, trail the rest with the first close under the 10- (or 20-) day average |
| **EP** | Qullamaggie's episodic pivot | Gap up ≥ 8% (NSE circuit bands; 10% in the US) on ≥ 3× the 50-day traded value that holds into the close; buy that close, stop = the gap day's low (max 12%); same partial / trailing exits |
| **ATH** | Wilcox & Crittenden (2005, Blackstar), Zarattini, Pagani & Wilcox (2025) | Close at an all-time high (≥ 1 year of history) → buy next open; trailing stop 10 × ATR(42) under the highest close since the buy, never lowered; a close under it sells at the next open |

Market filter for BO/EP (Qullamaggie): NIFTY 10-day average above its 20-day. Sizing: risk 1% of equity to the initial
stop; BO/EP max 20% per name and 10 names, ATH max 10% per name and 30 names; no leverage.

## 2. Data, costs, windows

Every NSE EQ-series instrument 2013-01 → 2026-09 **including later-delisted ones** (bhavcopy panel, corporate-action
adjusted) — no survivorship. Liquid = 20-day median traded value ≥ ₹5 crore and price ≥ ₹30, point in time.
Delivery charges (STT 0.1% both sides, stamp, exchange, SEBI, GST, DP) + slippage 12 bps at the open/close and 25 bps
for stop orders. ₹10 lakh book. One continuous run 2014-01 → 2026-09; WF = 2017-01 → 2025-06 (V7's walk-forward),
halves A (→ 2021-06) and B; PV = Jul 2025 → Sep 2026 (already viewed by V4/V5/V7). Keep rule (fixed in advance):
beat NIFTY in both WF halves **and** live V7's WF Sharpe. `python -m v8.research` (add `--v7` to rebuild V7's daily returns).

## 3. Results (after all costs; Sharpe above 6.5% risk-free)

| Variant | WF CAGR | WF Sharpe | WF Max DD | A / B CAGR | PV CAGR | Trades (WF) | Keep? |
|---|---|---|---|---|---|---|---|
| BO, 10-day trail | −8.0% | −0.81 | −68% | −4.3% / −11.9% | −16.5% | 1,249 | no |
| BO, 20-day trail | +1.2% | −0.21 | −50% | +13.9% / −11.4% | −13.7% | 1,091 | no |
| BO, no market filter | −15.6% | −1.11 | −87% | −2.8% / −28.0% | −34.3% | 1,778 | no |
| EP, 10-day trail | +0.3% | −1.13 | −19% | −3.8% / +5.1% | +0.3% | 107 | no |
| EP, 20-day trail | −0.5% | −1.22 | −19% | −4.0% / +3.5% | −0.9% | 107 | no |
| EP, not rallied before | −0.5% | −1.67 | −14% | −3.0% / +2.3% | +4.0% | 68 | no |
| BO + EP in one book | −7.1% | −0.72 | −69% | −6.1% / −8.2% | −16.8% | 1,304 | no |
| **ATH, 10 × ATR (as pre-registered)** | **+20.6%** | **0.82** | −32% | +18.7% / +22.7% | **+26.2%** | 236 | **yes** |
| **ATH 2-year high, 10 × ATR (live)** | **+21.3%** | **0.89** | −31% | +21.0% / +21.6% | **+23.7%** | 224 | **yes** |
| NIFTY 50 | +14.6% | 0.51 | −38% | +16.0% / +13.2% | −9.3% | | |
| V7 live (D1) | +19.4% | 0.76 | −26% | +18.6% / +20.3% | +7.8% | | |

Deflated Sharpe over all 56 trials ever recorded (V4/V5/V7/V8): ATH 0.55, ATH 2-year 0.66 — **not proven** (< 0.95).

**Why the breakouts fail here.** Short holds (median 3 sessions) turn the book over ~20× a year; at ~0.67% per delivery
round trip that is ~14%/yr of costs. Before costs and with the optimistic entry-day assumption BO makes +15.6% / +23.3%
(10- / 20-day trail) — an edge exists, NSE delivery costs eat it. Trend following holds ~8 months (median 176 sessions),
costs ~0.4%/yr, so it keeps its edge. Its trades: 47% winners, average win +93%, average loss −20%, profit factor 3.5.

**Checks on ATH (one change at a time, all positive vs NIFTY):** 8 or 12 ATR, 40 names, liquidity ₹2 cr / ₹10 cr, market
filter on, 2× costs (−0.4%/yr). Fewer names (20) or 0.5% risk lower the return (less money invested). Correlation with
V7's daily returns 0.80 (a cousin, not a hedge). Stocks only: WF +22.0%, PV +9.1% — **the 2025-26 lead came from gold /
silver ETFs** (trend following buys whatever trends). Weak year: 2025 (−6.5% vs NIFTY +10.5%, the momentum crash).

**Two fixes found while checking live parity (both in the numbers above):** money-market ETFs (LIQUIDBEES, LIQUID1, …)
make a new high almost every day; the first runs bought them and held them forever (never stopped → never in the closed
trade list). Cash / gilt / SDL / Bharat Bond ETFs are now excluded (`setups.cash_like`), which raised ATH's WF CAGR
from +18.4% to +20.6%. The live job keeps ~2 years of data, so it uses the 2-year high; its signals equal the research's
on the same dates (checked 2026-09-24 → 09-30: same instruments, ATR identical).

## 4. US (same code, `us_v8/` in the US repo) — nothing passed

S&P 500 (current members from their date added) and ~2,000 current US listings ≥ $300M (both survivorship-biased, so
optimistic): every BO / EP / ATH variant trails SPY (+14.6% WF). On US large caps the breakout has no edge even without
costs. A real US test of these traders' small-cap setups needs survivorship-free data (e.g. a free Alpaca account).

## 5. Live paper: V8 TREND (`python -m v8.live`)

The `v8` job of **NSE Alert Scan** (same cron-job.org trigger, Mon–Fri 09:20–15:30 IST, once a day), copied from V7:
reads V5's NSE data cache read-only, own state (`live_state/v8`, cache `nse-v8-`), own marker, Telegram tag
"V8 TREND PAPER". Each session: refresh NSE end-of-day files → at the previous close raise every holding's trailing
stop, find new 2-year closing highs → at today's 9:15 open sell holdings that closed under their stop, buy the new highs
(best relative strength first, ≤ 30 names, 1% risk, ≤ 10% per name, delivery costs) → Telegram. ₹10 lakh, paper only.
Dry run 2026-10-16 (real data, real 9:15 opens): 1,027 liquid names, 6 new 2-year highs → 6 buys (stops 28–36% below).
The book fills over days / weeks as new highs appear. The app does not show V8 yet.

## 6. Files

| | |
|---|---|
| `v8/setups.py` | indicators + signals (shared by research and live; same file in the US repo) |
| `v8/engine.py` | event-driven daily simulator + metrics (same file in the US repo) |
| `v8/research.py` | the NSE study above |
| `v8/live.py` | the paper job |
| `v8/tests/` | `python -m pytest v8/tests` |
