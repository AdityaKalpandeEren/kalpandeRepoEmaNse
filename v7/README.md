# V7 — NSE research: learn from the mistakes, keep only what holds up

Research only. Nothing here changes production (paper trading, Telegram).
Started 2026-10-12 after a review of every live paper result since 2026-09-28.

## 1. What went wrong (live paper, all-time to 2026-10-12)

| Mistake | Evidence |
|---|---|
| Too many low-edge trades | Score Engine + VWAP Band Reversion = ~90% of all paper losses (US −$342k of −$378k, NSE −₹6.04L of −₹6.71L). NSE Score Engine: 0 of 10 days up, day-t −5.8 |
| Stops inside the noise | Stops < 0.3% and exits within 10 min are the worst buckets in every strategy; a stopped trade averages −1.13R to −1.35R, not −1R |
| Time of day ignored | Mean reversion after 13:00 ET / 12:00 IST loses heavily (US VWAP −302R after 13:00 vs +22R at 11–12) |
| Too little data | US intraday models learned from 61 days of Yahoo bars; every "edge" was fragile |
| Costs ignored at design time | NSE intraday round trip ≈ 0.10–0.15% of notional: with a 0.25% stop, costs eat ~0.5R per trade |
| Late fills | Research filled at the breakout level; live alerts arrive one bar later |

## 2. Rules for V7

* Years of data, point-in-time universes (no survivorship).
* Only setups with outside evidence; every variant fixed before the run and all of them reported.
* Judged on daily returns after full costs; Deflated Sharpe over every trial ever run (V4 + V5 + V7).
* Splits: walk-forward / discovery first, the last period last; forward paper is the real test.

## 3. Experiments

### 3a. Intraday index momentum — REJECTED (`python -m v7.index_momentum`)
Zarattini, Aziz & Barbon (2024) "Beat the Market" noise-band momentum on NIFTY 50 / Nifty Bank,
5-minute bars 2022-01 → 2026-09, decisions every 30 min, futures costs 0.035% per round trip.

| | Discovery 2022–24 | Validation 2025 H1 |
|---|---|---|
| NIFTY 50 (paper stop) | Sharpe 0.30, t 0.5 | Sharpe −1.42 |
| Nifty Bank (paper stop) | Sharpe −0.74 | Sharpe −2.39 |

Alternates ±5% a year; negative at 2× costs. NIFTY's own intraday drift was negative in 2022–24
(Indian index returns came overnight). Not tuned further.

### 3b. Swing momentum — ACCEPTED as the V7 swing candidate (`python -m v7.swing_v7`)
Same engine, costs and walk-forward as V5 (delivery: 0.1% STT each side, 12 bps slippage, DP charges).
Candidates fixed in advance: NSE's own *Nifty200 Momentum 30* score (6- and 12-month return ÷ 1-year
volatility, z-scored), a 10% per-stock stop-loss (Han, Zhou & Zhu 2016), and the regime filter
(half exposure when India VIX > its 1y 80th percentile and NIFTY < 200-day average).

Walk-forward 2017-01 → 2025-06, after all costs (NIFTY 14.6% CAGR):

| Model | CAGR | Sharpe* | Max DD | Alpha | Turnover |
|---|---|---|---|---|---|
| m30_u200_regime | 19.1% | 0.69 | −35.3% | 6.8% | 2.3× |
| **m30_u200_both (stop + regime) — chosen** | **17.6%** | **0.67** | **−25.7%** | **7.3%** | 2.8× |
| m30_u200 | 18.7% | 0.67 | −35.3% | 6.4% | 2.3× |
| m30_u200_stop | 17.3% | 0.66 | −25.7% | 7.1% | 2.8× |
| v7_blend (V5 ML + M30) | 14.2% | 0.54 | −23.2% | 4.4% | 3.3× |
| v5_combo (live V5) | 13.2% | 0.50 | −24.1% | 3.2% | 1.6× |
| m30_u100 | 14.4% | 0.47 | −33.5% | 2.4% | 2.1× |

\* Sharpe above the 6.5% risk-free rate. Deflated Sharpe of the best: 0.79 over 30 trials — **not
statistically proven**. Chosen = stop + regime: Sharpe within 0.02 of the best with 10 points less drawdown.

Jul 2025 → Sep 2026 (previously viewed by V4/V5; NIFTY −9.4% a year): chosen +11.3% a year
(total +14.1%, max DD −8.2%); V5 combo −11.0%.

Robustness (one change at a time, `--sens`): every neighbour beats V5 in both periods —
walk-forward CAGR 16.8–19.7%, Sharpe 0.64–0.76; Jul 2025+ +6.4% to +11.3%. The chosen settings sit
mid-range, not at the peak (top 20 scored higher but was not picked after the fact).

Trades (walk-forward): rebalance exits +14.7% average (72% winners, ~4 months held); stop exits
−12.6% (sold at the next open, ~1 month). ML did not help: blending V5's ML score with momentum
lowered the Sharpe (0.54 vs 0.67).

### 3c. Opening-range breakout on stocks in play — REJECTED 2026-10-13 (`python -m v7.orb`)
Zarattini, Barbon & Aziz (2024), NSE MIS long + short, point-in-time top-200 (373 names incl. later
delisted), 5-minute bars 2022 → 2026 (`python -m v7.fetch_universe` fills the cache).
69,568 candidate symbol-days (348 symbols), 81% triggered. Net R per trade after MIS charges and
2 bps slippage; "best / worst" = the optimistic / pessimistic bound for a stop inside the entry bar:

| Pre-registered setting | Discovery 2022–24 | Validation 2025 H1 |
|---|---|---|
| N20, stop 0.10 ATR (the paper's) | +0.016 / −0.772 (day-t +0.3 / −26.7) | −0.164 / −0.750 |
| N20, stop 0.25 ATR | +0.044 / −0.115 (day-t +1.6 / −5.2) | −0.052 / −0.166 |
| N20, stop 0.50 ATR | +0.011 / −0.001 (day-t +0.7 / −0.1) | −0.034 / −0.039 |
| N10 (all three stops) | +0.013 … +0.052 best case | −0.071 … −0.161 best case |

About zero before costs at best, fading every year (N20 0.10 ATR best case: 2022 +0.011, 2023 +0.089,
2024 −0.049, 2025 −0.164 R) and negative in all 12 settings in 2025. The US result does not carry over
to NSE large caps. Holdout not opened.

### 3d. Swing round 2: other momentum scores — none kept (`python -m v7.swing_v7b`)
Four literature scores on top of the live V7 settings, fixed in advance; keep rule: beat live V7 in
BOTH windows (Sharpe). WF = walk-forward 2017-01 → 2025-06, PV = Jul 2025 → Sep 2026.

| Score | CAGR WF / PV | Sharpe WF / PV | Max DD WF / PV |
|---|---|---|---|
| **V7 live** | **17.6% / 11.3%** | **0.674 / 0.431** | −25.7% / −8.2% |
| A skip-month (Jegadeesh 1990) | 15.6% / 9.5% | 0.568 / 0.285 | −26.6% / −8.6% |
| B 52-week high (George & Hwang 2004) | 15.4% / 6.6% | 0.598 / 0.040 | −25.8% / −8.2% |
| C residual momentum (Blitz, Huij & Martens 2011) | 20.5% / 4.4% | 0.833 / −0.135 | −25.7% / −8.8% |
| D top-decile volatility excluded | 17.8% / 8.0% | 0.708 / 0.167 | −26.2% / −7.5% |

C and D look better in the walk-forward but lose after mid-2025 — kept out by the both-windows rule.

### 3e. Swing round 3: volatility-scaled momentum — REJECTED (`python -m v7.swing_v7_volscale`)
Barroso & Santa-Clara (2015): at each monthly rebalance scale exposure by min(1, target / realised
volatility of the strategy's own daily returns over 126 sessions); targets 15% and 20% fixed in advance,
no leverage, the engine's own costs. One continuous run 2017-01 → 2026-09 (so the PV slice carries the
June-2025 portfolio and differs from 3b's separately started PV window):

| | CAGR WF | Sharpe WF | Max DD WF | Sep 2024 – Mar 2025 | CAGR PV | Sharpe PV |
|---|---|---|---|---|---|---|
| **V7 live** | **17.6%** | **0.67** | −25.7% | −18.0% | 8.8% | 0.23 |
| vol target 15% | 16.4% | 0.65 | −26.5% | −15.5% | 8.8% | 0.23 |
| vol target 20% | 17.2% | 0.66 | −25.7% | −18.4% | 5.8% | −0.04 |

It softens the 2024–25 momentum crash a little but misses the 2020 rebound (+18.1% → +6.3% at 15%).
A long-only momentum book runs at ~16% volatility, so the scaling rarely binds (average scale 0.92 at
15%); the regime filter and the 10% stop already do this job.

### 3f. Swing round 4: delivery-% tilt — D1 passes, NOT proven (`python -m v7.swing_v7c`)
Abnormal delivery % (today's minus the stock's own 60-day average) predicts the next 5 sessions'
return in every period on the point-in-time top-200: Spearman IC +0.025 / +0.028 / +0.022
(t 4.5 / 3.6 / 3.1) in 2014–19 / 2020–22 / 2023–26 (top-decile minus bottom-decile ≈ 0.3% per 5 days).
V5's ML has delivery features, V7 had none. Three tilts of the V7 score, fixed in advance:

| Model | CAGR WF / PV | Sharpe WF / PV | Max DD WF / PV | Turnover WF | Both windows? | Deflated Sharpe |
|---|---|---|---|---|---|---|
| **V7 live** | 17.6% / 11.3% | 0.674 / 0.431 | −25.7% / −8.2% | 2.8× | — | 0.79 (30 trials) |
| **D1: 0.8 V7 + 0.2 delivery 20d−120d** | **19.4% / 11.8%** | **0.764 / 0.475** | −25.7% / −8.3% | 3.1× | **yes** | **0.86 (45 trials)** |
| D2: 0.6 V7 + 0.4 delivery 20d−120d | 18.6% / 9.6% | 0.748 / 0.303 | −28.4% / −7.7% | 4.1× | no | 0.86 |
| D3: 0.8 V7 + 0.2 abnormal delivery 5d | 16.4% / 10.0% | 0.614 / 0.325 | −27.7% / −7.9% | 3.1× | no | 0.75 |

21-session IC inside the walk-forward (monthly samples): delivery 20d−120d +0.017 (t 1.9), 5-day
abnormal +0.019 (t 2.1). D1 is the only tilt that beats live V7 in both windows; a modest, plausible
gain (same drawdown, ~10% more turnover, costs included) but not statistically proven (DSR 0.86 < 0.95).

## 4. Live paper: V7 momentum swing (`python -m v7.live`)

The `v7` job of **NSE Alert Scan** (same cron-job.org trigger, Mon–Fri 09:20–15:30 IST, once a day).
Separate from V5 and every other system: own state (`live_state/v7`, cache `nse-v7-`), own
once-a-day marker, own Telegram tag ("V7 MOMENTUM SWING PAPER"). It reads V5's NSE data cache
read-only and never saves over it (`swing/cache` is saved under `nse-v7-data-` only if V5's cache is
ever missing).

Each session (first run after 09:20): refresh NSE end-of-day files (V5's code) → M30 score of the
top-200 at the previous close → splits / bonuses applied to holdings (no fake stop on a split) →
10% stop-loss sells at today's open → every 21 sessions rebalance to the top 30 at today's open with
delivery costs (regime filter + drawdown breaker scale exposure) → Telegram + state. ₹10 lakh paper,
no orders sent. The app shows it as "V7 Momentum Swing" (same portfolio view as V5).

Dry run on 2026-10-12 (real data, real 9:15 opens): 195 stocks scored in 18 s; regime full exposure
(NIFTY 7.2% under its 200-day average but India VIX at the 71st percentile); 30 buys, 96% invested.

## 5. Files

| | |
|---|---|
| `v7/index_momentum.py` | 3a |
| `v7/swing_v7.py`, `v7/momentum.py` | 3b (+ `--sens`); the score is shared with live |
| `v7/live.py`, `v7/tests/` | 4 (`python -m pytest v7/tests`) |
| `v7/orb.py`, `v7/fetch_universe.py` | 3c |
| `v7/swing_v7b.py` | 3d |
| `v7/swing_v7_volscale.py` | 3e |
| `v7/swing_v7c.py` | 3f |
| `swing/backtest/engine.py` | new optional `risk.stop_loss` (off by default; V5 unchanged) + test |
| `app_data/export.py`, `.github/workflows/app-data.yml` | the app shows V7 (copies of nkas-app/pipeline) |

## 6. 2026-10-13: why NSE intraday loses, and what was tried

### 6a. Score Engine and VWAP Band Reversion over 3¾ years
`backtest.run_research` on the 48 watchlist large caps with unbroken 5-minute history, 2023-01 → 2026-09,
live settings (≈62,000 trades; gross R = net R + the 0.12% modelled cost in R):

| | Trades/day | Net R/trade | Gross R/trade | Day-level t | Up days | Every year negative? |
|---|---|---|---|---|---|---|
| VWAP Band Reversion | 29.4 | −0.470 | −0.051 | −29.0 | 16% | yes (2023–26) |
| VWAP, stop 3 × ATR | 28.9 | −0.250 | −0.037 | −19.4 | 25% | yes |
| Score Engine | 38.0 | −0.248 | +0.007 | −22.2 | 17% | yes |
| Score Engine, stop 3 × ATR | 29.4 | −0.126 | +0.027 | −11.7 | 28% | yes |

The Score Engine's edge before costs is zero; VWAP's is negative. No entry hour and no score band is
positive after costs; holding the signals overnight (delivery costs) does not rescue either.
Live (Sep 28 – Oct 12, 3,206 trades) agrees: gross −0.13R / −0.12R, costs 0.20R / 0.33R a trade.

### 6b. The intraday drift is against long-only intraday trading
NIFTY 50 2015–2025: overnight +29.9% a year (t 10.0), intraday −17.9% a year (t −4.4), intraday negative
every year 2015–2024. Point-in-time top-200 stocks 2022–2026: the 09:20 → 15:15 move is negative every
year (day-level t −2.7). The overnight premium faded in 2025–26 (NIFTY from 5-minute bars: overnight
+10.5% / −2.6%, intraday +0.3% / −11.0%), so buy-at-close/sell-at-open is not a strategy either.
(NIFTYBEES 5-minute opening prints are thin auction trades and unusable for this.)

### 6c. Other ideas tested, all rejected
| Idea | Result |
|---|---|
| Buy-on-gap (Chan 2013): open below prior low by > 1σ, above the 20-day MA, top 10, exit 15:15 | 0.3–0.9% a trade 2022–24 but ~40 trades a year, the edge sits in the first 5 minutes (09:20 entry +0.31% vs 09:15 +0.94%), Jul 2025+ −0.26%; the short mirror is unstable |
| Connors RSI-2 on daily bars (close > 200-day MA, RSI(2) < 10, exit close > 5-day MA, no stop), top-200 2014–26, 0.25% costs | 60–64% winners, ≈0 a trade; a 10-position portfolio loses in most years |
| Turn of the month on NIFTY 2015–2025 | +0.093% vs +0.036% a day, but negative in 2015–19; not stable |

### 6d. Loss-cut variants (paper, alongside the originals)
`SCORE_ENGINE_LC` (score ≥ 65, stop 3 × ATR) and `J_VWAP_LC` (fill before 12:00, stop 3 × ATR) run next
to the unchanged originals; same entries, research paper, Telegram-tagged "LC".
Replay of the LC code itself (same 48 stocks, 2023-01 → 2026-09):

| | Trades/day | Net R/trade | Gross R/trade | Median stop | Share of the original's loss | Up days | Day-level t |
|---|---|---|---|---|---|---|---|
| Score Engine (live settings) | 38.0 | −0.248 | +0.007 | 0.53% | 100% | 17% | −22.2 |
| **SCORE_ENGINE_LC** | **12.7** | **−0.110** | +0.032 | 0.92% | **15%** | 30% | −8.5 |
| VWAP Band Reversion (live settings) | 29.4 | −0.470 | −0.051 | 0.30% | 100% | 16% | −28.9 |
| **J_VWAP_LC** | **2.1** | **−0.183** | −0.026 | 0.80% | **3%** | 27% | −7.3 |

Negative in every year (SE_LC −0.09 … −0.14, VWAP_LC −0.15 … −0.22 R a trade): they cut the loss,
they do not create an edge. The Score Engine LC takes more trades than filtering the original's trades
suggested (7.8/day) because skipped low-score signals no longer use up the cooldown and the 3-a-day cap.
