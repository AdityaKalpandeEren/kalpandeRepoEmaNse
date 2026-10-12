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

### 3c. Opening-range breakout on stocks in play — PENDING (`python -m v7.orb`)
Zarattini, Barbon & Aziz (2024), NSE MIS long + short, point-in-time top-200 (373 names incl. later
delisted), 5-minute bars 2022 → 2026 (`python -m v7.fetch_universe` fills the cache).

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
| `swing/backtest/engine.py` | new optional `risk.stop_loss` (off by default; V5 unchanged) + test |
| `app_data/export.py`, `.github/workflows/app-data.yml` | the app shows V7 (copies of nkas-app/pipeline) |
