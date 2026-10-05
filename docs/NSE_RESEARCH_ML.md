# NSE research strategies, ML filters, live alerts + paper trading

Ported from `us_alert_bot` (see `HANDOFF_ML_V2_AND_LIVE_PAPER.md` for the original design) on 2026-09-27.
Production EMA-cross / VWAP alerts (`strategy/screener.py`) are unchanged; everything here is additive.

## What was added

| File | Role |
|---|---|
| `strategy/indicators.py` (appended) | research indicators + `enrich()` |
| `strategy/regime.py`, `strategy/strategies.py` | regime gate, 12 base models, `L_ML_META`, `L_ML_META_V2` |
| `strategy/session.py` | NSE session: minutes since 9:15, **15:15 MIS square-off** trimming |
| `strategy/trade_engine.py` (appended) | research fills/exits/costs + V2 exits (NIFTY / India VIX shock) |
| `strategy/market_context.py` | India context: India VIX, NIFTY 50, NIFTY BANK, NIFTY 50 breadth, earnings (yfinance `.NS`) — prices from Upstox |
| `strategy/ml_features.py`, `ml_features_v2.py` | V1 / V2 features |
| `strategy/news_catalyst.py` | live-only headline scoring with Indian catalyst terms (orders, SEBI/ED, QIP/OFS, promoter deals, RBI, FII) |
| `backtest/run_research.py`, `research_simulator.py`, `research_report.py` | research backtest (`--risk-sweep`, `--universe`) |
| `backtest/candle_cache.py` | resumable on-disk Upstox candle cache (`backtest/cache/`, gitignored) |
| `backtest/ml/*` | V1/V2 dataset builders (parallel) + trainers; models in `backtest/ml/model/` |
| `paper_trading/research_live.py` + hook in `scan_once.py` | live Telegram alerts + paper trading + EOD report |
| `live_ml_v2.py` | optional standalone V2 runner with open-trade news monitoring |
| workflow | **triggers unchanged** (same 9:15–15:30 IST schedule as main); added: state cache, no overlapping runs, 10-min timeout, EOD artifact |

## NSE-specific settings (config.py)

- `MIN_WARMUP_CANDLES = 12` (first signal 10:15 — no pre-market in NSE)
- `RESEARCH_SQUAREOFF_*` = 15:15 IST; V2 entries 9:30–14:45, flat at 15:15
- EOD report from 15:20 IST (`LIVE_EOD_REPORT_*`) — sent by the existing 15:20/15:25/15:30 runs
- Costs `SLIPPAGE_BPS=2`, `COMMISSION_BPS=4` per side (brokerage, STT, stamp, GST for ~Rs 1 lakh positions)
- V2 vetoes: India VIX ≥ 25 or +15% on the day; shock exit NIFTY −0.5% / India VIX +8% since entry

## Verified

- Live paper trading vs backtest replay (jittered/skipped cron, shortlist changing mid-day): **89/89 trades identical**
  (K, SCORE_ENGINE, J, L_ML_META, L_ML_META_V2).

## Results (12 months, 2025-09-26 → 2026-09-25, 83 stocks, long only, with costs)

| Strategy | Trades | Win % | Avg R | Before costs |
|---|---|---|---|---|
| SCORE_ENGINE | 15,973 | 38.0 | −0.26 | +0.01 |
| K_RSI2_REVERSION | 253 | 37.0 | −0.39 | +0.01 |
| J_VWAP_BAND_REVERSION | 13,146 | 34.3 | −0.49 | −0.07 |

- The signals have ~zero edge before costs; NSE charges (~0.32R per trade at median 0.375% stops) make them losers.
  Tight stops are worst (<0.2% stop: −0.74R), later entries worse (10:00 −0.18R → 14:00 −0.45R).
- **L_ML_META** (train Sep–Jun, test Jun 15–Sep 25): report shows 75 trades 65% win +0.17R, but that is 35 unique
  trades, 17/30 days positive, **day-level t = 0.32** → not proven.
- **L_ML_META_V2** (train 142 days, val 48, test 48): all 8 variants negative on validation; test = 2 trades, both
  losers → **no edge found**; it will rarely fire live.
- Nothing here is ready for real money. Paper-trade and judge on data after 2026-09-25 only.

## L_ML_META_V3 (added 2026-09-27)

V2 + (1) 13 NSE sector indices (sector move, stock vs sector, sector vs NIFTY), (2) overnight Brent / USD-INR /
S&P 500 / US 10y / gold, (3) rolling 120-day sensitivities of each **sector and each stock** to those moves,
re-estimated daily (e.g. crude beta ONGC +0.07, OIL +0.09, BPCL −0.06, INDIGO −0.09), (4) two relative-strength
breakout candidates (`RS_BREAKOUT`, `RS_20D_BREAKOUT`), (5) stops at least 0.5% (`ML_V3_MIN_STOP_PCT`),
(6) live-only catalyst reader `strategy/llm_catalyst.py`: market/oil/FX headlines → industry themes and symbols
(crude, RBI, IRDAI, SEBI, duties, tariffs, rupee, monsoon…), Claude when `ANTHROPIC_API_KEY` is set, rules otherwise.
Files: `strategy/sector_map.py`, `market_context_v3.py`, `v3_candidates.py`, `ml_features_v3.py`, `llm_catalyst.py`,
`backtest/ml/build_dataset_v3.py`, `train_meta_model_v3.py`. Live/backtest parity 8/8.

**Result (same 48 test days as V2, long only, after costs):** validation — all V3 variants ≈ 0 or negative, the new
inputs did not beat V2's inputs; test — V3 took **0 trades** (V2: 2 trades, −1.84R). No edge found.

Direct checks on all 12 months (214,779 candidates, V3 stops), avg R per trade:
RS_20D_BREAKOUT −0.16, RS_BREAKOUT −0.20, red-NIFTY-day relative-strength −0.27 (at new day high −0.25),
most favourable crude impulse −0.25, strongest sector vs NIFTY −0.21, all candidates −0.23.
Every slice loses ~0.15–0.30R after NSE charges: for 5-minute intraday entries squared off at 15:15, none of these
ideas has an edge, so no filter (V1/V2/V3) can create one.

**Retrain r2 (2026-10-05), 3x the data:** `build_dataset_v3.py` over 268 symbols, 2025-10-01..2026-10-01, long only →
`backtest/ml/data/dataset_v3_r2.csv` (639,425 candidates, 239 days; git-ignored). Split train 143 / val 48 / test 48 days
(test 2026-07-27..2026-10-01). Trained three ways, varying only `--min-trades` (minimum validation trades for the threshold):

| `--min-trades` | Chosen on val | Val trades / exp R | **Test trades** | **Test win %** | **Test exp R** | Test days + | Baseline exp R | Verdict |
|---|---|---|---|---|---|---|---|---|
| 30  | full_v3/hgb_shallow p≥0.50 | 49 / +0.22  | 20  | 30.0 | **−0.221** | 0/2  | −0.116 | FAILED |
| 100 | full_v3/hgb_leafy p≥0.50   | 107 / +0.07 | 68  | 33.8 | **−0.287** | 1/12 | −0.116 | FAILED |
| 200 | full_v3/hgb_shallow p≥0.45 | 214 / −0.09 | 143 | 39.9 | **−0.160** | 6/18 | −0.116 | FAILED |

Every version did worse out of sample than taking every candidate with no model (−0.116R, itself a loss), and with
200 trades even validation was negative. More data did not create an edge, consistent with the direct checks above.
**Decision: no new model is shipped.** Production `L_ML_META_V3` keeps its old model (threshold 0.675, 0 trades), and
it can be removed from the live research list. Top inputs by importance were overnight macro (USD-INR, S&P 500,
Brent, 1-day), then days to next earnings, so the stock-level intraday features added little.

## Swing strategies on daily candles (added 2026-09-27)

`strategy/swing_nse.py`, `backtest/run_swing_nse.py`, daily candles cached per year (`backtest/candle_cache.py`).
Signal on the close, entry next open, gap-through-stop fills at the open, delivery costs 0.20%/side
(`SWING_COST_PCT_PER_SIDE`), portfolio max 10 equal-weight positions. Parameters are textbook values fixed before testing.

| Strategy | Rule |
|---|---|
| SWING_DAYS_STR | US port: EMA30>50>60, 50-day high breakout on 1.5x vol, +8% / −2% |
| SWING_TREND_BREAKOUT | 50-day high breakout, >SMA200, SMA50>SMA200, 1.5x vol, beating NIFTY (3m); 2 ATR stop; exit close < prior 20-day low |
| SWING_RSI2_PULLBACK | Connors daily: >SMA200, RSI(2)<10; exit close > SMA5 (max 10 d); 3 ATR stop |
| SWING_RS_PULLBACK | leader (beating NIFTY 5%+ 3m, sector beating NIFTY 1m) dips to EMA20 and turns up; 3R target / close < SMA50 |

**Results 2015-01 → 2026-09 (11.7 years), after costs:**

| | Per trade | Portfolio CAGR / maxDD (83 stocks) | Portfolio CAGR / maxDD (NIFTY 50 names only) |
|---|---|---|---|
| NIFTY buy & hold | — | 9.2% / −38.4% | 9.2% / −38.4% |
| SWING_TREND_BREAKOUT | +2.51%, PF 1.66, t 4.25 | **11.6% / −28.8%** | 4.9% / −22.3% (57% invested) |
| SWING_RS_PULLBACK | +0.91%, PF 1.33, t 4.27 | 8.0% / −35.0% | 4.9% / −32.6% |
| SWING_DAYS_STR | −0.34% | −6.6% / −55.7% | — |
| SWING_RSI2_PULLBACK | +0.06% | −6.9% / −64.5% | — |

- TREND_BREAKOUT has a real per-trade edge that survives double costs, but the 11.6% CAGR is inflated by survivorship:
  watchlist names picked after they ran (SHAKTIPUMP, MAZDOCK, COCHINSHIP, SUZLON) average +4.7%/trade vs +1.3% for
  NIFTY 50 names. On NIFTY 50 names alone neither swing strategy beats buy-and-hold (≈ market return while invested,
  lower drawdown). Edge depends on the few biggest trends (without the best 2% of trades: +0.4%/trade).
- The NIFTY-above-200-day filter did not help.
- A fair test needs point-in-time index membership (e.g. NIFTY 200/500 as of each date) — not available here.

## L_ML_V32 — daily 9:45 ranking model (added 2026-09-27, testing only)

`strategy/v32_features.py`, `backtest/ml/build_dataset_v32.py`, `backtest/ml/train_v32.py`.
One row per (stock, day): at 9:45 IST rank all stocks using overnight macro × per-stock/sector sensitivities, prior-day
momentum/trend, earnings timing, gap, first-30-min move/volume vs NIFTY and sector, India VIX, breadth; buy top-K at the
9:45 open, sell at 15:15 (2% emergency stop, NSE intraday costs). Target = return minus the day's universe average.
Data: 5-min Jan 2022 → Sep 2026 (1,171 days, 83 stocks, 96,105 rows). Split by date: train 702 / val 234 / test 235 days.

**Test (2025-10-15 → 2026-09-25), per day after costs:** V3.2 top-3 −0.054% (t −0.86); bottom-3 −0.192%; equal-weight
−0.134%; one-line rule (top-3 by 30-min RS vs NIFTY) −0.028%; NIFTY 9:45→15:15 −0.033%.
Ranking skill is real (IC +0.054, t 5.6, positive 64% of days; validation IC +0.081, t 8.0) and V3.2 beats equal-weight
by +0.08%/day (t 1.7), but ~0.12% round-trip charges exceed the edge, and it does not beat the one-line 30-min RS rule.
Top inputs: move from prior close to 9:40, position in opening range, distance from opening VWAP, opening volume,
earnings timing, 1/5-day momentum, sector opening move, INR sensitivities.
Next options: hold V3.2 picks for several days; use V3.2 ranking to filter SWING_TREND_BREAKOUT; trade only high-conviction days.

## V3.3 (formerly V3.2+) — NSE market data, events and holding period (added 2026-09-27, testing only)

`data/nse_archives.py` (NSE bhavcopy with delivery, F&O bhavcopy OI/PCR, announcements, board meetings, insider PIT —
cached in `backtest/cache/nse/`), `backtest/ml/build_dataset_v33.py`, `backtest/ml/train_v33.py`.
33 new inputs, all known before 9:45 (prior-session bhavcopies; events by public timestamp). Holding periods tested:
same day, 3 days, 5 days (delivery costs for multi-day).

Useful on their own (rank IC vs next-period excess return, train+val): F&O-stock flag (+0.094 same-day, t 18),
previous-day delivery % (+0.070 same-day, t 10), delivery % vs its 20-day norm (+0.02, t 3–4, all horizons),
PCR (+0.036 at 3–5 days, t 7.6), announcement count 5d (+0.02, t 4–5), OI increases (−0.02 same-day), promoter net
buying 20d (+0.011 at 5 days, t 2.6). Not useful: unusual volume yesterday, trade size, order-win/fund-raise flags.

Results: the validation-chosen config (5-day hold) failed on test (−0.086%/day, IC −0.01) — multi-day models showed no
ranking skill. Descriptive second look: same-day V3.2 + NSE data raised test ranking skill (IC 0.055 → 0.067, t 6.3)
and moved top-3 from −0.039%/day to −0.002%/day (≈ breakeven after ~0.12% intraday charges). Still no profit.

## L_ML_V33 — live paper trading (V3.3 = V3.2 + NSE data, same-day)

Model: `backtest/ml/train_v33.py` → `backtest/ml/model/model_v33.joblib` (same-day hold, V3.2 + 33 NSE inputs,
hgb_small, K chosen on validation = 5). Research grid over 1/3/5-day holds kept as `backtest/ml/research_v33_grid.py`.
Exact costs: `strategy/v33_costs.py` (₹20/order + GST, STT 0.025% sell, exchange, SEBI, stamp 0.003% buy, slippage).
Validation/test at ₹5 lakh/stock: +0.044% / +0.029% per day (not yet statistically significant).

Runner: `live_v33.py` — separate from `scan_once.py` and production alerts.
- 9:45 IST: builds today's features with the SAME code as training (`strategy/v33_live.py` →
  `build_dataset_v32.row_features`, `build_dataset_v33.add_nse_features(as_of=today)`), paper-buys the top K at the
  9:45 candle open (`V33_NOTIONAL_PER_STOCK`, default ₹5 lakh), Telegram picks.
- Each run: 2% stop check. 15:15: square-off at the 15:10 candle close, exact charges, Telegram day P&L + all-time.
- State/log: `live_state/v33/state.json`, `live_state/v33/trades.csv`.
- Parity: replaying past days through the live builder reproduces the training features exactly (106/106).

```bash
python live_v33.py --loop               # run from ~9:40 IST; picks at 9:45, closes at 15:15
python live_v33.py                      # one pass (cron-friendly)
python live_v33.py --report             # all-time paper summary
V33_NOTIONAL_PER_STOCK=200000 python live_v33.py --loop    # other paper size
```
Judge after ~60+ trading days; compare with the backtest's +0.03–0.04%/day.

## Commands (run from repo root)

```bash
# backtest the five live strategies (ML models are IN-SAMPLE before 2026-09-26)
python -m backtest.run_research --universe --from 2025-09-26 --to 2026-09-25 \
  --strategies K_RSI2_REVERSION,SCORE_ENGINE,L_ML_META,L_ML_META_V2,J_VWAP_BAND_REVERSION \
  --directions long --risk-sweep

# swing (daily) backtest
python -m backtest.run_swing_nse                       # 83 stocks, 2015 -> today
python -m backtest.run_swing_nse --strategies SWING_TREND_BREAKOUT --symbols RELIANCE,TCS

# live: same as production
python scan_once.py
python -m paper_trading.research_live --report [--send]

# retrain (manual; read the printed verdict before committing new .joblib files)
python -m backtest.ml.build_dataset_v2 --days 365 --workers 7 && python -m backtest.ml.train_meta_model_v2 --directions long
python -m backtest.ml.build_dataset --days 365 --workers 4 && python -m backtest.ml.train_meta_model
python -m backtest.ml.build_dataset_v3 --days 365 --workers 8 && python -m backtest.ml.train_meta_model_v3
python -m backtest.candle_cache --from 2022-01-01 --to <today> --with-indices   # V3.2 data
python -m backtest.ml.build_dataset_v32 --workers 8 && python -m backtest.ml.train_v32
```

## Open items

- Invalid symbols in watchlist/universe (production errors on them too): `SHKTIPUMP`→`SHAKTIPUMP`,
  `RELIANCEPOWER`→`RPOWER`, `GMRINFRA`→`GMRAIRPORT`, `TATAMOTORS`→`TMPV`/`TMCV`, `ZOMATO`→`ETERNAL`.
- Consider a minimum stop width for NSE (costs dominate tight stops) — not implemented; would change strategy logic.
- The ML `.joblib` files are needed in production; the datasets (`backtest/ml/data/*.csv`) are gitignored.
