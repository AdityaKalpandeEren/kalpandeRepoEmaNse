# V5.0 — NSE swing ranking model (live paper)

Daily cross-sectional model for ~100 point-in-time NSE large caps. Runs live as
the `v5` job of **NSE Alert Scan** (same cron-job.org trigger): first run after
09:20 IST scores stocks from the previous close; every 5th session it rebalances
a ₹10 lakh paper portfolio (top 20, inverse-vol, 15% name / 30% sector caps,
hold while ranked ≤ 50) at today's open with full delivery costs. Telegram:
rebalance orders or a daily portfolio update vs NIFTY 50. Paper only.

| | |
|---|---|
| Live runner | `python -m v5.live` (state: `live_state/v5/`) |
| Model | `v5/model/v5_final_model.joblib` (LightGBM ×3 + XGBoost, trained through 2026-09-30) |
| Research | `python -m v5.run_v5` (walk-forward, ablations, Deflated Sharpe, report) |
| Shared core | `swing/` — NSE archives 2013→, corporate actions, point-in-time universe, features, purged walk-forward, Indian delivery costs, backtest engine, tests |
| Tests | `python -m pytest swing/tests v5/tests -q` |

Research result (after costs): walk-forward 2017–mid-2025 13.3% CAGR, Sharpe 0.53
(above 6.5% risk-free), max drawdown −26.7%, turnover 2.9×/yr; 2020–2026 10.8% vs
NIFTY 9.8% with −26.7% vs −38.4% max drawdown. Deflated Sharpe 0.80 over 23
trials — **not statistically proven**; it lagged NIFTY in 2025. The live paper
record is the real test. Details: research branch `nse_v5_ml`.

Notes: ETFs (GOLDBEES, LIQUIDBEES…) are excluded from live picks; opening
prices are looked up by bhavcopy ISIN (a symbol lookup can hit a debenture).
If NSE's corporate-actions API is unreachable, the committed `v5/seed/` list is
used and Telegram shows a warning. LLM news is deliberately not part of V5
(historical LLM backtests carry lookahead bias). SEBI's retail-algo rules apply
before any automated order placement.
