# V4.0 — NSE daily ML ranking model (research + paper)

A cross-sectional model that ranks large-cap NSE stocks every evening and
holds the top N from the next open. Built for **honest, after-cost**
evaluation, not for a pretty backtest. Separate from the intraday bot
(`scan_once.py`) and from V1–V3.3; nothing here sends orders or alerts.

## Run it

```bash
pip install -r v4/requirements.txt
python -m v4.run_research --download        # fetch 2013→yesterday NSE archives (~15 min first time)
python -m v4.run_research                   # walk-forward only (holdout untouched)
python -m v4.run_research --holdout         # + the ONE-TIME holdout test, saves the final model
python -m v4.paper                          # after the close: tomorrow's targets -> live_state/v4/signals_*.csv
python -m pytest v4/tests -q                # no-lookahead, purging, costs, execution rules
```
Config: `v4/config.yaml` (override any key: `--set label.horizon=20 --set portfolio.top_n=15`).
Outputs: `v4/results/run_<hash>/report.html` (+ CSVs). Caches: `v4/cache/` (git-ignored).

## Layout
| Path | What |
|---|---|
| `data/nse_daily.py` | NSE bhavcopy (old + UDiFF), MTO delivery %, index closes, participant OI → parquet |
| `data/corporate_actions.py` | NSE corporate-actions feed → split / bonus / consolidation factors, demerger & rights flags |
| `data/panel.py` | rename linking by ISIN, corporate-action adjustment, point-in-time universe |
| `data/macro.py` | INDIA VIX, USD/INR, Brent, US 10Y, S&P 500 with publication-time alignment |
| `data/sectors.py` | NSE industry per stock |
| `features/` | technical, cross-sectional ranks, sector/market-relative, regime, macro, FII positioning, labels |
| `models/` | momentum / reversal baselines, LightGBM LambdaRank + regression, XGBoost, rank ensemble, Optuna-in-fold |
| `validation/` | purged + embargoed expanding walk-forward, metrics, Deflated Sharpe, per-year / per-regime |
| `backtest/` | delivery costs, next-open execution, circuit locks, ADV cap, T+1, delisting exits |
| `risk/` | inverse-vol sizing, per-stock + per-sector caps, rolling drawdown breaker, regime filter |
| `reports/` | self-contained HTML report |
| `tests/` | pytest suite (offline, synthetic data) |

## Design decisions (and why)
- **Universe**: NSE does not publish historical NIFTY-100 constituents for free, so the universe is a
  point-in-time proxy: each month, the top 100 EQ stocks by trailing 126-day median traded value,
  using data up to the previous session. Delisted / merged stocks (HDFC, DHFL, YESBANK…) are included
  while they qualified.
- **Prices**: the bhavcopy `PREVCLOSE` is *not* adjusted on ex-dates (RELIANCE's 1:1 bonus shows a fake
  −50%). Splits/bonuses/consolidations use NSE's corporate-actions feed; demerger and rights ex-dates have
  their opening gap neutralised. Dividends are not reinvested (price returns, like the NIFTY 50 price
  benchmark).
- **Timing**: features at date t use only data published by the NSE close of t (US/FX/crude: previous
  day). Trades happen at the open of t+1; labels are next-open → open H days later.
- **Validation**: expanding walk-forward, retrain yearly, purge (label windows can't reach the test block)
  + 5-session embargo. Optuna tunes inside each fold's training data only. The last ~15 months are a
  holdout that `run_research --holdout` evaluates **once** (a marker file blocks silent re-use).
- **Selection**: pre-registered rule — highest walk-forward net Sharpe. Every configuration ever run is
  logged in `results/trials.json` and feeds the Deflated Sharpe Ratio.

## SEBI note
Retail algorithmic trading through broker APIs falls under SEBI's framework (circular of 4 Feb 2025):
orders go through the broker's registered API with an algo identifier, and strategies above the
order-rate threshold need exchange registration. This project only writes research reports and a
signals CSV — confirm the current rules with your broker before automating anything.
