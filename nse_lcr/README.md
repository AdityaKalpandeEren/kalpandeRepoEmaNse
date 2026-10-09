# NSE LCR — Large-Cap Runners: intraday + swing (alerts + paper, UNPROVEN)

The NSE version of US LCR, for **MEGA** (NIFTY 50), **LARGE** (the rest of NIFTY 100) and **MID** (NIFTY Midcap 150) stocks. It runs as the `lcr` job of **NSE Alert Scan**, every ~2 min from 09:20 to 15:30 IST, and sends alerts to your chat + the group.

| Part | What it does |
|---|---|
| Dynamic list | All ~250 stocks are scanned every run (today's 5-min candles fetched in parallel). The ones with a **volume shock right now** are kept: **time-adjusted volume ≥ 2x** (vs what NSE stocks normally trade by that time), up ≥ 1.5%, ≥ ₹20 cr turnover, above VWAP |
| 📈 EMA rule | **Price above the 10- and 20-day daily EMAs.** Alerts show all six (10/20/30/40/60/180) as ✅/❌; **above all six = ⭐ PERFECT TRADE** |
| 🏛️ Paper trades | **Pullback continuation**. Buy at the next candle's open (+5 bps); no fill on a circuit-locked candle; **stop under the pullback low (max 4%), no intraday trail**. **Stopped today = ⚡ INTRADAY trade** (intraday charges). **Still open at 15:15 = 🌙 carried as SWING** (delivery charges), held **up to 5 sessions**: exit on a gap below the stop, at the stop, or at 15:15 on day 5. ₹2 lakh per trade, max 13 new entries a day, max 30 open swings |
| 🏛️📊 Separate report (15:20) | **NSE LCR REPORT**: today's ⚡ intraday trades, 🌙 carried tonight, 🏁 swing closed today, 📂 swing open (day n/5, unrealised P&L), today's realised P&L, all-time **INTRADAY vs SWING** |
| 🏛️📡 Volume watch | Time-adjusted volume ≥ 3x while the price is within ±1.5% (watch only), at most every 30 min |

## Research (2026-10-09, `python -m nse_lcr.research`)
- **Data:** 5,256 large/mid volume-shocker days (Apr–Oct 2026), Upstox 5-min candles.
- **Every variant loses in both halves:** **−0.09% to −0.18% per trade** after charges, about 0% before costs.
- **⭐ PERFECT is worse on NSE** (−0.14% to −0.18%), the opposite of US LCR (+0.35% / +0.24%).
- **Shipped at the user's request,** labelled **UNPROVEN**. The volume-watch radar and the EMA tags are the useful parts.

## Intraday + swing backtest (2026-10-09, `python -m nse_lcr.research_swing [--hold]`, max hold 5 sessions)
| Version | First half / trade | Second half / trade | Overall |
|---|---|---|---|
| Intraday only (old live) | −0.08% | −0.13% | ≈ −0.10% |
| Intraday, then carry everything still open at 15:15 | −0.10% | −0.08% | ≈ −0.09% |
| **Swing from entry, stop ≤ 4%, hold to day 5 (LIVE)** | **+0.18%** | **−0.15%** | **≈ +0.05%** |
| Same, with a stop that trails the day lows | −0.08% | −0.21% | ≈ −0.13% |

**How the live version's trades ended:** 55% stop out on day 0 (about −1.0%); 145 held to day 5 made **+5.2%** each; 159 swing stops at −1.6%; 21 gaps at −2.5%. It's the best overall shape, but **inconsistent between halves, so UNPROVEN**.
