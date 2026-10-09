# NSE LCR — Large-Cap Runners (alerts + paper, UNPROVEN)

The NSE version of US LCR, for **MEGA** (NIFTY 50), **LARGE** (the rest of NIFTY 100) and **MID** (NIFTY Midcap 150) stocks. It runs as the `lcr` job of **NSE Alert Scan**, every ~2 min from 09:20 to 15:30 IST, and sends alerts to your chat + the group.

| Part | What it does |
|---|---|
| Dynamic list | All ~250 stocks are scanned every run (today's 5-min candles fetched in parallel). The ones with a **volume shock right now** are kept: **time-adjusted volume ≥ 2x** (vs what NSE stocks normally trade by that time), up ≥ 1.5%, ≥ ₹20 cr turnover, above VWAP |
| 📈 EMA rule | **Price above the 10- and 20-day daily EMAs.** Alerts show all six (10/20/30/40/60/180) as ✅/❌; **above all six = ⭐ PERFECT TRADE** |
| 🏛️ Paper trades | **Pullback continuation**. Buy at the next candle's open (+5 bps); no fill on a circuit-locked candle; stop under the pullback low (max 2%); breakeven after +1R then trail; square-off 15:15; exact NSE charges; ₹2 lakh per trade; max 13 a day |
| 🏛️📡 Volume watch | Time-adjusted volume ≥ 3x while the price is within ±1.5% (watch only), at most every 30 min |

## Research (2026-10-09, `python -m nse_lcr.research`)
- **Data:** 5,256 large/mid volume-shocker days (Apr–Oct 2026), Upstox 5-min candles.
- **Every variant loses in both halves:** **−0.09% to −0.18% per trade** after charges, about 0% before costs.
- **⭐ PERFECT is worse on NSE** (−0.14% to −0.18%), the opposite of US LCR (+0.35% / +0.24%).
- **Shipped at the user's request,** labelled **UNPROVEN**. The volume-watch radar and the EMA tags are the useful parts.
