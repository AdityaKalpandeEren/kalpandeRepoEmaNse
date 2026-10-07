# NSE SCR — Volume Shockers & Runners (alerts + paper)

The NSE version of US SCR. It runs as the `scr` job of **NSE Alert Scan**, every ~2 min from 09:20 to 15:30 IST, and sends alerts to every Telegram receiver (your chat + the group).

| Part | What it does |
|---|---|
| Dynamic watchlist | Rebuilt every run from NSE's free live screeners: **volume gainers** (shockers vs 1-week/2-week volume), **top gainers** (all securities), **most active** by value and by volume. ETFs excluded. Tagged **LARGE** (NIFTY 100) / **MID** (Midcap 150) / **SMALL** (the rest) |
| ⚡📡 Shocker watch | Volume ≥ 5x the 1-week average and turnover ≥ ₹5 cr. Batched at most every 30 min, one alert per stock per day. Watch only |
| ⚡ Paper trades | Volume traded so far ≥ 3x the 20-day average daily volume, up ≥ 3%, turnover ≥ ₹5 cr, price ≥ ₹20, above VWAP → a **new high of day on ≥ 2x candle volume**. Buy at the next candle's open (+10 bps); **no fill on a circuit-locked candle**; stop at the breakout candle's low (max 2%); breakeven after +1R, then trail; **square-off 15:15**. ₹1 lakh per trade, exact NSE intraday charges, max 10 a day |

## Research (2026-10-07, `python -m nse_scr.research`)
- **Data:** 7,628 volume-shocker stock-days (Apr–Oct 2026, bhavcopy: ≥ 3x volume, high ≥ +3%, ≥ ₹5 cr) with Upstox 5-min candles. There's no lookahead: the live trigger itself needs ≥ 3x volume by the signal candle.
- **Every variant loses in both halves:** **−0.36% to −0.54% per trade** after costs, day-level t −7 to −9. Before costs it's still about −0.12%.
- **The bigger the shock, the worse:** volume > 25x gives −0.8% to −0.9% per trade, and already up > 12% gives −0.8%. By the time NSE lists a shocker, the move is mostly done.
- **Paper trades run anyway at the user's request,** on the least-bad rule (HOD breakout, 2% stop) and **labelled UNPROVEN**. The shocker alerts are the useful part: a radar for manual review.
