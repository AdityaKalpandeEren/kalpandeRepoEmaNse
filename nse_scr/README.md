# NSE SCR — Volume Shockers & Runners (alerts + paper)

The NSE version of US SCR. It runs as the `scr` job of **NSE Alert Scan**, every ~2 min from 09:20 to 15:30 IST, and sends alerts to every Telegram receiver (your chat + the group).

| Part | What it does |
|---|---|
| Dynamic watchlist | Rebuilt every run from NSE's free live screeners: **volume gainers** (shockers vs 1-week/2-week volume), **top gainers** (all securities), **most active** by value and by volume. ETFs excluded. Tagged **LARGE** (NIFTY 100) / **MID** (Midcap 150) / **SMALL** (the rest) |
| ⚡📡 Shocker watch | Volume ≥ 5x the 1-week average and turnover ≥ ₹5 cr. Batched at most every 30 min, one alert per stock per day. Watch only |
| ⚡ Paper trades | Volume traded so far ≥ 3x the 20-day average daily volume, up ≥ 3%, turnover ≥ ₹5 cr, price ≥ ₹20, above VWAP → a **new high of day on ≥ 2x candle volume**. Buy at the next candle's open (+10 bps); **no fill on a circuit-locked candle**; stop at the breakout candle's low (max 2%); breakeven after +1R, then trail; **square-off 15:15**. ₹1 lakh per trade, exact NSE intraday charges, max 10 a day |
| 🔻 Paper SHORTS (intraday sell) | **Failed spike:** a shocker (≥ 3x 20-day volume, turnover ≥ ₹5 cr) that spiked **≥ 8%** above the previous close and now closes back **below VWAP** → sell at the next candle's open (−10 bps). **EQ series only** (BE/BZ can't be shorted intraday); no fill on a lower-circuit-locked candle. Stop above the last 6 candles' high (max 3%); breakeven after +1R, then trail above candle highs; **cover by 15:15**. Max **10 shorts a day**, separate from the longs. `NSE_SCR_SHORTS=0` turns them off |

## Short research (2026-10-09, `python -m nse_scr.research_short`)
Same window (Apr–Oct 2026), halves A/B (split 2026-07-07), exact NSE charges (STT on the sell) + 10 bps slippage per side.

| Variant | Half A net/trade | Half B net/trade | Setups/day |
|---|---|---|---|
| New low of day on a falling shocker (1,856 stock-days), 1.5 / 2 / 3% stop | −0.23 / −0.27 / −0.31% | −0.35 / −0.40 / −0.47% | 3 |
| Fade, 2% stop, spike ≥ 3 / 5 / 8% | −0.23% | −0.06 to −0.07% | 22–34 |
| **Fade, 3% stop, spike ≥ 8% (live)** | **−0.11%** | **+0.02%** | 21 |

**Breakdowns lose. Fading failed spikes is close to break-even:** better than the longs, but **not a proven edge**, so it's labelled UNPROVEN. Of the falling-shocker days, 105 symbols had no Upstox history and were skipped.

## Research (2026-10-07, `python -m nse_scr.research`)
- **Data:** 7,628 volume-shocker stock-days (Apr–Oct 2026, bhavcopy: ≥ 3x volume, high ≥ +3%, ≥ ₹5 cr) with Upstox 5-min candles. There's no lookahead: the live trigger itself needs ≥ 3x volume by the signal candle.
- **Every variant loses in both halves:** **−0.36% to −0.54% per trade** after costs, day-level t −7 to −9. Before costs it's still about −0.12%.
- **The bigger the shock, the worse:** volume > 25x gives −0.8% to −0.9% per trade, and already up > 12% gives −0.8%. By the time NSE lists a shocker, the move is mostly done.
- **Paper trades run anyway at the user's request,** on the least-bad rule (HOD breakout, 2% stop) and **labelled UNPROVEN**. The shocker alerts are the useful part: a radar for manual review.
