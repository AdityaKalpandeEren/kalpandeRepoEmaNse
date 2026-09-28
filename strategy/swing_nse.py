"""
NSE swing / positional strategies on DAILY candles (long only).

Separate from the intraday research engine: signals are evaluated on a
day's CLOSE, the trade is entered at the NEXT day's open and held for
days to weeks. Every parameter is a standard published value fixed in
config.SWING_* before any backtest (see the config comment for why).

  SWING_DAYS_STR        identical to us_alert_bot's: EMA 30>50>60, close
                        above the prior 50-day high on >=1.5x the 50-day
                        average volume, >=25% above the 52-week low and
                        within 25% of the 52-week high (Minervini trend
                        template). Exit +8% target / -2% stop, max 120 days.
  SWING_TREND_BREAKOUT  Donchian/Minervini trend breakout: close above the
                        prior 50-day high, above SMA200, SMA50 > SMA200,
                        >=1.5x volume, and stronger than NIFTY over ~3
                        months. Stop 2x ATR(14); exit when a close breaks
                        the prior 20-day low (trailing), max 120 days.
  SWING_RSI2_PULLBACK   Connors RSI(2) on DAILY bars (its original
                        setting): close above SMA200 and RSI(2) < 10. Exit
                        when a close is back above SMA5 (max 10 days);
                        emergency stop 3x ATR.
  SWING_RS_PULLBACK     relative-strength pullback (the V3 idea on daily
                        bars): close > SMA50 > SMA200, beating NIFTY by 5%+
                        over ~3 months, its SECTOR index beating NIFTY over
                        ~1 month, price dips to the 20-day EMA and closes
                        back above it on a green day. Stop under the last 3
                        lows minus 0.5 ATR; target 3R; exit on a close
                        below SMA50; max 60 days.

No lookahead: every rolling level that a signal compares against is the
PRIOR days' (shift(1)); indicators on the signal day itself use only
that day's own close/high/low/volume, which are final at the close when
the signal is evaluated.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config


@dataclass
class SwingPlan:
    stop_mode: str            # "pct" | "atr" | "struct"
    stop_value: float         # pct, ATR multiple, or absolute price
    target_mode: str          # "pct" | "r" | "none"
    target_value: float
    max_hold: int
    reason: str


def _rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def enrich_daily(df: pd.DataFrame, nifty_close: pd.Series, sector_close: pd.Series = None) -> pd.DataFrame:
    """df: one symbol's daily bars (timestamp, open, high, low, close,
    volume). nifty_close / sector_close: close series indexed by date."""
    df = df.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    # Upstox occasionally returns two bars for one date (e.g. special
    # sessions); keep the last so every lookup by date is unambiguous.
    df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
    nifty_close = nifty_close[~nifty_close.index.duplicated(keep="last")]
    if sector_close is not None:
        sector_close = sector_close[~sector_close.index.duplicated(keep="last")]
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    for p in (config.SWING_EMA_FAST, config.SWING_EMA_MID, config.SWING_EMA_SLOW, 20):
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    for p in (5, 50, 200):
        df[f"sma_{p}"] = c.rolling(p, min_periods=p).mean()
    df["avg_vol_50"] = v.rolling(50, min_periods=40).mean().shift(1)
    df["high_50_prior"] = h.rolling(config.SWING_STRUCT_LOOKBACK, min_periods=40).max().shift(1)
    df["low_20_prior"] = l.rolling(20, min_periods=15).min().shift(1)
    df["low_52w"] = l.rolling(252, min_periods=120).min().shift(1)
    df["high_52w"] = h.rolling(252, min_periods=120).max().shift(1)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / config.SWING_ATR_PERIOD, adjust=False,
                       min_periods=config.SWING_ATR_PERIOD).mean()
    df["rsi_2"] = _rsi(c, 2)
    lb = config.SWING_RS_LOOKBACK
    n = nifty_close.reindex(df["date"]).ffill().values
    df["nifty_close"] = n
    df["rs_63"] = (c / c.shift(lb) - 1) - (df["nifty_close"] / df["nifty_close"].shift(lb) - 1)
    if sector_close is not None and len(sector_close):
        sc = pd.Series(sector_close.reindex(df["date"]).ffill().values)
        df["sector_rs_21"] = (sc / sc.shift(21) - 1) - (df["nifty_close"] / df["nifty_close"].shift(21) - 1)
    else:
        df["sector_rs_21"] = np.nan
    return df


def _ok(*vals) -> bool:
    return all(v == v and v is not None for v in vals)


# ═══════════════════════════════════════════════════════════════════
# Entry rules - each looks at row i (the signal day) of an enriched df
# ═══════════════════════════════════════════════════════════════════

def swing_days_str(df, i) -> Optional[SwingPlan]:
    r = df.iloc[i]
    ef, em, es = r[f"ema_{config.SWING_EMA_FAST}"], r[f"ema_{config.SWING_EMA_MID}"], r[f"ema_{config.SWING_EMA_SLOW}"]
    if not _ok(r["high_50_prior"], r["avg_vol_50"], r["low_52w"], r["high_52w"]) or r["avg_vol_50"] <= 0:
        return None
    vol_ratio = r["volume"] / r["avg_vol_50"]
    if not (ef > em > es and r["close"] > r["high_50_prior"]
            and vol_ratio >= config.SWING_MIN_VOLUME_RATIO
            and r["close"] >= r["low_52w"] * (1 + config.SWING_MIN_ABOVE_52W_LOW_PCT)
            and r["close"] >= r["high_52w"] * (1 - config.SWING_MAX_BELOW_52W_HIGH_PCT)):
        return None
    return SwingPlan("pct", config.SWING_STOP_PCT, "pct", config.SWING_TARGET_PCT, config.SWING_MAX_HOLD_DAYS,
                     f"EMA30>50>60, broke 50d high {r['high_50_prior']:.2f} on {vol_ratio:.1f}x vol")


def swing_trend_breakout(df, i) -> Optional[SwingPlan]:
    r = df.iloc[i]
    if not _ok(r["high_50_prior"], r["sma_200"], r["sma_50"], r["avg_vol_50"], r["rs_63"], r["atr"]):
        return None
    vol_ratio = r["volume"] / r["avg_vol_50"] if r["avg_vol_50"] > 0 else 0
    if not (r["close"] > r["high_50_prior"] and r["close"] > r["sma_200"] and r["sma_50"] > r["sma_200"]
            and vol_ratio >= 1.5 and r["rs_63"] > 0):
        return None
    return SwingPlan("atr", 2.0, "none", 0.0, 120,
                     f"50d breakout {r['high_50_prior']:.2f} on {vol_ratio:.1f}x vol, "
                     f"RS vs NIFTY {r['rs_63']*100:+.1f}% (3m)")


def swing_rsi2_pullback(df, i) -> Optional[SwingPlan]:
    r = df.iloc[i]
    if not _ok(r["sma_200"], r["rsi_2"], r["atr"]):
        return None
    if not (r["close"] > r["sma_200"] and r["rsi_2"] < 10):
        return None
    return SwingPlan("atr", 3.0, "none", 0.0, 10, f"RSI(2) {r['rsi_2']:.1f} above SMA200")


def swing_rs_pullback(df, i) -> Optional[SwingPlan]:
    r = df.iloc[i]
    if i < 3 or not _ok(r["sma_50"], r["sma_200"], r["rs_63"], r["ema_20"], r["atr"]):
        return None
    touched = r["low"] <= r["ema_20"] * 1.01
    # Stocks without their own sector index (sector "OTHER") skip the
    # sector condition rather than being excluded outright.
    sector_ok = (not _ok(r["sector_rs_21"])) or r["sector_rs_21"] > 0
    if not (r["close"] > r["sma_50"] > r["sma_200"] and r["rs_63"] > 0.05 and sector_ok
            and touched and r["close"] > r["ema_20"] and r["close"] > r["open"]):
        return None
    stop = float(df["low"].iloc[i - 2: i + 1].min()) - 0.5 * r["atr"]
    return SwingPlan("struct", stop, "r", 3.0, 60,
                     f"pullback to EMA20 in leader: RS {r['rs_63']*100:+.1f}% (3m), "
                     f"sector RS {r['sector_rs_21']*100:+.1f}% (1m)")


# ═══════════════════════════════════════════════════════════════════
# Close-based exits: evaluated on day j's close -> exit at day j+1 open
# ═══════════════════════════════════════════════════════════════════

def _exit_none(df, j):
    return False


def _exit_trend(df, j):
    r = df.iloc[j]
    return _ok(r["low_20_prior"]) and r["close"] < r["low_20_prior"]


def _exit_rsi2(df, j):
    r = df.iloc[j]
    return _ok(r["sma_5"]) and r["close"] > r["sma_5"]


def _exit_rs(df, j):
    r = df.iloc[j]
    return _ok(r["sma_50"]) and r["close"] < r["sma_50"]


SWING_STRATEGIES = {
    "SWING_DAYS_STR": (swing_days_str, _exit_none),
    "SWING_TREND_BREAKOUT": (swing_trend_breakout, _exit_trend),
    "SWING_RSI2_PULLBACK": (swing_rsi2_pullback, _exit_rsi2),
    "SWING_RS_PULLBACK": (swing_rs_pullback, _exit_rs),
}


def stop_and_target(plan: SwingPlan, fill: float, signal_row) -> tuple:
    if plan.stop_mode == "pct":
        stop = fill * (1 - plan.stop_value)
    elif plan.stop_mode == "atr":
        stop = fill - plan.stop_value * float(signal_row["atr"])
    else:
        stop = plan.stop_value
    risk = fill - stop
    if plan.target_mode == "pct":
        target = fill * (1 + plan.target_value)
    elif plan.target_mode == "r":
        target = fill + plan.target_value * risk
    else:
        target = None
    return stop, target
