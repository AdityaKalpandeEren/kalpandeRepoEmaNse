"""
V3-only candidate generators and trade geometry (model L_ML_META_V3).

The 12 base models keep proposing candidates exactly as before; V3 adds
two relative-strength breakout setups that the base models don't have -
the "stock breaking out even though the market is weak" case - and puts
EVERY V3 candidate (base or new) on a cost-aware stop.

RS_BREAKOUT       long only: the candle closes at a new high of the day
                  (above every earlier regular-session candle's high), on
                  >= config.ML_V3_RS_MIN_RVOL relative volume, while the
                  stock is beating BOTH NIFTY and its own sector today.
                  Not before 9:45 (the first highs of the day are noise).
RS_20D_BREAKOUT   long only: the candle closes above the prior 20-session
                  high for the first time today (previous candle was at or
                  below it), on >= config.ML_V3_RS_20D_MIN_RVOL, while
                  beating NIFTY today.

Neither needs the market to be green: on a red day a stock beating NIFTY
and its sector is exactly what they look for (weak_mkt_rs feature).

COST-AWARE GEOMETRY: at NSE's ~12 bps round-trip charges, a 0.2% stop
pays ~0.6R per trade before the market moves at all. V3 widens any stop
tighter than config.ML_V3_MIN_STOP_PCT to that width and re-derives the
target at the same config.RISK_REWARD_RATIO; stops wider than
config.MAX_RISK_PCT are still skipped. This is arithmetic from the cost
model, chosen before looking at any V3 result.
"""
from dataclasses import replace

import config
from strategy.strategies import _build, _f


def apply_v3_geometry(sig):
    """Signal with the V3 minimum stop width, or None if it becomes too wide."""
    entry = float(sig.entry)
    if entry <= 0:
        return None
    min_risk = entry * config.ML_V3_MIN_STOP_PCT
    if sig.direction == "long":
        risk = entry - float(sig.stop_loss)
        risk = max(risk, min_risk)
        stop, target = entry - risk, entry + risk * config.RISK_REWARD_RATIO
    else:
        risk = float(sig.stop_loss) - entry
        risk = max(risk, min_risk)
        stop, target = entry + risk, entry - risk * config.RISK_REWARD_RATIO
    if risk <= 0 or risk / entry > config.MAX_RISK_PCT:
        return None
    return replace(sig, stop_loss=round(stop, 4), target=round(target, 4))


def rs_breakout(symbol, df, direction, regime, f):
    if direction != "long" or len(df) < 3:
        return None
    if _f(f.get("minutes_since_open")) < 30:
        return None
    row = df.iloc[-1]
    if not (f.get("new_day_high") == 1.0
            and _f(row.get("rvol")) >= config.ML_V3_RS_MIN_RVOL
            and _f(f.get("rs_vs_nifty_day")) > 0
            and _f(f.get("rs_vs_sector_day")) > 0
            and _f(row.get("close")) > _f(row.get("open"))):
        return None
    struct = float(df["low"].iloc[-3:].min())
    return _build(symbol, "RS_BREAKOUT", direction, row, regime,
                  f"new day high on {_f(row.get('rvol')):.1f}x vol, beating NIFTY "
                  f"({_f(f.get('rs_vs_nifty_day'))*100:+.2f}%) and sector "
                  f"({_f(f.get('rs_vs_sector_day'))*100:+.2f}%)", 75, struct)


def rs_20d_breakout(symbol, df, direction, regime, f):
    if direction != "long" or len(df) < 3:
        return None
    row, prev = df.iloc[-1], df.iloc[-2]
    dist = _f(f.get("sym_dist_high20"), default=-1.0)          # close / prior 20d high - 1
    close, prev_close = _f(row.get("close")), _f(prev.get("close"))
    if close <= 0 or dist <= 0:
        return None
    high20 = close / (1.0 + dist)
    if not (prev_close <= high20
            and _f(row.get("rvol")) >= config.ML_V3_RS_20D_MIN_RVOL
            and _f(f.get("rs_vs_nifty_day")) > 0):
        return None
    return _build(symbol, "RS_20D_BREAKOUT", direction, row, regime,
                  f"broke 20-day high {high20:.2f} on {_f(row.get('rvol')):.1f}x vol, "
                  f"beating NIFTY {_f(f.get('rs_vs_nifty_day'))*100:+.2f}%", 80,
                  min(float(df["low"].iloc[-3:].min()), high20))


V3_MODELS = {"RS_BREAKOUT": rs_breakout, "RS_20D_BREAKOUT": rs_20d_breakout}
