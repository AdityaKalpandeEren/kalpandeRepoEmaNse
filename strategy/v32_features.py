"""
L_ML_V32 - daily cross-sectional stock RANKING model (NSE), testing only.

V3 asked "will this 5-minute setup win by 15:15?" - a per-candle question
where V3's strongest inputs (overnight macro, sector, catalysts) are the
same value all day and got drowned out. V3.2 asks the question those
inputs actually answer:

    At 9:45 IST, across the whole universe, which stocks will do BEST
    from now until the 15:15 square-off?

One row per (stock, day), features known at 9:45:
  OVERNIGHT   Brent / USD-INR / S&P 500 / US 10y / gold moves of the prior
              US session, and those moves x each STOCK's and SECTOR's own
              rolling 120-day sensitivity (re-estimated daily, prior data
              only) - the "crude up -> ONGC up, BPCL down" channel.
  DAILY       the stock's 1/5/20-day returns, 3-month strength vs NIFTY,
              distance from 20-day / 52-week highs, SMA50/SMA200 distance,
              ATR %, RSI(14), 5-vs-20-day volume trend (all prior days).
  CATALYST    earnings: days since/until, last EPS surprise, reaction day;
              today's GAP; first-30-min volume vs its own 20-day norm;
              first-30-min move vs NIFTY and vs its sector; where it sits in
              its opening range; distance from the opening VWAP.
  MARKET      NIFTY and its sector: prior-day and 5-day moves, gap,
              first-30-min move; India VIX level, 5-day and opening change;
              NIFTY 50 breadth (% above SMA50); NIFTY vs SMA200.
Label: the NET return (NSE intraday costs) of buying at the 9:45 open and
selling at the 15:10 candle's close (the 15:15 square-off), with a 2%
emergency stop - and that return minus the day's universe average
("excess"), which is what the model learns: WHICH stocks, not whether the
market goes up.

No lookahead: daily values come from sessions strictly before the day;
intraday values only from the six 9:15-9:40 candles (closed by 9:45);
macro from the prior US session; betas use data up to the prior day.
"""
import math

import numpy as np
import pandas as pd

import config

DECISION_BARS = 6                 # 9:15 ... 9:40 candles, all closed by 9:45
STOP_PCT = 0.02                   # emergency stop from the 9:45 fill
MACROS = ("BRENT", "USDINR", "SPX", "US10Y", "GOLD")
BETA_MACROS = ("BRENT", "USDINR", "SPX")


def cost_per_side() -> float:
    return (config.SLIPPAGE_BPS + config.COMMISSION_BPS) / 10000.0


# ═══════════════════════════════════════════════════════════════════
# Per-day intraday summaries
# ═══════════════════════════════════════════════════════════════════

def opening_stats(op: pd.DataFrame) -> dict:
    """Stats of the six 9:15-9:40 candles (all closed by 9:45). Shared by
    the dataset builder and the live V3.3 runner, so both compute the
    opening features identically."""
    o915, c940 = float(op["open"].iat[0]), float(op["close"].iat[-1])
    hi, lo = float(op["high"].max()), float(op["low"].min())
    vol = float(op["volume"].sum())
    tp = (op["high"] + op["low"] + op["close"]) / 3
    vwap30 = float((tp * op["volume"]).sum() / vol) if vol > 0 else np.nan
    return {"o915": o915, "c940": c940, "hi30": hi, "lo30": lo, "vol30": vol, "vwap30": vwap30}


def intraday_day_table(df5: pd.DataFrame, is_index: bool = False) -> pd.DataFrame:
    """One row per trading day from 5-min candles: opening-window stats and
    (for tradables) the 9:45 -> 15:15 outcome."""
    from strategy.session import trim_to_research_session
    df5 = df5.copy()
    df5["timestamp"] = pd.to_datetime(df5["timestamp"])
    df5["date"] = df5["timestamp"].dt.date
    rows = []
    cps = cost_per_side()
    for d, g in df5.groupby("date", sort=True):
        g = trim_to_research_session(g.sort_values("timestamp").reset_index(drop=True))
        if len(g) < DECISION_BARS + 3:
            continue
        op = g.iloc[:DECISION_BARS]
        rest = g.iloc[DECISION_BARS:]
        rec = {"date": d, **opening_stats(op), "entry": float(rest["open"].iat[0]),
               "close_1510": float(rest["close"].iat[-1])}
        if not is_index:
            entry = rec["entry"]
            stop = entry * (1 - STOP_PCT)
            exit_px, outcome = rec["close_1510"], "CLOSE"
            for k, (o, l) in enumerate(zip(rest["open"].to_numpy(float), rest["low"].to_numpy(float))):
                if l <= stop:
                    exit_px = o if (k > 0 and o < stop) else stop
                    outcome = "STOP"
                    break
            rec["exit"] = exit_px
            rec["outcome"] = outcome
            rec["net_ret"] = exit_px * (1 - cps) / (entry * (1 + cps)) - 1
        else:
            rec["ret_945_1515"] = rec["close_1510"] / rec["entry"] - 1
        rows.append(rec)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Daily indicators (values as of each day's close; callers shift by one
# session to use them only on FOLLOWING days)
# ═══════════════════════════════════════════════════════════════════

def daily_table(dd: pd.DataFrame) -> pd.DataFrame:
    dd = dd.copy()
    dd["date"] = pd.to_datetime(dd["timestamp"]).dt.date
    dd = dd.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    c, h, l, v = dd["close"], dd["high"], dd["low"], dd["volume"]
    out = pd.DataFrame({"date": dd["date"], "close": c})
    out["ret1"] = c.pct_change()
    out["ret5"] = c / c.shift(5) - 1
    out["ret20"] = c / c.shift(20) - 1
    out["ret63"] = c / c.shift(63) - 1
    out["dist_high20"] = c / h.rolling(20, min_periods=10).max() - 1
    out["dist_high252"] = c / h.rolling(252, min_periods=120).max() - 1
    out["sma50_dist"] = c / c.rolling(50, min_periods=40).mean() - 1
    out["sma200_dist"] = c / c.rolling(200, min_periods=150).mean() - 1
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=10).mean() / c
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["rsi14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    out["vol5_20"] = v.rolling(5).mean() / v.rolling(20, min_periods=15).mean()
    return out.set_index("date")


def lagged_macro_returns(macro_close: pd.Series, dates) -> pd.Series:
    """For each Indian date t: the macro's return on its last session
    strictly before t."""
    r = macro_close.pct_change().dropna()
    idx = np.array(r.index, dtype="datetime64[D]")
    vals = []
    for t in dates:
        i = int(np.searchsorted(idx, np.datetime64(t, "D"), side="left")) - 1
        vals.append(float(r.iloc[i]) if i >= 0 else np.nan)
    return pd.Series(vals, index=list(dates))


def lagged_macro_level_change(macro_close: pd.Series, dates, lag: int) -> pd.Series:
    idx = np.array(macro_close.index, dtype="datetime64[D]")
    vals = []
    for t in dates:
        i = int(np.searchsorted(idx, np.datetime64(t, "D"), side="left")) - 1
        vals.append(float(macro_close.iloc[i] / macro_close.iloc[i - lag] - 1) if i - lag >= 0 else np.nan)
    return pd.Series(vals, index=list(dates))


def rolling_betas(daily_ret: pd.Series, macro_lag: dict, extend_to=None) -> dict:
    """{macro: beta series by date}, estimated on data up to the PRIOR day.
    extend_to (live only): also give the value for that later date - the
    estimate through the last available day, exactly what the shifted
    series holds for the next session in training."""
    w, mn = config.ML_V3_BETA_WINDOW, config.ML_V3_BETA_MIN_OBS
    out = {}
    for m in BETA_MACROS:
        x = macro_lag[m].reindex(daily_ret.index)
        df = pd.DataFrame({"y": daily_ret, "x": x}).dropna()
        if len(df) < mn:
            continue
        beta = df["y"].rolling(w, min_periods=mn).cov(df["x"]) / df["x"].rolling(w, min_periods=mn).var()
        out[m] = beta.shift(1).reindex(daily_ret.index).ffill()
        if extend_to is not None and extend_to not in out[m].index:
            out[m] = pd.concat([out[m], pd.Series([beta.iloc[-1]], index=[extend_to])])
    return out


def earnings_features(earn: pd.DataFrame, d) -> dict:
    out = {"earn_days_since": 120.0, "earn_days_to_next": 120.0, "earn_surprise": np.nan,
           "earn_reaction_today": 0.0}
    if earn is None or earn.empty:
        return out
    past = earn[earn["reaction"] <= d]
    if not past.empty:
        last = past.iloc[-1]
        out["earn_days_since"] = float(min(120, (d - last["reaction"]).days))
        s = last["surprise"]
        out["earn_surprise"] = float(np.clip(s, -100, 100)) if s == s else np.nan
        out["earn_reaction_today"] = 1.0 if last["reaction"] == d else 0.0
    fut = earn[earn["reaction"] > d]
    if not fut.empty:
        out["earn_days_to_next"] = float(min(120, (fut.iloc[0]["reaction"] - d).days))
    return out


def safe_ratio(a, b) -> float:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return np.nan
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return np.nan
    return a / b - 1
