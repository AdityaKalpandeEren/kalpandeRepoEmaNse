"""
Live feature builder for L_ML_V33 - computes TODAY's V3.3 rows at 9:45
with exactly the code that built the training data:

  backtest.ml.build_dataset_v32.load_market / symbol_context / row_features
  strategy.v32_features.opening_stats / intraday_day_table
  backtest.ml.build_dataset_v33.add_nse_features (as_of=today)

The only live-specific part is where TODAY's 5-min candles come from:
`today_5m(key_or_symbol)` - the Upstox intraday API in production, cached
history in a replay (tests/parity). Everything else - prior sessions,
daily history, NSE bhavcopies, events - comes from the same caches and
loaders as training.
"""
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from backtest.candle_cache import load_daily_cached, load_symbol_history_cached
from backtest.ml.build_dataset_v32 import DAILY_FROM, load_market, row_features, symbol_context
from backtest.ml.build_dataset_v33 import add_nse_features
from backtest.ml.common import access_token
from strategy import sector_map
from strategy.session import trim_to_research_session
from strategy.v32_features import DECISION_BARS, intraday_day_table, opening_stats


def opening_row(df5_today: pd.DataFrame):
    """Opening stats from today's candles, or None before the six 9:15-9:40
    candles are all present. 'entry' = the 9:45 candle's open when it exists."""
    if df5_today is None or df5_today.empty:
        return None
    g = df5_today.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"])
    g = trim_to_research_session(g.sort_values("timestamp").reset_index(drop=True))
    if len(g) < DECISION_BARS:
        return None
    rec = opening_stats(g.iloc[:DECISION_BARS])
    rec["entry"] = float(g["open"].iat[DECISION_BARS]) if len(g) > DECISION_BARS else np.nan
    return rec


def _add_today_index_rows(m: dict, today: date, today_5m) -> None:
    """Put today's opening stats for NIFTY, India VIX and every sector index
    into the market dict (training had them from the day tables)."""
    def add(table_key, sub_key, key):
        rec = opening_row(today_5m(key))
        if rec is None:
            return
        row = pd.DataFrame([rec], index=[today])
        if sub_key is None:
            m[table_key] = pd.concat([m[table_key][m[table_key].index != today], row])
        else:
            t = m[table_key][sub_key]
            m[table_key][sub_key] = pd.concat([t[t.index != today], row])
    add("nifty_day", None, "NSE_INDEX|Nifty 50")
    add("ivix_day", None, "NSE_INDEX|India VIX")
    for code, key in sector_map.SECTORS.items():
        if code == "OTHER":
            m["sector_day"][code] = m["nifty_day"]
        else:
            add("sector_day", code, key)


LAST_TIMINGS: dict = {}     # seconds per phase of the last build_rows() call (for the run log)


def build_rows(today: date, symbols: list, today_5m, verbose=True) -> pd.DataFrame:
    """One V3.3 feature row per symbol for `today` (labels are NaN)."""
    t_start = time.time()
    tok = access_token()
    frm = (today - timedelta(days=45)).strftime("%Y-%m-%d")
    prev = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    m = load_market(frm, prev, extend_to=today)
    _add_today_index_rows(m, today, today_5m)
    rows = []
    tm = {"market": time.time() - t_start, "today_5m": 0.0, "history_5m": 0.0, "daily": 0.0, "context": 0.0}
    for sym in symbols:
        try:
            t = time.time()
            rec = opening_row(today_5m(sym))
            tm["today_5m"] += time.time() - t
            if rec is None:
                continue
            t = time.time()
            hist = load_symbol_history_cached(sym, 5, frm, prev, tok)
            tm["history_5m"] += time.time() - t
            past = intraday_day_table(hist)
            prior_vol = past["vol30"].tail(20) if len(past) else pd.Series(dtype=float)
            rec["rvol30"] = rec["vol30"] / prior_vol.mean() if len(prior_vol) >= 10 and prior_vol.mean() > 0 else np.nan
            t = time.time()
            daily = load_daily_cached(sym, DAILY_FROM, prev, tok)
            tm["daily"] += time.time() - t
            if len(daily) < 260:
                continue
            t = time.time()
            f = row_features(sym, today, rec, symbol_context(sym, daily, m), m)   # incl. earnings dates (Yahoo)
            tm["context"] += time.time() - t
            if f is not None:
                rows.append(f)
        except Exception as e:
            if verbose:
                print(f"  {sym}: skipped ({str(e)[:70]})")
    ds = pd.DataFrame(rows)
    if ds.empty:
        LAST_TIMINGS.clear(); LAST_TIMINGS.update({k: round(v) for k, v in tm.items()})
        return ds
    t = time.time()
    out = add_nse_features(ds, today - timedelta(days=45), today, as_of=today, verbose=False)
    tm["nse_archives"] = time.time() - t
    LAST_TIMINGS.clear(); LAST_TIMINGS.update({k: round(v) for k, v in tm.items()})
    return out


def score(rows: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X = rows.reindex(columns=bundle["features"])
    return rows.assign(score=bundle["model"].predict(X)).sort_values("score", ascending=False)
