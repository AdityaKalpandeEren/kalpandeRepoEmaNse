"""
Daily macro series from Yahoo Finance, cached as parquet.

Timing (what is known at the NSE close of Indian date t, 15:30 IST):
  INDIA VIX   closes with the NSE session          -> its date <= t
  USD/INR, Brent, US 10Y, S&P 500 settle in the London / New York day,
  which ends after the NSE close                    -> only dates  < t
`align()` applies exactly these rules, so a feature row for t never sees
a value that was published after 15:30 IST on t.

GIFT Nifty has no free history; the previous US session's S&P 500 return
(known before the NSE close of t) is used as the global-risk proxy.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)

TICKERS = {
    "india_vix": ("^INDIAVIX", "same_day"),
    "usdinr": ("INR=X", "prior_day"),
    "brent": ("BZ=F", "prior_day"),
    "us10y": ("^TNX", "prior_day"),
    "spx": ("^GSPC", "prior_day"),
}


def fetch(cache_dir: str, start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Long frame: date, name, close, timing. Cached at cache/macro.parquet."""
    f = os.path.join(cache_dir, "macro.parquet")
    if os.path.exists(f) and not refresh:
        return pd.read_parquet(f)
    import yfinance as yf
    frames = []
    for name, (ticker, timing) in TICKERS.items():
        d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if d.empty:
            log.warning("macro: no data for %s", ticker)
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        frames.append(pd.DataFrame({"date": pd.to_datetime(d.index.date), "name": name,
                                    "close": d["Close"].astype(float).values, "timing": timing}))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(f, index=False)
    return out


def align(macro: pd.DataFrame, nse_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide frame indexed by NSE trading dates: for each series the last value
    whose publication is before the NSE close of that date."""
    out = pd.DataFrame(index=pd.DatetimeIndex(nse_dates, name="date"))
    for name, g in macro.groupby("name"):
        s = g.set_index("date")["close"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        timing = g["timing"].iloc[0]
        if timing == "same_day":
            idx = s.index.searchsorted(out.index, side="right") - 1   # date <= t
        else:
            idx = s.index.searchsorted(out.index, side="left") - 1    # date <  t
        vals = pd.Series(s.values[idx.clip(min=0)], index=out.index)
        vals[idx < 0] = float("nan")
        out[name] = vals
        out[f"{name}_asof"] = pd.Series(s.index.values[idx.clip(min=0)], index=out.index)
    return out
