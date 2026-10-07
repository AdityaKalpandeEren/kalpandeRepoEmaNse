"""
NSE SCR data - NSE's own free live screeners (same session / cookies as
data.nse_archives), rebuilt every run:

  shockers()     "volume gainers": today's volume vs the 1-week / 2-week
                 average volume (NSE's volume-shocker list)
  movers()       top gainers (all securities) + most active by value and
                 by volume
  watchlist()    the union, as one frame: symbol, ltp, pct, volume,
                 vol_x_week (shocker ratio, when known), turnover, sources
  cap_buckets()  LARGE = NIFTY 100, MID = NIFTY MIDCAP 150, else SMALL
                 (NSE index constituents, cached for the day)
"""
from __future__ import annotations

import json
import os
import time
from datetime import date

import pandas as pd

from data.nse_archives import _api_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "nse_scr", "cache")
BASE = "https://www.nseindia.com/api/"
# ETFs / funds trade in the EQ series but are not companies (same rule as V5) + index-tracking ETF names
ETF_PATTERN = r"(BEES|ETF|IETF)$|^(LIQUID|GOLD|SILVER)|^(PSUBANK|BANKNIFTY|NIFTY|SENSEX|CPSE|BHARAT)"


def _get(path: str):
    for attempt in range(3):
        try:
            return _api_json(BASE + path)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def shockers() -> pd.DataFrame:
    js = _get("live-analysis-volume-gainers") or {}
    rows = []
    for r in js.get("data", []) or []:
        avg = r.get("week1AvgVolume") or 0
        rows.append({"symbol": r.get("symbol"), "ltp": r.get("ltp"), "pct": r.get("pChange"), "volume": r.get("volume"),
                     "vol_x_week": (r.get("volume") or 0) / avg if avg else r.get("week1volChange"),
                     "turnover": (r.get("turnover") or 0) * 1e5,          # NSE reports turnover in Rs lakh
                     "name": r.get("companyName"), "source": "shocker"})
    return pd.DataFrame(rows)


def movers() -> pd.DataFrame:
    rows = []
    js = _get("live-analysis-variations?index=gainers") or {}
    for r in (js.get("allSec") or {}).get("data", []) or []:
        if r.get("series") not in (None, "EQ", "BE"):
            continue
        rows.append({"symbol": r.get("symbol"), "ltp": r.get("ltp"), "pct": r.get("perChange"),
                     "volume": r.get("trade_quantity"), "turnover": (r.get("turnover") or 0) * 1e5, "source": "gainer"})
    for kind in ("value", "volume"):
        js = _get(f"live-analysis-most-active-securities?index={kind}") or {}
        for r in js.get("data", []) or []:
            rows.append({"symbol": r.get("symbol"), "ltp": r.get("lastPrice"), "pct": r.get("pChange"),
                         "volume": r.get("totalTradedVolume"), "turnover": r.get("totalTradedValue"),
                         "source": f"active_{kind}"})
    return pd.DataFrame(rows)


def watchlist() -> pd.DataFrame:
    parts = [p for p in (shockers(), movers()) if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["symbol"])
    w = pd.concat(parts, ignore_index=True).dropna(subset=["symbol"])
    src = w.groupby("symbol")["source"].agg(lambda s: "+".join(sorted(set(s))))
    w = w.sort_values("vol_x_week", ascending=False, na_position="last").drop_duplicates("symbol")
    w["sources"] = w["symbol"].map(src)
    etf = w["symbol"].str.contains(ETF_PATTERN, regex=True, na=False) | \
        w.get("name", pd.Series("", index=w.index)).fillna("").str.contains(r"\bETF\b|Exchange Traded|Fund", case=False, regex=True)
    return w[~etf].drop(columns=["source"]).reset_index(drop=True)


def cap_buckets() -> dict:
    """symbol -> LARGE (NIFTY 100) / MID (NIFTY MIDCAP 150); others are SMALL.
    From NSE's published constituent CSVs, cached for the day."""
    f = os.path.join(CACHE, f"caps_{date.today():%Y%m%d}.json")
    if os.path.exists(f):
        return json.load(open(f))
    import io
    import requests
    out = {}
    for name, tag in (("ind_nifty100list.csv", "LARGE"), ("ind_niftymidcap150list.csv", "MID")):
        try:
            r = requests.get(f"https://archives.nseindia.com/content/indices/{name}",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            for s in pd.read_csv(io.StringIO(r.text))["Symbol"]:
                out.setdefault(str(s).strip(), tag)
        except Exception:
            continue
    if out:
        os.makedirs(CACHE, exist_ok=True)
        json.dump(out, open(f, "w"))
    return out
