"""
NSE corporate actions -> per-(symbol, ex-date) price adjustment.

The bhavcopy's PREVCLOSE is NOT adjusted on an ex-date (RELIANCE's 1:1
bonus on 2024-10-28 shows close 1334 vs prevclose 2656), so the raw
close/prevclose return would book a fake -50%. This module reads NSE's
corporate-actions feed (exact ex-dates and terms) and returns, per
(symbol, ex_date):

  factor   multiply PREVCLOSE by this to make it comparable with the ex-date
           close: bonus a:b -> b/(a+b); split / consolidation FV X -> Y -> Y/X
  demerger True for demergers / schemes of arrangement / rights issues: the
           value that left the share isn't in the feed in usable form, so the
           ex-date's opening gap is neutralised (return = close/open - 1)

Dividends are not adjusted (left out on purpose: price returns, like the
benchmark).
"""
from __future__ import annotations

import logging
import os
import re
from datetime import date

import pandas as pd

from data.nse_archives import _api_json

log = logging.getLogger(__name__)
URL = ("https://www.nseindia.com/api/corporates-corporateActions?index=equities"
       "&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}")


def fetch(cache_dir: str, start: date, end: date, refresh: bool = False) -> pd.DataFrame:
    """Raw feed (symbol, isin, ex_date, subject), one request per month, cached."""
    f = os.path.join(cache_dir, "corporate_actions.parquet")
    if os.path.exists(f) and not refresh:
        return pd.read_parquet(f)
    rows = []
    for m in pd.period_range(start, end, freq="M"):
        a, b = m.start_time.date(), min(m.end_time.date(), end)
        j = _api_json(URL.format(a=a, b=b))
        if j is None:
            raise RuntimeError(f"corporate actions: no response for {m}")
        for r in j:
            rows.append({"symbol": str(r.get("symbol", "")).strip(), "isin": r.get("isin"),
                         "series": r.get("series"), "ex_date": pd.to_datetime(r.get("exDate"), format="%d-%b-%Y",
                                                                               errors="coerce"),
                         "subject": str(r.get("subject", "")).strip()})
    out = pd.DataFrame(rows).dropna(subset=["ex_date"]).drop_duplicates()
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(f, index=False)
    log.info("corporate actions: %d rows %s .. %s", len(out), start, end)
    return out


_BONUS = re.compile(r"bonus[^0-9]{0,12}?(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.I)
_NOT_EQUITY = ("debenture", "preference", "ncrps", "ncrp", "redeemable")
_FV = re.compile(r"(?:rs\.?|re\.?|inr)\s*(\d+(?:\.\d+)?)\s*/?-?.*?to\s*(?:rs\.?|re\.?|inr)\s*(\d+(?:\.\d+)?)", re.I)


def parse(subject: str) -> tuple[float, bool]:
    """(price factor, is_demerger) for one subject line."""
    s = subject.lower()
    factor = 1.0
    bonus_hits = [] if any(w in s for w in _NOT_EQUITY) else _BONUS.findall(subject)
    for a, b in bonus_hits:
        a, b = float(a), float(b)
        if a > 0 and b > 0:
            factor *= b / (a + b)
    if "split" in s or "sub-division" in s or "subdivision" in s or "consolidat" in s:
        m = _FV.search(subject)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            if x > 0 and y > 0:
                factor *= y / x
    # demergers and rights issues: the value moving out of the share isn't in
    # the feed in usable form -> the ex-date opening gap is neutralised instead
    demerger = "demerger" in s or "arrangement" in s or "spin" in s or s.startswith("rights") or " rights " in f" {s} "
    return factor, demerger


def adjustments(actions: pd.DataFrame) -> pd.DataFrame:
    """symbol, isin, date, factor, demerger - one row per (isin, ex date)."""
    a = actions[actions["series"].fillna("EQ").isin(["EQ", "BE", "-", ""])].copy()
    parsed = a["subject"].map(parse)
    a["factor"] = [p[0] for p in parsed]
    a["demerger"] = [p[1] for p in parsed]
    a = a[(a["factor"] != 1.0) | a["demerger"]]
    g = a.groupby(["isin", "ex_date"]).agg(symbol=("symbol", "first"), factor=("factor", "prod"),
                                          demerger=("demerger", "max")).reset_index()
    return g.rename(columns={"ex_date": "date"})
