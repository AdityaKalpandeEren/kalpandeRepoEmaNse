"""
NSE industry per stock, from NSE's index constituent files (NIFTY Total
Market = ~750 stocks, plus NIFTY 500), matched by ISIN first, then symbol.

These lists are CURRENT, so a stock that left the market years ago may be
missing; it gets an override below or "UNKNOWN". Industry is a slow-moving
static attribute, so using today's label for history is a small, stated
approximation (not a return lookahead).
"""
from __future__ import annotations

import io
import os

import pandas as pd

from data.nse_archives import _get

LISTS = ["https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
         "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"]

# Large former index members no longer listed (merged / delisted / renamed).
OVERRIDES = {
    "HDFC": "Financial Services", "MINDTREE": "Information Technology", "LTI": "Information Technology",
    "IBULHSGFIN": "Financial Services", "ZEEL": "Media, Entertainment & Publication",
    "CAIRN": "Oil Gas & Consumable Fuels", "IDEA": "Telecommunication", "BHEL": "Capital Goods",
    "MCDOWELL-N": "Fast Moving Consumer Goods", "UNITDSPR": "Fast Moving Consumer Goods",
    "INFRATEL": "Telecommunication", "SRTRANSFIN": "Financial Services", "CADILAHC": "Healthcare",
    "ZYDUSLIFE": "Healthcare", "MOTHERSUMI": "Automobile and Auto Components",
    "PEL": "Financial Services", "YESBANK": "Financial Services", "RCOM": "Telecommunication",
    "JPASSOCIAT": "Construction", "RELINFRA": "Power", "RPOWER": "Power", "UNITECH": "Realty",
    "DHFL": "Financial Services", "RELCAPITAL": "Financial Services", "SUZLON": "Capital Goods",
    "ADANIPOWER": "Power", "GMRINFRA": "Construction", "JSWENERGY": "Power", "IDFC": "Financial Services",
    "BHARATFIN": "Financial Services", "KOTAKBANK": "Financial Services", "TATAMTRDVR": "Automobile and Auto Components",
    "GRUH": "Financial Services", "HEXAWARE": "Information Technology", "WABCOINDIA": "Automobile and Auto Components",
    "CROMPGREAV": "Capital Goods", "OFSS": "Information Technology", "IGL": "Oil Gas & Consumable Fuels",
    "ZOMATO": "Consumer Services", "ETERNAL": "Consumer Services", "TATAMOTORS": "Automobile and Auto Components",
}


def load(cache_dir: str, refresh: bool = False) -> pd.DataFrame:
    """isin, symbol, industry - cached at cache/industries.parquet."""
    f = os.path.join(cache_dir, "industries.parquet")
    if os.path.exists(f) and not refresh:
        return pd.read_parquet(f)
    frames = []
    for url in LISTS:
        r = _get(url)
        if r is None:
            continue
        d = pd.read_csv(io.StringIO(r.text))
        frames.append(pd.DataFrame({"isin": d["ISIN Code"].str.strip(), "symbol": d["Symbol"].str.strip(),
                                    "industry": d["Industry"].str.strip()}))
    out = pd.concat(frames).drop_duplicates("isin")
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(f, index=False)
    return out


def assign(entities: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """entities: columns entity, symbols (set of all symbols used), isins (set).
    Returns industry per entity."""
    by_isin = dict(zip(table["isin"], table["industry"]))
    by_sym = dict(zip(table["symbol"], table["industry"]))
    out = {}
    for _, e in entities.iterrows():
        ind = next((by_isin[i] for i in e["isins"] if i in by_isin), None)
        ind = ind or next((by_sym[s] for s in e["symbols"] if s in by_sym), None)
        ind = ind or next((OVERRIDES[s] for s in e["symbols"] if s in OVERRIDES), None)
        out[e["entity"]] = ind or "UNKNOWN"
    return pd.Series(out, name="industry")
