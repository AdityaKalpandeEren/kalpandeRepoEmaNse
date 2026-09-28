"""
Symbol -> sector index + industry themes, for model L_ML_META_V3.

Two levels, because they answer different questions:
  SECTOR  one of the NSE sector indices Upstox serves intraday - the
          stock's "own market" for relative-strength and for the rolling
          macro betas (strategy/market_context_v3.py).
  THEMES  finer industry tags ("OMC", "UPSTREAM", "INSURANCE", ...) that
          news rules and the Claude catalyst reader use to say WHO a
          headline hits: "crude spikes" is good for UPSTREAM and bad for
          OMC/PAINTS/AVIATION even though several share a sector.

Built automatically from Yahoo's sector/industry for `SYMBOL.NS`, cached
on disk (config.ML_V2_CACHE_DIR/sector_map.json) so each symbol is looked
up once. A stock added to the watchlist gets mapped on first use.
config.ML_V3_SECTOR_OVERRIDES wins over the automatic mapping.
"""
import json
import os
import threading

import config

# code -> Upstox index instrument key
SECTORS = {
    "BANK": "NSE_INDEX|Nifty Bank",
    "PSUBANK": "NSE_INDEX|Nifty PSU Bank",
    "FIN": "NSE_INDEX|Nifty Fin Service",
    "IT": "NSE_INDEX|Nifty IT",
    "ENERGY": "NSE_INDEX|Nifty Energy",
    "METAL": "NSE_INDEX|Nifty Metal",
    "PHARMA": "NSE_INDEX|Nifty Pharma",
    "FMCG": "NSE_INDEX|Nifty FMCG",
    "AUTO": "NSE_INDEX|Nifty Auto",
    "REALTY": "NSE_INDEX|Nifty Realty",
    "MEDIA": "NSE_INDEX|Nifty Media",
    "INFRA": "NSE_INDEX|Nifty Infra",
    "COMMOD": "NSE_INDEX|Nifty Commodities",
    "OTHER": "NSE_INDEX|Nifty 50",
}
SECTOR_CODES = list(SECTORS)

# (substring in Yahoo industry, lower-case) -> theme
_THEME_RULES = [
    ("refining & marketing", "OMC"), ("e&p", "UPSTREAM"), ("oil & gas integrated", "UPSTREAM"),
    ("oil & gas drilling", "UPSTREAM"), ("oil & gas equipment", "UPSTREAM"),
    ("utilities - regulated gas", "GAS"), ("natural gas", "GAS"), ("insurance", "INSURANCE"), ("airlines", "AVIATION"),
    ("specialty chemicals", "CHEMICALS"), ("chemicals", "CHEMICALS"), ("paint", "PAINTS"),
    ("steel", "STEEL"), ("aluminum", "NONFERROUS"), ("copper", "NONFERROUS"),
    ("other industrial metals", "NONFERROUS"), ("coal", "COAL"), ("gold", "JEWELLERY"),
    ("luxury goods", "JEWELLERY"), ("banks", "BANKS"), ("credit services", "NBFC"),
    ("mortgage", "NBFC"), ("capital markets", "BROKING"), ("asset management", "AMC"),
    ("financial data", "BROKING"), ("real estate", "REALTY"), ("auto manufacturers", "AUTO"),
    ("auto parts", "AUTO_ANCILLARY"), ("farm & heavy", "CAPITAL_GOODS"),
    ("information technology", "IT"), ("software", "IT"), ("drug manufacturers", "PHARMA"),
    ("biotechnology", "PHARMA"), ("medical", "HEALTHCARE"), ("telecom", "TELECOM"),
    ("utilities", "POWER"), ("renewable", "POWER"), ("building materials", "CEMENT"),
    ("tobacco", "FMCG"), ("packaged foods", "FMCG"), ("household", "FMCG"),
    ("beverages", "FMCG"), ("aerospace & defense", "DEFENCE"), ("engineering & construction", "INFRA"),
    ("railroads", "RAILWAYS"), ("marine shipping", "SHIPPING"), ("internet", "INTERNET"),
    ("specialty retail", "RETAIL"), ("department stores", "RETAIL"), ("fertilizers", "FERTILISER"),
    ("agricultural inputs", "FERTILISER"), ("textile", "TEXTILES"), ("electrical equipment", "CAPITAL_GOODS"),
    ("specialty industrial machinery", "CAPITAL_GOODS"), ("travel", "TRAVEL"), ("lodging", "HOTELS"),
]
# Names Yahoo classifies too broadly for the themes that matter to news.
_THEME_OVERRIDES = {
    "ASIANPAINT": ["PAINTS"], "BERGEPAINT": ["PAINTS"], "INDIGO": ["AVIATION"],
    "POLICYBZR": ["INSURANCE", "INTERNET"], "ETERNAL": ["INTERNET"], "NYKAA": ["INTERNET", "RETAIL"],
    "PAYTM": ["INTERNET", "NBFC"], "IRCTC": ["RAILWAYS", "TRAVEL"], "HAL": ["DEFENCE"], "BEL": ["DEFENCE"],
}

_lock = threading.Lock()
_cache = None


def _path() -> str:
    os.makedirs(config.ML_V2_CACHE_DIR, exist_ok=True)
    return os.path.join(config.ML_V2_CACHE_DIR, "sector_map.json")


def _load():
    global _cache
    if _cache is None:
        try:
            with open(_path()) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def _save():
    tmp = _path() + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({k: v for k, v in _cache.items() if v.get("resolved")}, f, indent=1, sort_keys=True)
    os.replace(tmp, _path())


def _sector_from(yf_sector: str, industry: str) -> str:
    s, ind = (yf_sector or "").lower(), (industry or "").lower()
    if "bank" in ind:
        return "BANK"
    if s == "financial services":
        return "FIN"
    if s == "technology":
        return "IT"
    if s in ("energy", "utilities"):
        return "ENERGY"
    if any(k in ind for k in ("steel", "aluminum", "copper", "metal", "mining", "coal")):
        return "METAL"
    if s == "basic materials":
        return "COMMOD"
    if s == "healthcare":
        return "PHARMA"
    if s == "consumer defensive":
        return "FMCG"
    if "auto" in ind:
        return "AUTO"
    if s == "real estate":
        return "REALTY"
    if s == "communication services" and "telecom" not in ind:
        return "MEDIA"
    if s == "industrials":
        return "INFRA"
    return "OTHER"


def _themes_from(industry: str) -> list:
    ind = (industry or "").lower()
    out = []
    for key, theme in _THEME_RULES:
        if key in ind and theme not in out:
            out.append(theme)
    return out


def lookup(symbol: str) -> dict:
    """{'sector': code, 'index_key': ..., 'themes': [...], 'industry': str,
    'yf_sector': str} - fetched once per symbol, then cached."""
    with _lock:
        cache = _load()
        rec = cache.get(symbol)
    if rec is None:
        yf_sector = industry = ""
        try:
            import yfinance as yf
            info = yf.Ticker(f"{symbol}.NS").get_info() or {}
            yf_sector, industry = info.get("sector") or "", info.get("industry") or ""
        except Exception:
            pass
        rec = {"yf_sector": yf_sector, "industry": industry, "resolved": bool(yf_sector or industry)}
        with _lock:
            cache = _load()
            cache[symbol] = rec
            # Only PERSIST successful lookups, so a network blip is retried on
            # the next run; a miss is remembered in memory for this run only.
            if rec["resolved"]:
                try:
                    _save()
                except Exception:
                    pass
    # Sector/themes are derived at lookup time from the cached raw Yahoo
    # fields, so rule changes apply without re-fetching anything.
    rec = dict(rec)
    rec["sector"] = config.ML_V3_SECTOR_OVERRIDES.get(
        symbol, _sector_from(rec.get("yf_sector", ""), rec.get("industry", "")))
    rec["themes"] = _themes_from(rec.get("industry", ""))
    if symbol in _THEME_OVERRIDES:
        rec["themes"] = list(dict.fromkeys(_THEME_OVERRIDES[symbol] + rec["themes"]))
    rec["index_key"] = SECTORS[rec["sector"]]
    return rec
