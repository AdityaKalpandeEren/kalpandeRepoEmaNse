"""
Upstox needs an 'instrument_key' (e.g. NSE_EQ|INE848E01016) rather than a
plain trading symbol for API calls. This module downloads Upstox's NSE
instrument master file and looks up the key for a given symbol.

Run refresh_instrument_master() whenever lookups start failing (the file
is small and changes rarely - weekly refresh is more than enough).
"""
import os
import gzip
import csv
import threading
import requests

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nse_instruments.csv")
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"

_cache = {}
_lock = threading.RLock()   # parallel fetchers (prefetch_today) all load on first use


def refresh_instrument_master():
    with _lock:
        resp = requests.get(INSTRUMENTS_URL)
        resp.raise_for_status()
        raw = gzip.decompress(resp.content).decode("utf-8")
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(raw)
        _cache.clear()
        print(f"Saved instrument master to {CACHE_FILE}")


def _load_cache():
    # Locked, and filled in one step: other threads must never see a half-loaded
    # cache (lookups failed for most symbols when 6 threads loaded at once).
    with _lock:
        if _cache:
            return
        if not os.path.exists(CACHE_FILE):
            refresh_instrument_master()
        keys, names = {}, {}
        with open(CACHE_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only keep NSE cash-market equities
                if row.get("instrument_type") == "EQUITY":
                    keys[row["tradingsymbol"]] = row["instrument_key"]
                    names[row["tradingsymbol"]] = row.get("name") or ""
        _names.update(names)
        _cache.update(keys)


_names = {}


def get_company_name(trading_symbol: str) -> str:
    """Registered company name, e.g. RELIANCE -> 'RELIANCE INDUSTRIES LTD'
    ('' if unknown). Used to find the company in news headlines."""
    if not _cache:
        _load_cache()
    return _names.get(trading_symbol, "")


def get_instrument_key(trading_symbol: str) -> str:
    if not _cache:
        _load_cache()
    if trading_symbol not in _cache:
        raise ValueError(
            f"Instrument key not found for '{trading_symbol}'. "
            f"Check the spelling matches the NSE trading symbol exactly, "
            f"or call refresh_instrument_master() if the file may be stale."
        )
    return _cache[trading_symbol]
