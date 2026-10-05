"""
Indian news headlines for the live catalyst readers (L_ML_META_V2's
strategy/news_catalyst.py and L_ML_META_V3's strategy/llm_catalyst.py).

Why not Yahoo: Yahoo's `.NS` / ^NSEI feeds were checked live and are days
to weeks stale and often about other companies (e.g. RELIANCE -> "Reliance
(NYSE:RS)"), so every NSE news read came back "no catalyst". These free
RSS feeds carry minutes-old Indian market news:

  MARKET  config.INDIA_NEWS_MARKET_FEEDS - Economic Times, Business
          Standard, Mint (markets, stocks, economy, companies, industry).
          (Moneycontrol's RSS was tested too - it stopped updating years
          ago, so it isn't used.)
  STOCK   Google News RSS search on the registered company name (from the
          Upstox instrument master), plus any market-feed headline that
          names the company.

No API key, no cost. Every fetch failure just means fewer headlines;
nothing here can stop the bot.
"""
import html
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import config

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/124.0 Safari/537.36"}
_FEED_TTL = 240          # seconds a fetched feed is reused within one run
_feeds = {}              # url -> (fetched_at, [(ts, title, summary)])
_TAG_RX = re.compile(r"<[^>]+>")

# Stricter than news_catalyst._MACRO_RX: the Indian feeds are full of single-
# stock stories ("X shares rise 3%..."), which must not count as MARKET news.
MARKET_RX = re.compile(
    r"\b(sensex|nifty|dalal street|indian (shares|stocks|equities|markets?)|market (crash|rally|sell-?off)|"
    r"rbi|repo|monetary policy|mpc|inflation|cpi|wpi|gdp|budget|fiscal deficit|gst council|"
    r"fiis?|fpis?|diis?|foreign (investors|funds|inflows|outflows)|rupee|crude|oil prices?|brent|"
    r"monsoon|tariffs?|us fed|federal reserve|fed (rate|chair|meeting)|us yields?|treasury yields?|"
    r"recession|war|geopolitic\w*|sebi)\b", re.I)
BANK_RX = re.compile(r"\b(banks?|banking|rbi|repo|nbfcs?|lenders?|credit growth|npa|deposit)\b", re.I)

# "RELIANCE INDUSTRIES LTD" -> "Reliance Industries" for the search query.
_SUFFIX_RX = re.compile(r"\b(LTD|LIMITED|LTD\.|PVT|PRIVATE|CO|CORPN|CORPORATION|INDIA LTD|THE)\b\.?", re.I)


def _parse_ts(text):
    try:
        ts = parsedate_to_datetime(text)
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _clean(text: str) -> str:
    return html.unescape(_TAG_RX.sub(" ", text or "")).replace("\xa0", " ").strip()


def fetch_feed(url: str, google: bool = False) -> list:
    """[(published_utc, title, summary)] from one RSS feed, cached for the run."""
    hit = _feeds.get(url)
    if hit and time.time() - hit[0] < _FEED_TTL:
        return hit[1]
    import requests
    out = []
    try:
        r = requests.get(url, headers=_UA, timeout=15)
        r.raise_for_status()
        for item in ET.fromstring(r.content).iter("item"):
            title = _clean(item.findtext("title"))
            ts = _parse_ts(item.findtext("pubDate") or "")
            if not title or ts is None:
                continue
            if google:
                # Google News titles end with " - Source"; its description is
                # just links, not a summary.
                title = re.sub(r"\s+-\s+[^-]{2,60}$", "", title)
                summary = ""
            else:
                summary = _clean(item.findtext("description"))[:400]
            out.append((ts, title, summary))
    except Exception as e:
        print(f"[india-news] feed failed {url[:70]}: {str(e)[:80]}")
    _feeds[url] = (time.time(), out)
    return out


def _fresh(items, lookback_h, now) -> list:
    seen, out = set(), []
    for ts, title, summary in sorted(items, key=lambda x: x[0], reverse=True):
        age_h = (now - ts).total_seconds() / 3600.0
        key = re.sub(r"\W+", " ", title.lower()).strip()
        if -0.25 <= age_h <= lookback_h and key not in seen:
            seen.add(key)
            out.append((ts, title, summary))
    return out


def all_market_headlines(lookback_h: float, now: datetime = None) -> list:
    """Every fresh headline from the Indian market feeds (deduped, newest
    first) - market-wide AND single-stock/sector stories."""
    now = now or datetime.now(timezone.utc)
    items = []
    for url in config.INDIA_NEWS_MARKET_FEEDS:
        items += fetch_feed(url)
    return _fresh(items, lookback_h, now)


def market_headlines(lookback_h: float, now: datetime = None, banking: bool = False) -> list:
    """Only headlines about the market / macro backdrop (or banking)."""
    rx = BANK_RX if banking else MARKET_RX
    return [x for x in all_market_headlines(lookback_h, now) if rx.search(x[1])]


def search_name(symbol: str) -> str:
    """Google News search phrase for a company: config.INDIA_NEWS_NAMES if
    set, else the first two words of its registered name ("RELIANCE
    INDUSTRIES LTD" -> "Reliance Industries"; registered names are often
    abbreviated after that, e.g. "SUN PHARMACEUTICAL IND L")."""
    if symbol in config.INDIA_NEWS_NAMES:
        return config.INDIA_NEWS_NAMES[symbol][0]
    from data.instruments import get_company_name
    words = [w for w in _SUFFIX_RX.sub(" ", get_company_name(symbol) or symbol).replace(".", " ").split()
             if len(w) > 1]
    return " ".join(w.capitalize() if w.isalpha() else w for w in words[:2]) or symbol


def stock_headlines(symbol: str, lookback_h: float, now: datetime = None) -> list:
    """Fresh headlines that name this company: Google News search on its
    registered name + matching market-feed headlines."""
    from strategy.news_catalyst import is_relevant
    now = now or datetime.now(timezone.utc)
    items = []
    if config.INDIA_NEWS_GOOGLE_ENABLED:
        days = max(1, int(-(-lookback_h // 24)))
        # registered names are often truncated ("DIXON TECHNO") - also search
        # the name token used for relevance ("DIXON")
        from strategy.news_catalyst import _company_tokens
        phrases = [search_name(symbol)]
        for tok in _company_tokens(symbol)[1:]:
            if tok.lower() not in {p.lower() for p in phrases}:
                phrases.append(tok)
        q = quote_plus(" OR ".join(f'"{p}"' for p in phrases) + f" when:{days}d")
        items += fetch_feed(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en", google=True)
    items += all_market_headlines(lookback_h, now)
    return [x for x in _fresh(items, lookback_h, now) if is_relevant(x[1], symbol)]
