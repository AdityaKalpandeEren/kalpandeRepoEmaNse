"""
Live catalyst reader for model L_ML_META_V3 (NSE) - LIVE ONLY.

Answers, at the moment V3 is about to take a trade: "is there fresh news
that is good or bad for THIS stock - directly, or through its industry?"

Two layers:
  MARKET  one read every config.ML_V3_NEWS_CACHE_SECONDS of the market,
          oil and currency news feeds (^NSEI, ^NSEBANK, Brent, WTI,
          USD/INR). Output: an overall market score plus per-INDUSTRY-
          THEME and per-SYMBOL impacts - "IRDAI caps commissions" ->
          INSURANCE negative -> POLICYBZR; "crude spikes 5%" -> UPSTREAM
          positive, OMC / AVIATION / PAINTS negative.
  STOCK   the stock's own headlines (company-name matched, opinion pieces
          dropped - strategy/news_catalyst.py), read only when needed.

Headlines: Indian RSS feeds (ET, Business Standard, Mint) + Google News
per company - strategy/india_news.py (config.INDIA_NEWS_SOURCE="rss").

Readers:
  - LLM (config.ML_V3_LLM = "auto" + a key for config.NEWS_LLM_PROVIDER -
    Gemini free tier by default - or "true"; strategy/llm_client.py):
    reads the headlines with the stock list and industry themes and
    returns structured JSON. Understands context keywords can't ("profit
    falls less than feared", "RBI holds but signals cuts"). Answers are
    stored between runs so the same headlines are never asked twice.
  - Rules fallback (no key, quota hit, or any error): the V2 lexicon for
    stock headlines + a table of unambiguous Indian sector rules below.
The bot never fails because of this module; worst case it returns
"no catalyst".

WHY LIVE ONLY: Yahoo keeps no headline archive, so none of this can be
backtested or trained on. The PRICE-based versions of the same ideas
(overnight crude x each stock's crude sensitivity, sector strength) ARE
in V3's model and were validated; this layer only adds what prices
haven't shown yet. Its value can only be measured from paper results.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config
from strategy import news_catalyst, sector_map

MARKET_FEEDS = ["^NSEI", "^NSEBANK", "BZ=F", "CL=F", "USDINR=X"]
_MARKET_LOOKBACK_H = 10

# (regex on the headline, {theme: score}) - only unambiguous, well-known
# transmission channels. Everything subtler is left to Claude.
_SECTOR_RULES = [
    (r"\b(crude|oil prices?|brent)\b.{0,40}\b(surges?|jumps?|spikes?|soars?|rall(y|ies)|climbs?|rises?|hits? .{0,15}high)",
     {"UPSTREAM": 0.4, "OMC": -0.4, "AVIATION": -0.4, "PAINTS": -0.3, "CHEMICALS": -0.2}),
    (r"\b(crude|oil prices?|brent)\b.{0,40}\b(falls?|slumps?|plunges?|drops?|tumbles?|slides?|sinks?)",
     {"UPSTREAM": -0.4, "OMC": 0.4, "AVIATION": 0.4, "PAINTS": 0.3, "CHEMICALS": 0.2}),
    (r"\brupee\b.{0,30}\b(falls?|weakens?|slumps?|slides?|record low|hits? .{0,10}low|depreciat)",
     {"IT": 0.3, "PHARMA": 0.2, "OMC": -0.2, "AVIATION": -0.2}),
    (r"\brupee\b.{0,30}\b(gains?|strengthens?|jumps?|rises?|appreciat)",
     {"IT": -0.2, "OMC": 0.2}),
    (r"\brbi\b.{0,40}\b(cuts?|lowers?|slashes?)\b.{0,20}\b(repo|rates?)\b|\brepo rate cut\b",
     {"BANKS": 0.3, "NBFC": 0.4, "REALTY": 0.4, "AUTO": 0.2}),
    (r"\brbi\b.{0,40}\b(hikes?|raises?)\b.{0,20}\b(repo|rates?)\b|\brepo rate hike\b",
     {"BANKS": -0.2, "NBFC": -0.4, "REALTY": -0.4, "AUTO": -0.2}),
    (r"\bwindfall (tax|levy)\b.{0,30}\b(cut|scrap|remov|abolish)", {"UPSTREAM": 0.5}),
    (r"\bwindfall (tax|levy)\b", {"UPSTREAM": -0.4}),
    (r"\birdai\b.{0,50}\b(caps?|curbs?|restricts?|bans?|tightens?|cuts?|limits?)\b", {"INSURANCE": -0.5}),
    (r"\bsebi\b.{0,50}\b(f&o|derivatives?)\b.{0,30}\b(curbs?|tighten|restrict|new rules)", {"BROKING": -0.5}),
    (r"\b(safeguard|anti-dumping|import) duty\b.{0,30}\bsteel\b|\bsteel\b.{0,30}\b(safeguard|anti-dumping|import) duty\b",
     {"STEEL": 0.4}),
    (r"\bexport duty\b.{0,30}\bsteel\b", {"STEEL": -0.4}),
    (r"\btariff (hike|increase)s?\b.{0,30}\b(telecom|mobile|jio|airtel|vodafone)\b|\b(telecom|mobile)\b.{0,30}\btariff (hike|increase)",
     {"TELECOM": 0.4}),
    (r"\bh-?1b\b.{0,40}\b(curbs?|fee|restrict|ban|tighten)", {"IT": -0.4}),
    # Pharma / healthcare. The exemption rule sits BEFORE the generic pharma-
    # tariff rule (first matching rule wins per headline): "US exempts Indian
    # drugs from pharma tariff" is good news, not a tariff hit.
    (r"\b(exempts?|exemption|spares?|relief)\b.{0,60}\b(pharma|drugs?|medicines?|generics?)\b|"
     r"\b(pharma|drugs?|medicines?|generics?)\b.{0,60}\b(exempt\w*|spared)\b", {"PHARMA": 0.4}),
    (r"\b(nppa|drug price control|price cap)\b.{0,40}\b(drugs?|medicines?|stents?|devices?)\b", {"PHARMA": -0.3}),
    (r"\b(health insurance|ayushman|pm-jay)\b.{0,40}\b(expan\w*|hike|raise|cover)", {"HEALTHCARE": 0.3}),
    # Sugar / ethanol
    (r"\bsugar\b.{0,40}\bexports?\b.{0,30}\b(allow\w*|permit\w*|quota (raised|hiked|increased)|opens?)\b|"
     r"\b(allow\w*|permit\w*)\b.{0,30}\bsugar exports?\b", {"SUGAR": 0.4}),
    (r"\bsugar\b.{0,40}\bexports?\b.{0,30}\b(ban\w*|curbs?|restrict\w*|halt\w*)\b|\bban on sugar exports?\b|"
     r"\b(bans?|curbs?|halts?|restricts?)\b.{0,20}\bsugar exports?\b", {"SUGAR": -0.4}),
    (r"\bethanol\b.{0,50}\b(price|procurement)\b.{0,30}\b(hike|raised?|increase\w*)\b|"
     r"\b(raises?|hikes?|increases?)\b.{0,30}\bethanol (price|procurement)", {"SUGAR": 0.4}),
    (r"\b(minimum selling price|msp)\b.{0,30}\bsugar\b.{0,30}\b(hike|raised?|increase\w*)\b|"
     r"\bsugar\b.{0,30}\b(msp|minimum selling price)\b.{0,30}\b(hike|raised?|increase\w*)\b", {"SUGAR": 0.4}),
    (r"\b(frp|fair and remunerative price|cane price)\b.{0,40}\b(hike|raised?|increase\w*)\b|"
     r"\b(hikes?|raises?|increases?)\b.{0,20}\b(frp|fair and remunerative price|cane price)\b", {"SUGAR": -0.2}),
    (r"\bsugar prices?\b.{0,30}\b(rise|rises|jump\w*|surge\w*|firm\w*|climb\w*)\b", {"SUGAR": 0.3}),
    (r"\bsugar prices?\b.{0,30}\b(fall|falls|drop\w*|slump\w*|slide\w*|decline\w*)\b", {"SUGAR": -0.3}),
    (r"\btariffs?\b.{0,40}\b(pharma|drugs?|medicines?)\b|\b(pharma|drugs?)\b.{0,40}\btariffs?\b", {"PHARMA": -0.4}),
    (r"\bdefen[cs]e\b.{0,40}\b(budget|orders?|procurement|contracts?|deals?)\b", {"DEFENCE": 0.3}),
    (r"\b(gst|excise)\b.{0,30}\b(cut|reduc)\w*\b.{0,30}\b(cars?|autos?|vehicles?|two-wheelers?)\b", {"AUTO": 0.4}),
    (r"\bmonsoon\b.{0,30}\b(deficient|weak|below normal|delayed)", {"FMCG": -0.3, "FERTILISER": -0.2, "AUTO": -0.2}),
    (r"\bmonsoon\b.{0,30}\b(above normal|good|surplus|strong)", {"FMCG": 0.2, "FERTILISER": 0.2}),
]
_SECTOR_RX = [(re.compile(p, re.I), impacts) for p, impacts in _SECTOR_RULES]


@dataclass
class MarketRead:
    market_score: float = 0.0
    themes: dict = field(default_factory=dict)     # THEME -> (score, reason)
    symbols: dict = field(default_factory=dict)    # SYMBOL -> (score, reason)
    method: str = "none"
    n_headlines: int = 0


@dataclass
class StockRead:
    score: float = 0.0
    materiality: str = "low"
    catalyst_type: str = "none"
    reason: str = ""
    method: str = "none"
    n_headlines: int = 0


def llm_enabled() -> bool:
    mode = config.ML_V3_LLM
    if mode == "false":
        return False
    from strategy import llm_client
    if mode == "true":
        return True
    return llm_client.available()


# ═══════════════════════════════════════════════════════════════════
# LLM (strategy/llm_client.py - Gemini by default, Claude optional)
# ═══════════════════════════════════════════════════════════════════

def _llm_json(system: str, user: str, schema: dict, max_tokens: int = 4096, tag: str = ""):
    """Structured JSON from the configured LLM, or None on ANY failure (-> rules)."""
    from strategy import llm_client
    return llm_client.json_call(system, user, schema, tag=tag or "V3 catalyst", max_tokens=max_tokens)


_MARKET_SCHEMA = {
    "type": "object",
    "properties": {
        "market_score": {"type": "number"},
        "theme_impacts": {"type": "array", "items": {
            "type": "object",
            "properties": {"theme": {"type": "string"}, "score": {"type": "number"},
                           "reason": {"type": "string"}},
            "required": ["theme", "score", "reason"], "additionalProperties": False}},
        "symbol_impacts": {"type": "array", "items": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "score": {"type": "number"},
                           "reason": {"type": "string"}},
            "required": ["symbol", "score", "reason"], "additionalProperties": False}},
    },
    "required": ["market_score", "theme_impacts", "symbol_impacts"],
    "additionalProperties": False,
}
_MARKET_SYSTEM = (
    "You are an Indian equity (NSE) intraday catalyst analyst. From the headlines, identify NEW, "
    "MATERIAL events and how they affect Indian stocks over the next few trading hours: government "
    "policy and regulation (ministries, RBI, SEBI, IRDAI, TRAI, DGCA, tariffs, duties, taxes), commodity "
    "moves (crude oil, metals, gold), the rupee, global risk, and sector-specific news. Score -1.0 "
    "(strongly negative) to +1.0 (strongly positive). Use only the listed industry themes and symbols. "
    "Think through the transmission channel: e.g. higher crude helps upstream producers but hurts "
    "refiners/OMCs, airlines and paint makers; an insurance regulator capping commissions hurts "
    "insurance distributors. Ignore opinion pieces, stock tips and recaps of old moves. Be "
    "conservative: most headlines deserve no impact; only clear new events above |0.5|."
)

_STOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "materiality": {"type": "string", "enum": ["low", "medium", "high"]},
        "catalyst_type": {"type": "string", "enum": [
            "earnings", "order_contract", "regulation_policy", "legal_probe", "management",
            "corporate_action", "rating_broker", "commodity_macro", "product_business", "none"]},
        "reason": {"type": "string"},
    },
    "required": ["score", "materiality", "catalyst_type", "reason"],
    "additionalProperties": False,
}
_STOCK_SYSTEM = (
    "You judge whether fresh news is a NEW, MATERIAL catalyst for one Indian (NSE) stock over the next "
    "few trading hours. Score -1.0 (strong negative) to +1.0 (strong positive); 0 if the headlines are "
    "opinion, tips, old news, or about other companies. Consider the company's industry and today's "
    "macro context. Be conservative."
)


# ═══════════════════════════════════════════════════════════════════
# Market layer
# ═══════════════════════════════════════════════════════════════════
_market_cache = {"at": 0.0, "read": None}


def _all_themes() -> list:
    themes = {t for _, t in sector_map._THEME_RULES}
    for v in sector_map._THEME_OVERRIDES.values():
        themes.update(v)
    return sorted(themes)


def _market_headlines(now) -> list:
    if config.INDIA_NEWS_SOURCE == "rss":
        # Indian market / economy / companies feeds (strategy/india_news.py):
        # market-wide AND sector/company stories, since this layer's job is
        # to find WHO a headline hits.
        from strategy import india_news
        return [x for x in india_news.all_market_headlines(_MARKET_LOOKBACK_H, now)
                if not news_catalyst._OPINION_RX.search(x[1])][:60]
    items = []
    for feed in MARKET_FEEDS:
        items += news_catalyst.fetch_headlines(feed, _MARKET_LOOKBACK_H, now, market=True) \
            if feed.startswith("^") else _raw_feed(feed, now)
    seen, out = set(), []
    for ts, title, summary in sorted(items, key=lambda x: x[0], reverse=True):
        if title not in seen:
            seen.add(title)
            out.append((ts, title, summary))
    return out[:40]


def _raw_feed(ticker: str, now) -> list:
    """Commodity / FX feeds: every fresh, non-opinion headline counts."""
    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw:
        title, summary, ts = news_catalyst._parse_item(item)
        if title and ts and 0 <= (now - ts).total_seconds() / 3600 <= _MARKET_LOOKBACK_H:
            if not news_catalyst._OPINION_RX.search(title):
                out.append((ts, title, summary))
    return out


def _rules_market(headlines) -> MarketRead:
    read = MarketRead(method="rules", n_headlines=len(headlines))
    # The market score reads only market-wide headlines: the Indian feeds
    # are full of single-stock stories that say nothing about the market.
    from strategy.india_news import MARKET_RX
    scores = [news_catalyst.score_headline(t) for _, t, _ in headlines if MARKET_RX.search(t)]
    nz = [s for s in scores if s]
    read.market_score = round(sum(nz) / len(nz), 3) if nz else 0.0
    for _, title, _ in headlines:
        for rx, impacts in _SECTOR_RX:
            if rx.search(title):
                for theme, sc in impacts.items():
                    prev = read.themes.get(theme, (0.0, ""))[0]
                    if abs(sc) > abs(prev):
                        read.themes[theme] = (sc, title[:120])
                break
    return read


def _read_to_dict(read: MarketRead) -> dict:
    return {"market_score": read.market_score, "themes": read.themes, "symbols": read.symbols,
            "method": read.method, "n_headlines": read.n_headlines}


def _read_from_dict(d: dict) -> MarketRead:
    return MarketRead(market_score=d["market_score"],
                      themes={k: tuple(v) for k, v in d["themes"].items()},
                      symbols={k: tuple(v) for k, v in d["symbols"].items()},
                      method=d["method"], n_headlines=d["n_headlines"])


def market_read(universe=None) -> MarketRead:
    """Market / industry read. The LLM answer is stored between runs
    (strategy/llm_client.py cache): re-asked only when the headlines have
    changed AND the stored read is older than config.ML_V3_NEWS_CACHE_SECONDS."""
    from strategy import llm_client
    now_t = time.time()
    if _market_cache["read"] is not None and now_t - _market_cache["at"] < config.ML_V3_NEWS_CACHE_SECONDS:
        return _market_cache["read"]
    now = datetime.now(timezone.utc)
    headlines = _market_headlines(now)
    read = None
    if headlines and llm_enabled():
        universe = universe or []
        key = llm_client.make_key("v3m", sorted(universe), [t for _, t, _ in headlines])
        same, _ = llm_client.cache_get("v3m", key)
        last, age = llm_client.cache_get("v3m", "latest", max_age_s=config.ML_V3_NEWS_CACHE_SECONDS)
        if same or last:
            read = _read_from_dict(same or last)
        else:
            stock_lines = "\n".join(f"{s}: {sector_map.lookup(s).get('industry') or 'unknown'}" for s in universe)
            listing = "\n".join(f"- {t}" + (f" — {s[:200]}" if s else "") for _, t, s in headlines)
            data = _llm_json(
                _MARKET_SYSTEM,
                f"Industry themes: {', '.join(_all_themes())}\n\nStocks (symbol: industry):\n{stock_lines}\n\n"
                f"Headlines (last {_MARKET_LOOKBACK_H}h):\n{listing}",
                _MARKET_SCHEMA, tag="V3 market")
            if data:
                read = MarketRead(market_score=max(-1, min(1, float(data.get("market_score", 0)))),
                                  method=config.NEWS_LLM_PROVIDER, n_headlines=len(headlines))
                for it in data.get("theme_impacts", []):
                    read.themes[str(it["theme"]).upper()] = (max(-1, min(1, float(it["score"]))), it["reason"][:160])
                for it in data.get("symbol_impacts", []):
                    read.symbols[str(it["symbol"]).upper()] = (max(-1, min(1, float(it["score"]))), it["reason"][:160])
                llm_client.cache_put("v3m", key, _read_to_dict(read))
                llm_client.cache_put("v3m", "latest", _read_to_dict(read))
    if read is None:
        read = _rules_market(headlines)
    _market_cache.update(at=now_t, read=read)
    return read


# ═══════════════════════════════════════════════════════════════════
# Stock layer
# ═══════════════════════════════════════════════════════════════════
_stock_cache = {}


def stock_read(symbol: str, macro_note: str = "") -> StockRead:
    from strategy import llm_client
    hit = _stock_cache.get(symbol)
    if hit and time.time() - hit[0] < config.ML_V3_NEWS_CACHE_SECONDS:
        return hit[1]
    now = datetime.now(timezone.utc)
    items = news_catalyst.fetch_headlines(symbol, config.ML_V2_NEWS_SYMBOL_LOOKBACK_HOURS, now)
    read = StockRead(n_headlines=len(items))
    if items and llm_enabled():
        # Same headlines -> same answer: stored between runs, asked once.
        key = llm_client.make_key("v3s", symbol, [t for _, t, _ in items[:12]])
        stored, _ = llm_client.cache_get("v3s", key)
        if stored:
            read = StockRead(**stored)
        else:
            info = sector_map.lookup(symbol)
            listing = "\n".join(f"- {t}" + (f" — {s[:300]}" if s else "") for _, t, s in items[:12])
            data = _llm_json(
                _STOCK_SYSTEM,
                f"Stock: {symbol} ({info.get('industry') or 'unknown industry'}, themes {info['themes']})\n"
                f"Today's macro: {macro_note or 'n/a'}\nHeadlines (last 24h):\n{listing}",
                _STOCK_SCHEMA, max_tokens=2048, tag=f"V3 stock {symbol}")
            if data:
                read = StockRead(score=max(-1, min(1, float(data["score"]))), materiality=data["materiality"],
                                 catalyst_type=data["catalyst_type"], reason=data["reason"][:200],
                                 method=config.NEWS_LLM_PROVIDER, n_headlines=len(items))
                llm_client.cache_put("v3s", key, read.__dict__)
    if read.method == "none" and items:
        lex = news_catalyst._aggregate(symbol, items, now)
        read = StockRead(score=lex.score, materiality="medium" if abs(lex.score) >= 0.5 else "low",
                         catalyst_type="unknown", reason=lex.top_headline[:200], method="rules",
                         n_headlines=len(items))
    _stock_cache[symbol] = (time.time(), read)
    return read


# ═══════════════════════════════════════════════════════════════════
# Decision used by the V3 model
# ═══════════════════════════════════════════════════════════════════

def entry_decision_v3(symbol: str, direction: str, universe=None, macro_note: str = ""):
    """(veto_reason_or_None, prob_adjustment <= 0, summary_str)."""
    sign = 1.0 if direction == "long" else -1.0
    info = sector_map.lookup(symbol)
    mk = market_read(universe)
    st = stock_read(symbol, macro_note)

    theme_scores = [(mk.themes[t][0], t, mk.themes[t][1]) for t in info["themes"] if t in mk.themes]
    sym_imp = mk.symbols.get(symbol)
    industry = 0.0
    why_ind = ""
    if theme_scores:
        worst = min(theme_scores, key=lambda x: sign * x[0])
        best = max(theme_scores, key=lambda x: sign * x[0])
        pick = worst if sign * worst[0] < 0 else best
        industry, why_ind = pick[0], f"{pick[1]}: {pick[2]}"
    if sym_imp and abs(sym_imp[0]) > abs(industry):
        industry, why_ind = sym_imp[0], f"{symbol}: {sym_imp[1]}"

    summary = (f"stock {st.score:+.2f} ({st.method}, {st.catalyst_type}) | industry {industry:+.2f} | "
               f"market {mk.market_score:+.2f} ({mk.method})")
    if sign * st.score <= -config.ML_V3_STOCK_VETO:
        return f"bad stock catalyst: {st.reason}", 0.0, summary
    if sign * industry <= -config.ML_V3_SECTOR_VETO:
        return f"bad industry catalyst: {why_ind}", 0.0, summary
    if sign * mk.market_score <= -config.ML_V2_NEWS_MARKET_VETO:
        return "bad market news", 0.0, summary
    combined = sign * (st.score + 0.6 * industry)
    adj = 0.0
    if combined >= config.ML_V3_BOOST_AT:
        adj = -config.ML_V3_BOOST_PROB * min(1.0, combined)
    return None, adj, summary
