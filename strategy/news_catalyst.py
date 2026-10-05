"""
Live news-catalyst reader for model L_ML_META_V2 (NSE port).

NSE differences from the us_alert_bot version: Yahoo tickers are the
`.NS` symbol for stocks and ^NSEI / ^NSEBANK for the market feed; the
lexicon adds Indian-market catalyst language (order wins, block deals,
promoter pledges, SEBI/ED actions, QIP/OFS dilution, RBI repo moves,
FII flows); conglomerate names (Tata, Adani, Bajaj...) are matched on
their first TWO words so one group company's news doesn't count for all.

Answers one question for a symbol at a moment in time: "is there fresh,
clearly good or clearly bad news for this stock - or for the market as a
whole - right now?" as a score in [-1, +1] (negative = bearish).

Used in two places, LIVE ONLY:
  - entry: a strong opposing catalyst vetoes the trade (bad news vetoes
    a long, good news vetoes a short); an aligned catalyst lowers the
    V2 probability bar slightly (config.ML_V2_NEWS_*).
  - open-trade monitoring: fresh opposing news on a symbol you are
    already in raises an exit alert (see live_ml_v2.py).

Why live only: Yahoo serves just the latest ~10 headlines per ticker,
with no archive. There is no way to know what the headlines said on a
past date, so this cannot be backtested or trained on - see the
config.ML_V2_* block comment. Every backtest number reported for V2 is
therefore WITHOUT the news overlay; its live effect is unmeasured until
paper/live results accumulate.

Scoring:
  - default: a finance keyword/phrase lexicon (no API key, no cost).
    Deliberately conservative - it only reacts to unambiguous catalyst
    language (beats/misses, guidance, upgrades/downgrades, probes,
    offerings, bankruptcy...) and ignores opinion-piece noise.
  - optional (config.ML_V2_NEWS_LLM_ENABLED): Claude reads the same
    headlines and returns a structured score. Falls back to the lexicon
    on any error, so an API problem can never block the bot.
"""
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

import config


@dataclass
class CatalystRead:
    score: float = 0.0            # -1 (very bearish) .. +1 (very bullish)
    n_items: int = 0              # fresh headlines considered
    top_headline: str = ""        # the one that moved the score most
    method: str = "none"          # "lexicon" | "llm" | "none"
    headlines: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.n_items:
            return "no fresh news"
        return f"{self.score:+.2f} ({self.n_items} items, {self.method}): {self.top_headline[:90]}"


# ═══════════════════════════════════════════════════════════════════
# Lexicon. Weights are per-phrase; a headline's score is the clipped
# sum. Multi-word phrases are listed so e.g. "cuts guidance" isn't read
# as neutral, and "beats" isn't matched inside unrelated words.
# ═══════════════════════════════════════════════════════════════════
_BULLISH = {
    r"beats? (estimates|expectations|forecasts?|consensus)": 0.8,
    r"tops? (estimates|expectations|forecasts?)": 0.7,
    r"(raises|lifts|boosts|hikes) (its )?(guidance|forecast|outlook)": 0.9,
    r"record (revenue|sales|profit|quarter)": 0.6,
    r"upgrade[sd]? (to )?(buy|outperform|overweight)": 0.7,
    r"\bupgrade[sd]?\b": 0.4,
    r"price target (raised|hiked|increased|boosted)": 0.4,
    r"(raises|lifts|hikes) price target": 0.4,
    r"(fda|regulatory) approv": 0.8,
    r"\bapproval\b": 0.3,
    r"(buyback|share repurchase)": 0.5,
    r"(dividend (hike|increase)|raises dividend)": 0.4,
    r"(to acquire|agrees to buy|takeover bid|buyout offer)": 0.5,
    r"(wins|awarded|secures|lands) .{0,30}(contract|deal|order)": 0.5,
    r"(strategic )?partnership with": 0.3,
    r"\b(surges?|soars?|jumps?|rall(y|ies)|skyrockets?)\b": 0.4,
    r"\bstrong demand\b": 0.4,
    r"(rate cut|cuts rates)": 0.4,
    r"(cooler|softer)[- ]than[- ]expected (inflation|cpi)": 0.5,
    r"\bstimulus\b": 0.3,
    r"(trade deal|tariff relief|tariffs? (paused|lifted|eased))": 0.5,
    # Indian-market catalyst language
    r"\b(bags|wins|secures|receives|gets) .{0,25}(order|contract)s?\b": 0.5,
    r"\border (win|inflow)s?\b": 0.5,
    r"(net )?profit (rises|jumps|surges|soars|up|climbs|doubles)": 0.6,
    r"\bq[1-4] (beat|beats)\b": 0.6,
    r"(rating|outlook) upgrade|upgrades? (rating|outlook)": 0.5,
    r"\bbonus (issue|shares)\b": 0.4,
    r"promoters? (buys?|hikes? stake|increases? stake|raises? stake)": 0.5,
    r"\bfii (buying|inflows?)\b|foreign (investors|funds) (buy|return)": 0.4,
    r"rbi (cuts|lowers) (repo|rates?)|repo rate cut": 0.5,
    # Tariff EXEMPTION is good news - outweighs the generic "tariffs" -0.3 below.
    r"\bexempt\w*\b.{0,50}\btariffs?\b|\btariffs?\b.{0,50}\bexempt\w*\b": 0.6,
    # Pharma (USFDA)
    r"\b(us ?fda|usfda)\b.{0,30}\b(nod|approval|approves|clears?|green ?light)\b|\banda approval\b": 0.5,
    r"\b(eir|establishment inspection report)\b|\bzero (observations|483)\b|\bvai status\b": 0.4,
}
_BEARISH = {
    r"miss(es|ed)? (estimates|expectations|forecasts?|consensus)": -0.8,
    r"(cuts|lowers|slashes|trims|withdraws) (its )?(guidance|forecast|outlook)": -0.9,
    r"(weak|disappointing|soft) (guidance|outlook|forecast|results|quarter)": -0.7,
    r"profit warning": -0.9,
    r"downgrade[sd]? (to )?(sell|underperform|underweight|neutral|hold)": -0.7,
    r"\bdowngrade[sd]?\b": -0.4,
    r"price target (cut|lowered|reduced|slashed)": -0.4,
    r"(cuts|lowers|slashes) price target": -0.4,
    r"(sec|doj|ftc|antitrust) (probe|investigation|lawsuit|charges|sues)": -0.8,
    r"\b(probe|investigation|subpoena)\b": -0.5,
    r"\b(lawsuit|sued|class action)\b": -0.4,
    r"\brecall(s|ed)?\b": -0.5,
    r"\b(fda rejects?|rejection|complete response letter|clinical hold)\b": -0.8,
    r"(secondary|stock|share|equity) offering": -0.6,
    r"\b(dilution|dilutive)\b": -0.5,
    r"(bankruptcy|chapter 11|going concern|default(s|ed)? on)": -1.0,
    r"\b(layoffs?|job cuts)\b": -0.3,
    r"(ceo|cfo) (resigns|steps down|ousted|departs)": -0.5,
    r"\b(accounting|restatement|fraud|short seller|short report)\b": -0.8,
    r"\b(plunges?|plummets?|tumbles?|sinks?|crash(es)?|sell-?off|slumps?)\b": -0.5,
    r"\bdata breach\b|\bhack(ed)?\b|\boutage\b": -0.4,
    r"(hotter|higher)[- ]than[- ]expected (inflation|cpi|ppi)": -0.6,
    r"(rate hike|hikes rates)": -0.4,
    r"(new |more )?tariffs?\b(?! relief)": -0.3,
    r"\b(export ban|export controls?|sanctions?)\b": -0.5,
    r"\b(recession|shutdown|downturn)\b": -0.4,
    r"\b(war|missiles?|airstrikes?|(drone|missile) strikes?|invasion|escalat\w*)\b": -0.4,
    # Indian-market catalyst language
    r"(net )?profit (falls|drops|declines|slumps|plunges|down|slides|halves)": -0.6,
    r"\bq[1-4] (miss|misses)\b": -0.6,
    r"(rating|outlook) downgrade|downgrades? (rating|outlook)": -0.5,
    r"\b(block|bulk) deal\b": -0.3,
    r"promoters? (sells?|offloads?|pledges?|cuts? stake|trims? stake)|stake sale": -0.5,
    r"\b(qip|offer for sale|ofs)\b": -0.5,
    r"\bsebi (probe|order|ban|bars|notice|penalty|action)\b": -0.8,
    r"\b(ed|income tax|it department|cbi) (raid|raids|probe|searches|search)\b": -0.8,
    r"\bfii (selling|outflows?)\b|foreign (investors|funds) (sell|dump|exit|retreat|cut)": -0.4,
    r"rbi (hikes|raises) (repo|rates?)|repo rate hike": -0.5,
    r"rupee (hits|slumps|falls|plunges) .{0,20}(low|record)": -0.3,
    # Pharma (USFDA)
    r"\bwarning letter\b|\bimport alert\b|\boai (status|classification)\b": -0.7,
    r"\bform 483\b|\b\d+ observations\b": -0.4,
}
_BULL_RX = [(re.compile(p, re.I), w) for p, w in _BULLISH.items()]
_BEAR_RX = [(re.compile(p, re.I), w) for p, w in _BEARISH.items()]


# Headlines that are commentary, listicles or explainers rather than a
# new event - scored 0 whatever words they contain ("60 years of market
# crashes" is not a crash).
_OPINION_RX = re.compile(
    r"^(why|here'?s|is|are|should|what|how|can|could|will|would|do|does|\d+ )\b"
    r"|\?|stocks? to (buy|sell|watch)|should you|buy (now|today)|millionaire|forever"
    r"|(my|our) (top|favorite)|prediction|opinion|here'?s (why|what|how)",
    re.I)

# A market-feed headline only counts if it is actually about the market
# or macro backdrop, not one stock that happened to land in the SPY feed.
_MACRO_RX = re.compile(
    r"\b(indian (shares|stocks|equities|markets?)|stocks|markets?|sensex|nifty|dalal street|"
    r"rbi|repo|monetary policy|inflation|cpi|wpi|gdp|budget|fiscal|sebi|fii|fiis|fpi|fpis|dii|"
    r"foreign (investors|funds)|rupee|crude|oil prices?|monsoon|tariffs?|fed|us yields?|"
    r"recession|war|geopolitic\w*)\b",
    re.I)

_NAME_SUFFIX_RX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|plc|holdings?|group|"
    r"platforms|technologies|technology|systems|class [a-c]|the)\b\.?|[,.]", re.I)
_names = {}


# Group names shared by many listed companies: matching "Tata" alone would
# make every Tata company's headline count for TCS, TATASTEEL, TMPV...
_GROUP_PREFIXES = {"TATA", "ADANI", "BAJAJ", "HDFC", "ICICI", "MAHINDRA", "JSW", "BHARAT",
                   "HINDUSTAN", "INDIAN", "STATE", "BANK", "RELIANCE", "BIRLA", "GODREJ",
                   "KOTAK", "AXIS", "LARSEN", "SHRIRAM", "SUN", "POWER", "OIL", "GAIL"}


def _company_tokens(ticker: str) -> list:
    """Words that identify a company in a headline: its NSE symbol plus the
    distinctive part of its name (e.g. SBIN -> ['SBIN', 'State Bank'],
    ONGC -> ['ONGC', 'Oil and Natural'...])."""
    if ticker in _names:
        return _names[ticker]
    tokens = [ticker]
    if ticker in config.INDIA_NEWS_NAMES:     # abbreviated / ambiguous registered names
        tokens.extend(config.INDIA_NEWS_NAMES[ticker][1:])   # one or more name tokens
        _names[ticker] = tokens
        return tokens
    try:
        # Registered name from the Upstox instrument master (no network
        # call); Yahoo only as a fallback for a symbol it doesn't have.
        from data.instruments import get_company_name
        name = get_company_name(ticker)
        if not name:
            import yfinance as yf
            info = yf.Ticker(f"{ticker}.NS").get_info() or {}
            name = info.get("longName") or info.get("shortName") or ""
        core = _NAME_SUFFIX_RX.sub(" ", name).split()
        if core:
            # group prefix ("TATA STEEL") or a short first word ("PB FINTECH"):
            # the first word alone is ambiguous / too short - keep two
            if (core[0].upper() in _GROUP_PREFIXES or len(core[0]) < 3) and len(core) >= 2:
                tokens.append(f"{core[0]} {core[1]}")
            elif len(core[0]) >= 3:
                tokens.append(core[0])
    except Exception:
        pass
    _names[ticker] = tokens
    return tokens


def is_relevant(text: str, ticker: str) -> bool:
    for i, tok in enumerate(_company_tokens(ticker)):
        if i == 0:
            # The bare ticker is matched case-sensitively and only when long
            # enough not to be an ordinary word ('ON', 'IT', 'ALL'); short
            # ones must appear as (T) or $T.
            if re.search(rf"[($]{re.escape(tok)}\b", text):
                return True
            if len(tok) >= 3 and re.search(rf"\b{re.escape(tok)}\b", text):
                return True
        elif " " in tok:
            if re.search(rf"\b{re.escape(tok)}".replace("\\ ", r"\s+"), text, re.I):
                return True
        # A one-word company name - often identical to the ticker (MPHASIS,
        # FORTIS, TITAN), which used to send it down the case-sensitive
        # ticker path and miss "Mphasis". Title case or CAPS only ("eternal"
        # is an ordinary word), and only in a stock-market headline: "Trent
        # Williams", "Willmott Dixon", "Titan International" are not news
        # about TRENT, DIXON or TITAN.
        elif (re.search(rf"\b({re.escape(tok.capitalize())}|{re.escape(tok.upper())})\b", text)
              and _MARKET_CONTEXT_RX.search(text)):
            return True
    return False


# Words that put a headline on the stock-market page - required for matches
# on a bare one-word company name (see is_relevant).
_MARKET_CONTEXT_RX = re.compile(
    r"\b(shares?|stocks?|share price|stock price|scrip|equity|market cap|m-?cap|nse|bse|sensex|nifty|"
    r"q[1-4]|quarter(ly)?|results?|earnings|profit|revenue|ebitda|margins?|guidance|outlook|"
    r"orders?|contracts?|deals?|merger|acquisitions?|acquires?|stake|ipo|ofs|qip|dividend|bonus|buyback|"
    r"brokerage|target|rating|upgrade[sd]?|downgrade[sd]?|buy|sell|hold|"
    r"rall(y|ies)|surges?|jumps?|soars?|gains?|falls?|slumps?|plunges?|tumbles?|slides?|crash(es)?|"
    r"52-week|block deal|fii|dii|mutual funds?|investors?|sebi|irdai|rbi|cci|nclt|"
    r"business update|sales|volumes?|launch(es)?|plant|capex|expansion)\b", re.I)


def _is_counterparty(text: str, ticker: str) -> bool:
    """True when the company is only the OTHER side of the deal in the
    headline - "Sonu Infratech wins order from Reliance Industries" is an
    order win for Sonu, not for RELIANCE. The keyword scorer can't tell who
    won, so such headlines are neutral for the named company."""
    if ticker.startswith("^"):
        return False
    for tok in _company_tokens(ticker):
        pat = re.escape(tok).replace("\\ ", r"\s+")
        if re.search(rf"\b(orders?|contracts?|deals?)\b.{{0,60}}\b(from|by|with)\b.{{0,15}}\b{pat}", text, re.I):
            if not re.search(rf"^\W*{pat}", text, re.I):     # the company isn't the subject
                return True
    return False


def score_headline(text: str) -> float:
    """Lexicon score of one headline TITLE, clipped to [-1, 1]. Titles
    only: summaries add more unrelated vocabulary than signal."""
    if not text:
        return 0.0
    if _OPINION_RX.search(text):
        return 0.0
    s = 0.0
    for rx, w in _BULL_RX:
        if rx.search(text):
            s += w
    for rx, w in _BEAR_RX:
        if rx.search(text):
            s += w
    return max(-1.0, min(1.0, s))


# ═══════════════════════════════════════════════════════════════════
# Fetch
# ═══════════════════════════════════════════════════════════════════

def _parse_item(item: dict):
    """Yahoo's news payload has changed shape before; handle both the
    current {'content': {...}} form and the older flat form."""
    c = item.get("content", item) if isinstance(item, dict) else {}
    title = c.get("title") or ""
    summary = c.get("summary") or c.get("description") or ""
    pub = c.get("pubDate") or c.get("displayTime")
    ts = None
    if pub:
        try:
            ts = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
        except ValueError:
            ts = None
    elif c.get("providerPublishTime"):
        ts = datetime.fromtimestamp(int(c["providerPublishTime"]), tz=timezone.utc)
    return title, summary, ts


def fetch_headlines(ticker: str, lookback_hours: float, now: datetime = None,
                    market: bool = False) -> list:
    """[(published_utc, title, summary)] newer than lookback_hours and
    relevant: about this company (symbol feeds) or about the market /
    macro backdrop (market feeds)."""
    now = now or datetime.now(timezone.utc)
    if config.INDIA_NEWS_SOURCE == "rss":
        # Indian RSS feeds + Google News (strategy/india_news.py) - Yahoo's
        # NSE feeds are stale. ^NSEBANK reads the banking subset.
        from strategy import india_news
        if market or ticker.startswith("^"):
            return india_news.market_headlines(lookback_hours, now, banking=(ticker == "^NSEBANK"))
        return india_news.stock_headlines(ticker, lookback_hours, now)

    import yfinance as yf
    yf_ticker = ticker if (market or ticker.startswith("^")) else f"{ticker}.NS"
    try:
        raw = yf.Ticker(yf_ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw:
        title, summary, ts = _parse_item(item)
        if not title or ts is None:
            continue
        age_h = (now - ts).total_seconds() / 3600.0
        if not (0 <= age_h <= lookback_hours):
            continue
        # Relevance is judged on the TITLE only: Yahoo summaries routinely
        # name other companies ("...alongside Meta and Nvidia"), which let
        # unrelated stories leak into a symbol's read.
        if market:
            if not _MACRO_RX.search(title):
                continue
        elif not is_relevant(title, ticker):
            continue
        out.append((ts, title, summary))
    return sorted(out, key=lambda x: x[0], reverse=True)


# ═══════════════════════════════════════════════════════════════════
# Optional LLM scoring (Claude)
# ═══════════════════════════════════════════════════════════════════
_llm_client = None

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["index", "score"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}

_LLM_SYSTEM = (
    "You score financial news headlines for their likely effect on an Indian (NSE) stock's price over "
    "the next few trading hours. For each numbered headline return a score from -1.0 "
    "(clearly bearish catalyst: earnings miss, guidance cut, downgrade, investigation, "
    "offering, recall, macro shock) to +1.0 (clearly bullish catalyst: beat-and-raise, "
    "upgrade, approval, big contract, buyback). Opinion pieces, listicles, recaps of old "
    "moves, and anything not about a new event score 0. Be conservative: only a genuine, "
    "new, material event deserves a magnitude above 0.5."
)


def _llm_scores(ticker: str, titles: list):
    """Per-headline scores from the configured LLM (strategy/llm_client.py -
    Gemini by default), or None on any failure (-> lexicon)."""
    from strategy import llm_client
    subject = "the Indian stock market overall (NIFTY)" if ticker.startswith("^") else f"{ticker} (NSE)"
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
    data = llm_client.json_call(_LLM_SYSTEM, f"Stock: {subject}\nHeadlines:\n{listing}", _LLM_SCHEMA,
                                tag=f"V2 news {ticker}", max_tokens=2048)
    if not data:
        return None
    scores = [0.0] * len(titles)
    for s in data.get("scores", []):
        i = int(s.get("index", -1))
        if 0 <= i < len(titles):
            scores[i] = max(-1.0, min(1.0, float(s.get("score", 0.0))))
    return scores


def _cached_llm_scores(ticker: str, texts: list) -> list:
    """LLM score per headline, or None where there is none. Each headline is
    sent once; later runs reuse the stored score (llm_client cache)."""
    from strategy import llm_client
    keys = [llm_client.make_key("v2", ticker, t) for t in texts]
    out = [llm_client.cache_get("v2", k)[0] for k in keys]
    todo = [i for i, s in enumerate(out) if s is None]
    if todo:
        fresh = _llm_scores(ticker, [texts[i] for i in todo])
        if fresh is not None:
            for i, s in zip(todo, fresh):
                out[i] = s
                llm_client.cache_put("v2", keys[i], s)
    return out


# ═══════════════════════════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════════════════════════
_cache = {}   # (ticker, lookback) -> (fetched_at, CatalystRead)


def _aggregate(ticker: str, items: list, now: datetime) -> CatalystRead:
    if not items:
        return CatalystRead(method="none")
    titles = [f"{t}. {s}" if s else t for _, t, s in items]
    lexicon = [0.0 if _is_counterparty(t, ticker) else score_headline(t) for _, t, _ in items]
    scores, method = lexicon, "lexicon"
    if config.ML_V2_NEWS_LLM_ENABLED:
        llm = _cached_llm_scores(ticker, titles)
        n_llm = sum(s is not None for s in llm)
        if n_llm:
            # LLM score where one exists, lexicon for any headline the LLM
            # couldn't score this run (quota) - same veto/boost rules either way.
            scores = [s if s is not None else x for s, x in zip(llm, lexicon)]
            method = "llm" if n_llm == len(llm) else "llm+lexicon"

    # Recency weighting: a 1-hour-old headline counts ~2x a 6-hour-old one.
    num = den = 0.0
    best_i, best_mag = 0, -1.0
    for i, ((ts, title, _), sc) in enumerate(zip(items, scores)):
        age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        w = math.exp(-age_h / 6.0)
        if sc != 0.0:
            num += w * sc
            den += w
        if abs(sc) * w > best_mag:
            best_i, best_mag = i, abs(sc) * w
    # Averaging only over NON-neutral headlines, then shrinking by how many
    # there were: one strong headline is a signal, but it is not as strong
    # a signal as three agreeing ones.
    avg = num / den if den else 0.0
    n_signal = sum(1 for s in scores if s != 0.0)
    confidence = 1.0 - math.exp(-n_signal / 1.5)
    score = max(-1.0, min(1.0, avg * confidence))
    return CatalystRead(score=round(score, 3), n_items=len(items),
                        top_headline=items[best_i][1], method=method,
                        headlines=[t for _, t, _ in items[:5]])


def read_catalyst(ticker: str, lookback_hours: float = None, market: bool = False) -> CatalystRead:
    """Cached catalyst read for one ticker."""
    lookback_hours = lookback_hours or config.ML_V2_NEWS_SYMBOL_LOOKBACK_HOURS
    key = (ticker, lookback_hours, market)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < config.ML_V2_NEWS_CACHE_SECONDS:
        return hit[1]
    now = datetime.now(timezone.utc)
    read = _aggregate(ticker, fetch_headlines(ticker, lookback_hours, now, market), now)
    _cache[key] = (time.time(), read)
    return read


def read_market_catalyst() -> CatalystRead:
    """Broad-market read from the index ETFs' own news feeds. The
    strongest reading (by magnitude) wins, so one clear macro shock in
    either feed isn't averaged away by the other feed's noise."""
    reads = [read_catalyst(t, config.ML_V2_NEWS_MARKET_LOOKBACK_HOURS, market=True)
             for t in config.ML_V2_NEWS_MARKET_TICKERS]
    reads = [r for r in reads if r.n_items]
    if not reads:
        return CatalystRead(method="none")
    return max(reads, key=lambda r: abs(r.score))


def entry_decision(symbol: str, direction: str):
    """(veto_reason_or_None, prob_adjustment, symbol_read, market_read).

    prob_adjustment is <= 0: subtracted from the probability floor when
    the catalyst is aligned with the trade."""
    sym = read_catalyst(symbol)
    mkt = read_market_catalyst()
    sign = 1.0 if direction == "long" else -1.0
    s_al, m_al = sign * sym.score, sign * mkt.score
    if s_al <= -config.ML_V2_NEWS_VETO:
        return f"opposing {symbol} news {sym.summary}", 0.0, sym, mkt
    if m_al <= -config.ML_V2_NEWS_MARKET_VETO:
        return f"opposing market news {mkt.summary}", 0.0, sym, mkt
    adj = -config.ML_V2_NEWS_BOOST_PROB if s_al >= config.ML_V2_NEWS_BOOST else 0.0
    return None, adj, sym, mkt


def exit_check(symbol: str, direction: str):
    """Reason string if fresh news now argues for closing an open trade."""
    sym = read_catalyst(symbol)
    mkt = read_market_catalyst()
    sign = 1.0 if direction == "long" else -1.0
    if sign * sym.score <= -config.ML_V2_NEWS_EXIT:
        return f"bad {symbol} catalyst: {sym.summary}"
    if sign * mkt.score <= -config.ML_V2_NEWS_EXIT:
        return f"bad market catalyst: {mkt.summary}"
    return None


# ═══════════════════════════════════════════════════════════════════
# Decision log: every live news check L_ML_META_V2 / V3 make (veto, boost,
# plain trade or skip) with the scores and the headline behind them -
# vetoed candidates otherwise leave no trace, so whether news ever protects
# a trade could not be measured. live_state/reports/ is uploaded with the
# day's paper-trading artifact.
# ═══════════════════════════════════════════════════════════════════
DECISION_FIELDS = ["logged_at", "candle", "model", "symbol", "direction", "setup", "prob", "base_bar",
                   "final_bar", "decision", "veto_reason", "news", "news_method", "news_items", "news_headline",
                   "mkt", "mkt_method", "mkt_items", "summary"]


def _decisions_path() -> str:
    d = os.path.join(config.LIVE_STATE_DIR, "reports")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "news_decisions.csv")


def log_decision(model: str, symbol: str, direction: str, candle, setup: str, prob: float, base_bar: float,
                 veto, adj: float, sym: CatalystRead = None, mkt: CatalystRead = None, summary: str = "") -> str:
    """Record one news-checked candidate; returns the decision label:
    VETO | SKIP (below the bar even after news) | TRADE_BOOSTED (only
    trades because aligned news lowered the bar) | TRADE."""
    final_bar = base_bar + adj
    if veto:
        decision = "VETO"
    elif prob < final_bar:
        decision = "SKIP"
    elif prob < base_bar:
        decision = "TRADE_BOOSTED"
    else:
        decision = "TRADE"
    sym, mkt = sym or CatalystRead(), mkt or CatalystRead()
    detail = summary or (f"news {sym.score:+.2f} ({sym.method}, {sym.n_items}) "
                         f"mkt {mkt.score:+.2f} ({mkt.method}, {mkt.n_items})")
    print(f"[news-decision] {model} {symbol} {direction} {setup} p={prob:.2f} bar {base_bar:.2f}->{final_bar:.2f} "
          f"{detail} -> {decision}" + (f": {veto}" if veto else ""), flush=True)
    row = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "candle": str(candle),
           "model": model, "symbol": symbol, "direction": direction, "setup": setup, "prob": round(prob, 4),
           "base_bar": round(base_bar, 4), "final_bar": round(final_bar, 4), "decision": decision,
           "veto_reason": veto or "", "news": sym.score, "news_method": sym.method, "news_items": sym.n_items,
           "news_headline": sym.top_headline[:200], "mkt": mkt.score, "mkt_method": mkt.method,
           "mkt_items": mkt.n_items, "summary": summary[:300]}
    try:
        import csv
        path = _decisions_path()
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[news] could not log decision: {e!r}")
    return decision


def decisions_summary(day: str) -> str:
    """EOD report block: per model, how news changed today's candidates."""
    path = _decisions_path()
    if not os.path.exists(path):
        return ""
    try:
        import pandas as pd
        d = pd.read_csv(path)
    except Exception:
        return ""
    d = d[d["candle"].astype(str).str[:10] == day]
    if d.empty:
        return ""
    lines = ["📰 NEWS DECISIONS today (candidates near the bar):"]
    for model, g in d.groupby("model"):
        c = g["decision"].value_counts()
        line = (f"{model}: {len(g)} checked | TRADE {c.get('TRADE', 0)} | BOOSTED {c.get('TRADE_BOOSTED', 0)} | "
                f"VETO {c.get('VETO', 0)} | SKIP {c.get('SKIP', 0)}")
        if model == "L_ML_META_V2":
            m = g["news_method"].astype(str)
            line += (f" | stock news found {(g['news_items'] > 0).mean() * 100:.0f}%, "
                     f"LLM-scored {m.str.startswith('llm').sum()}/{(m != 'none').sum()}")
        lines.append(line)
        for _, v in g[g["decision"] == "VETO"].head(5).iterrows():
            lines.append(f"   veto {v['symbol']} p={v['prob']:.2f}: {str(v['veto_reason'])[:120]}")
    return "\n".join(lines)
