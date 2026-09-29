"""
One place for the LLM calls made by the live news/catalyst readers
(strategy/news_catalyst.py for L_ML_META_V2, strategy/llm_catalyst.py for
L_ML_META_V3): structured JSON from Google Gemini (default - free tier,
GEMINI_API_KEY) or Claude (ANTHROPIC_API_KEY), plus a small persistent
cache.

Why a persistent cache: scan_once.py is a fresh process every run, so an
in-memory cache never survives from one run to the next - every run
would re-send the same headlines and a free-tier quota runs out within
minutes (seen live on the US bot). Answers are stored in
config.LIVE_STATE_DIR (carried between GitHub runs by the workflow
cache), keyed by exactly what was asked, so the same headlines are never
sent twice.

Every failure returns None and the caller falls back to its rules /
keyword scorer - an API problem can never block the bot. After a
rate-limit or overload error (429/500/503) the LLM is skipped for the
rest of the run instead of firing more calls that would fail the same way.
"""
import hashlib
import json
import os
import time

import config

_CACHE_FILE = "news_llm_cache.json"
_CACHE_DAYS = 3
_cache = None            # "ns|key" -> [value, stored_at_epoch]
_blocked = False         # set on 429/500/503: rest of this process skips the LLM
_claude = None


def provider() -> str:
    return config.NEWS_LLM_PROVIDER


def available() -> bool:
    """An LLM is configured and has a key."""
    if provider() == "gemini":
        return bool(config.GEMINI_API_KEY)
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ═══════════════════════════════════════════════════════════════════
# Persistent cache
# ═══════════════════════════════════════════════════════════════════

def _cache_path() -> str:
    os.makedirs(config.LIVE_STATE_DIR, exist_ok=True)
    return os.path.join(config.LIVE_STATE_DIR, _CACHE_FILE)


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_cache_path()) as f:
                data = json.load(f)
        except Exception:
            data = {}
        cutoff = time.time() - _CACHE_DAYS * 86400
        _cache = {k: v for k, v in data.items() if isinstance(v, list) and len(v) == 2 and v[1] >= cutoff}
    return _cache


def _save():
    try:
        path = _cache_path()
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[llm] could not save cache: {e!r}")


def make_key(*parts) -> str:
    raw = "|".join([provider(), model()] + [str(p) for p in parts])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def cache_get(ns: str, key: str, max_age_s: float = None):
    """(value, age_seconds) or (None, None)."""
    hit = _load().get(f"{ns}|{key}")
    if not hit:
        return None, None
    age = time.time() - hit[1]
    if max_age_s is not None and age > max_age_s:
        return None, None
    return hit[0], age


def cache_put(ns: str, key: str, value):
    _load()[f"{ns}|{key}"] = [value, time.time()]
    _save()


# ═══════════════════════════════════════════════════════════════════
# Calls
# ═══════════════════════════════════════════════════════════════════

def model() -> str:
    return config.NEWS_LLM_MODEL


def json_call(system: str, user: str, schema: dict, tag: str = "", max_tokens: int = 4096):
    """Structured JSON (a dict) matching `schema`, or None on any failure."""
    if _blocked or not available():
        return None
    if provider() == "gemini":
        return _gemini(system, user, schema, tag)
    return _claude_call(system, user, schema, tag, max_tokens)


def _to_gemini_schema(s):
    """JSON Schema -> Gemini responseSchema (OpenAPI subset: upper-case
    types, no additionalProperties)."""
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            if k == "additionalProperties":
                continue
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            elif k in ("properties",):
                out[k] = {pk: _to_gemini_schema(pv) for pk, pv in v.items()}
            else:
                out[k] = _to_gemini_schema(v)
        return out
    if isinstance(s, list):
        return [_to_gemini_schema(x) for x in s]
    return s


def _error_detail(resp) -> str:
    """Status, message and which quota was hit (per-minute vs per-day)."""
    try:
        err = resp.json().get("error", {})
    except Exception:
        return resp.text[:1000]
    parts = [f"{err.get('status', '')}: {err.get('message', '')}"]
    for d in err.get("details", []):
        for v in d.get("violations", []):
            parts.append(f"quota={v.get('quotaId') or v.get('quotaMetric')} limit={v.get('quotaValue', '?')}")
        if d.get("retryDelay"):
            parts.append(f"retryDelay={d['retryDelay']}")
    return " | ".join(parts)[:1000]


def _gemini(system: str, user: str, schema: dict, tag: str):
    global _blocked
    import requests
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model()}:generateContent"
        resp = requests.post(url, headers={"x-goog-api-key": config.GEMINI_API_KEY}, timeout=30, json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": _to_gemini_schema(schema)},
        })
        if resp.status_code != 200:
            if resp.status_code in (429, 500, 503):
                _blocked = True
                print(f"[llm] Gemini {resp.status_code} ({tag}) - LLM paused for the rest of this run, "
                      f"using rules. {_error_detail(resp)}")
            else:
                print(f"[llm] Gemini {resp.status_code} ({tag}), using rules: {_error_detail(resp)}")
            return None
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return json.loads(text) if text else None
    except Exception as e:
        print(f"[llm] Gemini call failed ({tag}), using rules: {e!r}")
        return None


def _claude_call(system: str, user: str, schema: dict, tag: str, max_tokens: int):
    global _claude
    try:
        import anthropic
    except ImportError:
        return None
    try:
        if _claude is None:
            _claude = anthropic.Anthropic(timeout=30.0, max_retries=1)
        response = _claude.beta.messages.create(
            model=model(),
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        return json.loads(text) if text else None
    except Exception as e:
        print(f"[llm] Claude call failed ({tag}), using rules: {e!r}")
        return None
