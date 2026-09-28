"""
Point-in-time market + catalyst context for model L_ML_META_V2 (NSE).

Port of us_alert_bot's strategy/market_context.py with India's inputs:

  MARKET   India VIX level vs its 50-day SMA and its move today / last
           30 min; NIFTY 50 and NIFTY BANK intraday trend (day return,
           30-min return, move since the 9:15 open, position in the
           day's range); NIFTY vs its 50/200-day SMA; % of the NIFTY 50
           constituents above their own 50-day SMA (breadth).
  CATALYST earnings proximity + last EPS surprise (yfinance, `.NS`),
           today's gap vs the prior close, time-of-day relative volume
           ("is it in play"), relative strength vs NIFTY, distance from
           recent highs.

Differences from the US version, and why:
  - no VIX3M / VXN: India has no VIX term-structure series, so there is
    no backwardation feature or veto.
  - index candles carry no volume, so "index vs its VWAP" (US: QQQ vs
    VWAP) becomes "index vs its own session open" + range position.
  - all prices come from Upstox (5-min history goes back years), not
    Yahoo - so the backtest/training window can be many months long.

NO LOOKAHEAD - identical rules to the US module:
  - intraday series: the last 5-min bar whose timestamp is <= the signal
    candle's timestamp (both close at the same moment).
  - daily series: only sessions strictly BEFORE the signal's date.
  - earnings: a report counts from its reaction session (before 9:15 IST
    -> that day; otherwise the next weekday).
"""
import os
import pickle
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from strategy.session import minutes_since_open, session_open

IST = ZoneInfo(config.MARKET_TIMEZONE)
NAN = float("nan")

INDEX_KEYS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANK": "NSE_INDEX|Nifty Bank",
    "IVIX": "NSE_INDEX|India VIX",
}
SESSION_MINUTES = 375.0          # 9:15 -> 15:30

_INTRADAY_TTL = 60 * 60
_INTRADAY_TTL_LIVE = 4 * 60
_DAILY_TTL = 12 * 60 * 60
_EARNINGS_TTL = 3 * 24 * 60 * 60
_DAILY_LOOKBACK_DAYS = 420
_CHUNK_DAYS = 25                 # Upstox minute data: ~1 month per request


def _token() -> str:
    if config.UPSTOX_ACCESS_TOKEN:
        return config.UPSTOX_ACCESS_TOKEN
    import json
    with open(config.TOKEN_FILE) as f:
        return json.load(f)["access_token"]


def _nan_safe(v) -> float:
    try:
        if v is None or pd.isna(v):
            return NAN
        return float(v)
    except (TypeError, ValueError):
        return NAN


def _ratio(a, b) -> float:
    a, b = _nan_safe(a), _nan_safe(b)
    if np.isnan(a) or np.isnan(b) or b == 0:
        return NAN
    return a / b - 1.0


def _to_ist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ═══════════════════════════════════════════════════════════════════
# Disk cache (same scheme as the US module: merge, never truncate)
# ═══════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> str:
    os.makedirs(config.ML_V2_CACHE_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return os.path.join(config.ML_V2_CACHE_DIR, f"{safe}.pkl")


def _cache_load(name: str):
    try:
        with open(_cache_path(name), "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_save(name: str, payload: dict):
    path = _cache_path(name)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def _fresh_daily(cached) -> bool:
    """Daily bars fetched mid-session hold a partial bar, so a daily cache
    is only reusable on the same IST date it was fetched."""
    if cached is None:
        return False
    fetched = cached.get("fetched", 0)
    if time.time() - fetched >= _DAILY_TTL:
        return False
    return datetime.fromtimestamp(fetched, IST).date() == datetime.now(IST).date()


def _merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    out = pd.concat([old, new])
    return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# Fetchers (Upstox for prices, yfinance for earnings dates)
# ═══════════════════════════════════════════════════════════════════

def _retry(fn, *args, attempts: int = 4):
    last = None
    for k in range(attempts):
        try:
            return fn(*args)
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    raise last


def _fetch_minutes(instrument_key: str, from_dt, to_dt) -> pd.DataFrame:
    from data.upstox_client import get_historical_candles
    frames = []
    cur_to = to_dt
    while cur_to >= from_dt:
        cur_from = max(from_dt, cur_to - timedelta(days=_CHUNK_DAYS))
        df = _retry(get_historical_candles, instrument_key, "minutes", config.CANDLE_INTERVAL_MINUTES,
                    cur_to.strftime("%Y-%m-%d"), cur_from.strftime("%Y-%m-%d"), _token())
        if not df.empty:
            frames.append(df)
        cur_to = cur_from - timedelta(days=1)
        time.sleep(0.2)
    return _to_ist(pd.concat(frames, ignore_index=True)) if frames else _to_ist(None)


def _fetch_today(instrument_key: str) -> pd.DataFrame:
    from data.upstox_client import get_intraday_candles
    return _to_ist(get_intraday_candles(instrument_key, config.CANDLE_INTERVAL_MINUTES, _token()))


def _fetch_daily(instrument_key: str) -> pd.DataFrame:
    from data.upstox_client import get_historical_candles
    today = datetime.now(IST)
    start = today - timedelta(days=_DAILY_LOOKBACK_DAYS)
    return _to_ist(_retry(get_historical_candles, instrument_key, "days", 1, today.strftime("%Y-%m-%d"),
                          start.strftime("%Y-%m-%d"), _token()))


def _yf_symbol(symbol: str) -> str:
    return f"{symbol}.NS"


def _fetch_earnings(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    try:
        e = yf.Ticker(_yf_symbol(symbol)).get_earnings_dates(limit=24)
    except Exception:
        return pd.DataFrame(columns=["ts", "surprise"])
    if e is None or e.empty:
        return pd.DataFrame(columns=["ts", "surprise"])
    out = pd.DataFrame({
        "ts": pd.to_datetime(e.index),
        "surprise": pd.to_numeric(e.get("Surprise(%)"), errors="coerce").values,
    })
    if out["ts"].dt.tz is None:
        out["ts"] = out["ts"].dt.tz_localize("America/New_York")
    out["ts"] = out["ts"].dt.tz_convert(IST)     # yfinance reports US-Eastern times
    return out.sort_values("ts").reset_index(drop=True)


def _next_weekday(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _reaction_date(ts: pd.Timestamp):
    """First session that trades on this report: same day only if it came
    out before the 9:15 open; a report during or after market hours is
    counted from the next session (conservative - no lookahead)."""
    if ts.hour * 60 + ts.minute < config.MARKET_OPEN_HOUR * 60 + config.MARKET_OPEN_MINUTE:
        return ts.date()
    return _next_weekday(ts.date())


# ═══════════════════════════════════════════════════════════════════
# As-of lookups
# ═══════════════════════════════════════════════════════════════════

class _AsOfSeries:
    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        self.ts = (df["timestamp"].dt.tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
                   if len(df) else np.array([], dtype=np.int64))

    def idx_at(self, t, max_age_minutes: float = 30.0) -> int:
        if not len(self.ts):
            return -1
        key = pd.Timestamp(t).tz_convert("UTC").value
        i = int(np.searchsorted(self.ts, key, side="right")) - 1
        if i < 0 or (key - self.ts[i]) / 6e10 > max_age_minutes:
            return -1
        return i

    def value(self, t, col: str = "close", max_age_minutes: float = 30.0) -> float:
        i = self.idx_at(t, max_age_minutes)
        return NAN if i < 0 else _nan_safe(self.df[col].iat[i])


class _DailySeries:
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["date"] = df["timestamp"].dt.tz_convert(IST).dt.date
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        c = df["close"]
        df["sma50"] = c.rolling(50, min_periods=40).mean()
        df["sma200"] = c.rolling(200, min_periods=150).mean()
        df["high20"] = df["high"].rolling(20, min_periods=10).max()
        df["vol20"] = df["volume"].rolling(20, min_periods=10).mean()
        prev_c = c.shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(),
                        (df["low"] - prev_c).abs()], axis=1).max(axis=1)
        df["atr14_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=10).mean() / c
        df["ret5"] = c / c.shift(5) - 1
        df["ret20"] = c / c.shift(20) - 1
        self.df = df
        self.dates = np.array(df["date"].values, dtype="datetime64[D]")

    def prior_idx(self, d) -> int:
        if not len(self.dates):
            return -1
        return int(np.searchsorted(self.dates, np.datetime64(d, "D"), side="left")) - 1

    def prior(self, d, col: str) -> float:
        i = self.prior_idx(d)
        return NAN if i < 0 else _nan_safe(self.df[col].iat[i])


def _add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day open and running high/low (inclusive of the current bar,
    which closes at the same moment as the signal candle)."""
    df = df.copy()
    day = df["timestamp"].dt.date
    df["day_open"] = df.groupby(day)["open"].transform("first")
    df["day_high"] = df.groupby(day)["high"].cummax()
    df["day_low"] = df.groupby(day)["low"].cummin()
    return df


# ═══════════════════════════════════════════════════════════════════
# The context object
# ═══════════════════════════════════════════════════════════════════

class MarketContext:
    def __init__(self, live: bool = False, history_start=None, verbose: bool = False):
        self.live = live
        self.verbose = verbose
        # How far back intraday index data must reach (backtests / dataset
        # builds over long windows). Live only needs today.
        self.history_start = (pd.Timestamp(history_start).date() if history_start
                              else (datetime.now(IST) - timedelta(days=60)).date())
        self._intraday = {}
        self._daily = {}
        self._earnings = {}
        self._breadth = None
        self._market_loaded = False
        self._last_live_refresh = 0.0

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ---------- loading ----------

    def _load_intraday(self, name: str, ttl: float):
        self._load_intraday_key(name, INDEX_KEYS[name], ttl)

    def _load_intraday_key(self, name: str, key: str, ttl: float):
        """Load/refresh one index's 5-min history into self._intraday[name]."""
        cached = _cache_load(f"intraday_{name}")
        df = cached["df"] if cached else None
        fresh = cached is not None and (time.time() - cached.get("fetched", 0)) < ttl
        try:
            changed = False
            today = datetime.now(IST).date()
            if df is None or df.empty or df["timestamp"].min().date() > self.history_start:
                # (Re)build the history back to history_start.
                end = df["timestamp"].min().date() if (df is not None and not df.empty) else today
                df = _merge_frames(df, _fetch_minutes(key, self.history_start, end))
                changed = True
            if not fresh:
                last_day = df["timestamp"].max().date() if not df.empty else self.history_start
                if last_day < today:
                    df = _merge_frames(df, _fetch_minutes(key, last_day, today))
                df = _merge_frames(df, _fetch_today(key))
                changed = True
            if changed:
                _cache_save(f"intraday_{name}", {"fetched": time.time(), "df": df})
        except Exception as e:
            self._log(f"[context] intraday {name} fetch failed: {e}")
        if df is None or df.empty:
            self._intraday[name] = _AsOfSeries(pd.DataFrame(columns=["timestamp", "close"]))
            return
        self._intraday[name] = _AsOfSeries(_add_session_columns(df))

    def _load_daily(self, name: str, instrument_key: str):
        cached = _cache_load(f"daily_{name}")
        if _fresh_daily(cached):
            df = cached["df"]
        else:
            try:
                df = _fetch_daily(instrument_key)
                _cache_save(f"daily_{name}", {"fetched": time.time(), "df": df})
            except Exception as e:
                self._log(f"[context] daily {name} fetch failed: {e}")
                df = cached["df"] if cached else _to_ist(None)
        ser = _DailySeries(df) if not df.empty else None
        self._daily[name] = ser
        return ser

    def _load_breadth(self):
        cached = _cache_load("breadth_closes")
        if _fresh_daily(cached):
            closes = cached["df"]
        else:
            from data.instruments import get_instrument_key
            cols = {}
            for sym in config.ML_V2_BREADTH_UNIVERSE:
                try:
                    d = _fetch_daily(get_instrument_key(sym))
                    if len(d) > 60:
                        cols[sym] = d.set_index(d["timestamp"].dt.date)["close"]
                except Exception as e:
                    self._log(f"[context] breadth {sym}: {e}")
                time.sleep(0.1)
            closes = pd.DataFrame(cols).sort_index()
            if not closes.empty:
                _cache_save("breadth_closes", {"fetched": time.time(), "df": closes})
            elif cached:
                closes = cached["df"]
        if closes is None or closes.empty:
            self._breadth = pd.Series(dtype=float)
            return
        sma = closes.rolling(50, min_periods=40).mean()
        above = (closes > sma).where(sma.notna())
        pct = above.mean(axis=1, skipna=True)
        pct.index = pd.to_datetime(pd.Index(pct.index)).date
        self._breadth = pct.sort_index()

    def ensure_market(self):
        if self._market_loaded:
            return
        ttl = _INTRADAY_TTL_LIVE if self.live else _INTRADAY_TTL
        for name in INDEX_KEYS:
            self._load_intraday(name, ttl)
            self._load_daily(f"MKT_{name}", INDEX_KEYS[name])
        self._load_breadth()
        self._market_loaded = True
        self._last_live_refresh = time.time()

    def refresh_live(self):
        if not self._market_loaded:
            self.ensure_market()
            return
        if time.time() - self._last_live_refresh < _INTRADAY_TTL_LIVE:
            return
        for name in INDEX_KEYS:
            self._load_intraday(name, _INTRADAY_TTL_LIVE)
        self._last_live_refresh = time.time()

    def ensure_symbol(self, symbol: str):
        if symbol in self._daily and symbol in self._earnings:
            return
        if symbol not in self._daily:
            from data.instruments import get_instrument_key
            try:
                self._load_daily(symbol, get_instrument_key(symbol))
            except ValueError:
                self._daily[symbol] = None
        if symbol not in self._earnings:
            cached = _cache_load(f"earnings_{symbol}")
            if cached is not None and (time.time() - cached.get("fetched", 0)) < _EARNINGS_TTL:
                e = cached["df"]
            else:
                e = _fetch_earnings(symbol)
                try:
                    _cache_save(f"earnings_{symbol}", {"fetched": time.time(), "df": e})
                except Exception:
                    pass
            if not e.empty:
                e = e.copy()
                e["reaction"] = e["ts"].map(_reaction_date)
            self._earnings[symbol] = e

    # ---------- lookups ----------

    def asof(self, name: str, t, col: str = "close") -> float:
        self.ensure_market()
        ser = self._intraday.get(name)
        return NAN if ser is None else ser.value(t, col)

    def _earnings_features(self, symbol: str, d) -> dict:
        e = self._earnings.get(symbol)
        out = {"earn_days_since": 120.0, "earn_days_to_next": 120.0,
               "earn_surprise": NAN, "earn_reaction_today": 0.0}
        if e is None or e.empty:
            return out
        past = e[e["reaction"] <= d]
        if not past.empty:
            last = past.iloc[-1]
            out["earn_days_since"] = float(min(120, (d - last["reaction"]).days))
            s = _nan_safe(last["surprise"])
            out["earn_surprise"] = float(np.clip(s, -100, 100)) if not np.isnan(s) else NAN
            out["earn_reaction_today"] = 1.0 if last["reaction"] == d else 0.0
        future = e[e["reaction"] > d]
        if not future.empty:
            out["earn_days_to_next"] = float(min(120, (future.iloc[0]["reaction"] - d).days))
        return out

    def features(self, symbol: str, df: pd.DataFrame, direction: str) -> dict:
        """Every V2 context/catalyst feature for the signal candle df.iloc[-1]."""
        self.ensure_market()
        self.ensure_symbol(symbol)
        row = df.iloc[-1]
        ts = pd.Timestamp(row["timestamp"]).tz_convert(IST)
        d = ts.date()
        close = _nan_safe(row["close"])
        sign = 1.0 if direction == "long" else -1.0
        mso = minutes_since_open(ts)
        t30 = ts - pd.Timedelta(minutes=30)

        f = {"minutes_since_open": mso}

        # --- India VIX ---
        ivix = self.asof("IVIX", ts)
        dv = self._daily.get("MKT_IVIX")
        f["ivix"] = ivix
        f["ivix_vs_sma50"] = _ratio(ivix, dv.prior(d, "sma50")) if dv else NAN
        f["ivix_chg_day"] = _ratio(ivix, dv.prior(d, "close")) if dv else NAN
        f["ivix_chg_30m"] = _ratio(ivix, self.asof("IVIX", t30))

        # --- index tape ---
        nifty = self.asof("NIFTY", ts)
        dn, db = self._daily.get("MKT_NIFTY"), self._daily.get("MKT_BANK")
        f["nifty_ret_day"] = _ratio(nifty, dn.prior(d, "close")) if dn else NAN
        f["nifty_ret_30m"] = _ratio(nifty, self.asof("NIFTY", t30))
        f["nifty_ret_since_open"] = _ratio(nifty, self.asof("NIFTY", ts, "day_open"))
        hi, lo = self.asof("NIFTY", ts, "day_high"), self.asof("NIFTY", ts, "day_low")
        f["nifty_range_pos"] = ((nifty - lo) / (hi - lo)) if (hi == hi and lo == lo and hi > lo) else NAN
        f["bank_ret_day"] = _ratio(self.asof("BANK", ts), db.prior(d, "close")) if db else NAN
        f["nifty_vs_sma50_d"] = _ratio(dn.prior(d, "close"), dn.prior(d, "sma50")) if dn else NAN
        f["nifty_vs_sma200_d"] = _ratio(dn.prior(d, "close"), dn.prior(d, "sma200")) if dn else NAN

        # --- breadth (prior close) ---
        b = self._breadth
        if b is not None and len(b):
            prior = b[b.index < d]
            f["breadth_pct50"] = _nan_safe(prior.iloc[-1]) if len(prior) else NAN
            f["breadth_chg5"] = (_nan_safe(prior.iloc[-1]) - _nan_safe(prior.iloc[-6])) if len(prior) >= 6 else NAN
        else:
            f["breadth_pct50"] = f["breadth_chg5"] = NAN

        # --- symbol daily structure (prior sessions only) ---
        sd = self._daily.get(symbol)
        prev_close = sd.prior(d, "close") if sd else NAN
        f["sym_ret5_d"] = sd.prior(d, "ret5") if sd else NAN
        f["sym_ret20_d"] = sd.prior(d, "ret20") if sd else NAN
        f["sym_atr_pct_d"] = sd.prior(d, "atr14_pct") if sd else NAN
        f["sym_vs_sma50_d"] = _ratio(prev_close, sd.prior(d, "sma50")) if sd else NAN
        f["sym_dist_high20"] = _ratio(close, sd.prior(d, "high20")) if sd else NAN
        f["sym_dist_prev_high"] = _ratio(close, sd.prior(d, "high")) if sd else NAN

        # --- today's catalyst footprint ---
        reg = df[df["timestamp"] >= session_open(ts)]
        day_open = _nan_safe(reg["open"].iat[0]) if len(reg) else close
        f["gap_pct"] = _ratio(day_open, prev_close)
        f["sym_ret_day"] = _ratio(close, prev_close)
        f["sym_ret_since_open"] = _ratio(close, day_open)
        f["rs_vs_nifty_day"] = (f["sym_ret_day"] - f["nifty_ret_day"]
                                if not (np.isnan(f["sym_ret_day"]) or np.isnan(f["nifty_ret_day"])) else NAN)
        past30 = df[df["timestamp"] <= t30]
        sym30 = _ratio(close, past30["close"].iat[-1]) if len(past30) else NAN
        f["rs_vs_nifty_30m"] = (sym30 - f["nifty_ret_30m"]
                                if not (np.isnan(sym30) or np.isnan(f["nifty_ret_30m"])) else NAN)
        vol20 = sd.prior(d, "vol20") if sd else NAN
        if len(reg) and vol20 and not np.isnan(vol20) and vol20 > 0:
            elapsed = min(1.0, max(mso + config.CANDLE_INTERVAL_MINUTES, 5) / SESSION_MINUTES)
            f["tod_rvol"] = float(reg["volume"].sum()) / (vol20 * elapsed)
        else:
            f["tod_rvol"] = NAN
        if len(reg):
            h, l = float(reg["high"].max()), float(reg["low"].min())
            f["day_range_pos"] = (close - l) / (h - l) if h > l else 0.5
        else:
            f["day_range_pos"] = NAN

        f.update(self._earnings_features(symbol, d))

        def _al(v):
            return sign * v if not np.isnan(v) else NAN
        f["align_nifty_day"] = _al(f["nifty_ret_day"])
        f["align_nifty_30m"] = _al(f["nifty_ret_30m"])
        f["align_rs_day"] = _al(f["rs_vs_nifty_day"])
        f["align_gap"] = _al(f["gap_pct"])
        f["align_ivix_chg_day"] = _al(-f["ivix_chg_day"]) if not np.isnan(f["ivix_chg_day"]) else NAN
        f["align_earn_surprise"] = _al(f["earn_surprise"]) if not np.isnan(f["earn_surprise"]) else NAN
        return f

    def risk_veto(self, f: dict, direction: str):
        """Hard market-risk vetoes for longs (config.ML_V2_IVIX_*)."""
        if not config.ML_V2_VETO_ENABLED or direction != "long":
            return None
        ivix, spike = f.get("ivix", NAN), f.get("ivix_chg_day", NAN)
        if not np.isnan(ivix) and ivix >= config.ML_V2_IVIX_MAX_LONG:
            return f"India VIX {ivix:.1f} >= {config.ML_V2_IVIX_MAX_LONG}"
        if not np.isnan(spike) and spike >= config.ML_V2_IVIX_DAY_SPIKE_MAX_LONG:
            return f"India VIX +{spike*100:.0f}% today"
        return None


_shared = {}
_mode = {"live": False, "history_start": None}


def set_live_mode(flag: bool):
    _mode["live"] = bool(flag)


def is_live_mode() -> bool:
    return _mode["live"]


def set_history_start(date_str):
    """Backtests/dataset builds call this with their --from date so the
    intraday index history reaches back far enough."""
    _mode["history_start"] = date_str
    _shared.pop("backtest", None)


def reset_context(live: bool = None):
    if live is None:
        live = _mode["live"]
    _shared.pop("live" if live else "backtest", None)


def get_context(live: bool = None) -> MarketContext:
    if live is None:
        live = _mode["live"]
    key = "live" if live else "backtest"
    if key not in _shared:
        start = None if live else _mode["history_start"]
        _shared[key] = MarketContext(live=live, history_start=start)
    return _shared[key]
