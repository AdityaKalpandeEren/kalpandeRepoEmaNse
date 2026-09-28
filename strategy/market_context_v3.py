"""
Point-in-time context for model L_ML_META_V3 (NSE): everything V2's
MarketContext computes, plus

  SECTOR   the stock's own NSE sector index (strategy/sector_map.py):
           its move today / last 30 min / since the open, range position,
           stock-vs-sector and sector-vs-NIFTY strength.
  MACRO    the PRIOR US session's Brent crude, USD/INR, S&P 500, US 10y
           yield and gold moves (Yahoo daily) - what happened overnight,
           known before the 9:15 open.
  DYNAMIC  each sector's own rolling 120-day sensitivity (beta) of its
           daily return to the previous session's crude / USD-INR / S&P
           move, recomputed daily from data up to the prior day only, and
           the resulting "impulse" = beta x last night's move. This is how
           "crude +3% overnight" becomes a positive push for upstream
           energy and a negative one for refiners - learned from how those
           sectors actually traded, not typed in by hand.
  BREAKOUT new day high / distance to it, and relative strength on a red
           market day (the "breaking out while the market is weak" case).

No lookahead, same rules as V2: intraday values are "last bar at or
before the signal candle"; daily/macro values use sessions strictly
before the signal date (a Yahoo daily bar dated D closes during the
night before Indian date D+1, so it is known before that open).
"""
import time

import numpy as np
import pandas as pd

import config
from strategy import sector_map
from strategy.market_context import (MarketContext, _AsOfSeries, _cache_load, _cache_save,
                                     _fresh_daily, _nan_safe, _ratio, NAN, IST, _mode, _INTRADAY_TTL,
                                     _INTRADAY_TTL_LIVE)
from strategy.session import session_open

_BETA_MACROS = ("BRENT", "USDINR", "SPX")


def _fetch_macro(ticker: str) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=False)
    if h is None or h.empty:
        return pd.Series(dtype=float)
    s = h["Close"].astype(float)
    s.index = pd.to_datetime(s.index).date      # the bar's own (exchange-local) date
    return s[~pd.Index(s.index).duplicated(keep="last")].sort_index()


class MarketContextV3(MarketContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._macro = {}          # name -> pd.Series(date -> close)
        self._betas = {}          # (sector, macro) -> pd.Series(indian date -> beta as of prior day)
        self._stock_betas = {}    # symbol -> {macro: pd.Series}
        self._v3_loaded = False

    # ---------- loading ----------

    def _load_macro(self, name: str, ticker: str):
        cached = _cache_load(f"macro_{name}")
        if _fresh_daily(cached):
            s = cached["s"]
        else:
            try:
                s = _fetch_macro(ticker)
                if len(s):
                    _cache_save(f"macro_{name}", {"fetched": time.time(), "s": s})
            except Exception as e:
                self._log(f"[context-v3] macro {name}: {e}")
                s = cached["s"] if cached else pd.Series(dtype=float)
        self._macro[name] = s

    def _sector_daily_returns(self, code: str) -> pd.Series:
        ser = self._daily.get("MKT_NIFTY" if code == "OTHER" else f"SEC_{code}")
        if ser is None:
            return pd.Series(dtype=float)
        df = ser.df.set_index("date")
        return df["close"].pct_change().dropna()

    def _macro_lagged_returns(self, name: str, dates) -> pd.Series:
        """For each Indian trading date t: the macro's return on its latest
        session strictly BEFORE t (what happened 'overnight')."""
        m = self._macro.get(name)
        if m is None or len(m) < 3:
            return pd.Series(NAN, index=dates)
        r = m.pct_change().dropna()
        rd = np.array(r.index, dtype="datetime64[D]")
        out = []
        for t in dates:
            i = int(np.searchsorted(rd, np.datetime64(t, "D"), side="left")) - 1
            out.append(float(r.iloc[i]) if i >= 0 else NAN)
        return pd.Series(out, index=dates)

    def _rolling_betas(self, rets: pd.Series) -> dict:
        """{macro: beta series} of `rets` (daily, by Indian date) on each
        macro's previous-session return. shift(1): the beta used ON day t
        is estimated from data up to t-1 only."""
        w, mn = config.ML_V3_BETA_WINDOW, config.ML_V3_BETA_MIN_OBS
        out = {}
        for mac in _BETA_MACROS:
            x = self._macro_lagged_returns(mac, list(rets.index))
            df = pd.DataFrame({"y": rets.values, "x": x.values}, index=rets.index).dropna()
            if len(df) < mn:
                continue
            cov = df["y"].rolling(w, min_periods=mn).cov(df["x"])
            var = df["x"].rolling(w, min_periods=mn).var()
            out[mac] = (cov / var.replace(0, np.nan)).shift(1)
        return out

    def _compute_betas(self):
        for code in sector_map.SECTOR_CODES:
            sec = self._sector_daily_returns(code)
            if sec.empty:
                continue
            for mac, beta in self._rolling_betas(sec).items():
                self._betas[(code, mac)] = beta

    def _stock_beta(self, symbol: str, mac: str, d) -> float:
        """Stock-level sensitivity. Sector betas blur opposite reactions
        inside one index (refiners vs upstream in NIFTY ENERGY both sit in
        'energy' but move opposite ways on crude), so V3 measures each
        stock's own sensitivity too."""
        if symbol not in self._stock_betas:
            ser = self._daily.get(symbol)
            rets = (ser.df.set_index("date")["close"].pct_change().dropna()
                    if ser is not None else pd.Series(dtype=float))
            self._stock_betas[symbol] = self._rolling_betas(rets) if len(rets) else {}
        b = self._stock_betas[symbol].get(mac)
        if b is None or not len(b):
            return NAN
        idx = np.array(b.index, dtype="datetime64[D]")
        i = int(np.searchsorted(idx, np.datetime64(d, "D"), side="right")) - 1
        return _nan_safe(b.iloc[i]) if i >= 0 else NAN

    def ensure_v3(self):
        if self._v3_loaded:
            return
        self.ensure_market()
        ttl = _INTRADAY_TTL_LIVE if self.live else _INTRADAY_TTL
        for code, key in sector_map.SECTORS.items():
            if code == "OTHER":
                continue                         # = NIFTY 50, already loaded by V2
            if code == "BANK":
                self._intraday["SEC_BANK"] = self._intraday.get("BANK")
                self._daily["SEC_BANK"] = self._daily.get("MKT_BANK")
                continue
            self._load_intraday_key(f"SEC_{code}", key, ttl)
            self._load_daily(f"SEC_{code}", key)
        for name, ticker in config.ML_V3_MACRO_TICKERS.items():
            self._load_macro(name, ticker)
        self._compute_betas()
        self._v3_loaded = True

    def refresh_live(self):
        super().refresh_live()
        if self._v3_loaded and self.live:
            for code, key in sector_map.SECTORS.items():
                if code not in ("OTHER", "BANK"):
                    self._load_intraday_key(f"SEC_{code}", key, _INTRADAY_TTL_LIVE)

    # ---------- lookups ----------

    def _macro_prior(self, name: str, d, lag_days: int = 1) -> float:
        """Return over `lag_days` sessions ending at the macro's last session
        strictly before Indian date d."""
        m = self._macro.get(name)
        if m is None or len(m) <= lag_days:
            return NAN
        idx = np.array(m.index, dtype="datetime64[D]")
        i = int(np.searchsorted(idx, np.datetime64(d, "D"), side="left")) - 1
        if i - lag_days < 0:
            return NAN
        return _ratio(m.iloc[i], m.iloc[i - lag_days])

    def _beta(self, code: str, mac: str, d) -> float:
        b = self._betas.get((code, mac))
        if b is None or not len(b):
            return NAN
        idx = np.array(b.index, dtype="datetime64[D]")
        i = int(np.searchsorted(idx, np.datetime64(d, "D"), side="right")) - 1
        return _nan_safe(b.iloc[i]) if i >= 0 else NAN

    def features_v3(self, symbol: str, df: pd.DataFrame, direction: str) -> dict:
        self.ensure_v3()
        f = self.features(symbol, df, direction)        # every V2 feature
        info = sector_map.lookup(symbol)
        code = info["sector"]
        row = df.iloc[-1]
        ts = pd.Timestamp(row["timestamp"]).tz_convert(IST)
        d = ts.date()
        close = _nan_safe(row["close"])
        sign = 1.0 if direction == "long" else -1.0
        t30 = ts - pd.Timedelta(minutes=30)

        for c in sector_map.SECTOR_CODES:
            f[f"sec_{c}"] = 1.0 if c == code else 0.0

        # --- sector tape ---
        name = "NIFTY" if code == "OTHER" else f"SEC_{code}"
        dser = self._daily.get("MKT_NIFTY" if code == "OTHER" else f"SEC_{code}")
        sec_now = self.asof(name, ts)
        f["sector_ret_day"] = _ratio(sec_now, dser.prior(d, "close")) if dser else NAN
        f["sector_ret_30m"] = _ratio(sec_now, self.asof(name, t30))
        f["sector_ret_since_open"] = _ratio(sec_now, self.asof(name, ts, "day_open"))
        hi, lo = self.asof(name, ts, "day_high"), self.asof(name, ts, "day_low")
        f["sector_range_pos"] = ((sec_now - lo) / (hi - lo)) if (hi == hi and lo == lo and hi > lo) else NAN

        def _diff(a, b):
            return a - b if not (np.isnan(a) or np.isnan(b)) else NAN
        f["rs_vs_sector_day"] = _diff(f["sym_ret_day"], f["sector_ret_day"])
        past30 = df[df["timestamp"] <= t30]
        sym30 = _ratio(close, past30["close"].iat[-1]) if len(past30) else NAN
        f["rs_vs_sector_30m"] = _diff(sym30, f["sector_ret_30m"])
        f["sector_vs_nifty_day"] = _diff(f["sector_ret_day"], f["nifty_ret_day"])

        # --- overnight macro (prior US / FX session) ---
        f["brent_ret_1d"] = self._macro_prior("BRENT", d, 1)
        f["brent_ret_5d"] = self._macro_prior("BRENT", d, 5)
        f["usdinr_ret_1d"] = self._macro_prior("USDINR", d, 1)
        f["spx_ret_1d"] = self._macro_prior("SPX", d, 1)
        f["us10y_chg_1d"] = self._macro_prior("US10Y", d, 1)
        f["gold_ret_1d"] = self._macro_prior("GOLD", d, 1)

        # --- dynamic sector sensitivities x overnight moves ---
        for mac, feat, move in (("BRENT", "crude", f["brent_ret_1d"]),
                                ("USDINR", "inr", f["usdinr_ret_1d"]),
                                ("SPX", "us", f["spx_ret_1d"])):
            b = self._beta(code, mac, d)
            f[f"sector_{feat}_beta"] = b
            f[f"{feat}_impulse"] = b * move if not (np.isnan(b) or np.isnan(move)) else NAN
            sb = self._stock_beta(symbol, mac, d)
            f[f"stock_{feat}_beta"] = sb
            f[f"stock_{feat}_impulse"] = sb * move if not (np.isnan(sb) or np.isnan(move)) else NAN

        # --- breakout structure ---
        reg = df[df["timestamp"] >= session_open(ts)]
        prev = reg.iloc[:-1]
        prev_hi = float(prev["high"].max()) if len(prev) else NAN
        f["dist_day_high"] = _ratio(close, prev_hi)
        f["new_day_high"] = 1.0 if (prev_hi == prev_hi and close > prev_hi) else 0.0
        nd = f["nifty_ret_day"]
        f["weak_mkt_rs"] = f["rs_vs_nifty_day"] if (nd == nd and nd < 0 and f["rs_vs_nifty_day"] == f["rs_vs_nifty_day"]) else 0.0

        def _al(v):
            return sign * v if v == v else NAN
        f["align_sector_day"] = _al(f["sector_ret_day"])
        f["align_rs_sector"] = _al(f["rs_vs_sector_day"])
        f["align_crude_impulse"] = _al(f["crude_impulse"])
        f["align_inr_impulse"] = _al(f["inr_impulse"])
        f["align_us_impulse"] = _al(f["us_impulse"])
        f["align_stock_crude_impulse"] = _al(f["stock_crude_impulse"])
        f["align_stock_inr_impulse"] = _al(f["stock_inr_impulse"])
        f["align_stock_us_impulse"] = _al(f["stock_us_impulse"])
        return f


_shared_v3 = {}


def get_context_v3(live: bool = None) -> MarketContextV3:
    if live is None:
        live = _mode["live"]
    key = "live" if live else f"backtest|{_mode['history_start']}"
    if key not in _shared_v3:
        start = None if live else _mode["history_start"]
        _shared_v3[key] = MarketContextV3(live=live, history_start=start)
    return _shared_v3[key]


def reset_context_v3():
    _shared_v3.clear()
