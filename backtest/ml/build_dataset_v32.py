"""
Builds the (stock, day) dataset for L_ML_V32 - the daily 9:45 ranking
model (strategy/v32_features.py explains the design).

    python -m backtest.ml.build_dataset_v32                      # 2022-01-03 -> today
    python -m backtest.ml.build_dataset_v32 --from 2023-01-01 --symbols RELIANCE,TCS

Reads 5-min candles from backtest/cache (fill it first with
`python -m backtest.candle_cache --from 2022-01-01 --to <today> --with-indices`),
daily candles from 2015 (for 52-week / 200-day / beta history) and the
macro series from Yahoo. Output: config.ML_V32_DATASET_PATH, one row per
(stock, trading day), with `excess` = the stock's net 9:45->15:15 return
minus that day's universe average.
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

import config
from backtest.candle_cache import load_daily_cached, load_symbol_history_cached
from backtest.ml.common import access_token, load_symbol_list
from strategy import sector_map
from strategy.v32_features import (MACROS, BETA_MACROS, daily_table, earnings_features,
                                   intraday_day_table, lagged_macro_level_change,
                                   lagged_macro_returns, rolling_betas, safe_ratio)

DAILY_FROM = "2015-01-01"
MARKET = {}                     # filled in each worker by _init


def _macro_series() -> dict:
    path = os.path.join(config.ML_V2_CACHE_DIR, "v32_macro.pkl")
    try:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if datetime.fromtimestamp(cached["fetched"]).date() == datetime.now().date():
            return cached["series"]
    except Exception:
        pass
    import yfinance as yf
    series = {}
    for name, ticker in config.ML_V3_MACRO_TICKERS.items():
        h = yf.Ticker(ticker).history(start="2014-06-01", interval="1d", auto_adjust=False)
        s = h["Close"].astype(float)
        s.index = pd.to_datetime(s.index).date
        series[name] = s[~pd.Index(s.index).duplicated(keep="last")].sort_index()
    os.makedirs(config.ML_V2_CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"fetched": time.time(), "series": series}, f)
    return series


def load_market(from_date, to_date, extend_to=None) -> dict:
    """extend_to (live only): also compute every date-indexed market value
    for that date (the next session after to_date), exactly as training
    computed it for a day inside the calendar."""
    tok = access_token()
    m = {}
    nd = load_daily_cached("NSE_INDEX|Nifty 50", DAILY_FROM, to_date, tok, is_key=True)
    m["nifty_daily"] = daily_table(nd)
    m["calendar"] = list(m["nifty_daily"].index)
    m["nifty_day"] = intraday_day_table(
        load_symbol_history_cached("NSE_INDEX|Nifty 50", 5, from_date, to_date, tok), is_index=True).set_index("date")
    vd = load_daily_cached("NSE_INDEX|India VIX", DAILY_FROM, to_date, tok, is_key=True)
    vd["date"] = pd.to_datetime(vd["timestamp"]).dt.date
    m["ivix_close"] = vd.drop_duplicates("date", keep="last").set_index("date")["close"]
    # prior-session values, precomputed per date (value known before day d opens)
    iv = m["ivix_close"]
    m["ivix_prior"] = iv.shift(1)
    m["ivix_chg5d_prior"] = (iv / iv.shift(5) - 1).shift(1)
    m["ivix_day"] = intraday_day_table(
        load_symbol_history_cached("NSE_INDEX|India VIX", 5, from_date, to_date, tok), is_index=True).set_index("date")
    m["sector_daily"], m["sector_day"] = {}, {}
    for code, key in sector_map.SECTORS.items():
        if code == "OTHER":
            m["sector_daily"][code], m["sector_day"][code] = m["nifty_daily"], m["nifty_day"]
            continue
        m["sector_daily"][code] = daily_table(load_daily_cached(key, DAILY_FROM, to_date, tok, is_key=True))
        m["sector_day"][code] = intraday_day_table(
            load_symbol_history_cached(key, 5, from_date, to_date, tok), is_index=True).set_index("date")
    # breadth: % of NIFTY 50 names above their own SMA50, per day's close
    closes = {}
    for s in config.ML_V2_BREADTH_UNIVERSE:
        try:
            dd = load_daily_cached(s, DAILY_FROM, to_date, tok)
            dd["date"] = pd.to_datetime(dd["timestamp"]).dt.date
            closes[s] = dd.drop_duplicates("date", keep="last").set_index("date")["close"]
        except Exception:
            pass
    cl = pd.DataFrame(closes).sort_index()
    sma = cl.rolling(50, min_periods=40).mean()
    m["breadth"] = (cl > sma).where(sma.notna()).mean(axis=1)
    m["breadth_prior"] = m["breadth"].shift(1)
    # macro: lagged (prior US session) returns per Indian date
    macro = _macro_series()
    cal = m["calendar"]
    cal_x = cal + ([extend_to] if extend_to is not None and extend_to not in cal else [])
    m["macro_lag1"] = {k: lagged_macro_returns(macro[k], cal_x) for k in MACROS}
    m["macro_lag5"] = {k: lagged_macro_level_change(macro[k], cal_x, 5) for k in ("BRENT",)}
    if extend_to is not None:
        iv = m["ivix_close"]
        m["ivix_prior"] = pd.concat([m["ivix_prior"], pd.Series([iv.iloc[-1]], index=[extend_to])])
        m["ivix_chg5d_prior"] = pd.concat([m["ivix_chg5d_prior"],
                                           pd.Series([iv.iloc[-1] / iv.iloc[-6] - 1], index=[extend_to])])
        m["breadth_prior"] = pd.concat([m["breadth_prior"], pd.Series([m["breadth"].iloc[-1]], index=[extend_to])])
    m["extend_to"] = extend_to
    # sector betas (per sector, per macro), prior-day estimates
    m["sector_betas"] = {code: rolling_betas(t["ret1"].dropna(), m["macro_lag1"], extend_to)
                         for code, t in m["sector_daily"].items()}
    return m


def _init(market):
    MARKET.update(market)


def _prior(table: pd.DataFrame, d):
    """The row of `table` (indexed by date) for the last session BEFORE d.
    The numpy index is stored on the table itself (not in a cache keyed by
    object id, which Python can reuse for a different table)."""
    idx = table.attrs.get("_npidx")
    if idx is None or len(idx) != len(table):
        idx = np.array(table.index, dtype="datetime64[D]")
        table.attrs["_npidx"] = idx
    i = int(np.searchsorted(idx, np.datetime64(d, "D"), side="left")) - 1
    return table.iloc[i] if i >= 0 else None


def _at(series, d):
    try:
        v = series.get(d, np.nan)
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan


def load_earnings(symbol):
    from strategy.market_context import _fetch_earnings, _reaction_date, _cache_load, _cache_save
    cached = _cache_load(f"earnings_{symbol}")
    earn = cached["df"] if cached and time.time() - cached.get("fetched", 0) < 3 * 86400 else None
    if earn is None:
        earn = _fetch_earnings(symbol)
        try:
            _cache_save(f"earnings_{symbol}", {"fetched": time.time(), "df": earn})
        except Exception:
            pass
    if earn is not None and not earn.empty:
        earn = earn.copy()
        earn["reaction"] = earn["ts"].map(_reaction_date)
    return earn


def symbol_context(symbol, daily: pd.DataFrame, m: dict) -> dict:
    """Per-symbol inputs that don't depend on the day: daily indicator
    table, sector, rolling macro betas, earnings calendar."""
    dt = daily_table(daily)
    code = sector_map.lookup(symbol)["sector"]
    return {"dt": dt, "code": code,
            "stock_betas": rolling_betas(dt["ret1"].dropna(), m["macro_lag1"], m.get("extend_to")),
            "sec_betas": m["sector_betas"].get(code, {}), "earn": load_earnings(symbol)}


def row_features(symbol, d, r, sc: dict, m: dict):
    """ALL V3.2 features for one (symbol, day). `r` = that day's opening
    stats (+ rvol30, and entry/exit/outcome/net_ret when building labels).
    Shared by the dataset builder and the live V3.3 runner."""
    dt, code = sc["dt"], sc["code"]
    p = _prior(dt, d)
    np_ = _prior(m["nifty_daily"], d)
    sp = _prior(m["sector_daily"][code], d)
    if p is None or np_ is None or not (p["close"] > 0):
        return None
    prev_close = float(p["close"])
    nd = m["nifty_day"].loc[d] if d in m["nifty_day"].index else None
    sd = m["sector_day"][code].loc[d] if d in m["sector_day"][code].index else None
    vd = m["ivix_day"].loc[d] if d in m["ivix_day"].index else None
    stock_betas, sec_betas = sc["stock_betas"], sc["sec_betas"]

    f = {"symbol": symbol, "date": str(d), "sector": code, "entry": r.get("entry", np.nan),
         "exit": r.get("exit", np.nan), "outcome": r.get("outcome", None), "net_ret": r.get("net_ret", np.nan),
         "nifty_ret_945_1515": float(nd["ret_945_1515"]) if (nd is not None and "ret_945_1515" in nd) else np.nan,
         "dow": float(pd.Timestamp(d).dayofweek)}
    # overnight macro + sensitivities
    for mac in MACROS:
        f[f"{mac.lower()}_ret1d"] = _at(m["macro_lag1"][mac], d)
    f["brent_ret5d"] = _at(m["macro_lag5"]["BRENT"], d)
    for mac, nm in (("BRENT", "crude"), ("USDINR", "inr"), ("SPX", "us")):
        move = f[f"{mac.lower()}_ret1d"]
        sb = _at(stock_betas.get(mac, pd.Series(dtype=float)), d)
        xb = _at(sec_betas.get(mac, pd.Series(dtype=float)), d)
        f[f"stock_{nm}_beta"], f[f"sector_{nm}_beta"] = sb, xb
        f[f"stock_{nm}_impulse"] = sb * move if sb == sb and move == move else np.nan
        f[f"sector_{nm}_impulse"] = xb * move if xb == xb and move == move else np.nan
    # stock daily (prior sessions)
    for k in ("ret1", "ret5", "ret20", "dist_high20", "dist_high252", "sma50_dist", "sma200_dist",
              "atr_pct", "rsi14", "vol5_20"):
        f[f"d_{k}"] = float(p[k]) if p[k] == p[k] else np.nan
    f["d_rs63"] = (p["ret63"] - np_["ret63"]) if (p["ret63"] == p["ret63"] and np_["ret63"] == np_["ret63"]) else np.nan
    # opening 30 minutes (9:15-9:40 candles)
    f["gap"] = safe_ratio(r["o915"], prev_close)
    f["r30"] = safe_ratio(r["c940"], r["o915"])
    f["ret_prev_to_940"] = safe_ratio(r["c940"], prev_close)
    f["range30"] = (r["hi30"] - r["lo30"]) / r["o915"] if r["o915"] else np.nan
    f["pos30"] = (r["c940"] - r["lo30"]) / (r["hi30"] - r["lo30"]) if r["hi30"] > r["lo30"] else 0.5
    f["rvol30"] = float(r["rvol30"]) if r["rvol30"] == r["rvol30"] else np.nan
    f["vwap30_dist"] = safe_ratio(r["c940"], r["vwap30"])
    # market + sector
    f["nifty_ret1d"], f["nifty_ret5d"] = float(np_["ret1"]), float(np_["ret5"])
    f["nifty_sma200_dist"] = float(np_["sma200_dist"]) if np_["sma200_dist"] == np_["sma200_dist"] else np.nan
    f["nifty_gap"] = safe_ratio(nd["o915"], np_["close"]) if nd is not None else np.nan
    f["nifty_r30"] = safe_ratio(nd["c940"], nd["o915"]) if nd is not None else np.nan
    f["ivix_prior"] = _at(m["ivix_prior"], d)
    f["ivix_chg5d"] = _at(m["ivix_chg5d_prior"], d)
    f["ivix_open_chg"] = safe_ratio(vd["c940"], f["ivix_prior"]) if vd is not None else np.nan
    f["breadth_pct50"] = _at(m["breadth_prior"], d)
    f["sector_ret1d"] = float(sp["ret1"]) if sp is not None else np.nan
    f["sector_ret5d"] = float(sp["ret5"]) if sp is not None else np.nan
    f["sector_rs20"] = (sp["ret20"] - np_["ret20"]) if sp is not None else np.nan
    f["sector_gap"] = safe_ratio(sd["o915"], sp["close"]) if (sd is not None and sp is not None) else np.nan
    f["sector_r30"] = safe_ratio(sd["c940"], sd["o915"]) if sd is not None else np.nan
    f["rs30_vs_nifty"] = f["r30"] - f["nifty_r30"] if f["nifty_r30"] == f["nifty_r30"] else np.nan
    f["rs30_vs_sector"] = f["r30"] - f["sector_r30"] if f["sector_r30"] == f["sector_r30"] else np.nan
    f["gap_vs_nifty"] = f["gap"] - f["nifty_gap"] if f["nifty_gap"] == f["nifty_gap"] else np.nan
    f["gap_vs_sector"] = f["gap"] - f["sector_gap"] if f["sector_gap"] == f["sector_gap"] else np.nan
    f.update(earnings_features(sc["earn"], d))
    for c in sector_map.SECTOR_CODES:
        f[f"sec_{c}"] = 1.0 if c == code else 0.0
    return f


def build_symbol(symbol, from_date, to_date):
    m = MARKET
    tok = access_token()
    try:
        intraday = load_symbol_history_cached(symbol, 5, from_date, to_date, tok)
        daily = load_daily_cached(symbol, DAILY_FROM, to_date, tok)
    except Exception as e:
        return symbol, [], f"FAILED: {str(e)[:80]}"
    if intraday.empty or len(daily) < 260:
        return symbol, [], "not enough history"
    days = intraday_day_table(intraday)
    if days.empty:
        return symbol, [], "no usable days"
    days = days.set_index("date")
    days["rvol30"] = days["vol30"] / days["vol30"].rolling(20, min_periods=10).mean().shift(1)
    sc = symbol_context(symbol, daily, m)
    rows = []
    for d, r in days.iterrows():
        f = row_features(symbol, d, r, sc, m)
        if f is not None:
            rows.append(f)
    return symbol, rows, f"{len(rows)} days"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="from_date", default="2022-01-03")
    p.add_argument("--to", dest="to_date", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--symbols")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=config.ML_V32_DATASET_PATH)
    args = p.parse_args()
    symbols = ([s.strip().upper() for s in args.symbols.split(",")] if args.symbols else
               list(dict.fromkeys(load_symbol_list("watchlist.txt") + load_symbol_list("universe.txt")
                                  + list(config.ML_V2_BREADTH_UNIVERSE))))
    print(f"V3.2 dataset: {len(symbols)} symbols | {args.from_date}..{args.to_date}")
    print("Loading market, sector, India VIX, breadth and macro history...", flush=True)
    market = load_market(args.from_date, args.to_date)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(market,)) as pool:
        futs = {pool.submit(build_symbol, s, args.from_date, args.to_date): s for s in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                sym, sym_rows, status = fut.result()
            except Exception as e:
                sym, sym_rows, status = futs[fut], [], f"CRASHED: {e!r}"
            rows += sym_rows
            print(f"[{n}/{len(symbols)}] {sym}: {status}", flush=True)
    df = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    # cross-sectional target: net return minus the day's universe average
    df["day_mean"] = df.groupby("date")["net_ret"].transform("mean")
    df["excess"] = df["net_ret"] - df["day_mean"]
    df["n_day"] = df.groupby("date")["net_ret"].transform("size")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = f"{args.out}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, args.out)
    print(f"\nRows: {len(df)} | days: {df['date'].nunique()} | symbols: {df['symbol'].nunique()}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
