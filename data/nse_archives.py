"""
NSE's own daily market data + corporate events, for RESEARCH (V3.2).

    python -m data.nse_archives --from 2022-01-01 --to 2026-09-25     # download + cache

Sources (public NSE archives / APIs, no login):
  CM bhavcopy "sec_bhavdata_full" (daily)  per stock: traded qty, DELIVERY
      qty and DELIVERY %, turnover, number of trades. Published after the
      close -> usable from the NEXT session.
  F&O bhavcopy (daily; old format until 2024-07-05, UDiFF after) per
      underlying: stock-futures OI and OI change, futures contracts
      traded, stock-options call/put OI (-> PCR) and volumes. Also after
      the close -> next session.
  Corporate announcements, board meetings, insider (PIT) disclosures -
      each with the timestamp it became PUBLIC (an_dt / bm_timestamp /
      disclosure date), so research can use only what was known before a
      given moment.

Raw files are reduced on download to per-symbol rows and cached under
backtest/cache/nse/ (holidays cached as empty); re-runs resume. The
production scanner does not use this module.
"""
import argparse
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
import requests

CACHE = os.path.join("backtest", "cache", "nse")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
      "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
UDIFF_FROM = date(2024, 7, 8)          # F&O bhavcopy format switch

_local = {}


def _session() -> requests.Session:
    import threading
    k = threading.get_ident()
    if k not in _local:
        s = requests.Session()
        s.headers.update(UA)
        _local[k] = s
    return _local[k]


def _get(url, attempts=4):
    last = None
    for i in range(attempts):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code == 404:
                return None                      # holiday / no file
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def _path(kind, key):
    d = os.path.join(CACHE, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.pkl")


def _cached(kind, key, fn, cache_empty=True):
    p = _path(kind, key)
    if os.path.exists(p):
        return pd.read_pickle(p)
    df = fn()
    if df.empty and not cache_empty:
        return df
    tmp = f"{p}.tmp.{os.getpid()}"
    df.to_pickle(tmp)
    os.replace(tmp, p)
    return df


# ═══════════════════════════════════════════════════════════════════
# Daily bhavcopies
# ═══════════════════════════════════════════════════════════════════

def cm_day(d: date) -> pd.DataFrame:
    def fetch():
        r = _get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv")
        if r is None:
            return pd.DataFrame()
        if not r.text.lstrip().upper().startswith("SYMBOL"):
            # an error / throttling page, not a bhavcopy: fail -> not cached, retried next run
            raise ValueError(f"CM {d}: unexpected response ({r.text[:60]!r})")
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df["SERIES"] = df["SERIES"].astype(str).str.strip()
        # NSE can serve another session's file for a holiday URL - trust
        # the file's own date column, not the URL.
        fdate = pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce").dt.date
        if fdate.dropna().empty or fdate.dropna().iloc[0] != d:
            return pd.DataFrame()
        df = df[df["SERIES"] == "EQ"]
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].astype(str).str.strip(),
            "qty": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
            "deliv_qty": pd.to_numeric(df["DELIV_QTY"], errors="coerce"),
            "deliv_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
            "turnover_lacs": pd.to_numeric(df["TURNOVER_LACS"], errors="coerce"),
            "trades": pd.to_numeric(df["NO_OF_TRADES"], errors="coerce"),
        })
        out["date"] = d
        return out.reset_index(drop=True)
    # An empty result is only a HOLIDAY for past dates; today's / future
    # bhavcopies just aren't published yet and must not be cached as empty.
    return _cached("cm", f"{d:%Y%m%d}", fetch, cache_empty=d < date.today())


def _fo_old(d):
    mon = d.strftime("%b").upper()
    r = _get(f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d:%Y}/{mon}/"
             f"fo{d:%d}{mon}{d:%Y}bhav.csv.zip")
    if r is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df.columns = [c.strip() for c in df.columns]
    fdate = pd.to_datetime(df["TIMESTAMP"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce").dt.date
    if fdate.dropna().empty or fdate.dropna().iloc[0] != d:
        return None
    return pd.DataFrame({"inst": df["INSTRUMENT"].str.strip(), "symbol": df["SYMBOL"].str.strip(),
                         "opt": df["OPTION_TYP"].astype(str).str.strip(), "oi": df["OPEN_INT"],
                         "oi_chg": df["CHG_IN_OI"], "vol": df["CONTRACTS"]}).assign(
        kind=lambda x: x["inst"].map({"FUTSTK": "FUT", "OPTSTK": "OPT"}))


def _fo_udiff(d):
    r = _get(f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
    if r is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    fdate = pd.to_datetime(df["TradDt"].astype(str).str.strip(), errors="coerce").dt.date
    if fdate.dropna().empty or fdate.dropna().iloc[0] != d:
        return None
    return pd.DataFrame({"inst": df["FinInstrmTp"].astype(str).str.strip(), "symbol": df["TckrSymb"].astype(str).str.strip(),
                         "opt": df["OptnTp"].astype(str).str.strip(), "oi": df["OpnIntrst"],
                         "oi_chg": df["ChngInOpnIntrst"], "vol": df["TtlTradgVol"]}).assign(
        kind=lambda x: x["inst"].map({"STF": "FUT", "STO": "OPT"}))


def fo_day(d: date) -> pd.DataFrame:
    def fetch():
        raw = _fo_udiff(d) if d >= UDIFF_FROM else _fo_old(d)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = raw[raw["kind"].notna()]
        for c in ("oi", "oi_chg", "vol"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)
        fut = raw[raw["kind"] == "FUT"].groupby("symbol")[["oi", "oi_chg", "vol"]].sum().add_prefix("fut_")
        opt = raw[raw["kind"] == "OPT"]
        ce = opt[opt["opt"] == "CE"].groupby("symbol")[["oi", "oi_chg", "vol"]].sum().add_prefix("call_")
        pe = opt[opt["opt"] == "PE"].groupby("symbol")[["oi", "oi_chg", "vol"]].sum().add_prefix("put_")
        out = fut.join(ce, how="outer").join(pe, how="outer").fillna(0).reset_index()
        out["date"] = d
        return out
    return _cached("fo", f"{d:%Y%m%d}", fetch, cache_empty=d < date.today())


# ═══════════════════════════════════════════════════════════════════
# Corporate events (API; cached per month)
# ═══════════════════════════════════════════════════════════════════

def _api_json(url):
    for i in range(4):
        try:
            r = _session().get(url, timeout=40)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(3 * (i + 1))
    return None


def _weeks(m_start: date, m_end: date):
    cur = m_start
    while cur <= m_end:
        yield cur, min(cur + timedelta(days=6), m_end)
        cur += timedelta(days=7)


def announcements_month(m_start: date, use_cache: bool = None) -> pd.DataFrame:
    """use_cache=None: cache only COMPLETE past months (the current month
    is still growing and is always fetched fresh)."""
    m_end = (m_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

    def fetch():
        rows = []
        for a, b in _weeks(m_start, m_end):
            js = _api_json("https://www.nseindia.com/api/corporate-announcements?index=equities"
                           f"&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}")
            if js is None:
                raise RuntimeError(f"announcements {a}..{b} unavailable - not caching this month")
            for x in js:
                rows.append({"symbol": x.get("symbol"), "time": x.get("sort_date") or x.get("an_dt"),
                             "desc": (x.get("desc") or "")[:120], "text": (x.get("attchmntText") or "")[:300]})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df
    if use_cache is None:
        today = date.today()
        use_cache = (m_start.year, m_start.month) < (today.year, today.month)
    return _cached("announcements", f"{m_start:%Y%m}", fetch) if use_cache else fetch()


def board_meetings_month(m_start: date, use_cache: bool = None) -> pd.DataFrame:
    """use_cache=None: cache only COMPLETE past months (the current month
    is still growing and is always fetched fresh)."""
    m_end = (m_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

    def fetch():
        js = _api_json("https://www.nseindia.com/api/corporate-board-meetings?index=equities"
                       f"&from_date={m_start:%d-%m-%Y}&to_date={m_end:%d-%m-%Y}")
        if js is None:
            raise RuntimeError(f"board meetings {m_start:%Y-%m} unavailable - not caching")
        df = pd.DataFrame([{"symbol": x.get("bm_symbol"), "meeting": x.get("bm_date"),
                            "announced": x.get("bm_timestamp"), "purpose": (x.get("bm_purpose") or "")[:200]}
                           for x in js])
        if not df.empty:
            df["meeting"] = pd.to_datetime(df["meeting"], format="%d-%b-%Y", errors="coerce")
            df["announced"] = pd.to_datetime(df["announced"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
        return df
    if use_cache is None:
        today = date.today()
        use_cache = (m_start.year, m_start.month) < (today.year, today.month)
    return _cached("board_meetings", f"{m_start:%Y%m}", fetch) if use_cache else fetch()


def insider_month(m_start: date, use_cache: bool = None) -> pd.DataFrame:
    """use_cache=None: cache only COMPLETE past months (the current month
    is still growing and is always fetched fresh)."""
    m_end = (m_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

    def fetch():
        rows = []
        for a, b in _weeks(m_start, m_end):
            js = _api_json("https://www.nseindia.com/api/corporates-pit?index=equities"
                           f"&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}")
            if js is None:
                raise RuntimeError(f"insider {a}..{b} unavailable - not caching this month")
            for x in js.get("data", []):
                rows.append({"symbol": x.get("symbol"), "time": x.get("date"),
                             "category": x.get("personCategory"), "type": x.get("tdpTransactionType"),
                             "mode": x.get("acqMode"), "value": x.get("secVal")})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], format="%d-%b-%Y %H:%M", errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df
    if use_cache is None:
        today = date.today()
        use_cache = (m_start.year, m_start.month) < (today.year, today.month)
    return _cached("insider", f"{m_start:%Y%m}", fetch) if use_cache else fetch()


# ═══════════════════════════════════════════════════════════════════
# Panels for research
# ═══════════════════════════════════════════════════════════════════

def weekdays(a: date, b: date):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def months(a: date, b: date):
    cur = date(a.year, a.month, 1)
    while cur <= b:
        yield cur
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)


def _safe(fn):
    def run(d):
        try:
            fn(d)
            return None
        except Exception as e:
            return f"{d}: {str(e)[:80]}"
    return run


def download(a: date, b: date, workers: int = 6, events: bool = True):
    days = list(weekdays(a, b))
    for fn, name in ((cm_day, "CM bhavcopy"), (fo_day, "F&O bhavcopy")):
        failed = []
        with ThreadPoolExecutor(workers) as ex:
            for n, err in enumerate(ex.map(_safe(fn), days), 1):
                if err:
                    failed.append(err)
                if n % 200 == 0:
                    print(f"  {name} {n}/{len(days)}", flush=True)
        print(f"  {name}: {len(days) - len(failed)}/{len(days)} days ok"
              + (f"; {len(failed)} failed, e.g. {failed[:3]} (re-run to retry)" if failed else ""), flush=True)
    if events:
        ms = list(months(a, b))
        for fn, name in ((board_meetings_month, "board meetings"), (announcements_month, "announcements"),
                         (insider_month, "insider")):
            failed = []
            for m in ms:
                try:
                    fn(m)
                except Exception:
                    failed.append(f"{m:%Y-%m}")
            print(f"  {name}: {len(ms) - len(failed)}/{len(ms)} months cached"
                  + (f", failed {failed} (re-run to retry)" if failed else ""), flush=True)


def panel(fn, a: date, b: date) -> pd.DataFrame:
    """All cached days in [a, b]; a day that can't be fetched (e.g. NSE served
    a non-CSV file for it) is left out rather than failing the whole panel."""
    frames = []
    for d in weekdays(a, b):
        try:
            frames.append(fn(d))
        except Exception as e:
            print(f"  (skipping {d}: {str(e)[:60]})")
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def event_panel(fn, a: date, b: date) -> pd.DataFrame:
    frames = []
    for m in months(a, b):
        try:
            frames.append(fn(m))
        except Exception as e:
            print(f"  (skipping {m:%Y-%m}: {str(e)[:60]})")
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="a", required=True)
    p.add_argument("--to", dest="b", required=True)
    p.add_argument("--no-events", action="store_true")
    args = p.parse_args()
    a, b = (datetime.strptime(x, "%Y-%m-%d").date() for x in (args.a, args.b))
    download(a, b, events=not args.no_events)
    print("done")


if __name__ == "__main__":
    main()
