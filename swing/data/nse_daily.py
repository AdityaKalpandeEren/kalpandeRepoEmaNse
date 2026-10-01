"""
Free NSE end-of-day archives -> per-day parquet cache.

  equity      CM bhavcopy, EQ series: OHLC, PREVCLOSE, volume, traded value,
              trades, ISIN. Old format (cmDDMONYYYYbhav.csv.zip) until
              2024-07-05, UDiFF (BhavCopy_NSE_CM_...) from 2024-07-08.
  delivery    MTO_DDMMYYYY.DAT: delivery % per EQ symbol.
  indices     ind_close_all_DDMMYYYY.csv: closes of NIFTY 50 and the sector
              indices (old "CNX ..." names normalised to "NIFTY ...").
  participant fao_participant_oi_DDMMYYYY.csv: index-futures long/short
              contracts of FII, DII, Pro and Client.

All of these are published after the close of day t, so a model may use
them from the session of t+1 on - features at t are computed after the
close of t and only drive trades at the open of t+1.

Holidays return 404 and are cached as empty files so re-runs skip them.
"""
from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from data.nse_archives import _get   # shared session with browser headers + retries

log = logging.getLogger(__name__)

BASE = "https://nsearchives.nseindia.com"
UDIFF_FROM = date(2024, 7, 8)
KINDS = ("equity", "delivery", "indices", "participant")


# ─── per-kind fetch + parse ─────────────────────────────────────────────

def _equity(d: date) -> pd.DataFrame:
    if d < UDIFF_FROM:
        mon = f"{d:%b}".upper()
        url = f"{BASE}/content/historical/EQUITIES/{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip"
        r = _get(url)
        if r is None:
            return pd.DataFrame()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))
        df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].astype(str).str.strip(), "isin": df["ISIN"].astype(str).str.strip(),
            "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"], "close": df["CLOSE"],
            "prevclose": df["PREVCLOSE"], "volume": df["TOTTRDQTY"], "value": df["TOTTRDVAL"],
            "trades": df.get("TOTALTRADES"),
        })
        file_date = pd.to_datetime(df["TIMESTAMP"].astype(str).str.strip(), format="%d-%b-%Y",
                                   errors="coerce").dt.date
    else:
        url = f"{BASE}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
        r = _get(url)
        if r is None:
            return pd.DataFrame()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))
        df = df[df["SctySrs"].astype(str).str.strip() == "EQ"]
        out = pd.DataFrame({
            "symbol": df["TckrSymb"].astype(str).str.strip(), "isin": df["ISIN"].astype(str).str.strip(),
            "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"], "close": df["ClsPric"],
            "prevclose": df["PrvsClsgPric"], "volume": df["TtlTradgVol"], "value": df["TtlTrfVal"],
            "trades": df["TtlNbOfTxsExctd"],
        })
        file_date = pd.to_datetime(df["TradDt"].astype(str).str.strip(), errors="coerce").dt.date
    # NSE sometimes serves another session's file for a holiday URL.
    if file_date.dropna().empty or file_date.dropna().iloc[0] != d:
        return pd.DataFrame()
    for c in ("open", "high", "low", "close", "prevclose", "volume", "value", "trades"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out.reset_index(drop=True)


def _delivery(d: date) -> pd.DataFrame:
    r = _get(f"{BASE}/archives/equities/mto/MTO_{d:%d%m%Y}.DAT")
    if r is None:
        return pd.DataFrame()
    head = r.text[:400]
    if f"{d:%d-%b-%Y}".upper() not in head.upper():
        return pd.DataFrame()
    rows = []
    for line in r.text.splitlines():
        p = line.split(",")
        if len(p) >= 7 and p[0] == "20" and p[3].strip() == "EQ":
            rows.append((p[2].strip(), pd.to_numeric(p[6], errors="coerce")))
    return pd.DataFrame(rows, columns=["symbol", "deliv_pct"])


def normalise_index_name(name: str) -> str:
    n = " ".join(str(name).upper().replace("S&P ", "").split())
    if n in ("CNX NIFTY", "NIFTY", "NIFTY 50"):
        return "NIFTY 50"
    if n.startswith("CNX "):
        n = "NIFTY " + n[4:]
    return {"NIFTY NIFTY JUNIOR": "NIFTY NEXT 50", "NIFTY JUNIOR": "NIFTY NEXT 50",
            "NIFTY FINANCE": "NIFTY FINANCIAL SERVICES", "NIFTY FIN SERVICE": "NIFTY FINANCIAL SERVICES",
            "NIFTY 500": "NIFTY 500"}.get(n, n)


def _indices(d: date) -> pd.DataFrame:
    r = _get(f"{BASE}/content/indices/ind_close_all_{d:%d%m%Y}.csv")
    if r is None or not r.text.lstrip().startswith("Index Name"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text))
    fdate = pd.to_datetime(df["Index Date"].astype(str), dayfirst=True, errors="coerce").dt.date
    if fdate.dropna().empty or fdate.dropna().iloc[0] != d:
        return pd.DataFrame()
    return pd.DataFrame({"index": df["Index Name"].map(normalise_index_name),
                         "close": pd.to_numeric(df["Closing Index Value"], errors="coerce")})


def _participant(d: date) -> pd.DataFrame:
    r = _get(f"{BASE}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv")
    if r is None:
        return pd.DataFrame()
    lines = r.text.splitlines()
    m = re.search(r"as on ([A-Za-z]+ \d{1,2}, \d{4})", lines[0]) if lines else None
    if m is None or pd.to_datetime(m.group(1), errors="coerce") != pd.Timestamp(d):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines[1:])))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Client Type"].astype(str).str.strip().isin(["Client", "DII", "FII", "Pro"])]
    keep = ["Future Index Long", "Future Index Short", "Option Index Call Long", "Option Index Put Long",
            "Option Index Call Short", "Option Index Put Short"]
    out = df[["Client Type"] + keep].copy()
    out.columns = ["participant", "fut_idx_long", "fut_idx_short", "opt_idx_call_long",
                   "opt_idx_put_long", "opt_idx_call_short", "opt_idx_put_short"]
    out["participant"] = out["participant"].str.strip()
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return out.reset_index(drop=True)


FETCHERS: dict[str, Callable[[date], pd.DataFrame]] = {
    "equity": _equity, "delivery": _delivery, "indices": _indices, "participant": _participant,
}


# ─── cache ──────────────────────────────────────────────────────────────

def _file(cache_dir: str, kind: str, d: date) -> str:
    p = os.path.join(cache_dir, "raw", kind)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"{d:%Y%m%d}.parquet")


def get_day(cache_dir: str, kind: str, d: date) -> pd.DataFrame:
    """One day's file for `kind`, from cache or NSE. Past empty days (holidays)
    are cached; today's not-yet-published file is not."""
    f = _file(cache_dir, kind, d)
    if os.path.exists(f):
        return pd.read_parquet(f)
    df = FETCHERS[kind](d)
    if df.empty and d >= date.today():
        return df
    tmp = f"{f}.tmp{os.getpid()}"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, f)
    return df


def weekdays(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def download(cache_dir: str, start: date, end: date, workers: int = 6,
             kinds: tuple[str, ...] = KINDS) -> dict[str, int]:
    """Fetch every missing (kind, day) in [start, end]. Returns failures per kind."""
    jobs = [(k, d) for d in weekdays(start, end) for k in kinds
            if not os.path.exists(_file(cache_dir, k, d))]
    log.info("download: %d missing files (%s .. %s)", len(jobs), start, end)
    failures = {k: 0 for k in kinds}
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get_day, cache_dir, k, d): (k, d) for k, d in jobs}
        for fut in as_completed(futs):
            k, d = futs[fut]
            done += 1
            try:
                fut.result()
            except Exception as e:                     # not cached -> retried next run
                failures[k] += 1
                log.warning("fetch failed %s %s: %r", k, d, e)
            if done % 500 == 0:
                log.info("download: %d/%d", done, len(jobs))
    return failures


def load_kind(cache_dir: str, kind: str, start: date, end: date) -> pd.DataFrame:
    """Concatenate cached days of one kind into a long frame with a `date` column."""
    frames = []
    for d in weekdays(start, end):
        f = _file(cache_dir, kind, d)
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        if not df.empty:
            df["date"] = pd.Timestamp(d)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
