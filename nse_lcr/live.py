"""
NSE LCR (Large-Cap Runners) - live paper trading, one step per run
(09:20-15:30 IST, same cron-job.org trigger).

    python -m nse_lcr.live                 # what the NSE Alert Scan `lcr` job runs
    python -m nse_lcr.live --no-telegram

Each run:
  1. DYNAMIC universe, rebuilt daily: NIFTY 50 (MEGA) + rest of NIFTY 100
     (LARGE) + NIFTY Midcap 150 (MID) from NSE's constituent CSVs, PLUS every
     other NSE stock with market cap >= Rs 25,000 cr (Yahoo screener; tagged
     "MID · off-index" etc.). Today's 5-min
     candles for all ~250 are fetched in parallel (as V3.3's pick) and the
     DYNAMIC list = the stocks having a volume shock right now (time-adjusted
     volume >= RVOL_MIN, up >= WATCH_PCT) + open positions.
  2. 🏛️📡 VOLUME WATCH: time-adjusted volume >= 3x while the price is within
     +-1.5% (unusual volume, no move yet) - batched at most every 30 min.
  3. Every NEW setup on a closed candle (nse_lcr/strategy.py: pullback
     continuation above the 10 & 20-day EMAs) -> paper BUY at the next
     candle's open; ⭐ PERFECT when above all six daily EMAs.
  4. INTRADAY + SWING (S1): stop under the pullback low (max 4%), no
     intraday trail. Stopped today = INTRADAY trade (intraday charges);
     still open at 15:15 = carried as SWING (delivery charges), held up to
     5 sessions: exit at a gap below the stop, at the stop, or day-5 15:15.
  5. Separate NSE LCR end-of-day report at 15:20: intraday trades, swing
     carried / closed / open (day n/5, unrealised P&L), all-time by kind.
Alerts go to every Telegram receiver (personal + group).
Paper only.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from nse_lcr import strategy as S

IST = ZoneInfo("Asia/Kolkata")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("NSE_LCR_STATE_DIR", os.path.join(ROOT, "live_state", "nse_lcr"))
V_MAX = int(os.environ.get("NSE_LCR_MAX_TRADES_PER_DAY", "13"))
WATCH_X = 3.0
HEADER = "🏛️🏛️ NSE LARGE-CAP RUNNER (NSE LCR) 🏛️🏛️"


def _p(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def load_state():
    try:
        return json.load(open(_p("state.json")))
    except Exception:
        return {}


def save_state(st):
    tmp = _p("state.json.tmp")
    json.dump(st, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, _p("state.json"))


def notify(text, telegram):
    print(text, flush=True)
    if telegram:
        from paper_trading.research_live import send_text
        send_text(text)


MIN_MCAP_CR = float(os.environ.get("NSE_LCR_MIN_MCAP_CR", "25000"))   # off-index stocks: market cap >= Rs 25,000 cr


def _cap_bucket(mcap_cr: float) -> str:
    return "MEGA" if mcap_cr >= 500_000 else ("LARGE" if mcap_cr >= 100_000 else "MID")


def off_index_large_caps() -> dict:
    """Every NSE stock with market cap >= MIN_MCAP_CR from Yahoo's free
    screener (region in, exchange NSI) - catches large / mid companies that
    are not in NIFTY 50 / 100 / Midcap 150 (new listings, recent risers).
    Yahoo's live % change lags, so it is used for SIZE only; the volume
    shock is checked on real-time Upstox candles. {symbol: (bucket, mcap_cr)}"""
    import yfinance as yf
    from yfinance import EquityQuery as EQ
    q = EQ("and", [EQ("eq", ["region", "in"]), EQ("eq", ["exchange", "NSI"]), EQ("gte", ["intradaymarketcap", MIN_MCAP_CR * 1e7])])
    out, offset = {}, 0
    while True:
        try:
            r = yf.screen(q, sortField="intradaymarketcap", sortAsc=False, size=250, offset=offset)
        except Exception as e:
            print(f"[NSE LCR] Yahoo NSE screener failed: {e!r}")
            break
        qs = r.get("quotes", [])
        for x in qs:
            sym, mc = str(x.get("symbol", "")), x.get("marketCap") or 0
            if sym.endswith(".NS") and mc:
                out[sym[:-3]] = (_cap_bucket(mc / 1e7), round(mc / 1e7))
        offset += len(qs)
        if not qs or offset >= (r.get("total") or 0) or offset >= 1000:
            break
    return out


def universe() -> dict:
    """symbol -> MEGA / LARGE / MID (index stocks) or 'MID · off-index' etc.
    (off-index, by market cap). Rebuilt once a day:
      1. NSE's NIFTY 50 / NIFTY 100 / Midcap 150 constituent CSVs
      2. + every other NSE stock with market cap >= MIN_MCAP_CR (Yahoo)
    Only symbols with an Upstox instrument key are kept."""
    f = os.path.join(ROOT, "nse_lcr", "cache", f"universe_{datetime.now(IST):%Y%m%d}.json")
    if os.path.exists(f):
        return json.load(open(f))
    import requests
    from data.instruments import get_instrument_key
    caps = {}
    for name, tag in (("ind_nifty50list.csv", "MEGA"), ("ind_nifty100list.csv", "LARGE"), ("ind_niftymidcap150list.csv", "MID")):
        try:
            r = requests.get(f"https://archives.nseindia.com/content/indices/{name}", headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            for s in pd.read_csv(io.StringIO(r.text))["Symbol"]:
                caps.setdefault(str(s).strip(), tag)
        except Exception:
            continue
    n_index = len(caps)
    for sym, (bucket, _mc) in off_index_large_caps().items():
        if sym in caps:
            continue
        try:
            get_instrument_key(sym)
        except Exception:
            continue
        caps[sym] = f"{bucket} · off-index"                        # not in NIFTY 50 / 100 / Midcap 150; sized by market cap
    print(f"[NSE LCR] universe: {n_index} index stocks + {len(caps) - n_index} off-index large/mid caps "
          f"(market cap >= Rs {MIN_MCAP_CR:,.0f} cr)", flush=True)
    if caps:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        json.dump(caps, open(f, "w"))
    return caps


def daily_ctx(syms, st) -> dict:
    """prev close, 20-day average volume and daily EMAs (Upstox daily, once a day)."""
    from backtest.candle_cache import load_daily_cached
    from backtest.ml.common import access_token
    cache = st.setdefault("daily", {})
    need = [s for s in syms if s not in cache]
    if need:
        tok = access_token()
        today = datetime.now(IST).date()
        frm = (today - timedelta(days=400)).strftime("%Y-%m-%d")
        prev = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        for s in need:
            try:
                d = load_daily_cached(s, frm, prev, tok)
                if len(d) < 60:
                    cache[s] = None
                    continue
                cache[s] = {"prev_close": float(d["close"].iloc[-1]), "avg_vol20": float(d["volume"].tail(20).mean()),
                            "emas": S.daily_emas(d["close"])}
            except Exception:
                cache[s] = None
    return cache


MAX_OPEN_SWING = int(os.environ.get("NSE_LCR_MAX_OPEN_SWING", "30"))


def _pnl(p, net):
    return round(p["qty"] * p["entry_raw"] * net, 0)


def _log_trade(st, p):
    f = _p("trades.csv")
    new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        row = {"exit_date": st["date"], **{k: p.get(k) for k in ("symbol", "bucket", "kind", "entry_date", "entry_ts", "entry_raw",
                                                                   "stop", "exit_ts", "exit", "outcome", "sessions", "net_pct",
                                                                   "pnl", "pct", "rvol", "qty", "perfect")}}
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run(telegram: bool) -> bool:
    from live_v33 import _c5, prefetch_today
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    st = load_state()
    today = str(now.date())
    if st.get("date") != today:
        swing = st.get("swing", [])                                     # carried positions survive the day change
        st = {"date": today, "positions": [], "last_ts": {}, "taken": 0, "swing": swing, "closed_today": [],
              "carried_today": []}
    if mins < 9 * 60 + 20 or st.get("reported"):
        return False
    caps = universe()
    syms = sorted(set(caps) | {p["symbol"] for p in st["swing"]})
    ctx = daily_ctx(syms, st)
    prefetch_today(syms)
    bars, feats, watch_lines = {}, {}, []
    for s in syms:
        hit, c = _c5.get(s), ctx.get(s)
        if not hit or hit[1] is None or hit[1].empty:
            continue
        x = hit[1].set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        x = x[x.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]
        if len(x) < 1:
            continue
        bars[s] = x
        if c is None or len(x) < 3:
            continue
        f = S.bar_features(x, c["prev_close"], c["avg_vol20"])
        feats[s] = f
        r = f.iloc[-1]
        if r["rvol"] >= WATCH_X and abs(r["pct"]) <= 0.015 and s not in st.setdefault("watch_sent", []):
            watch_lines.append((s, r))
    last_w = st.get("watch_at")
    if watch_lines and mins <= 15 * 60 and (not last_w or now - datetime.fromisoformat(last_w) >= timedelta(minutes=30)):
        lines = []
        for s, r in sorted(watch_lines, key=lambda t: -t[1]["rvol"])[:10]:
            st["watch_sent"].append(s)
            lines.append(f"📡 {s} [{caps.get(s, '?')}] {r['pct'] * 100:+.1f}% | volume {r['rvol']:.1f}x normal for this time | "
                         f"turnover Rs {r['turnover'] / 1e7:,.0f} cr")
        st["watch_at"] = now.isoformat()
        notify("🏛️📡 NSE LCR VOLUME WATCH - large / mid caps with unusual volume, no move yet (watch only)\n" + "\n".join(lines), telegram)
    busy = {p["symbol"] for p in st["positions"] if p["status"] in ("PENDING", "OPEN")} | {p["symbol"] for p in st["swing"]}
    for s, f in feats.items():
        x, c = bars[s], ctx[s]
        last = st["last_ts"].get(s)
        new = [ts for ts in S.setups(x, f, c["emas"]) if last is None or str(ts) > last]
        st["last_ts"][s] = str(x.index[-1])
        for ts in new:
            if st["taken"] >= V_MAX or len(st["swing"]) >= MAX_OPEN_SWING or ts < x.index[-1] - pd.Timedelta(minutes=10) or s in busy:
                continue
            r, close = f.loc[ts], float(x.loc[ts, "close"])
            stop = S.swing_stop(x, x.index.get_loc(ts), close)
            perfect = S.is_perfect(close, c["emas"])
            st["taken"] += 1
            busy.add(s)
            st["positions"].append({"symbol": s, "status": "PENDING", "signal_ts": str(ts), "bucket": caps.get(s, "?"),
                                    "pct": round(float(r["pct"]) * 100, 2), "rvol": round(float(r["rvol"]), 1), "perfect": perfect})
            tag = ("⭐ PERFECT TRADE - above ALL daily EMAs (10/20/30/40/60/180)" if perfect
                   else "✳️ minimum trend - above the 10 & 20-day EMAs, not all longer ones")
            notify(f"{HEADER}\n{tag}\n🚀 BUY {s} [{caps.get(s, '?')}] ~Rs {close:.2f} (next 5-min candle open)\n"
                   f"⚡ {float(r['pct']) * 100:+.1f}% today | volume {float(r['rvol']):.1f}x normal for this time | "
                   f"turnover Rs {float(r['turnover']) / 1e7:,.0f} cr\n📈 daily EMAs: {S.ema_tags(close, c['emas'])}\n"
                   f"🛑 stop ~Rs {stop:.2f} ({(stop / close - 1) * 100:+.1f}%) | INTRADAY + SWING: if not stopped today it is "
                   f"carried and held up to {S.SWING_DAYS} sessions\ntrade {st['taken']}/{V_MAX} today | "
                   f"🧪 UNPROVEN (backtest ~+0.05%/trade, inconsistent halves). Paper only.", telegram)
    manage_today(st, bars, mins, telegram)
    manage_swing(st, bars, mins, telegram)
    if mins >= 15 * 60 + 20 and not st.get("reported"):
        eod_report(st, bars, telegram)
        st["reported"] = True
    save_state(st)
    return True


def manage_today(st, bars, mins, telegram):
    """Today's entries: stop-only on day 0; still open at 15:15 -> carried as swing."""
    for p in st["positions"]:
        if p["status"] not in ("PENDING", "OPEN"):
            continue
        x = bars.get(p["symbol"])
        sig = pd.Timestamp(p["signal_ts"])
        if x is None or sig not in x.index:
            continue
        r = S.day0_status(x, sig)
        if r is None:
            continue
        if r["status"] == "NOFILL":
            p["status"] = "NOFILL"
            continue
        if p["status"] == "PENDING":
            p.update(status="OPEN", entry_raw=round(r["entry_raw"], 2), entry_ts=str(r["entry_ts"]), entry_date=st["date"],
                     stop=round(r["stop"], 2), qty=max(int(S.NOTIONAL // r["entry_raw"]), 1))
        if r["status"] == "STOPPED":
            net = S.intraday_net(p["entry_raw"], r["exit"])
            p.update(status="CLOSED", kind="INTRADAY", exit=round(r["exit"], 2), exit_ts=str(r["exit_ts"]), outcome=r["outcome"],
                     sessions=0, net_pct=round(net * 100, 2), pnl=_pnl(p, net))
            _log_trade(st, p)
            st["closed_today"].append(p)
            notify(f"🏛️ NSE LCR EXIT ❌{'⭐' if p.get('perfect') else ''} {p['symbol']} INTRADAY stop: Rs {p['entry_raw']:.2f} -> "
                   f"Rs {r['exit']:.2f} ({net * 100:+.2f}% net) | paper P&L Rs {p['pnl']:+,.0f}", telegram)
        elif mins >= 15 * 60 + 15:
            p["status"] = "CARRIED"
            st["swing"].append({**p, "kind": "SWING", "sessions": 0, "carried_on": st["date"], "last_session": st["date"]})
            st["carried_today"].append(p["symbol"])
            last = float(x["close"].iloc[-1])
            notify(f"🏛️🌙 NSE LCR CARRY {p['symbol']} as SWING (up to {S.SWING_DAYS} sessions): entry Rs {p['entry_raw']:.2f}, "
                   f"now Rs {last:.2f} ({(last / p['entry_raw'] - 1) * 100:+.2f}%), stop Rs {p['stop']:.2f}", telegram)


def manage_swing(st, bars, mins, telegram):
    """Carried positions on later sessions: gap / stop exit, or the day-5 close."""
    keep = []
    for p in st["swing"]:
        if p.get("carried_on") == st["date"]:                       # carried today - nothing more today
            keep.append(p)
            continue
        x = bars.get(p["symbol"])
        if x is not None and len(x) and p.get("last_session") != st["date"]:
            p["sessions"] = p.get("sessions", 0) + 1                    # count only days with market data (not holidays)
            p["last_session"] = st["date"]
        hit = S.swing_day_exit(x, p["stop"]) if x is not None and len(x) else None
        if hit is None and p.get("sessions", 0) >= S.SWING_DAYS and mins >= 15 * 60 + 15 and x is not None and len(x):
            hit = (float(x["close"].iloc[-1]), x.index[-1], "SWING_DAY5")
        if hit is None:
            p["last_px"] = float(x["close"].iloc[-1]) if x is not None and len(x) else p.get("last_px")
            keep.append(p)
            continue
        exit_px, ts, outcome = hit
        net = S.delivery_net(p["entry_raw"], exit_px)
        p.update(status="CLOSED", exit=round(exit_px, 2), exit_ts=str(ts), outcome=outcome, net_pct=round(net * 100, 2),
                 pnl=_pnl(p, net))
        _log_trade(st, p)
        st["closed_today"].append(p)
        notify(f"🏛️ NSE LCR SWING EXIT {'✅' if net > 0 else '❌'}{'⭐' if p.get('perfect') else ''} {p['symbol']} after "
               f"{p.get('sessions', 0)} session(s): Rs {p['entry_raw']:.2f} -> Rs {exit_px:.2f} ({net * 100:+.2f}% net, {outcome}) | "
               f"paper P&L Rs {p['pnl']:+,.0f}", telegram)
    st["swing"] = keep


def eod_report(st, bars, telegram):
    """Separate NSE LCR end-of-day report: intraday + swing."""
    lines = [f"🏛️📊 NSE LCR REPORT {st['date']} - intraday + swing (paper, UNPROVEN)"]
    intra = [p for p in st["closed_today"] if p.get("kind") == "INTRADAY"]
    sw_closed = [p for p in st["closed_today"] if p.get("kind") == "SWING"]
    lines.append(f"\n⚡ INTRADAY (entered today, stopped today): {len(intra)}")
    for p in intra:
        lines.append(f"  ❌{'⭐' if p.get('perfect') else ''} {p['symbol']} [{p['bucket']}] {p['entry_raw']:.2f} -> {p['exit']:.2f} "
                     f"{p['net_pct']:+.2f}% Rs {p['pnl']:+,.0f}")
    lines.append(f"\n🌙 CARRIED TONIGHT as swing: {len(st['carried_today'])} {', '.join(st['carried_today'])}")
    lines.append(f"\n🏁 SWING CLOSED TODAY: {len(sw_closed)}")
    for p in sw_closed:
        lines.append(f"  {'✅' if p['net_pct'] > 0 else '❌'}{'⭐' if p.get('perfect') else ''} {p['symbol']} day {p.get('sessions', 0)} "
                     f"{p['entry_raw']:.2f} -> {p['exit']:.2f} {p['net_pct']:+.2f}% Rs {p['pnl']:+,.0f} ({p['outcome']})")
    open_sw = st["swing"]
    unreal = 0.0
    lines.append(f"\n📂 SWING OPEN: {len(open_sw)}")
    for p in sorted(open_sw, key=lambda q: q.get("carried_on", "")):
        x = bars.get(p["symbol"])
        px = float(x["close"].iloc[-1]) if x is not None and len(x) else p.get("last_px", p["entry_raw"])
        u = p["qty"] * (px - p["entry_raw"])
        unreal += u
        lines.append(f"  {'⭐' if p.get('perfect') else '·'} {p['symbol']} [{p['bucket']}] day {p.get('sessions', 0)}/{S.SWING_DAYS} | "
                     f"{p['entry_raw']:.2f} -> {px:.2f} ({(px / p['entry_raw'] - 1) * 100:+.2f}%) | stop {p['stop']:.2f} | Rs {u:+,.0f}")
    realised = sum(p["pnl"] for p in st["closed_today"])
    lines.append(f"\nToday realised: Rs {realised:+,.0f} | open swing (unrealised, before sell charges): Rs {unreal:+,.0f}")
    f = _p("trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        for kind in ("INTRADAY", "SWING"):
            k = t[t["kind"] == kind]
            if len(k):
                lines.append(f"ALL-TIME {kind}: {len(k)} trades, Rs {k['pnl'].sum():+,.0f}, win {(k['net_pct'] > 0).mean() * 100:.0f}%, "
                             f"avg {k['net_pct'].mean():+.2f}%")
        lines.append(f"ALL-TIME total: {len(t)} closed trades, Rs {t['pnl'].sum():+,.0f}")
    notify("\n".join(lines), telegram)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    changed = False
    try:
        changed = run(not a.no_telegram)
    finally:
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as fh:
                fh.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    main()
