"""
NSE LCR (Large-Cap Runners) - live paper trading, one step per run
(09:20-15:30 IST, same cron-job.org trigger).

    python -m nse_lcr.live                 # what the NSE Alert Scan `lcr` job runs
    python -m nse_lcr.live --no-telegram

Each run:
  1. Universe = NIFTY 50 (MEGA) + rest of NIFTY 100 (LARGE) + NIFTY Midcap
     150 (MID) from NSE's constituent CSVs (cached daily). Today's 5-min
     candles for all ~250 are fetched in parallel (as V3.3's pick) and the
     DYNAMIC list = the stocks having a volume shock right now (time-adjusted
     volume >= RVOL_MIN, up >= WATCH_PCT) + open positions.
  2. 🏛️📡 VOLUME WATCH: time-adjusted volume >= 3x while the price is within
     +-1.5% (unusual volume, no move yet) - batched at most every 30 min.
  3. Every NEW setup on a closed candle (nse_lcr/strategy.py: pullback
     continuation above the 10 & 20-day EMAs) -> paper BUY at the next
     candle's open; ⭐ PERFECT when above all six daily EMAs; exits
     re-simulated each run with the SAME function as research; day report
     after 15:20. Alerts go to every Telegram receiver (personal + group).
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


def universe() -> dict:
    """symbol -> MEGA / LARGE / MID, cached for the day."""
    f = os.path.join(ROOT, "nse_lcr", "cache", f"universe_{datetime.now(IST):%Y%m%d}.json")
    if os.path.exists(f):
        return json.load(open(f))
    import requests
    caps = {}
    for name, tag in (("ind_nifty50list.csv", "MEGA"), ("ind_nifty100list.csv", "LARGE"), ("ind_niftymidcap150list.csv", "MID")):
        try:
            r = requests.get(f"https://archives.nseindia.com/content/indices/{name}", headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            for s in pd.read_csv(io.StringIO(r.text))["Symbol"]:
                caps.setdefault(str(s).strip(), tag)
        except Exception:
            continue
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


def run(telegram: bool) -> bool:
    from live_v33 import _c5, prefetch_today
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    st = load_state()
    today = str(now.date())
    if st.get("date") != today:
        st = {"date": today, "positions": [], "last_ts": {}, "taken": 0}
    if mins < 9 * 60 + 20 or st.get("reported"):
        return False
    caps = universe()
    syms = sorted(caps)
    ctx = daily_ctx(syms, st)
    prefetch_today(syms)
    bars, feats, watch_lines = {}, {}, []
    for s in syms:
        hit, c = _c5.get(s), ctx.get(s)
        if not hit or c is None or hit[1] is None or hit[1].empty:
            continue
        x = hit[1].set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        x = x[x.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]
        if len(x) < 3:
            continue
        f = S.bar_features(x, c["prev_close"], c["avg_vol20"])
        bars[s], feats[s] = x, f
        r = f.iloc[-1]
        if r["rvol"] >= WATCH_X and abs(r["pct"]) <= 0.015 and s not in st.setdefault("watch_sent", []):
            watch_lines.append((s, r))
    last_w = st.get("watch_at")
    if watch_lines and mins <= 15 * 60 and (not last_w or now - datetime.fromisoformat(last_w) >= timedelta(minutes=30)):
        lines = []
        for s, r in sorted(watch_lines, key=lambda t: -t[1]["rvol"])[:10]:
            st["watch_sent"].append(s)
            lines.append(f"📡 {s} [{caps[s]}] {r['pct'] * 100:+.1f}% | volume {r['rvol']:.1f}x normal for this time | "
                         f"turnover Rs {r['turnover'] / 1e7:,.0f} cr")
        st["watch_at"] = now.isoformat()
        notify("🏛️📡 NSE LCR VOLUME WATCH - large / mid caps with unusual volume, no move yet (watch only)\n" + "\n".join(lines), telegram)
    held = {p["symbol"] for p in st["positions"] if p["status"] in ("PENDING", "OPEN")}
    for s, x in bars.items():
        f, c = feats[s], ctx[s]
        last = st["last_ts"].get(s)
        new = [ts for ts in S.setups(x, f, c["emas"]) if last is None or str(ts) > last]
        st["last_ts"][s] = str(x.index[-1])
        for ts in new:
            if st["taken"] >= V_MAX or ts < x.index[-1] - pd.Timedelta(minutes=10) or s in held:
                continue
            if sum(p["symbol"] == s for p in st["positions"]) >= S.MAX_PER_SYMBOL_DAY:
                continue
            r, close = f.loc[ts], float(x.loc[ts, "close"])
            stop = S.initial_stop(x, x.index.get_loc(ts), close)
            perfect = S.is_perfect(close, c["emas"])
            st["taken"] += 1
            held.add(s)
            st["positions"].append({"symbol": s, "status": "PENDING", "signal_ts": str(ts), "bucket": caps[s],
                                    "pct": round(float(r["pct"]) * 100, 2), "rvol": round(float(r["rvol"]), 1), "perfect": perfect})
            tag = ("⭐ PERFECT TRADE - above ALL daily EMAs (10/20/30/40/60/180)" if perfect
                   else "✳️ minimum trend - above the 10 & 20-day EMAs, not all longer ones")
            notify(f"{HEADER}\n{tag}\n🚀 BUY {s} [{caps[s]}] ~Rs {close:.2f} (next 5-min candle open)\n"
                   f"⚡ {float(r['pct']) * 100:+.1f}% today | volume {float(r['rvol']):.1f}x normal for this time | "
                   f"turnover Rs {float(r['turnover']) / 1e7:,.0f} cr\n📈 daily EMAs: {S.ema_tags(close, c['emas'])}\n"
                   f"🛑 stop ~Rs {stop:.2f} ({(stop / close - 1) * 100:+.1f}%) | trail after +1R | square-off 15:15\n"
                   f"trade {st['taken']}/{V_MAX} today | 🧪 UNPROVEN - backtest -0.1%/trade on NSE. Paper only.", telegram)
    manage(st, bars, telegram)
    if mins >= 15 * 60 + 20 and not st.get("reported"):
        day_report(st, telegram)
        st["reported"] = True
    st.pop("daily_full", None)
    save_state(st)
    return True


def manage(st, bars, telegram):
    for p in st["positions"]:
        if p["status"] not in ("PENDING", "OPEN"):
            continue
        x = bars.get(p["symbol"])
        sig = pd.Timestamp(p["signal_ts"])
        if x is None or sig not in x.index:
            continue
        tr = S.simulate(p["symbol"], x, sig, final=False)
        if tr is None:
            if x.index[-1] > sig + pd.Timedelta(minutes=5):
                p["status"] = "NOFILL"
            continue
        if p["status"] == "PENDING":
            p.update(status="OPEN", entry=round(tr.entry, 2), entry_ts=str(tr.entry_ts),
                     stop_initial=round(S.initial_stop(x, x.index.get_loc(sig), tr.entry), 2),
                     qty=max(int(S.NOTIONAL // tr.entry), 1))
        if tr.outcome == "OPEN":
            p["stop_now"] = round(tr.stop0, 2)
            continue
        pnl = p["qty"] * tr.entry * tr.ret_pct / 100
        p.update(status="CLOSED", exit=round(tr.exit, 2), exit_ts=str(tr.exit_ts), outcome=tr.outcome,
                 R=round(tr.R, 2), ret_pct=round(tr.ret_pct, 2), pnl=round(pnl, 0))
        f = _p("trades.csv")
        new = not os.path.exists(f)
        with open(f, "a", newline="") as fh:
            row = {"date": st["date"], **{k: p.get(k) for k in ("symbol", "bucket", "signal_ts", "entry_ts", "entry",
                                                                  "stop_initial", "exit_ts", "exit", "outcome", "R",
                                                                  "ret_pct", "pnl", "pct", "rvol", "qty", "perfect")}}
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        mins = int((tr.exit_ts - tr.entry_ts).total_seconds() // 60)
        notify(f"🏛️ NSE LCR EXIT {'✅' if tr.R > 0 else '❌'}{'⭐' if p.get('perfect') else ''} {p['symbol']} "
               f"Rs {tr.entry:.2f} -> Rs {tr.exit:.2f} ({tr.ret_pct:+.2f}% net, {tr.R:+.2f}R) {tr.outcome} after {mins} min | "
               f"paper P&L Rs {pnl:+,.0f}", telegram)


def day_report(st, telegram):
    ps = [p for p in st["positions"] if p["status"] == "CLOSED"]
    lines = [f"🏛️📊 NSE LCR LARGE-CAP RUNNERS - PAPER RESULT {st['date']}"]
    if not ps:
        lines.append("No large-cap runner trade today.")
    for p in ps:
        lines.append(f"{'✅' if p['R'] > 0 else '❌'}{'⭐' if p.get('perfect') else ''} {p['symbol']} [{p['bucket']}]: "
                     f"{p['entry']:.2f} -> {p['exit']:.2f} ({p['ret_pct']:+.2f}%, {p['R']:+.2f}R, {p['outcome']}) Rs {p['pnl']:+,.0f}")
    if ps:
        lines.append(f"Day: Rs {sum(p['pnl'] for p in ps):+,.0f} on Rs {S.NOTIONAL:,.0f}/trade (net of charges)")
    f = _p("trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        lines.append(f"ALL-TIME ({t['date'].nunique()} days, {len(t)} trades): Rs {t['pnl'].sum():+,.0f} | "
                     f"win {(t['R'] > 0).mean() * 100:.0f}% | avg {t['ret_pct'].mean():+.2f}%/trade")
        pf = t[t["perfect"].astype(str).str.lower() == "true"] if "perfect" in t else t.iloc[0:0]
        if len(pf):
            lines.append(f"⭐ PERFECT: {len(pf)} trades, Rs {pf['pnl'].sum():+,.0f} | other: {len(t) - len(pf)} trades, "
                         f"Rs {t['pnl'].sum() - pf['pnl'].sum():+,.0f}")
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
