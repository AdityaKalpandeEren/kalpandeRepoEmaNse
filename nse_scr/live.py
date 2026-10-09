"""
NSE SCR (Volume Shockers & Runners) - live paper trading, one step per run.

    python -m nse_scr.live                 # what the NSE Alert Scan `scr` job runs
    python -m nse_scr.live --no-telegram   # console only

Each run (~every 2 min, 09:20-15:30 IST):
  1. DYNAMIC WATCHLIST from NSE's live screeners: volume gainers (shockers),
     top gainers (all securities), most active by value and by volume - plus
     open positions. Tagged LARGE (NIFTY 100) / MID (MIDCAP 150) / SMALL.
  2. ⚡📡 SHOCKER WATCH: names trading >= SHOCK_X x their 1-week average
     volume with turnover >= Rs 5 cr - one alert per name per day, batched
     at most every 30 min (watch only).
  3. Today's 5-min candles (Upstox) for the list; every NEW setup on a
     closed candle (nse_scr/strategy.py) -> paper BUY at the next candle's
     open (+10 bps), structural stop (max STOP_PCT), breakeven at +1R then
     trail under candle lows, square-off 15:15. Max V_MAX trades a day.
     SHORTS (intraday sell, EQ series only): a failed spike - a shocker up
     >= 8% at its high closing back below VWAP (short_setups, mode "fade")
     -> paper SELL at the next open (-10 bps), stop above, cover by 15:15.
     Max SHORT_MAX shorts a day (separate from the longs).
  4. Exits re-simulated each run with the SAME function as research ->
     Telegram exit alert; day report after 15:20.
Alerts go to every receiver (personal chat + group) via research_live.
Paper only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nse_scr import data as D
from nse_scr import strategy as S

IST = ZoneInfo("Asia/Kolkata")
STATE_DIR = os.environ.get("NSE_SCR_STATE_DIR", os.path.join(D.ROOT, "live_state", "nse_scr"))
V_MAX = int(os.environ.get("NSE_SCR_MAX_TRADES_PER_DAY", "10"))
MAX_WATCH = 45                                                          # per side (longs / shorts)
SHORTS_ON = os.environ.get("NSE_SCR_SHORTS", "1") == "1"
SHORT_MAX = int(os.environ.get("NSE_SCR_MAX_SHORTS_PER_DAY", "10"))
SHORT_HEADER = "🔻🔻 NSE SHOCKER SHORT (NSE SCR · intraday sell) 🔻🔻"
SHORT_RISK_LINE = ("🧪 UNPROVEN - backtest: fading failed spikes ~break-even (-0.11% / +0.02% per trade, two halves). "
                   "Paper only. Intraday short (MIS): must be covered the same day.")
SHOCK_X = 5.0
HEADER = "⚡⚡ NSE VOLUME-SHOCKER RUNNER (NSE SCR) ⚡⚡"
RISK_LINE = ("🧪 UNPROVEN - backtest: buying shockers lost -0.4%/trade (both halves). "
             "⚠️ Shockers reverse fast and hit circuit limits. Paper only.")


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


def _log_trade(row):
    f = _p("trades.csv")
    if os.path.exists(f):
        old = pd.read_csv(f)
        if "side" not in old.columns:                                 # before shorts: every trade was a long
            old["side"] = "LONG"
            old.to_csv(f, index=False)
        row = {c: row.get(c) for c in list(old.columns) + [k for k in row if k not in old.columns]}
    new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _cr(x):
    return f"Rs {x / 1e7:,.0f} cr" if x and x == x else "?"


def daily_ctx(syms, st) -> dict:
    """prev close + 20-day average daily volume per symbol (Upstox daily candles, once a day)."""
    from backtest.candle_cache import load_daily_cached
    from backtest.ml.common import access_token
    cache = st.setdefault("daily", {})
    need = [s for s in syms if s not in cache]
    if need:
        tok = access_token()
        today = datetime.now(IST).date()
        frm = (today - timedelta(days=60)).strftime("%Y-%m-%d")
        prev = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        for s in need:
            try:
                d = load_daily_cached(s, frm, prev, tok)
                if len(d) >= 10:
                    cache[s] = {"prev_close": float(d["close"].iloc[-1]), "avg_vol20": float(d["volume"].tail(20).mean())}
            except Exception:
                cache[s] = None
    return cache


def shocker_watch(st, w, caps, telegram, now):
    last = st.get("shock_sent_at")
    if last and now - datetime.fromisoformat(last) < timedelta(minutes=30):
        return
    sent = set(st.setdefault("shock_sent", []))
    x = w[(w["vol_x_week"] >= SHOCK_X) & (w["turnover"] >= S.MIN_TURNOVER) & ~w["symbol"].isin(sent)]
    if x.empty:
        return
    lines = []
    for _, r in x.head(12).iterrows():
        sent.add(r["symbol"])
        lines.append(f"📡 {r['symbol']} [{caps.get(r['symbol'], 'SMALL')}] {r['pct']:+.1f}% @ Rs {r['ltp']} | volume "
                     f"{r['vol_x_week']:.0f}x 1-week avg | turnover {_cr(r['turnover'])}")
    st["shock_sent"], st["shock_sent_at"] = sorted(sent), now.isoformat()
    notify("⚡📡 NSE VOLUME SHOCKERS (watch only, no trade)\n" + "\n".join(lines), telegram)


def run(telegram: bool) -> bool:
    from live_v33 import today_5m
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
    w = D.watchlist()
    caps = D.cap_buckets()
    if not w.empty and mins <= 15 * 60:
        shocker_watch(st, w, caps, telegram, now)
    held = [p["symbol"] for p in st["positions"] if p["status"] in ("PENDING", "OPEN")]
    cand = w[(w["pct"] >= S.WATCH_PCT * 100) & (w["ltp"] >= S.MIN_PRICE)] if not w.empty else w
    syms = list(cand.get("symbol", []))[:MAX_WATCH]
    eq = D.eq_series() if SHORTS_ON else set()
    if SHORTS_ON and not w.empty:                     # falling shockers + names that spiked (fade candidates)
        sc = w[((w["pct"] <= -S.WATCH_PCT * 100) | (w["pct"] >= 1)) & (w["ltp"] >= S.MIN_PRICE)]
        sc = sc[sc["symbol"].isin(eq) & ~sc["symbol"].isin(syms)] if eq else sc.iloc[0:0]
        syms += list(sc["symbol"])[:MAX_WATCH]
    syms = list(dict.fromkeys(syms + held))
    ctx = daily_ctx(syms, st)
    bars = {}
    for s in syms:
        try:
            x = today_5m(s)
        except Exception:
            continue
        if x is None or x.empty:
            continue
        x = x.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        x = x[x.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]            # closed candles only
        if len(x):
            bars[s] = x
    for s in syms:
        x, c = bars.get(s), ctx.get(s)
        if x is None or not c:
            continue
        f = S.bar_features(x, c["prev_close"], c["avg_vol20"])
        last = st["last_ts"].get(s)
        new = [(ts, "LONG") for ts in S.setups(x, f) if last is None or str(ts) > last]
        if SHORTS_ON and s in eq:
            new += [(ts, "SHORT") for ts in S.short_setups(x, f) if last is None or str(ts) > last]
        st["last_ts"][s] = str(x.index[-1])
        for ts, side in sorted(new):
            if ts < x.index[-1] - pd.Timedelta(minutes=10):
                continue                                                  # stale backlog setup
            if side == "LONG" and st["taken"] >= V_MAX:
                continue
            if side == "SHORT" and st.get("shorts", 0) >= SHORT_MAX:
                continue
            if any(p["symbol"] == s and p["status"] in ("PENDING", "OPEN") for p in st["positions"]):
                continue
            if sum(p["symbol"] == s for p in st["positions"]) >= S.MAX_PER_SYMBOL_DAY:
                continue
            r = f.loc[ts]
            close = float(x.loc[ts, "close"])
            if side == "SHORT":
                open_short(st, s, ts, x, r, close, caps, telegram)
                continue
            stop = S.initial_stop(x, x.index.get_loc(ts), close)
            st["taken"] += 1
            st["positions"].append({"symbol": s, "status": "PENDING", "signal_ts": str(ts), "cap": caps.get(s, "SMALL"),
                                    "pct": round(float(r["pct"]) * 100, 1), "rvol": round(float(r["rvol"]), 1)})
            meta = w.set_index("symbol").loc[s] if s in set(w["symbol"]) else {}
            vx = meta.get("vol_x_week") if hasattr(meta, "get") else None
            notify(f"{HEADER}\n🚀 BUY {s} [{caps.get(s, 'SMALL')}] ~Rs {close:.2f} (next 5-min candle open)\n"
                   f"⚡ {float(r['pct']) * 100:+.1f}% today | volume {float(r['rvol']):.1f}x 20-day avg"
                   f"{f' ({vx:.0f}x 1-week)' if vx == vx and vx else ''} | turnover {_cr(float(r['turnover']))}\n"
                   f"🛑 stop ~Rs {stop:.2f} ({(stop / close - 1) * 100:+.1f}%) | trail after +1R | square-off 15:15\n"
                   f"trade {st['taken']}/{V_MAX} today | {RISK_LINE}", telegram)
    manage(st, bars, telegram)
    if mins >= 15 * 60 + 20 and not st.get("reported"):
        day_report(st, telegram)
        st["reported"] = True
    save_state(st)
    return True


def open_short(st, s, ts, x, r, close, caps, telegram):
    stop = S.initial_stop_short(x, x.index.get_loc(ts), close)
    st["shorts"] = st.get("shorts", 0) + 1
    st["positions"].append({"symbol": s, "side": "SHORT", "status": "PENDING", "signal_ts": str(ts),
                            "cap": caps.get(s, "SMALL"), "pct": round(float(r["pct"]) * 100, 1),
                            "rvol": round(float(r["rvol"]), 1)})
    why = (f"spiked {float(r['hod_pct']) * 100:+.1f}% and lost VWAP (failed spike)"
           if S.SHORT_MODE == "fade" else "new low of day on heavy volume")
    notify(f"{SHORT_HEADER}\n🔻 SHORT SELL {s} [{caps.get(s, 'SMALL')}] ~Rs {close:.2f} (next 5-min candle open)\n"
           f"⚡ {float(r['pct']) * 100:+.1f}% today, {why} | volume {float(r['rvol']):.1f}x 20-day avg | "
           f"turnover {_cr(float(r['turnover']))}\n"
           f"🛑 stop (buy back) ~Rs {stop:.2f} ({(stop / close - 1) * 100:+.1f}%) | trail after +1R | cover by 15:15\n"
           f"short {st['shorts']}/{SHORT_MAX} today | {SHORT_RISK_LINE}", telegram)


def manage(st, bars, telegram):
    for p in st["positions"]:
        if p["status"] not in ("PENDING", "OPEN"):
            continue
        x = bars.get(p["symbol"])
        sig = pd.Timestamp(p["signal_ts"])
        if x is None or sig not in x.index:
            continue
        short = p.get("side") == "SHORT"
        tr = (S.simulate_short if short else S.simulate)(p["symbol"], x, sig, final=False)
        if tr is None:
            if x.index[-1] > sig + pd.Timedelta(minutes=5):
                p["status"] = "NOFILL"                         # next candle was circuit-locked / past square-off
            continue
        if p["status"] == "PENDING":
            p.update(status="OPEN", entry=round(tr.entry, 2), entry_ts=str(tr.entry_ts),
                     stop_initial=round((S.initial_stop_short if short else S.initial_stop)(x, x.index.get_loc(sig), tr.entry), 2))
            p["qty"] = max(int(S.NOTIONAL // tr.entry), 1)
        if tr.outcome == "OPEN":
            p["stop_now"] = round(tr.stop0, 2)
            continue
        pnl = p["qty"] * tr.entry * tr.ret_pct / 100
        p.update(status="CLOSED", exit=round(tr.exit, 2), exit_ts=str(tr.exit_ts), outcome=tr.outcome,
                 R=round(tr.R, 2), ret_pct=round(tr.ret_pct, 2), pnl=round(pnl, 0))
        _log_trade({"date": st["date"], "side": p.get("side", "LONG"), **{k: p.get(k) for k in ("symbol", "cap", "signal_ts", "entry_ts", "entry",
                                                                  "stop_initial", "exit_ts", "exit", "outcome", "R",
                                                                  "ret_pct", "pnl", "pct", "rvol", "qty")}})
        mins = int((tr.exit_ts - tr.entry_ts).total_seconds() // 60)
        notify(f"{'🔻 NSE SCR SHORT COVER' if short else '⚡ NSE SCR EXIT'} {'✅' if tr.R > 0 else '❌'} {p['symbol']} Rs {tr.entry:.2f} -> Rs {tr.exit:.2f} "
               f"({tr.ret_pct:+.2f}% net, {tr.R:+.2f}R) {tr.outcome} after {mins} min | paper P&L Rs {pnl:+,.0f}", telegram)


def day_report(st, telegram):
    ps = [p for p in st["positions"] if p["status"] == "CLOSED"]
    lines = [f"⚡📊 NSE SCR VOLUME-SHOCKER RUNNERS - PAPER RESULT {st['date']}"]
    if not ps:
        lines.append("No runner trade today.")
    for p in ps:
        lines.append(f"{'✅' if p['R'] > 0 else '❌'} {'🔻SHORT ' if p.get('side') == 'SHORT' else ''}{p['symbol']} [{p['cap']}]: {p['entry']:.2f} -> {p['exit']:.2f} "
                     f"({p['ret_pct']:+.2f}%, {p['R']:+.2f}R, {p['outcome']}) Rs {p['pnl']:+,.0f}")
    if ps:
        lines.append(f"Day: Rs {sum(p['pnl'] for p in ps):+,.0f} on Rs {S.NOTIONAL:,.0f}/trade (net of charges)")
    f = _p("trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        lines.append(f"ALL-TIME ({t['date'].nunique()} days, {len(t)} trades): Rs {t['pnl'].sum():+,.0f} | "
                     f"win {(t['R'] > 0).mean() * 100:.0f}% | avg {t['ret_pct'].mean():+.2f}%/trade")
        if "side" in t.columns:
            for side, g in t.groupby(t["side"].fillna("LONG")):
                lines.append(f"  {side}: {len(g)} trades, Rs {g['pnl'].sum():+,.0f}, avg {g['ret_pct'].mean():+.2f}%/trade")
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
