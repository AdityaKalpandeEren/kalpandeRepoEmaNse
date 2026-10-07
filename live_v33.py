"""
L_ML_V33 paper-trading runner (NSE) - separate from scan_once.py and the
production / research alerts.

    python live_v33.py                 # one pass: does whatever is due right now
    python live_v33.py --loop          # keep running (every 60 s) until the day is done
    python live_v33.py --report        # print the all-time paper summary
    python live_v33.py --no-telegram   # console only
    python live_v33.py --warm          # fill the data caches (no pick, no state change)

Daily cycle (IST):
  from 9:45   PICK: score every stock (watchlist + universe + NIFTY 50) with
              the V3.3 model from the 9:15-9:40 candles, yesterday's NSE
              delivery / F&O OI-PCR, events public before now, overnight
              macro, sector and market context. Paper-BUY the top K (from
              the model bundle, or config.V33_TOP_K) at the 9:45 candle's
              open, config.V33_NOTIONAL_PER_STOCK each (whole shares).
              Telegram: the picks. A pick made after 9:55 (runner started
              late) fills at the last closed candle's close and is flagged.
  every run   STOP check: 2% below the fill, on the candles since entry
              (a candle opening below the stop fills at that open).
  from 15:15  CLOSE: anything still open exits at the 15:10 candle's close
              (the MIS square-off). Exact charges (strategy/v33_costs.py:
              Rs config.V33_BROKERAGE_PER_ORDER/order + GST, STT, exchange,
              SEBI, stamp, config.V33_SLIPPAGE_BPS slippage). Telegram: the
              day's P&L per stock, total, and all-time stats.

State: config.V33_STATE_DIR/state.json (today) and trades.csv (all closed
trades). Every step is idempotent - running twice never double-buys.
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import config

IST = ZoneInfo("Asia/Kolkata")
TRADE_FIELDS = ["date", "rank", "symbol", "score", "qty", "entry_time", "entry", "price_at_alert", "alert_drift_pct",
                "stop", "exit_time", "exit", "outcome", "gross", "charges", "net", "net_pct", "late_fill"]


def _path(name):
    os.makedirs(config.V33_STATE_DIR, exist_ok=True)
    return os.path.join(config.V33_STATE_DIR, name)


def load_state():
    try:
        with open(_path("state.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    tmp = _path("state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, default=str)
    os.replace(tmp, _path("state.json"))


def notify(text, telegram):
    print(text)
    if telegram:
        from paper_trading.research_live import send_text
        send_text(text)


def universe():
    from backtest.ml.common import load_symbol_list
    from data.instruments import get_instrument_key
    syms = list(dict.fromkeys(load_symbol_list("watchlist.txt") + load_symbol_list("universe.txt")
                              + list(config.ML_V2_BREADTH_UNIVERSE)))
    ok = []
    for s in syms:
        try:
            get_instrument_key(s)
            ok.append(s)
        except ValueError:
            pass
    return ok


_c5 = {}
_http = None


def today_5m(key_or_symbol):
    """Today's 5-min candles (incl. the still-forming one) from the Upstox
    intraday API, over one reused HTTP session, cached 50 s."""
    global _http
    import requests
    from data.instruments import get_instrument_key
    hit = _c5.get(key_or_symbol)
    if hit and time.time() - hit[0] < 50:
        return hit[1]
    if _http is None:
        _http = requests.Session()
    key = key_or_symbol if "|" in key_or_symbol else get_instrument_key(key_or_symbol)
    r = _http.get(f"https://api.upstox.com/v3/historical-candle/intraday/{key}/minutes/5",
                  headers={"Accept": "application/json", "Authorization": f"Bearer {config.UPSTOX_ACCESS_TOKEN}"},
                  timeout=20)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST)
        df = df.sort_values("timestamp").reset_index(drop=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c])
    _c5[key_or_symbol] = (time.time(), df)
    return df


def prefetch_today(symbols, workers: int = 6) -> None:
    """Fetch today's 5-min candles for every symbol in parallel into the
    today_5m cache (one HTTP session per thread) - was ~90 s sequential."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from data.instruments import get_instrument_key

    def one(sym):
        try:
            key = get_instrument_key(sym)
            r = requests.get(f"https://api.upstox.com/v3/historical-candle/intraday/{key}/minutes/5",
                             headers={"Accept": "application/json", "Authorization": f"Bearer {config.UPSTOX_ACCESS_TOKEN}"},
                             timeout=20)
            r.raise_for_status()
            df = pd.DataFrame(r.json().get("data", {}).get("candles", []),
                              columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST)
                df = df.sort_values("timestamp").reset_index(drop=True)
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = pd.to_numeric(df[c])
            _c5[sym] = (time.time(), df)
        except Exception:
            pass                                             # today_5m() fetches it again on demand
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, symbols))


def pick_with_sector_cap(ranked: pd.DataFrame, k: int, cap: int) -> pd.DataFrame:
    """Top k by score, at most `cap` from one sector (OTHER uncapped; cap 0 = off)."""
    if not cap or "sector" not in ranked:
        return ranked.head(k)
    keep, count = [], {}
    for idx, sec in ranked["sector"].items():
        if sec != "OTHER" and count.get(sec, 0) >= cap:
            continue
        keep.append(idx)
        count[sec] = count.get(sec, 0) + 1
        if len(keep) == k:
            break
    return ranked.loc[keep]


def closed(df, now):
    return df[df["timestamp"] + timedelta(minutes=5) <= now].reset_index(drop=True) if not df.empty else df


# ═══════════════════════════════════════════════════════════════════

def do_pick(st, now, telegram):
    import joblib
    from strategy import v33_live
    bundle = joblib.load(config.ML_V33_MODEL_PATH)
    k = config.V33_TOP_K or int(bundle.get("k", 5))
    print(f"[V3.3] scoring the universe ({now:%H:%M} IST) ...", flush=True)
    t0 = time.time()
    # Raw candles (incl. the forming 9:45 one): the opening features use only
    # the six closed 9:15-9:40 candles; the 9:45 candle supplies the fill
    # price, exactly as in training.
    syms = universe()
    t_pre = time.time()
    prefetch_today(syms)
    print(f"[V3.3] today's candles for {len(syms)} symbols in {time.time() - t_pre:.0f} s (parallel)", flush=True)
    rows = v33_live.build_rows(now.date(), syms, today_5m, verbose=False)
    print(f"[V3.3] features for {len(rows)}/{len(syms)} symbols in {time.time() - t0:.0f} s "
          f"(phases: {v33_live.LAST_TIMINGS})", flush=True)
    if rows.empty:
        print("[V3.3] no rows yet (opening candles missing?) - will retry next run")
        return
    ranked = v33_live.score(rows, bundle)
    done_at = datetime.now(IST)
    if done_at.hour * 60 + done_at.minute >= 14 * 60 + 30:
        # A pick this late would be entered and squared off minutes apart.
        st.update({"date": str(now.date()), "closed": True, "skipped": "pick finished after 14:30"})
        save_state(st)
        notify(f"V3.3 PAPER {now.date()}: pick finished too late ({done_at:%H:%M} IST) - no trades today.", telegram)
        return
    late = done_at.hour * 60 + done_at.minute >= 9 * 60 + 55
    top = pick_with_sector_cap(ranked, k, config.V33_MAX_PER_SECTOR)
    skipped = [s for s in ranked.head(len(top) + 10)["symbol"] if s not in set(top["symbol"])
               and ranked.set_index("symbol").loc[s, "score"] > top["score"].min()]
    if skipped:
        print(f"[V3.3] sector cap {config.V33_MAX_PER_SECTOR}: skipped {skipped}", flush=True)
    if not late and top["entry"].isna().any():
        print("[V3.3] the 9:45 candle hasn't printed yet for every pick - retrying next run")
        return
    picks = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        # What you could realistically buy at when the alert arrives - the
        # latest traded price, incl. the forming candle (fetched fresh, not
        # from the scoring pass). On time, the paper fill stays the 9:45 open
        # (what the model was trained on) and the gap is the entry drift to
        # watch. A LATE pick fills at this same live price - never at a
        # just-closed candle that Upstox may still be revising.
        _c5.pop(r["symbol"], None)
        raw = today_5m(r["symbol"])
        if late and raw.empty:
            continue
        at_alert = float(raw["close"].iat[-1]) if len(raw) else float(r["entry"])
        entry = float(r["entry"]) if not late else at_alert
        qty = int(config.V33_NOTIONAL_PER_STOCK // entry)
        picks.append({"rank": rank, "symbol": r["symbol"], "score": round(float(r["score"]), 5), "qty": qty,
                      "entry": round(entry, 2), "price_at_alert": round(at_alert, 2),
                      "alert_drift_pct": round((at_alert / entry - 1) * 100, 3),
                      "stop": round(entry * 0.98, 2), "late_fill": late,
                      "entry_time": (done_at if late else now.replace(hour=9, minute=45, second=0, microsecond=0)).isoformat(),
                      "status": "OPEN"})
    st.update({"date": str(now.date()), "picked": True, "picks": picks, "universe_scored": len(ranked),
               "model_trained_through": bundle.get("trained_through")})
    save_state(st)
    lines = [f"🧠 V3.3 PAPER PICKS {now.date()} (top {k} of {len(ranked)} scored, "
             f"Rs {config.V33_NOTIONAL_PER_STOCK:,.0f}/stock){' - LATE FILL' if late else ''}"]
    for p in picks:
        lines.append(f"{p['rank']}. {p['symbol']}  buy {p['qty']} @ {p['entry']} (now {p['price_at_alert']}, "
                     f"{p['alert_drift_pct']:+.2f}%)  stop {p['stop']}  (score {p['score']:+.4f})")
    if skipped:
        lines.append(f"(sector cap {config.V33_MAX_PER_SECTOR}/sector: skipped {', '.join(skipped)})")
    lines.append("Exit 15:15 IST or 2% stop. Paper only.")
    notify("\n".join(lines), telegram)


def check_stops(st, now, telegram):
    msgs = []
    for p in st.get("picks", []):
        if p["status"] != "OPEN":
            continue
        df = closed(today_5m(p["symbol"]), now)
        after = df[df["timestamp"] >= pd.Timestamp(p["entry_time"])]
        for k, (_, bar) in enumerate(after.iterrows()):
            if bar["low"] <= p["stop"]:
                px = float(bar["open"]) if (k > 0 and bar["open"] < p["stop"]) else p["stop"]
                _close(p, px, bar["timestamp"], "STOP")
                msgs.append(f"🛑 {p['symbol']} stopped @ {px:.2f} ({p['net_pct']*100:+.2f}% net)")
                break
    if msgs:
        save_state(st)
        notify("V3.3 PAPER\n" + "\n".join(msgs), telegram)


def _close(p, exit_px, when, outcome):
    from strategy.v33_costs import round_trip
    rt = round_trip(p["entry"], exit_px, p["qty"])
    p.update(status="CLOSED", exit=round(exit_px, 2), exit_time=str(when), outcome=outcome,
             gross=round(rt["gross"], 2), charges=round(rt["charges"], 2), net=round(rt["net"], 2),
             net_pct=rt["net_pct"])


def do_close(st, now, telegram):
    for p in st.get("picks", []):
        if p["status"] != "OPEN":
            continue
        df = today_5m(p["symbol"])
        bar = df[(df["timestamp"].dt.hour == 15) & (df["timestamp"].dt.minute == 10)]
        last = bar.iloc[-1] if len(bar) else closed(df, now).iloc[-1]
        _close(p, float(last["close"]), last["timestamp"], "SQUARE_OFF")
    new = not os.path.exists(_path("trades.csv"))
    with open(_path("trades.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for p in st.get("picks", []):
            w.writerow({"date": st["date"], **p})
    st["closed"] = True
    save_state(st)
    day_net = sum(p["net"] for p in st["picks"])
    late_day = any(p.get("late_fill") for p in st["picks"])
    lines = [f"📊 V3.3 PAPER RESULT {st['date']}"
             + (" - LATE-FILL TEST DAY (not the 9:45 entry; excluded from all-time stats)" if late_day else "")]
    for p in st["picks"]:
        mark = "✅" if p["net"] > 0 else "❌"
        lines.append(f"{mark} {p['symbol']}: {p['entry']} -> {p['exit']} ({p['outcome']})  "
                     f"net Rs {p['net']:+,.0f} ({p['net_pct']*100:+.2f}%), charges Rs {p['charges']:,.0f}")
    lines.append(f"Day: Rs {day_net:+,.0f} on Rs {config.V33_NOTIONAL_PER_STOCK * len(st['picks']):,.0f} deployed")
    lines.append(summary_text())
    notify("\n".join(lines), telegram)


def summary_text() -> str:
    path = _path("trades.csv")
    if not os.path.exists(path):
        return "No closed V3.3 paper trades yet."
    t = pd.read_csv(path)
    late = t["late_fill"].astype(str).str.lower().eq("true") if "late_fill" in t else pd.Series(False, index=t.index)
    n_late = int(late.sum())
    t = t[~late]
    if t.empty:
        return f"No on-time V3.3 paper trades yet ({n_late} late-fill test trades excluded)."
    day = t.groupby("date").agg(net=("net", "sum"), pct=("net_pct", "mean"))
    tt = (day["pct"].mean() / (day["pct"].std(ddof=1) / math.sqrt(len(day)))) if len(day) > 2 else float("nan")
    return (f"ALL-TIME ({len(day)} days, {len(t)} trades): net Rs {t['net'].sum():+,.0f} | "
            f"win {100*(t['net'] > 0).mean():.0f}% of trades, {100*(day['net'] > 0).mean():.0f}% of days | "
            f"avg {day['pct'].mean()*100:+.3f}%/day | charges Rs {t['charges'].sum():,.0f} | t {tt:.2f}"
            + (f" | avg entry drift at alert {t['alert_drift_pct'].mean():+.3f}%" if "alert_drift_pct" in t else "")
            + ("  (need ~60+ days before judging)" if len(day) < 60 else "")
            + (f" | {n_late} late-fill test trades excluded" if n_late else ""))


def run_once(telegram=True) -> bool:
    """Returns True when today's cycle is complete (or it's not a trading day)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        print("Weekend - nothing to do.")
        return True
    st = load_state()
    if st.get("date") != str(now.date()):
        st = {"date": str(now.date())}
    mins = now.hour * 60 + now.minute
    if st.get("closed"):
        return True
    if not st.get("picked"):
        if mins < 9 * 60 + 45:
            if mins >= 9 * 60 + 20 and st.get("prewarmed") != str(now.date()):
                # Pre-warm: fetch everything up to yesterday now, so the 9:45
                # pick only needs today's candles (it took ~8 min on 2026-10-07
                # and filled late).
                from strategy import v33_live
                info = v33_live.prewarm(now.date(), universe())
                print(f"[V3.3] pre-warmed history for the 9:45 pick: {info}", flush=True)
                st["prewarmed"] = str(now.date())
                save_state(st)
            else:
                print(f"{now:%H:%M} - waiting for 9:45 IST.")
            return False
        if mins >= 14 * 60 + 30:
            print("Too late to pick today (after 14:30) - skipping the day.")
            return True
        do_pick(st, now, telegram)
        return False
    if mins >= 15 * 60 + 15:
        do_close(st, now, telegram)
        return True
    check_stops(st, now, telegram)
    return False


def warm():
    """Run the 9:45 feature build once without picking, so every history /
    NSE-archive cache the pick needs is on disk. On GitHub Actions this
    seeds the cached backtest/cache before the first live 9:45 pick,
    which would otherwise spend its first minutes downloading years of
    daily candles."""
    from strategy import v33_live
    now = datetime.now(IST)
    syms = universe()
    t0 = time.time()
    rows = v33_live.build_rows(now.date(), syms, today_5m, verbose=False)
    print(f"[V3.3] warm: {len(rows)}/{len(syms)} symbols built in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", action="store_true", help="run every 60 s until today's cycle is done")
    ap.add_argument("--report", action="store_true", help="print the all-time paper summary")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--warm", action="store_true", help="fill the data caches only (no pick, no state change)")
    args = ap.parse_args()
    if args.report:
        print(summary_text())
        return
    if not config.UPSTOX_ACCESS_TOKEN:
        raise SystemExit("UPSTOX_ACCESS_TOKEN not set")
    if args.warm:
        warm()
        return
    tg = not args.no_telegram
    before = _state_bytes()
    while True:
        try:
            done = run_once(tg)
        except Exception as e:
            print(f"[V3.3] error: {e!r}")
            done = False
        if done or not args.loop:
            break
        time.sleep(60)
    changed = _state_bytes() != before
    out = os.environ.get("GITHUB_OUTPUT")
    if out:                       # lets the workflow save its cache only when something happened
        with open(out, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")


def _state_bytes() -> bytes:
    try:
        with open(_path("state.json"), "rb") as f:
            return f.read()
    except OSError:
        return b""


if __name__ == "__main__":
    main()
