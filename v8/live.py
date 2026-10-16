"""
V8 TREND - live paper trading (NSE). One step per trading day.

    python -m v8.live                  # what the NSE Alert Scan `v8` job runs
    python -m v8.live --no-telegram    # console only
    python -m v8.live --report         # print the state

The model (research: v8/README.md, `python -m v8.research`, variant ATH504_10ATR):
  trend following after Wilcox & Crittenden (2005) / Zarattini, Pagani & Wilcox
  (2025): buy a stock (or ETF - gold / silver ETFs trend too) whose close is the
  highest of the last 2 years (504 sessions) at the next open; trailing stop
  10 x ATR(42) under the highest close since the buy, never lowered; a close
  under it sells at the next open. No profit target, no time limit.
  universe  every NSE EQ-series instrument with 20-day median traded value
            >= Rs 5 crore and price >= Rs 30 (point in time, same as research)
  sizing    risk 1% of equity to the initial stop, at most 10% of equity per
            name, at most 30 names, no leverage; best relative strength first
            when there are more signals than room
Backtest 2017-01 .. 2025-06 after delivery costs: +21.1%/yr, Sharpe 0.91 (6.5% rf),
max DD -30.7% vs NIFTY +14.6% and live V7 +19.4%; Jul 2025 .. Sep 2026 +20.5%
(gold ETFs; stocks only +9.1%) vs NIFTY -9.3%. Deflated Sharpe 0.68 - NOT proven.

Each trading day, on the first run after 09:20 IST:
  1. refresh NSE end-of-day data (V5's files and code, read-only cache)
  2. at the PREVIOUS close: apply splits / bonuses to holdings, raise each
     holding's trailing stop, find new 2-year-high breakouts
  3. at today's open (Upstox 9:15 candle): sell holdings that closed under their
     stop, then buy the new breakouts, with full delivery costs
  4. Telegram: orders + the book vs NIFTY 50
State: live_state/v8/state.json + trades.csv (V5's columns). Idempotent per day.
Paper only - no orders are sent.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("V8_STATE_DIR", os.path.join(ROOT, "live_state", "v8"))
WINDOW_DAYS = 800                  # V5's window: ~548 sessions >= the 505 a 2-year high needs
TAG = "V8 TREND PAPER"
CAPITAL = 1_000_000
RISK_PCT, MAX_WEIGHT, MAX_POS, ATR_K = 0.01, 0.10, 30, 10.0
MIN_TICKET = 0.005


def _path(name: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def load_state() -> dict:
    try:
        with open(_path("state.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st: dict) -> None:
    tmp = _path("state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, default=str)
    os.replace(tmp, _path("state.json"))


def _log_trade(day, side, sym, qty, px, cash_flow, pnl):
    f = _path("trades.csv")
    new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "side", "symbol", "qty", "price", "cash_flow", "realised_pnl"])
        w.writerow([day, side, sym, qty, round(px, 2), round(cash_flow, 2), round(pnl, 2)])


def _set_output(changed: bool) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")


# ─── signal (identical code path to the research: v8.setups) ─────────────

def signals_previous_close(cfg: dict) -> dict:
    """Everything the day needs, at the latest session before today."""
    from swing import config as cfgmod
    from swing.features.market import index_close
    from swing.research import load_data
    from v5.live import _clean_old
    from v8 import setups
    from v8.research import PARAMS
    data = load_data(cfg, download=False)
    _clean_old(cfgmod.path(cfg, "cache"), data["end"])
    p = data["panel"]
    p = p[~pd.Series(setups.cash_like(p["symbol"]), index=p.index)]   # no money-market / bond ETFs (as research)
    wide = {k: p.pivot(index="date", columns="entity", values=v)
            for k, v in (("o", "adj_open"), ("h", "adj_high"), ("l", "adj_low"), ("c", "adj_close"),
                         ("val", "value"), ("raw", "close"))}
    ind = setups.indicators(wide["o"], wide["h"], wide["l"], wide["c"], wide["val"], wide["raw"], PARAMS)
    last = wide["c"].index[-1]
    day = p[p["date"] == last].set_index("entity")
    to_raw = (wide["raw"].loc[last] / wide["c"].loc[last])               # adjusted -> today's rupees
    info = pd.DataFrame({"symbol": day["symbol"], "isin": day["isin"], "close": day["close"],
                         "atr_raw": ind["atr42"].loc[last] * to_raw, "rs": ind["rs"].loc[last]})
    entries = ind["ath_504"].loc[last]
    entries = list(entries[entries.fillna(False).astype(bool)].index)
    factors = p.assign(f=p["adj_close"] / p["close"])[["date", "entity", "f"]]
    nifty = index_close(data["indices"], "NIFTY 50")
    return {"last": pd.Timestamp(last), "info": info, "entries": entries, "factors": factors, "nifty": nifty,
            "n_liquid": int(ind["liquid"].loc[last].sum())}


def apply_corporate_actions(st: dict, factors: pd.DataFrame, last: pd.Timestamp) -> list[str]:
    """Splits / bonuses since a holding's reference date: shares up, every price
    (entry, last, stop, highest close) down by the same factor."""
    notes = []
    f = factors.set_index(["entity", "date"])["f"]
    for sym, p in st["positions"].items():
        try:
            ratio = float(f.loc[(p["entity"], pd.Timestamp(p["ref_date"]))]) / float(f.loc[(p["entity"], last)])
        except KeyError:
            continue
        if abs(ratio - 1) > 1e-4:
            p["qty"] = int(round(p["qty"] / ratio))
            for k in ("entry_px", "last_px", "stop", "hi"):
                p[k] *= ratio
            notes.append(f"{sym}: corporate action x{1 / ratio:g} - now {p['qty']} shares")
        p["ref_date"] = str(last.date())
    return notes


def _sell(st, sym, qty, px, c, today):
    from swing.backtest import costs as cost_mod
    p = st["positions"][sym]
    fill = cost_mod.fill_price(px, "sell", c)
    gross = qty * fill
    proceeds = gross - cost_mod.order_charges(gross, "sell", c)
    cost_basis = p["cost"] * qty / p["qty"]
    st["cash"] += proceeds
    p["qty"] -= qty
    p["cost"] -= cost_basis
    pnl = proceeds - cost_basis
    _log_trade(today, "SELL", sym, qty, fill, proceeds, pnl)
    if p["qty"] <= 0:
        del st["positions"][sym]
    return fill, pnl, pnl / cost_basis if cost_basis else 0.0


def run_day(telegram: bool) -> bool:
    """Returns True if state changed."""
    from swing import config as cfgmod
    from swing.backtest import costs as cost_mod
    from v5.live import equity_at, holdings_table, market_open_today, movers_table, notify, prepare_data, todays_open

    now = datetime.now(IST)
    today = now.date()
    if now.weekday() >= 5:
        print("Weekend - nothing to do.")
        return False
    st = load_state()
    if st.get("last_day") == str(today):
        print(f"[V8] already done for {today}.")
        return False
    if now.hour * 60 + now.minute < 9 * 60 + 20:
        print(f"{now:%H:%M} - waiting for 09:20 IST (needs today's opening prices).")
        return False
    if not market_open_today():
        if st.get("closed_day") != str(today):
            st.update(closed_day=str(today), last_day=str(today))
            save_state(st)
            notify(f"{TAG} {today}: NSE closed today (no 9:15 prices) - no trades.", telegram)
            return True
        return False

    cfg = cfgmod.load({"data.start": str(today - timedelta(days=WINDOW_DAYS)), "data.end": str(today - timedelta(days=1))})
    complete, warns = prepare_data(cfg, today)
    if not complete:
        return True                      # partial download saved; resume next trigger
    t0 = time.time()
    x = signals_previous_close(cfg)
    last, info = x["last"], x["info"]
    print(f"[V8] {x['n_liquid']} liquid names, {len(x['entries'])} 2-year-high breakouts at the {last.date()} close "
          f"({time.time() - t0:.0f} s)", flush=True)

    if not st or "positions" not in st:
        st = {"start": str(today), "cash": float(CAPITAL), "positions": {}, "nifty_start": None, "start_equity": None}
    c = cfg["costs"]
    lines = apply_corporate_actions(st, x["factors"], last)
    held_ents = {p["entity"] for p in st["positions"].values()}

    # 1) trailing stops at the previous close
    to_sell = []
    for sym, p in st["positions"].items():
        if p["entity"] not in info.index or pd.isna(info.loc[p["entity"], "close"]):
            lines.append(f"⚠️ {sym}: no close on {last.date()} - stop not updated")
            continue
        r = info.loc[p["entity"]]
        p["last_px"] = float(r["close"])
        p["hi"] = max(p["hi"], p["last_px"])
        if pd.notna(r["atr_raw"]):
            p["stop"] = max(p["stop"], p["hi"] - ATR_K * float(r["atr_raw"]))
        if p["last_px"] < p["stop"]:
            to_sell.append(sym)
    prices_prev = {sym: p["last_px"] for sym, p in st["positions"].items()}
    eq_prev = equity_at(st, prices_prev)
    nifty_last = float(x["nifty"].loc[:last].iloc[-1])
    st["nifty_start"] = st["nifty_start"] or nifty_last
    st["start_equity"] = st["start_equity"] or eq_prev

    # 2) sells at today's open
    sells, opens = [], {}
    for sym in to_sell:
        p = st["positions"][sym]
        px = todays_open(sym, p.get("isin"))
        if px is None:
            lines.append(f"⚠️ {sym}: closed under its stop but no opening price - sell postponed")
            continue
        opens[sym] = px
        days = int(np.busday_count(pd.Timestamp(p["since"]).date(), today))
        fill, pnl, ret = _sell(st, sym, p["qty"], px, c, today)
        sells.append(f"{sym} @ {fill:.2f} ({pnl:+,.0f}, {ret * 100:+.1f}%, {days} sessions)")

    # 3) buys at today's open: best relative strength first
    cands = [e for e in x["entries"] if e not in held_ents and e in info.index]
    cands.sort(key=lambda e: -(info.loc[e, "rs"] if pd.notna(info.loc[e, "rs"]) else -1))
    buys, skipped = [], 0
    for e in cands:
        if len(st["positions"]) >= MAX_POS:
            skipped += 1
            continue
        r = info.loc[e]
        sym = r["symbol"]
        if pd.isna(r["atr_raw"]) or pd.isna(r["close"]):
            continue
        px = todays_open(sym, r["isin"])
        if px is None:
            lines.append(f"⚠️ {sym}: 2-year high but no opening price - buy skipped")
            continue
        fill = cost_mod.fill_price(px, "buy", c)
        stop = float(r["close"]) - ATR_K * float(r["atr_raw"])
        d = (fill - stop) / fill
        if not d > 0:
            continue
        money = min(RISK_PCT * eq_prev / d, MAX_WEIGHT * eq_prev, st["cash"] / 1.002)
        qty = int(money // fill)
        if qty <= 0 or qty * fill < MIN_TICKET * eq_prev:
            continue
        gross = qty * fill
        fee = cost_mod.order_charges(gross, "buy", c)
        st["cash"] -= gross + fee
        st["positions"][sym] = {"entity": e, "isin": r["isin"], "qty": qty, "cost": gross + fee, "since": str(today),
                                "entry_px": fill, "last_px": px, "stop": stop, "hi": px, "ref_date": str(last.date())}
        opens[sym] = px
        _log_trade(today, "BUY", sym, qty, fill, -(gross + fee), 0.0)
        buys.append(f"{sym} {qty} @ {fill:.2f} (stop {stop:.2f}, -{d * 100:.0f}%, {gross / eq_prev * 100:.1f}%)")

    prices_now = {sym: opens.get(sym, p["last_px"]) for sym, p in st["positions"].items()}
    eq_now = equity_at(st, prices_now)
    head = f"📈 {TAG} {today} (signals: {last.date()} close; fills at today's open incl. delivery costs)"
    body = ([f"🟢 BUY ({len(buys)}) - new 2-year highs: " + "; ".join(buys)] if buys else []) \
        + ([f"🔴 SELL ({len(sells)}) - closed under the 10 x ATR trailing stop: " + "; ".join(sells)] if sells else []) \
        + ([f"({skipped} more breakouts skipped: book full, {MAX_POS} names)"] if skipped else [])
    if not body:
        body = ["No trades today - holdings above their stops, no new 2-year highs with room in the book."]
    tot = eq_now / st["start_equity"] - 1
    nf = nifty_last / st["nifty_start"] - 1
    invested = eq_now - st["cash"]
    lines = [head] + body + lines + [
        f"Book Rs {eq_now:,.0f} ({len(st['positions'])} names, {invested / eq_now * 100:.0f}% invested) | since "
        f"{st['start']}: {tot * 100:+.2f}% vs NIFTY 50 {nf * 100:+.2f}% | paper, not proven"]
    lines += [f"⚠️ {w_}" for w_ in warns]
    st["last_day"] = str(today)
    daily = st.setdefault("daily", [])
    if daily and daily[-1].get("date") == str(today):
        daily.pop()
    daily.append({"date": str(today), "equity": round(eq_now, 2), "nifty": nifty_last})
    save_state(st)
    notify("\n".join(lines), telegram, holdings_table(st, prices_now, eq_now))
    movers = movers_table(st, prices_now, eq_now)
    if movers and len(st["positions"]) >= 6:
        notify(f"🏆 {TAG} {today} - best and worst holdings (P&L since bought)", telegram, movers)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(json.dumps(load_state(), indent=1, default=str)[:4000])
        return
    changed = False
    try:
        changed = run_day(not a.no_telegram)
    finally:
        _set_output(changed)


if __name__ == "__main__":
    main()
