"""
V7 momentum swing - live paper trading (NSE). One step per trading day.

    python -m v7.live                  # what the NSE Alert Scan `v7` job runs
    python -m v7.live --no-telegram    # console only
    python -m v7.live --report         # print the state

The model (research: v7/README.md, `python -m v7.swing_v7`):
  universe  point-in-time top-200 NSE stocks by 6-month median traded value
            (ETFs excluded)
  score     D1 (v7.momentum, since 2026-10-13): 0.8 x NSE "Nifty200 Momentum
            30" style momentum (6- and 12-month return / 1-year volatility,
            z-scored) + 0.2 x z(delivery trend: 20-session mean delivery %
            minus the 120-session mean)
  portfolio top 30, inverse-volatility weights (15% name / 30% sector caps),
            rebalanced every 21 sessions at the open; a holding stays while
            ranked within the top 45
  stop      a close 10% under the average buy price sells at the next open;
            the name can't be bought again before the next rebalance
  regime    at a rebalance, half exposure when India VIX is above its 1-year
            80th percentile AND NIFTY 50 is below its 200-day average (plus
            the swing engine's drawdown breaker)

Each trading day, on the first run after 09:20 IST:
  1. refresh NSE end-of-day data (the same files and code as V5)
  2. score the universe at the PREVIOUS close, mark holdings at that close,
     apply splits / bonuses to holdings
  3. stop-loss sells at today's open; on a rebalance day trade to the new
     target at today's open (Upstox 9:15 candle) with full delivery costs
  4. Telegram: orders, or a daily update vs NIFTY 50
State: live_state/v7/state.json + trades.csv (same columns as V5, so the app
reads both the same way). Idempotent per day. Paper only - no orders are sent.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("V7_STATE_DIR", os.path.join(ROOT, "live_state", "v7"))
WINDOW_DAYS = 800                  # calendar days of history (> 252 sessions + warm-ups)
REWEIGHT_TOL = 0.25
OVERRIDES = {"universe.size": 200, "portfolio.top_n": 30, "portfolio.rebalance_days": 21,
             "portfolio.hold_buffer": 1.5, "risk.stop_loss": 0.10, "risk.regime_filter": True}
TAG = "V7 MOMENTUM SWING PAPER"


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


# ─── signal ──────────────────────────────────────────────────────────────

def regime(indices: pd.DataFrame, macro: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """The two regime inputs at the last date, defined exactly as in
    swing.features.market (what the research's regime filter read)."""
    from swing.data import macro as macro_mod
    from swing.features.market import index_close
    nifty = index_close(indices, "NIFTY 50").reindex(dates).ffill()
    dist = nifty / nifty.rolling(200, min_periods=150).mean() - 1
    m = macro_mod.align(macro, dates)
    vix_pct = m["india_vix"].rolling(252, min_periods=150).rank(pct=True) if "india_vix" in m else pd.Series(index=dates)
    return {"vix_pct_1y": float(vix_pct.iloc[-1]) if pd.notna(vix_pct.iloc[-1]) else None,
            "nifty_dist_ma200": float(dist.iloc[-1]) if pd.notna(dist.iloc[-1]) else None}


def score_previous_close(cfg: dict):
    """D1 scores of the top-200 universe at the latest session before today,
    plus that session's per-stock info, closes and corporate-action factors."""
    from swing import config as cfgmod
    from swing.data import panel as panel_mod
    from swing.features.market import index_close
    from swing.research import load_data
    from v5.live import ETF_PATTERN, _clean_old
    from v7.momentum import d1_score
    data = load_data(cfg, download=False)
    _clean_old(cfgmod.path(cfg, "cache"), data["end"])
    panel = panel_mod.add_universe(data["panel"].drop(columns=["in_universe"]), cfg)
    last = panel["date"].max()
    day = panel[panel["date"] == last].set_index("entity")
    etf = day["symbol"].str.contains(ETF_PATTERN.replace("(", "(?:"), regex=True, na=False)   # non-capturing: no pandas warning
    members = day.index[day["in_universe"] & ~etf]
    s = d1_score(panel, pd.DataFrame({"date": last, "entity": members}))
    scores = s.set_index("entity")["score"]
    vol = panel[["date", "entity", "ret_cc"]].sort_values(["entity", "date"])
    vol63 = vol.groupby("entity")["ret_cc"].apply(lambda r: r.tail(63).std() if r.notna().sum() >= 31 else float("nan"))
    info = day[["symbol", "isin", "close"]].copy()
    info["vol_63"] = vol63.reindex(info.index)
    info["industry"] = info.index.map(data["industry"]).fillna("UNKNOWN")
    factors = panel.assign(f=panel["adj_close"] / panel["close"])[["date", "entity", "f"]]
    ctx = regime(data["indices"], data["macro"], pd.DatetimeIndex(sorted(panel["date"].unique())))
    nifty = index_close(data["indices"], "NIFTY 50")
    return pd.Timestamp(last), scores, info, ctx, nifty, factors


def apply_corporate_actions(st: dict, factors: pd.DataFrame, last: pd.Timestamp) -> list[str]:
    """Splits / bonuses since a holding's reference date: shares up, entry and
    last price down by the same factor (value unchanged). Uses the panel's
    current adjustment factors (adj_close / close) of the reference date and
    the last date, so a later re-adjustment of history can't double-count."""
    notes = []
    f = factors.set_index(["entity", "date"])["f"]
    for sym, p in st["positions"].items():
        ref = pd.Timestamp(p.get("ref_date") or p["since"])
        try:
            ratio = float(f.loc[(p["entity"], ref)]) / float(f.loc[(p["entity"], last)])
        except KeyError:
            continue
        if abs(ratio - 1) > 1e-4:
            p["qty"] = int(round(p["qty"] / ratio))
            p["entry_px"] *= ratio
            p["last_px"] *= ratio
            notes.append(f"{sym}: corporate action x{1 / ratio:g} - now {p['qty']} shares")
        p["ref_date"] = str(last.date())
    return notes


def stops_hit(st: dict, closes: dict, stop: float) -> list[str]:
    """Holdings whose last close is at or below (1 - stop) x average buy price."""
    return [sym for sym, p in st["positions"].items()
            if sym in closes and closes[sym] <= p["entry_px"] * (1 - stop)]


# ─── trading ─────────────────────────────────────────────────────────────

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
    from swing.risk import sizing
    from v5.live import (equity_at, holdings_table, market_open_today, movers_table, notify, prepare_data,
                         todays_open)

    now = datetime.now(IST)
    today = now.date()
    if now.weekday() >= 5:
        print("Weekend - nothing to do.")
        return False
    st = load_state()
    if st.get("last_day") == str(today):
        print(f"[V7] already done for {today}.")
        return False
    if now.hour * 60 + now.minute < 9 * 60 + 20:
        print(f"{now:%H:%M} - waiting for 09:20 IST (needs today's opening prices).")
        return False
    if not market_open_today():
        st = st or {}
        if st.get("closed_day") != str(today):
            st.update(closed_day=str(today), last_day=str(today))
            save_state(st)
            notify(f"{TAG} {today}: NSE closed today (no 9:15 prices) - no session counted, no trades.", telegram)
            return True
        return False

    cfg = cfgmod.load({"data.start": str(today - timedelta(days=WINDOW_DAYS)), "data.end": str(today - timedelta(days=1)),
                       **OVERRIDES})
    complete, warns = prepare_data(cfg, today)
    if not complete:
        return True                      # partial download saved; resume next trigger
    t0 = time.time()
    last, scores, info, ctx, nifty, factors = score_previous_close(cfg)
    print(f"[V7] scored {len(scores)} stocks from the {last.date()} close in {time.time() - t0:.0f} s; regime {ctx}", flush=True)

    if not st or "positions" not in st:
        st = {"start": str(today), "cash": float(cfg["portfolio"]["capital"]), "positions": {},
              "sessions_since_rebalance": 10 ** 6, "equity_history": [], "nifty_start": None, "start_equity": None,
              "stopped": []}
    c = cfg["costs"]
    lines = apply_corporate_actions(st, factors, last)
    closes = info.set_index("symbol")["close"].to_dict()
    for sym, p in st["positions"].items():
        p["last_px"] = float(closes.get(sym, p["last_px"]))
    prices_prev = {sym: p["last_px"] for sym, p in st["positions"].items()}
    eq_prev = equity_at(st, prices_prev)
    nifty_last = float(nifty.loc[:last].iloc[-1])
    st["nifty_start"] = st["nifty_start"] or nifty_last
    st["start_equity"] = st["start_equity"] or eq_prev
    st["sessions_since_rebalance"] = st.get("sessions_since_rebalance", 0) + 1
    isin = dict(zip(info["symbol"], info["isin"]))
    isin.update({sym: p.get("isin") for sym, p in st["positions"].items() if p.get("isin")})

    # 1) stop-loss exits at today's open
    stop_out = []
    for sym in stops_hit(st, closes, cfg["risk"]["stop_loss"]):
        px = todays_open(sym, isin.get(sym))
        if px is None:
            lines.append(f"⚠️ {sym}: stop hit but no opening price - sell postponed")
            continue
        ent, qty = st["positions"][sym]["entity"], st["positions"][sym]["qty"]
        fill, pnl, ret = _sell(st, sym, qty, px, c, today)
        st.setdefault("stopped", []).append(ent)
        stop_out.append(f"{sym} {qty} @ {fill:.2f} ({pnl:+,.0f}, {ret * 100:+.1f}%)")
    if stop_out:
        lines.insert(0, f"🛑 STOP-LOSS ({len(stop_out)}, closed 10% under the buy price): " + "; ".join(stop_out))

    never_invested = not st["positions"] and not os.path.exists(_path("trades.csv"))
    rebalance = (st["sessions_since_rebalance"] >= cfg["portfolio"]["rebalance_days"]
                 or st.get("rebalance_pending", False) or never_invested)
    if rebalance:
        sc = scores.drop([e for e in st.get("stopped", []) if e in scores.index])
        held = {p["entity"] for p in st["positions"].values()}
        names = sizing.select(sc, held, cfg["portfolio"]["top_n"], cfg["portfolio"]["hold_buffer"])
        ex = sizing.Exposure(cfg)
        ex.history = [float(x) for x in st["equity_history"]][-ex.window:]
        scale = ex.scale(eq_prev, ctx)
        w = sizing.weights(names, info["vol_63"], info["industry"], cfg) * scale
        st["equity_history"] = ex.history
        st["stopped"] = []
        target = {info.loc[e, "symbol"]: float(wt) for e, wt in w.items()}
        opens = {sym: todays_open(sym, isin.get(sym)) for sym in set(target) | set(st["positions"])}
        eq_open = st["cash"] + sum(p["qty"] * (opens.get(sym) or p["last_px"]) for sym, p in st["positions"].items())
        buys, sells, holds = [], [], []
        for sym in list(st["positions"]):
            p, px = st["positions"][sym], opens.get(sym)
            tgt = target.get(sym, 0.0) * eq_open
            cur = p["qty"] * (px or p["last_px"])
            if tgt > 0 and (cur - tgt) / max(tgt, 1) <= REWEIGHT_TOL:
                holds.append(sym)
                continue
            if px is None:
                lines.append(f"⚠️ {sym}: no opening price - sell postponed")
                continue
            qty = p["qty"] if tgt <= 0 else int((cur - tgt) // px)
            if qty <= 0:
                holds.append(sym)
                continue
            fill, pnl, ret = _sell(st, sym, qty, px, c, today)
            sells.append(f"{sym} {qty} @ {fill:.2f} ({pnl:+,.0f}, {ret * 100:+.1f}%)")
            if sym in st["positions"]:
                holds.append(sym)
        for sym, wt in sorted(target.items(), key=lambda kv: -kv[1]):
            px = opens.get(sym)
            cur = st["positions"].get(sym, {}).get("qty", 0) * (px or 0)
            tgt = wt * eq_open
            if px is None or tgt <= 0 or (cur > 0 and (tgt - cur) / tgt < REWEIGHT_TOL):
                if px is None and sym not in st["positions"]:
                    lines.append(f"⚠️ {sym}: no opening price - buy skipped")
                continue
            fill = cost_mod.fill_price(px, "buy", c)
            qty = int((tgt - cur) // fill)
            if qty <= 0:
                continue
            gross = qty * fill
            fee = cost_mod.order_charges(gross, "buy", c)
            if gross + fee > st["cash"]:
                qty = int((st["cash"] / (1 + 0.002)) // fill)
                if qty <= 0:
                    continue
                gross, fee = qty * fill, cost_mod.order_charges(qty * fill, "buy", c)
            st["cash"] -= gross + fee
            ent = next(e for e in w.index if info.loc[e, "symbol"] == sym)
            p = st["positions"].setdefault(sym, {"entity": ent, "isin": info.loc[ent, "isin"], "qty": 0, "cost": 0.0,
                                                 "since": str(today), "last_px": px, "entry_px": fill,
                                                 "ref_date": str(today), "sector": info.loc[ent, "industry"]})
            p["entry_px"] = (p["entry_px"] * p["qty"] + gross) / (p["qty"] + qty)
            p["qty"] += qty
            p["cost"] += gross + fee
            p["last_px"] = px
            _log_trade(today, "BUY", sym, qty, fill, -(gross + fee), 0.0)
            buys.append(f"{sym} {qty} @ {fill:.2f} ({wt * 100:.1f}%)")
        st["sessions_since_rebalance"] = 0
        st["last_rebalance"] = str(today)
        st["rebalance_pending"] = bool(target) and not buys and not st["positions"]
        risk_off = scale < 1
        head = (f"📈 {TAG} - REBALANCE {today} (signal: {last.date()} close; fills at today's open incl. delivery costs)"
                + (f"\n⚖️ Exposure {scale * 100:.0f}%: " + ("India VIX high and NIFTY under its 200-day average"
                   if ctx.get("vix_pct_1y") and ctx["vix_pct_1y"] > 0.8 and (ctx.get("nifty_dist_ma200") or 0) < 0
                   else "drawdown breaker") if risk_off else ""))
        lines = [head] + ([f"BUY ({len(buys)}): " + "; ".join(buys)] if buys else []) \
            + ([f"SELL ({len(sells)}): " + "; ".join(sells)] if sells else []) \
            + ([f"HOLD ({len(holds)}): " + ", ".join(sorted(holds))] if holds else []) + lines
        prices_now = {sym: (opens.get(sym) or p["last_px"]) for sym, p in st["positions"].items()}
    else:
        prices_now = {sym: p["last_px"] for sym, p in st["positions"].items()}
        lines.insert(0 if not stop_out else 1,
                     f"📊 {TAG} {today} (marked at the {last.date()} close) - next rebalance in "
                     f"{cfg['portfolio']['rebalance_days'] - st['sessions_since_rebalance']} session(s)")
    eq_now = equity_at(st, prices_now)
    tot = eq_now / st["start_equity"] - 1
    nf = nifty_last / st["nifty_start"] - 1
    invested = eq_now - st["cash"]
    lines.append(f"Portfolio Rs {eq_now:,.0f} ({len(st['positions'])} stocks, {invested / eq_now * 100:.0f}% invested) | "
                 f"since {st['start']}: {tot * 100:+.2f}% vs NIFTY 50 {nf * 100:+.2f}% | paper only")
    lines += [f"⚠️ {w_}" for w_ in warns]
    st["last_day"] = str(today)
    daily = st.setdefault("daily", [])
    if daily and daily[-1].get("date") == str(today):
        daily.pop()
    daily.append({"date": str(today), "equity": round(eq_now, 2), "nifty": nifty_last})
    save_state(st)
    notify("\n".join(lines), telegram, holdings_table(st, prices_now, eq_now))
    movers = movers_table(st, prices_now, eq_now)
    if movers:
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
