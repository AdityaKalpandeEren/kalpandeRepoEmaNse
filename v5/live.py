"""
V5.0 live paper trading (NSE swing model) - one step per trading day.

    python -m v5.live                  # what the NSE Alert Scan `v5` job runs
    python -m v5.live --no-telegram    # console only
    python -m v5.live --report         # print the portfolio summary

Each trading day, on the first run after 09:20 IST:
  1. download any missing NSE end-of-day files (bhavcopy, delivery, index
     closes, participant OI) for a ~2-year window, refresh corporate actions
     (falls back to the committed seed if NSE's API is unreachable) and macro
  2. rebuild V5 features up to the PREVIOUS session's close and score the
     universe with the committed model (v5/model/v5_final_model.joblib),
     smoothed exactly as in research
  3. every 5th session (and on the first day): rebalance to the top-20
     target at TODAY'S OPEN (Upstox 9:15 candle open) with full delivery
     costs; holdings stay while ranked within top 50 (no-trade band)
  4. Telegram: rebalance orders (BUY / SELL / HOLD), or a daily portfolio
     update; all-time P&L vs NIFTY 50 since the start

State: live_state/v5/state.json (positions, cash, history) + trades.csv.
Idempotent: a second run on the same day does nothing. Paper only - no
orders are sent anywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("V5_STATE_DIR", os.path.join(ROOT, "live_state", "v5"))
MODEL = os.path.join(ROOT, "v5", "model", "v5_final_model.joblib")
SEED = os.path.join(ROOT, "v5", "seed")
WINDOW_DAYS = 800                  # calendar days of history (> 252 sessions + warm-ups)
DOWNLOAD_BUDGET_S = 25 * 60        # stop downloading after this and resume next run
REWEIGHT_TOL = 0.25
# One-off late start (user, 2026-10-05): the first run hit a holiday and
# bought nothing, so V5 rebalances once more on this date at the LIVE price
# (09:15 has passed) instead of waiting for the next session. Inert after.
FORCE_START_DAY = "2026-10-05"


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


def notify(text: str, telegram: bool) -> None:
    print(text, flush=True)
    if telegram:
        from paper_trading.research_live import send_text
        send_text(text)


def _set_output(changed: bool) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")


# ─── data ────────────────────────────────────────────────────────────────

def prepare_data(cfg: dict, today: date) -> tuple[bool, list[str]]:
    """Download / refresh everything V5 needs. Returns (complete, warnings)."""
    from swing import config as cfgmod
    from swing.data import corporate_actions, macro, nse_daily, sectors
    warnings = []
    cache = cfgmod.path(cfg, "cache")
    start = today - timedelta(days=WINDOW_DAYS)
    end = today - timedelta(days=1)
    # seeds (committed) so a fresh cache never starts empty
    for name in ("corporate_actions.parquet", "industries.parquet"):
        dst = os.path.join(cache, name)
        if not os.path.exists(dst) and os.path.exists(os.path.join(SEED, name)):
            shutil.copy(os.path.join(SEED, name), dst)
    t0 = time.time()
    missing = [(k, d) for d in nse_daily.weekdays(start, end) for k in nse_daily.KINDS
               if not os.path.exists(nse_daily._file(cache, k, d))]
    for i, (k, d) in enumerate(missing):
        if time.time() - t0 > DOWNLOAD_BUDGET_S:
            print(f"[V5] download budget used ({i}/{len(missing)} files) - resuming next run", flush=True)
            return False, warnings
        try:
            nse_daily.get_day(cache, k, d)
        except Exception as e:
            print(f"[V5] fetch failed {k} {d}: {e!r}", flush=True)
    if missing:
        print(f"[V5] downloaded {len(missing)} files in {time.time() - t0:.0f} s", flush=True)
    # corporate actions: refresh the last two months, keep the rest
    f = os.path.join(cache, "corporate_actions.parquet")
    old = pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()
    try:
        tmpdir = os.path.join(cache, "_ca_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        new = corporate_actions.fetch(tmpdir, (today.replace(day=1) - timedelta(days=40)), today, refresh=True)
        allca = pd.concat([old, new]).drop_duplicates() if not old.empty else new
        allca.to_parquet(f, index=False)
    except Exception as e:
        warnings.append(f"corporate-actions feed unavailable ({str(e)[:60]}) - using cached list")
    try:
        macro.fetch(cache, str(start - timedelta(days=400)), refresh=True)
    except Exception as e:
        warnings.append(f"macro refresh failed ({str(e)[:60]}) - using cached")
    try:
        sectors.load(cache)
    except Exception:
        pass
    return True, warnings


def _clean_old(cache: str, keep_end: date) -> None:
    """Drop panels / datasets built for earlier days (they are rebuilt daily)."""
    tag = f"{keep_end:%Y%m%d}"
    for n in os.listdir(cache):
        if n.startswith(("panel_", "dataset_", "v5_dataset_")) and n.endswith(".parquet") and tag not in n:
            os.remove(os.path.join(cache, n))


def score_previous_close(cfg: dict, bundle: dict, today: date):
    """Smoothed V5 scores at the latest session before today + that session's info."""
    from swing import config as cfgmod
    from swing.research import load_data
    from v5 import features as v5f
    from v5.run_v5 import blend, build_v5_dataset, predict_ml, smooth
    data = load_data(cfg, download=False)
    df = build_v5_dataset(cfg, data)
    _clean_old(cfgmod.path(cfg, "cache"), data["end"])
    days = sorted(df["date"].unique())[-30:]
    recent = df[df["date"].isin(days)]
    feats, v5 = bundle["features"], bundle["v5"]
    ml = predict_ml(bundle["models"], recent, feats)
    fac = pd.DataFrame({"date": recent["date"], "entity": recent["entity"],
                        "score": v5f.factor_score(recent).values})
    raw = {"v5_ml": ml, "v5_factor": fac, "v5_combo": blend(ml, fac, v5["combo_weight_ml"])}[bundle["selected"]]
    s = smooth(raw, v5["halflife"])
    last = s["date"].max()
    scores = s[s["date"] == last].set_index("entity")["score"]
    info = recent[recent["date"] == last].set_index("entity")
    panel = data["panel"]
    day = panel[panel["date"] == last]
    closes = day.set_index("symbol")["close"]
    info = info.join(day.set_index("entity")["isin"], how="left")
    # ETFs (GOLDBEES, LIQUIDBEES, ...) trade in the EQ series but are not companies
    etf = info["symbol"].str.contains(ETF_PATTERN, regex=True, na=False)
    scores = scores.drop(index=info.index[etf], errors="ignore")
    from swing.features.market import index_close
    nifty = index_close(data["indices"], "NIFTY 50")
    return pd.Timestamp(last), scores, info, closes, nifty, data["panel"]


# ─── trading ─────────────────────────────────────────────────────────────

ETF_PATTERN = r"(BEES|ETF|IETF)$|^(LIQUID|GOLD|SILVER)"


def todays_open(symbol: str, isin: str | None = None) -> float | None:
    """Today's 9:15 candle open from Upstox (None if not traded yet / missing).
    Looked up by the bhavcopy ISIN when known: a symbol lookup can resolve
    to a company's debenture instead of its equity (seen for CHOLAFIN)."""
    from live_v33 import today_5m
    for key in ([f"NSE_EQ|{isin}"] if isin else []) + [symbol]:
        try:
            df = today_5m(key)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        first = df.iloc[0]
        if first["timestamp"].hour == 9 and first["timestamp"].minute == 15:
            return float(first["open"])
    return None


def latest_price(symbol: str, isin: str | None = None) -> float | None:
    """Last traded price today (close of the latest 5-min candle) - fills
    for the one-off late start (FORCE_START_DAY)."""
    from live_v33 import today_5m
    for key in ([f"NSE_EQ|{isin}"] if isin else []) + [symbol]:
        try:
            df = today_5m(key)
        except Exception:
            continue
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
    return None


def market_open_today() -> bool:
    """False on an NSE holiday: no 9:15 candle today for two of the most
    liquid stocks (checked after 09:20). Without this check V5's first run,
    on the Gandhi Jayanti holiday (2026-10-02), 'rebalanced' with no
    opening prices, bought nothing and still started its 5-session wait."""
    return any(todays_open(s) is not None for s in ("RELIANCE", "HDFCBANK"))


def equity_at(st: dict, prices: dict) -> float:
    return st["cash"] + sum(p["qty"] * prices.get(sym, p["last_px"]) for sym, p in st["positions"].items())


def run_day(telegram: bool) -> bool:
    """Returns True if state changed."""
    import joblib
    from swing import config as cfgmod
    from swing.backtest import costs as cost_mod
    from swing.risk import sizing

    now = datetime.now(IST)
    today = now.date()
    if now.weekday() >= 5:
        print("Weekend - nothing to do.")
        return False
    st = load_state()
    force = str(today) == FORCE_START_DAY and not st.get("force_started")
    if st.get("last_day") == str(today) and not force:
        print(f"[V5] already done for {today}.")
        return False
    if now.hour * 60 + now.minute < 9 * 60 + 20:
        print(f"{now:%H:%M} - waiting for 09:20 IST (needs today's opening prices).")
        return False
    if not market_open_today():
        # Holiday (or no Upstox data at all): not a session - nothing is
        # counted or traded. last_day marks today done so later triggers skip.
        st = st or {}
        if st.get("closed_day") != str(today):
            st.update(closed_day=str(today), last_day=str(today))
            save_state(st)
            notify(f"V5.0 SWING PAPER {today}: NSE closed today (no 9:15 prices) - no session counted, "
                   f"no trades.", telegram)
            return True
        return False

    bundle = joblib.load(MODEL)
    cfg = cfgmod.load({"data.start": str(today - timedelta(days=WINDOW_DAYS)), "data.end": str(today - timedelta(days=1))})
    for k in ("universe", "label", "features", "portfolio", "risk", "costs"):
        cfg[k] = bundle["config"][k]
    cfg["v5"] = bundle["v5"]
    complete, warns = prepare_data(cfg, today)
    if not complete:
        return True                      # partial download saved; resume next trigger
    t0 = time.time()
    last, scores, info, closes, nifty, panel = score_previous_close(cfg, bundle, today)
    print(f"[V5] scored {len(scores)} stocks from the {last.date()} close in {time.time() - t0:.0f} s", flush=True)

    if not st:
        st = {"start": str(today), "cash": float(cfg["portfolio"]["capital"]), "positions": {},
              "sessions_since_rebalance": 10 ** 6, "equity_history": [], "nifty_start": None, "start_equity": None}
    prices_prev = {sym: float(closes.get(sym, p["last_px"])) for sym, p in st["positions"].items()}
    for sym, p in st["positions"].items():
        p["last_px"] = prices_prev[sym]
    eq_prev = equity_at(st, prices_prev)
    nifty_last = float(nifty.loc[:last].iloc[-1])
    st["nifty_start"] = st["nifty_start"] or nifty_last
    st["start_equity"] = st["start_equity"] or eq_prev
    st["sessions_since_rebalance"] = st.get("sessions_since_rebalance", 0) + 1
    # Also rebalance at once when the last rebalance could not buy anything
    # (missing opening prices) or the portfolio has never been invested.
    never_invested = not st["positions"] and not os.path.exists(_path("trades.csv"))
    rebalance = (st["sessions_since_rebalance"] >= cfg["portfolio"]["rebalance_days"]
                 or st.get("rebalance_pending", False) or never_invested or force)

    lines = []
    if rebalance:
        c = cfg["costs"]
        held = {p["entity"] for p in st["positions"].values()}
        names = sizing.select(scores, held, cfg["portfolio"]["top_n"], cfg["portfolio"]["hold_buffer"])
        ex = sizing.Exposure(cfg)
        ex.history = [float(x) for x in st["equity_history"]][-ex.window:]
        w = sizing.weights(names, info["vol_63"], info["industry"], cfg) * ex.scale(eq_prev)
        st["equity_history"] = ex.history
        target = {info.loc[e, "symbol"]: float(wt) for e, wt in w.items()}
        isin = dict(zip(info["symbol"], info["isin"]))
        isin.update({sym: p.get("isin") for sym, p in st["positions"].items() if p.get("isin")})
        price_fn = latest_price if force else todays_open
        opens = {sym: price_fn(sym, isin.get(sym)) for sym in set(target) | set(st["positions"])}
        eq_open = st["cash"] + sum(p["qty"] * (opens.get(sym) or p["last_px"]) for sym, p in st["positions"].items())
        buys, sells, holds = [], [], []
        # sells / reductions first
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
            fill = cost_mod.fill_price(px, "sell", c)
            gross = qty * fill
            proceeds = gross - cost_mod.order_charges(gross, "sell", c)
            cost_basis = p["cost"] * qty / p["qty"]
            st["cash"] += proceeds
            p["qty"] -= qty
            p["cost"] -= cost_basis
            pnl = proceeds - cost_basis
            _log_trade(today, "SELL", sym, qty, fill, proceeds, pnl)
            sells.append(f"{sym} {qty} @ {fill:.2f} ({pnl:+,.0f}, {pnl / cost_basis * 100:+.1f}%)")
            if p["qty"] <= 0:
                del st["positions"][sym]
            else:
                holds.append(sym)
        # buys / increases
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
            p = st["positions"].setdefault(sym, {"entity": ent, "isin": info.loc[ent, "isin"],
                                                 "qty": 0, "cost": 0.0, "since": str(today), "last_px": px})
            p["qty"] += qty
            p["cost"] += gross + fee
            p["last_px"] = px
            _log_trade(today, "BUY", sym, qty, fill, -(gross + fee), 0.0)
            buys.append(f"{sym} {qty} @ {fill:.2f} ({wt * 100:.1f}%)")
        st["sessions_since_rebalance"] = 0
        st["last_rebalance"] = str(today)
        # targets but no fills at all (no opening prices) -> try again next session
        st["rebalance_pending"] = bool(target) and not buys and not st["positions"]
        head = (f"📈 V5.0 SWING PAPER - REBALANCE {today} (signal: {last.date()} close; fills at today's open "
                f"incl. delivery costs)")
        if force:
            head = (f"📈 V5.0 SWING PAPER - LATE START {today} {now:%H:%M} IST (signal: {last.date()} close; "
                    f"fills at the LIVE price, not the 9:15 open - one-off, first run hit the 2-Oct holiday)")
            st["force_started"] = str(today)
        lines = [head] + ([f"BUY ({len(buys)}): " + "; ".join(buys)] if buys else []) \
            + ([f"SELL ({len(sells)}): " + "; ".join(sells)] if sells else []) \
            + ([f"HOLD ({len(holds)}): " + ", ".join(sorted(holds))] if holds else []) + lines
        prices_now = {sym: (opens.get(sym) or p["last_px"]) for sym, p in st["positions"].items()}
        eq_now = equity_at(st, prices_now)
    else:
        eq_now = eq_prev
        movers = sorted(((sym, p["last_px"] / (p["cost"] / p["qty"]) - 1) for sym, p in st["positions"].items()),
                        key=lambda kv: kv[1])
        lines = [f"📊 V5.0 SWING PAPER {today} (marked at the {last.date()} close) - next rebalance in "
                 f"{cfg['portfolio']['rebalance_days'] - st['sessions_since_rebalance']} session(s)"]
        if movers:
            lines.append("Best: " + ", ".join(f"{s} {r * 100:+.1f}%" for s, r in movers[::-1][:3])
                         + " | Worst: " + ", ".join(f"{s} {r * 100:+.1f}%" for s, r in movers[:3]))
    tot = eq_now / st["start_equity"] - 1
    nf = nifty_last / st["nifty_start"] - 1
    invested = eq_now - st["cash"]
    lines.append(f"Portfolio Rs {eq_now:,.0f} ({len(st['positions'])} stocks, {invested / eq_now * 100:.0f}% invested) | "
                 f"since {st['start']}: {tot * 100:+.2f}% vs NIFTY 50 {nf * 100:+.2f}% | paper only")
    lines += [f"⚠️ {w_}" for w_ in warns]
    st["last_day"] = str(today)
    daily = st.setdefault("daily", [])
    if daily and daily[-1].get("date") == str(today):
        daily.pop()                      # a second run today (late start) replaces the first mark
    daily.append({"date": str(today), "equity": round(eq_now, 2), "nifty": nifty_last})
    save_state(st)
    notify("\n".join(lines), telegram)
    return True


def _log_trade(day, side, sym, qty, px, cash_flow, pnl):
    f = _path("trades.csv")
    new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "side", "symbol", "qty", "price", "cash_flow", "realised_pnl"])
        w.writerow([day, side, sym, qty, round(px, 2), round(cash_flow, 2), round(pnl, 2)])


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
