"""
V6.0 research - NSE post-earnings drift ("earnings-reaction swing").

    python -m v6.research                 # event study + portfolio, discovery 2015-2021 only
    python -m v6.research --test          # + the ONE-TIME look at 2022-01 -> 2026-09

Idea (Ball-Brown / Bernard-Thomas post-earnings-announcement drift; documented
for India too): a stock whose price and volume react strongly to its quarterly
results keeps drifting the same way for weeks. Unlike the 5-min setups this
uses NEW information (the results surprise, read from the market reaction)
and holds for weeks, so delivery charges (~0.25% round trip) are small.

Timing (no lookahead):
  E      = first session on/after the board meeting that approved the results
           (NSE board-meetings feed, purpose contains "Result")
  signal = after the close of E+1: abnormal return AR2 = close(E+1)/close(E-1)
           minus NIFTY 50 over the same two days - covers results released
           before the open, during the session or after the close of E
  entry  = OPEN of E+2;  exit = OPEN of E+2+H  (H = 5/10/20/40/60 sessions)
Universe: point-in-time top-N by trailing median traded value (swing.data.panel,
corporate-action adjusted), membership as of E-1.

Honesty: thresholds / holding period are chosen on DISCOVERY (events before
2022-01-01) only. --test evaluates 2022-01-01 onward once and writes
v6/results/test_used.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime

import numpy as np
import pandas as pd

from data import nse_archives as A
from swing import config as cfgmod
from swing.backtest import costs as cost_mod
from swing.data import corporate_actions, nse_daily, panel as panel_mod

log = logging.getLogger("v6")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "v6", "results")
HORIZONS = (5, 10, 20, 40, 60)
TEST_START = pd.Timestamp("2022-01-01")
START, END = date(2014, 1, 1), date(2026, 10, 2)


# ─── data ────────────────────────────────────────────────────────────────

def build_panel(cfg: dict) -> pd.DataFrame:
    """Adjusted daily panel 2014-2026 for every stock that was ever in the
    daily top 500 by traded value (a superset of any top-N<=300 universe)."""
    cache = cfgmod.path(cfg, "cache")
    f = os.path.join(cache, f"v6_panel_{START:%Y%m%d}_{END:%Y%m%d}_n{cfg['universe']['size']}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    days = list(nse_daily.weekdays(START, END))
    cand: set[str] = set()
    for d in days:                                            # pass 1: candidate symbols
        p = nse_daily._file(cache, "equity", d)
        if os.path.exists(p):
            x = pd.read_parquet(p)
            if "symbol" in x and len(x):                      # holidays are cached as empty files
                cand |= set(x.nlargest(500, "value")["symbol"])
    log.info("candidate symbols (ever top-500 by value): %d", len(cand))
    eqs, dls = [], []
    for d in days:                                            # pass 2: their rows only
        p, q = nse_daily._file(cache, "equity", d), nse_daily._file(cache, "delivery", d)
        if os.path.exists(p):
            x = pd.read_parquet(p)
            if "symbol" not in x or x.empty:
                continue
            x = x[x["symbol"].isin(cand)]
            x["date"] = pd.Timestamp(d)
            eqs.append(x)
        if os.path.exists(q):
            y = pd.read_parquet(q)
            if "symbol" in y and not y.empty:
                y = y[y["symbol"].isin(cand)]
                y["date"] = pd.Timestamp(d)
                dls.append(y)
    eq = pd.concat(eqs, ignore_index=True).dropna(subset=["close", "prevclose"])
    eq = eq.sort_values(["date", "symbol"]).reset_index(drop=True)
    eq["entity"] = panel_mod.link_entities(eq)
    eq = eq.drop_duplicates(["entity", "date"], keep="last")
    dl = pd.concat(dls, ignore_index=True) if dls else None
    adj = corporate_actions.adjustments(corporate_actions.fetch(cache, START, END))
    p = panel_mod.assemble(eq, dl, cfg, adj)
    keep = ["date", "entity", "symbol", "open", "close", "adj_open", "adj_close", "value", "ret_cc",
            "in_universe", "n_days"] + (["deliv_pct"] if "deliv_pct" in p else [])
    p = p[keep]
    p.to_parquet(f, index=False)
    return p


def nifty_close() -> pd.Series:
    ix = nse_daily.load_kind(cfgmod.path(cfgmod.load(), "cache"), "indices", START, END)
    s = ix[ix["index"] == "NIFTY 50"].drop_duplicates("date").set_index("date")["close"].sort_index()
    return s


def result_dates() -> pd.DataFrame:
    """symbol, meeting - one row per results board meeting, 2014-07 -> 2026-09."""
    parts = []
    for m in A.months(date(2014, 7, 1), date(2026, 9, 1)):
        try:
            d = A.board_meetings_month(m)
        except Exception as e:
            log.warning("board meetings %s unavailable: %s", m, e)
            continue
        parts.append(d)
    b = pd.concat(parts, ignore_index=True)
    b = b[b["purpose"].str.contains("result", case=False, na=False)]
    b["meeting"] = pd.to_datetime(b["meeting"], errors="coerce").dt.normalize()
    return b.dropna(subset=["meeting"]).drop_duplicates(["symbol", "meeting"])[["symbol", "meeting"]]


# ─── events ──────────────────────────────────────────────────────────────

def build_events(p: pd.DataFrame, nifty: pd.Series, res: pd.DataFrame) -> pd.DataFrame:
    """One row per results event of a stock in the universe at E-1."""
    p = p.sort_values(["entity", "date"]).reset_index(drop=True)
    n = nifty.reindex(pd.DatetimeIndex(sorted(p["date"].unique()))).ffill()
    sym_ent = p.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])["entity"]
    rows = []
    groups = {e: g.reset_index(drop=True) for e, g in p.groupby("entity", sort=False)}
    # map each meeting to its entity: the symbol's entity on the nearest session on/after the meeting
    for sym, mtgs in res.groupby("symbol")["meeting"]:
        ents = p.loc[p["symbol"] == sym, ["date", "entity"]]
        if ents.empty:
            continue
        for mtg in mtgs:
            after = ents[ents["date"] >= mtg]
            if after.empty:
                continue
            ent = after["entity"].iloc[0]
            g = groups[ent]
            i = int(g["date"].searchsorted(mtg))           # E = first session on/after the meeting
            if i < 61 or i + 2 >= len(g) or not bool(g.at[i - 1, "in_universe"]):
                continue                                    # need 60 sessions of history and an entry day
            c, o, v, d = g["adj_close"].values, g["adj_open"].values, g["value"].values, g["date"].values
            nE1, nEm1 = n.get(pd.Timestamp(d[i + 1]), np.nan), n.get(pd.Timestamp(d[i - 1]), np.nan)
            row = {"entity": ent, "symbol": sym, "meeting": mtg, "E": pd.Timestamp(d[i]),
                   "signal_date": pd.Timestamp(d[i + 1]), "entry_date": pd.Timestamp(d[i + 2]),
                   "ret2": c[i + 1] / c[i - 1] - 1, "mkt2": nE1 / nEm1 - 1,
                   "pre20": c[i - 1] / c[i - 21] - 1, "mkt_pre20": n.get(pd.Timestamp(d[i - 1])) / n.get(pd.Timestamp(d[i - 21])) - 1,
                   "vol_ratio": np.mean(v[i:i + 2]) / np.median(v[i - 61:i - 1]),
                   "gap_E": o[i] / c[i - 1] - 1, "price": g.at[i + 1, "close"]}
            if "deliv_pct" in g:
                dp = g["deliv_pct"].values
                row["deliv_chg"] = np.nanmean(dp[i:i + 2]) - np.nanmean(dp[i - 61:i - 1])
            entry = o[i + 2]
            for h in HORIZONS:
                j = i + 2 + h
                if j < len(g):
                    row[f"ret_{h}"] = o[j] / entry - 1
                    row[f"mkt_{h}"] = n.get(pd.Timestamp(d[j - 1])) / nE1 - 1   # NIFTY close E+1 -> close before exit open
                    row[f"exit_date_{h}"] = pd.Timestamp(d[j])
            rows.append(row)
    ev = pd.DataFrame(rows)
    ev["ar2"] = ev["ret2"] - ev["mkt2"]
    ev["pre_ar20"] = ev["pre20"] - ev["mkt_pre20"]
    for h in HORIZONS:
        ev[f"car_{h}"] = ev[f"ret_{h}"] - ev[f"mkt_{h}"]
    return ev.drop_duplicates(["entity", "E"]).sort_values("E").reset_index(drop=True)


def clustered_t(x: pd.Series, by: pd.Series) -> float:
    """t-stat of the mean with events clustered by month (results seasons
    bunch together, so per-event t-stats overstate significance)."""
    m = x.groupby(by).mean().dropna()
    return float(m.mean() / m.std() * np.sqrt(len(m))) if len(m) > 2 and m.std() > 0 else np.nan


def event_study(ev: pd.DataFrame, label: str) -> pd.DataFrame:
    ev = ev.copy()
    ev["dec"] = pd.qcut(ev["ar2"], 10, labels=False, duplicates="drop") + 1
    month = ev["E"].dt.to_period("M")
    rows = []
    for dcl, g in ev.groupby("dec"):
        r = {"decile": int(dcl), "n": len(g), "ar2_mean": g["ar2"].mean()}
        for h in HORIZONS:
            r[f"car_{h}"] = g[f"car_{h}"].mean()
            r[f"t_{h}"] = clustered_t(g[f"car_{h}"], month[g.index])
        rows.append(r)
    t = pd.DataFrame(rows)
    log.info("%s event study (CAR after entry at E+2 open, vs NIFTY; t clustered by month):\n%s", label,
             t.round(4).to_string(index=False))
    return t


# ─── portfolio ───────────────────────────────────────────────────────────

def simulate(p: pd.DataFrame, nifty: pd.Series, trades: pd.DataFrame, h: int, max_pos: int, cfg: dict,
             start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, pd.DataFrame]:
    """Equal-slot portfolio: each signal takes one free slot (1/max_pos of equity
    at the entry open), best ar2 first; held h sessions; delivery costs + slippage.
    Returns daily returns and the trades taken."""
    c = cfg["costs"]
    days = pd.DatetimeIndex(sorted(p.loc[(p["date"] >= start) & (p["date"] <= end), "date"].unique()))
    ents = set(trades["entity"])
    sub = p[p["entity"].isin(ents) & p["date"].isin(days)]
    op = sub.pivot(index="date", columns="entity", values="adj_open").reindex(days).ffill()
    cl = sub.pivot(index="date", columns="entity", values="adj_close").reindex(days).ffill()
    by_entry = {d: g.sort_values("ar2", ascending=False) for d, g in trades.groupby("entry_date")}
    cash, pos, eq_prev, rets, taken = 1.0, {}, 1.0, [], []
    for d in days:
        # exits at the open
        for e in [e for e, q in pos.items() if q["exit"] <= d]:
            q = pos.pop(e)
            px = op.at[d, e] if not np.isnan(op.at[d, e]) else q["last"]
            gross = q["units"] * cost_mod.fill_price(px, "sell", c)
            proceeds = gross - cost_mod.order_charges(gross * 1e6, "sell", c) / 1e6
            cash += proceeds
            taken.append({**q["info"], "exit_px": px, "ret_net": proceeds / q["cost"] - 1})
        # entries at the open
        equity_open = cash + sum(q["units"] * (op.at[d, e] if not np.isnan(op.at[d, e]) else q["last"]) for e, q in pos.items())
        for _, t in by_entry.get(d, pd.DataFrame()).iterrows():
            if len(pos) >= max_pos or t["entity"] in pos or np.isnan(op.at[d, t["entity"]]):
                continue
            spend = min(equity_open / max_pos, cash)
            if spend <= 1e-6:
                break
            fee = cost_mod.order_charges(spend * 1e6, "buy", c) / 1e6
            fp = cost_mod.fill_price(op.at[d, t["entity"]], "buy", c)
            pos[t["entity"]] = {"units": (spend - fee) / fp, "cost": spend, "exit": t[f"exit_date_{h}"],
                                "last": fp, "info": {"entity": t["entity"], "symbol": t["symbol"], "entry_date": d,
                                                     "ar2": t["ar2"], "exit_date": t[f"exit_date_{h}"]}}
            cash -= spend
        for e, q in pos.items():
            if not np.isnan(cl.at[d, e]):
                q["last"] = cl.at[d, e]
        eq = cash + sum(q["units"] * q["last"] for q in pos.values())
        rets.append(eq / eq_prev - 1)
        eq_prev = eq
    return pd.Series(rets, index=days), pd.DataFrame(taken)


def perf(r: pd.Series, bench: pd.Series, rf: float = 0.065) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    b = bench.reindex(r.index).fillna(0)
    beq = (1 + b).cumprod()
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "Sharpe": (r.mean() - rf / 252) / r.std() * np.sqrt(252),
            "MaxDD": (eq / eq.cummax() - 1).min(), "NIFTY_CAGR": beq.iloc[-1] ** (1 / yrs) - 1,
            "NIFTY_MaxDD": (beq / beq.cummax() - 1).min(), "Total": eq.iloc[-1] - 1, "NIFTY_Total": beq.iloc[-1] - 1}


# ─── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="the one-time look at 2022-01 ->")
    ap.add_argument("--universe", type=int, default=200)
    a = ap.parse_args()
    cfgmod.setup_logging()
    os.makedirs(OUT, exist_ok=True)
    cfg = cfgmod.load({"universe.size": a.universe, "data.start": str(START), "data.end": str(END)})
    p = build_panel(cfg)
    nifty = nifty_close()
    res = result_dates()
    log.info("panel %s; results meetings %d (%s .. %s)", p.shape, len(res), res["meeting"].min().date(), res["meeting"].max().date())
    evf = os.path.join(OUT, f"events_n{a.universe}.parquet")
    ev = pd.read_parquet(evf) if os.path.exists(evf) else build_events(p, nifty, res)
    ev.to_parquet(evf, index=False)
    ev = ev[ev["E"] >= "2015-01-01"]
    disc, test = ev[ev["E"] < TEST_START], ev[ev["E"] >= TEST_START]
    log.info("events in universe: %d discovery (2015-2021), %d test (2022-)", len(disc), len(test))
    rt = cost_mod.round_trip_pct(50_000, cfg["costs"])
    log.info("round-trip delivery cost incl. slippage on Rs 50k: %.3f%%", rt * 100)
    event_study(disc, "DISCOVERY 2015-2021")

    # rule grid, chosen on discovery: top reaction events (+ volume confirmation), hold H
    bench = nifty.pct_change()
    grid = []
    for q in (0.80, 0.90, 0.95):
        thr = disc["ar2"].quantile(q)
        for vmin in (0.0, 1.5, 2.5):
            for h in (10, 20, 40, 60):
                for max_pos in (10, 20):
                    sig = disc[(disc["ar2"] >= thr) & (disc["vol_ratio"] >= vmin) & disc[f"exit_date_{h}"].notna()]
                    r, tk = simulate(p, nifty, sig, h, max_pos, cfg, pd.Timestamp("2015-01-01"), TEST_START - pd.Timedelta(days=1))
                    m = perf(r, bench)
                    grid.append({"ar2_q": q, "ar2_thr": thr, "vol_min": vmin, "H": h, "max_pos": max_pos,
                                 "signals": len(sig), "trades": len(tk), "win%": (tk["ret_net"] > 0).mean() * 100 if len(tk) else np.nan,
                                 "avg_net%": tk["ret_net"].mean() * 100 if len(tk) else np.nan, **m})
    g = pd.DataFrame(grid).sort_values("Sharpe", ascending=False)
    g.to_csv(os.path.join(OUT, "discovery_grid.csv"), index=False)
    log.info("DISCOVERY portfolio grid (top 12 by Sharpe; %d configs):\n%s", len(g),
             g.head(12)[["ar2_q", "ar2_thr", "vol_min", "H", "max_pos", "trades", "win%", "avg_net%", "CAGR",
                         "Sharpe", "MaxDD", "NIFTY_CAGR"]].round(3).to_string(index=False))
    best = g.iloc[0].to_dict()
    json.dump(best, open(os.path.join(OUT, "selected.json"), "w"), indent=1, default=str)
    log.info("selected on discovery: %s", {k: best[k] for k in ("ar2_q", "ar2_thr", "vol_min", "H", "max_pos")})

    if a.test:
        marker = os.path.join(OUT, "test_used.json")
        if os.path.exists(marker):
            log.warning("TEST PERIOD ALREADY VIEWED (%s) - this is a repeated look", open(marker).read().strip())
        event_study(test, "TEST 2022-2026")
        h, mp = int(best["H"]), int(best["max_pos"])
        sig = test[(test["ar2"] >= best["ar2_thr"]) & (test["vol_ratio"] >= best["vol_min"]) & test[f"exit_date_{h}"].notna()]
        r, tk = simulate(p, nifty, sig, h, mp, cfg, TEST_START, pd.Timestamp(END))
        m = perf(r, bench)
        log.info("TEST portfolio (selected rule): trades %d, win %.1f%%, avg net %.2f%%\n%s", len(tk),
                 (tk["ret_net"] > 0).mean() * 100, tk["ret_net"].mean() * 100, pd.Series(m).round(4).to_string())
        yr = pd.DataFrame({"V6": (1 + r).groupby(r.index.year).prod() - 1,
                           "NIFTY": (1 + bench.reindex(r.index).fillna(0)).groupby(r.index.year).prod() - 1})
        log.info("TEST per year:\n%s", yr.round(4).to_string())
        tk.to_csv(os.path.join(OUT, "test_trades.csv"), index=False)
        r.to_csv(os.path.join(OUT, "test_daily.csv"))
        if not os.path.exists(marker):
            json.dump({"used_at": datetime.now().isoformat(timespec="seconds"), "selected": best}, open(marker, "w"), default=str)


if __name__ == "__main__":
    main()
