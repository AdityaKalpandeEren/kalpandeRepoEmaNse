"""
V7 research (NSE swing): externally-motivated upgrades of V5, pre-registered
and all reported - no tuning on these results.

    python -m v7.swing_v7            # the pre-registered candidates
    python -m v7.swing_v7 --sens     # one-at-a-time neighbours of the chosen one

Candidates (each one is counted as a trial in swing/results/trials.json, so
the Deflated Sharpe covers every V4 + V5 + V7 trial):
  v5_combo        live V5 as-is (top 20, weekly, hold band 2.5) - the baseline
  m30_u100        NSE "Nifty200 Momentum 30" score on the V5 top-100 universe:
                  0.5 z(6m return / 1y vol) + 0.5 z(12m return / 1y vol);
                  top 20, monthly (21 sessions), hold band 1.5
  m30_u200        same score on a top-200 universe (NSE's choice), top 30
  m30_u200_stop   + 10% per-stock stop-loss (Han, Zhou & Zhu 2016)
  m30_u200_regime + regime filter (half exposure when India VIX is above its
                  1y 80th percentile AND NIFTY is below its 200-day average)
  m30_u200_both   stop + regime
  v7_blend        50/50 rank blend of the V5 ML score and M30 on top-100,
                  V5's portfolio rules + stop + regime
Windows: walk-forward 2017-01 .. 2025-06 (V5's), and Jul 2025 .. Sep 2026
reported separately as PREVIOUSLY VIEWED (V4/V5 already looked at it).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from swing import config as cfgmod
from swing.backtest import engine
from swing.data import panel as panel_mod
from swing.research import bench_returns, load_data, record_trials, summarise
from swing.validation import metrics
from v5 import features as v5f
from v5.run_v5 import V5_DEFAULTS, blend, build_v5_dataset, smooth
from v7.momentum import m30_score

log = logging.getLogger("v7")
V5_OVERRIDES = {"data.end": "2026-09-30", "label.horizon": 5, "portfolio.top_n": 20,
                "portfolio.rebalance_days": 5, "portfolio.hold_buffer": 2.5}
M30_PORT = {"portfolio.rebalance_days": 21, "portfolio.hold_buffer": 1.5}
WF = ("2017-01-01", "2025-06-30")
PV = ("2025-07-01", "2026-09-30")
V5_RUN = "swing/results/v5_65c945cb"


def run_bt(cfg, scores, prices, info, ctx, window):
    return engine.run(scores, prices, info, ctx, cfg, pd.Timestamp(window[0]), pd.Timestamp(window[1]))


def by_year(ret: pd.Series, bench: pd.Series) -> str:
    out = []
    for y, r in ret.groupby(ret.index.year):
        b = bench.reindex(r.index).fillna(0)
        out.append(f"{y}: {100 * ((1 + r).prod() - 1):+5.1f}% (N {100 * ((1 + b).prod() - 1):+5.1f}%)")
    return "  ".join(out)


def prepare() -> dict:
    """Shared inputs of every V7 swing run."""
    base = cfgmod.load(V5_OVERRIDES)
    data = load_data(base)
    df = build_v5_dataset(base, data)
    panel = data["panel"]
    bench = bench_returns(data)
    ctx = df.groupby("date")[["vix_pct_1y", "nifty_dist_ma200"]].first()

    # universes: V5's top-100 rows, and the same point-in-time rule at size 200
    u100 = df[["date", "entity"]].drop_duplicates()
    c200 = cfgmod.load({**V5_OVERRIDES, "universe.size": 200})
    eq200 = panel_mod.add_universe(panel.drop(columns=["in_universe"]), c200)
    u200 = eq200.loc[eq200["in_universe"], ["date", "entity"]]
    log.info("universes: top-100 %d rows (%d names), top-200 %d rows (%d names)",
             len(u100), u100["entity"].nunique(), len(u200), u200["entity"].nunique())

    vol = panel[["date", "entity", "ret_cc"]].sort_values(["entity", "date"])
    vol["vol_63"] = vol.groupby("entity")["ret_cc"].transform(lambda s: s.rolling(63, min_periods=31).std())
    info200 = u200.merge(vol[["date", "entity", "vol_63"]], on=["date", "entity"], how="left")
    info200["industry"] = info200["entity"].map(data["industry"]).fillna("UNKNOWN")
    info100 = df[["date", "entity", "vol_63", "industry"]]
    ents = set(u200["entity"]) | set(u100["entity"])
    prices = engine.prepare_prices(panel, ents)

    m100, m200 = m30_score(panel, u100), m30_score(panel, u200)
    ml_raw = pd.read_parquet(os.path.join(V5_RUN, "wf_scores_v5_ml_raw.parquet"))
    fac = df[["date", "entity"]].assign(score=v5f.factor_score(df).values)
    v5_combo = smooth(blend(ml_raw, fac[fac["date"] >= ml_raw["date"].min()], V5_DEFAULTS["combo_weight_ml"]),
                      V5_DEFAULTS["halflife"])
    v7_blend = blend(ml_raw, m100, 0.5)

    return {"base": base, "df": df, "bench": bench, "ctx": ctx, "prices": prices, "info100": info100,
            "panel": panel, "u200": u200,
            "info200": info200, "m100": m100, "m200": m200, "v5_combo": v5_combo, "v7_blend": v7_blend}


SENS = [("top 20", {"portfolio.top_n": 20}), ("top 40", {"portfolio.top_n": 40}),
        ("rebalance 10d", {"portfolio.rebalance_days": 10}), ("rebalance 42d", {"portfolio.rebalance_days": 42}),
        ("stop 8%", {"risk.stop_loss": 0.08}), ("stop 15%", {"risk.stop_loss": 0.15})]


def sensitivity(x: dict) -> None:
    """One-at-a-time neighbours of m30_u200_both - robustness, not selection (all counted as trials)."""
    base_over = {**M30_PORT, "portfolio.top_n": 30, "risk.stop_loss": 0.10, "risk.regime_filter": True}
    rows, rets = [], {}
    for label, over in [("m30_u200_both (chosen)", {})] + SENS:
        cfg = cfgmod.load({**V5_OVERRIDES, **base_over, **over})
        for wl, window in (("WF", WF), ("PV", PV)):
            res = run_bt(cfg, x["m200"], x["prices"], x["info200"], x["ctx"], window)
            s = summarise(res, x["bench"], cfg)
            rows.append({"variant": label, "window": wl, **{k: s.get(k) for k in ("CAGR", "Sharpe", "MaxDD", "Alpha_ann", "Turnover_ann")}})
            if wl == "WF":
                rets[f"sens: {label}"] = res.daily["ret"].iloc[1:]
    cfg_rec = cfgmod.load(V5_OVERRIDES)
    cfg_rec["v7_run"] = "swing_v7_sensitivity"
    record_trials(cfgmod.path(cfg_rec, "results", "v7_swing"), cfg_rec, rets)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.pivot(index="variant", columns="window", values=["CAGR", "Sharpe", "MaxDD", "Alpha_ann"]).round(3).to_string())


def main():
    cfgmod.setup_logging()
    if "--sens" in __import__("sys").argv:
        sensitivity(prepare())
        return
    x = prepare()
    base, bench, ctx, prices = x["base"], x["bench"], x["ctx"], x["prices"]
    info100, info200, m100, m200 = x["info100"], x["info200"], x["m100"], x["m200"]
    v5_combo, v7_blend = x["v5_combo"], x["v7_blend"]
    stop, regime = {"risk.stop_loss": 0.10}, {"risk.regime_filter": True}
    cands = {
        "v5_combo": (v5_combo, info100, {}),
        "m30_u100": (m100, info100, M30_PORT),
        "m30_u200": (m200, info200, {**M30_PORT, "portfolio.top_n": 30}),
        "m30_u200_stop": (m200, info200, {**M30_PORT, "portfolio.top_n": 30, **stop}),
        "m30_u200_regime": (m200, info200, {**M30_PORT, "portfolio.top_n": 30, **regime}),
        "m30_u200_both": (m200, info200, {**M30_PORT, "portfolio.top_n": 30, **stop, **regime}),
        "v7_blend": (v7_blend, info100, {**stop, **regime}),
    }
    out_dir = cfgmod.path(base, "results", "v7_swing")
    rows, wf_rets, pv_rows = [], {}, []
    for name, (scores, info, over) in cands.items():
        cfg = cfgmod.load({**V5_OVERRIDES, **over})
        for label, window in (("WF", WF), ("PV", PV)):
            res = run_bt(cfg, scores, prices, info, ctx, window)
            r = res.daily["ret"].iloc[1:]
            s = summarise(res, bench, cfg)
            stops = int((res.trades.get("reason") == "stop").sum()) if not res.trades.empty and "reason" in res.trades else 0
            row = {"model": name, "window": label, **{k: s.get(k) for k in ("CAGR", "Sharpe", "MaxDD", "Calmar", "Alpha_ann",
                                                                            "Beta", "Bench_CAGR", "Turnover_ann", "Cost_drag_ann",
                                                                            "HitRate_trades", "Trades")}, "stops": stops}
            (rows if label == "WF" else pv_rows).append(row)
            if label == "WF":
                wf_rets[name] = r
                log.info("%-16s WF by year: %s", name, by_year(r, bench))
            res.daily.to_csv(os.path.join(out_dir, f"{label.lower()}_daily_{name}.csv"))
            if not res.trades.empty:
                res.trades.to_csv(os.path.join(out_dir, f"{label.lower()}_trades_{name}.csv"), index=False)
    cfg_rec = cfgmod.load(V5_OVERRIDES)
    cfg_rec["v7_run"] = "swing_v7"
    trial_sr = record_trials(out_dir, cfg_rec, wf_rets)
    table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    best = table.iloc[0]["model"]
    dsr = metrics.deflated_sharpe(wf_rets[best], trial_sr)
    pd.set_option("display.width", 220)
    print("\nWALK-FORWARD 2017-01 .. 2025-06 (after all delivery costs)")
    print(table.drop(columns=["window"]).round(3).to_string(index=False))
    print(f"\nbest {best}: Deflated Sharpe {dsr['DSR']:.3f} over {dsr['N_trials']} trials (V4 + V5 + V7)")
    print("\nPREVIOUSLY VIEWED Jul 2025 .. Sep 2026")
    print(pd.DataFrame(pv_rows).drop(columns=["window"]).round(3).to_string(index=False))
    json.dump({"wf": rows, "pv": pv_rows, "best": best, "dsr": dsr}, open(os.path.join(out_dir, "summary.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
