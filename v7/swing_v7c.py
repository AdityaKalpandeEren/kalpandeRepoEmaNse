"""
V7 swing, round 4: delivery-% tilt on the live V7 score (top 30 of the top-200,
monthly, 10% stop, regime). Fixed in advance, all reported, all counted as
trials. Keep rule (same as round 2): beat live V7 in BOTH windows.

Why: on 2026-10-13 abnormal delivery % (today's delivery % minus the stock's
own 60-day average) predicted the next 5 sessions' return in every period
tested - Spearman IC +0.025 / +0.028 / +0.022 (t 4.5 / 3.6 / 3.1) in
2014-19 / 2020-22 / 2023-26 on the point-in-time top-200. V5's ML model has
delivery features; V7 has none.

  D1_tilt20    0.8 z(V7) + 0.2 z(deliv 20d mean - deliv 120d mean)   (V5's delivery feature)
  D2_tilt40    0.6 z(V7) + 0.4 z(deliv 20d mean - deliv 120d mean)
  D3_abn5      0.8 z(V7) + 0.2 z(5d mean of (deliv - own 60d mean))   (the 5-day signal tested)

z-scores are taken per date across the stocks that have a V7 score; a stock
with no delivery value keeps a delivery z of 0 (its V7 score alone decides).

    python -m v7.swing_v7c
"""
import numpy as np
import pandas as pd

from swing import config as cfgmod
from swing.research import record_trials, summarise
from swing.validation import metrics
from v7.swing_v7 import M30_PORT, PV, V5_OVERRIDES, WF, prepare, run_bt

LIVE = {**M30_PORT, "portfolio.top_n": 30, "risk.stop_loss": 0.10, "risk.regime_filter": True}
TILTS = {"D1_tilt20": ("d20_120", 0.2), "D2_tilt40": ("d20_120", 0.4), "D3_abn5": ("abn5", 0.2)}


def zs(m, col):
    g = m.groupby("date")[col]
    return (m[col] - g.transform("mean")) / g.transform("std")


def delivery_features(x) -> pd.DataFrame:
    """Causal delivery features (they use the signal day's own delivery %, known after
    that day's close, like every other V7 input)."""
    ents = set(x["u200"]["entity"])
    p = x["panel"][x["panel"]["entity"].isin(ents)][["date", "entity", "deliv_pct"]]
    p = p.sort_values(["entity", "date"]).copy()
    g = p.groupby("entity")["deliv_pct"]
    p["d20_120"] = g.transform(lambda s: s.rolling(20, min_periods=10).mean()) - \
        g.transform(lambda s: s.rolling(120, min_periods=60).mean())
    abn = p["deliv_pct"] - g.transform(lambda s: s.shift(1).rolling(60, min_periods=40).mean())
    p["abn5"] = abn.groupby(p["entity"]).transform(lambda s: s.rolling(5, min_periods=3).mean())
    return p[["date", "entity", "d20_120", "abn5"]]


def variants(x) -> dict:
    base = x["m200"].rename(columns={"score": "v7"})
    m = base.merge(delivery_features(x), on=["date", "entity"], how="left")
    m["zv7"] = zs(m, "v7")
    out = {}
    for name, (feat, w) in TILTS.items():
        a = m.copy()
        a["zd"] = zs(a, feat).fillna(0.0)
        a["score"] = (1 - w) * a["zv7"] + w * a["zd"]
        out[name] = a.dropna(subset=["score"])[["date", "entity", "score"]]
    return out, m


def ic_report(m, x):
    """Diagnostic only: 21-session forward IC of each delivery feature inside the walk-forward window."""
    px = x["panel"][["date", "entity", "adj_close"]].sort_values(["entity", "date"])
    px["fwd21"] = px.groupby("entity")["adj_close"].shift(-21) / px["adj_close"] - 1
    j = m.merge(px[["date", "entity", "fwd21"]], on=["date", "entity"], how="left")
    j = j[(j["date"] >= WF[0]) & (j["date"] <= WF[1])]
    for feat in ("d20_120", "abn5"):
        k = j.dropna(subset=[feat, "fwd21"])
        ic = k.groupby("date").apply(lambda d: d[feat].rank().corr(d["fwd21"].rank()) if len(d) > 30 else np.nan,
                                     include_groups=False).dropna()
        ic = ic.iloc[::21]                                           # non-overlapping months
        print(f"  21-session IC {feat:8s} (WF, monthly samples): mean {ic.mean():+.4f}  t {ic.mean() / (ic.std() / np.sqrt(len(ic))):+.2f}  months {len(ic)}")


def main():
    cfgmod.setup_logging()
    x = prepare()
    tilted, m = variants(x)
    cands = {"V7_live": x["m200"], **tilted}
    cfg = cfgmod.load({**V5_OVERRIDES, **LIVE})
    rows, rets = [], {}
    for name, sc in cands.items():
        for wl, w in (("WF", WF), ("PV", PV)):
            res = run_bt(cfg, sc, x["prices"], x["info200"], x["ctx"], w)
            s = summarise(res, x["bench"], cfg)
            rows.append({"model": name, "window": wl, "CAGR": s["CAGR"], "Sharpe": s["Sharpe"], "MaxDD": s["MaxDD"],
                         "Alpha": s["Alpha_ann"], "Turnover": s.get("Turnover_ann")})
            if wl == "WF" and name != "V7_live":
                rets[f"v7c: {name}"] = res.daily["ret"].iloc[1:]
    c = cfgmod.load(V5_OVERRIDES); c["v7_run"] = "swing_v7c"
    trials = record_trials(cfgmod.path(c, "results", "v7_swing"), c, rets)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.pivot(index="model", columns="window", values=["CAGR", "Sharpe", "MaxDD", "Alpha", "Turnover"]).round(3).to_string())
    live = t[t["model"] == "V7_live"].set_index("window")["Sharpe"]
    for name in tilted:
        s = t[t["model"] == name].set_index("window")["Sharpe"]
        keep = (s["WF"] > live["WF"]) and (s["PV"] > live["PV"])
        dsr = metrics.deflated_sharpe(rets[f"v7c: {name}"], trials)
        print(f"  {name:10s} beats live V7 in both windows: {'YES' if keep else 'no'}   "
              f"Deflated Sharpe {dsr['DSR']:.3f} over {dsr['N_trials']} trials (PSR vs 0: {dsr['PSR_vs_0']:.3f})")
    ic_report(m, x)
    print(f"trials recorded so far: {len(trials)}")


if __name__ == "__main__":
    main()
