"""
V7 swing, round 2: four literature-based score variants on top of the live V7
settings (top 30 of the top-200, monthly, 10% stop, regime). Fixed in advance,
all reported, all counted as trials. Keep rule: beat live V7 in BOTH windows.

  A skip_month   0.5 z(6-1m / vol) + 0.5 z(12-1m / vol)      (Jegadeesh 1990)
  B high52       50/50 rank blend of V7 score and price / 52-week high (George & Hwang 2004)
  C residual     12-1m momentum of NIFTY-beta residual returns / residual vol (Blitz, Huij & Martens 2011)
  D volcap       V7 score, top-decile 1-year volatility excluded

    python -m v7.swing_v7b
"""
import numpy as np
import pandas as pd

from swing import config as cfgmod
from swing.research import record_trials, summarise
from v7.swing_v7 import M30_PORT, PV, V5_OVERRIDES, WF, prepare, run_bt

LIVE = {**M30_PORT, "portfolio.top_n": 30, "risk.stop_loss": 0.10, "risk.regime_filter": True}


def zs(m, col):
    g = m.groupby("date")[col]
    return (m[col] - g.transform("mean")) / g.transform("std")


def variants(x):
    u, bench = x["u200"], x["bench"]
    p = x["panel"][x["panel"]["entity"].isin(set(u["entity"]))][["date", "entity", "adj_close", "ret_cc"]]
    p = p.sort_values(["entity", "date"]).copy()
    g = p.groupby("entity")
    lr = np.log1p(p["ret_cc"].clip(lower=-0.95))
    p["vol"] = lr.groupby(p["entity"]).transform(lambda s: s.rolling(252, min_periods=200).std()) * np.sqrt(252)
    p["m6_1"] = g["adj_close"].shift(21) / g["adj_close"].shift(126) - 1
    p["m12_1"] = g["adj_close"].shift(21) / g["adj_close"].shift(252) - 1
    p["hi52"] = p["adj_close"] / g["adj_close"].transform(lambda s: s.rolling(252, min_periods=200).max())
    mkt = p["date"].map(bench).fillna(0.0)
    r = p["ret_cc"].fillna(0.0)
    cov = (r * mkt).groupby(p["entity"]).transform(lambda s: s.rolling(252, min_periods=200).mean()) \
        - r.groupby(p["entity"]).transform(lambda s: s.rolling(252, min_periods=200).mean()) \
        * mkt.groupby(p["entity"]).transform(lambda s: s.rolling(252, min_periods=200).mean())
    var = mkt.groupby(p["entity"]).transform(lambda s: s.rolling(252, min_periods=200).var())
    p["res"] = r - (cov / var) * mkt
    rs = p.groupby("entity")["res"]
    p["res_mom"] = rs.transform(lambda s: s.shift(21).rolling(231, min_periods=180).sum()) / \
        rs.transform(lambda s: s.rolling(252, min_periods=200).std())
    m = u.merge(p, on=["date", "entity"], how="left")
    base = x["m200"].rename(columns={"score": "v7"})
    m = m.merge(base, on=["date", "entity"], how="left")
    out = {}
    a = m.dropna(subset=["m6_1", "m12_1", "vol"]).copy()
    a["m6"], a["m12"] = a["m6_1"] / a["vol"], a["m12_1"] / a["vol"]
    a["score"] = 0.5 * zs(a, "m6") + 0.5 * zs(a, "m12")
    out["A_skip_month"] = a[["date", "entity", "score"]]
    b = m.dropna(subset=["v7", "hi52"]).copy()
    b["score"] = 0.5 * b.groupby("date")["v7"].rank(pct=True) + 0.5 * b.groupby("date")["hi52"].rank(pct=True)
    out["B_high52"] = b[["date", "entity", "score"]]
    c = m.dropna(subset=["res_mom"]).copy()
    c["score"] = c["res_mom"]
    out["C_residual"] = c[["date", "entity", "score"]]
    d = m.dropna(subset=["v7", "vol"]).copy()
    d = d[d["vol"] < d.groupby("date")["vol"].transform(lambda s: s.quantile(0.9))]
    d["score"] = d["v7"]
    out["D_volcap"] = d[["date", "entity", "score"]]
    return out


def main():
    cfgmod.setup_logging()
    x = prepare()
    cands = {"V7_live": x["m200"], **variants(x)}
    cfg = cfgmod.load({**V5_OVERRIDES, **LIVE})
    rows, rets = [], {}
    for name, sc in cands.items():
        for wl, w in (("WF", WF), ("PV", PV)):
            res = run_bt(cfg, sc, x["prices"], x["info200"], x["ctx"], w)
            s = summarise(res, x["bench"], cfg)
            rows.append({"model": name, "window": wl, "CAGR": s["CAGR"], "Sharpe": s["Sharpe"], "MaxDD": s["MaxDD"],
                         "Alpha": s["Alpha_ann"]})
            if wl == "WF" and name != "V7_live":
                rets[f"v7b: {name}"] = res.daily["ret"].iloc[1:]
    c = cfgmod.load(V5_OVERRIDES); c["v7_run"] = "swing_v7b"
    trials = record_trials(cfgmod.path(c, "results", "v7_swing"), c, rets)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.pivot(index="model", columns="window", values=["CAGR", "Sharpe", "MaxDD", "Alpha"]).round(3).to_string())
    print(f"trials recorded so far: {len(trials)}")


if __name__ == "__main__":
    main()
