"""
V7 swing, round 3: volatility-scaled momentum (Barroso & Santa-Clara 2015,
"Momentum has its moments") on top of the live V7 settings. Fixed in advance:
target 15% and 20% a year, 126-session lookback, no leverage, exposure set at
each monthly rebalance. Not recorded as trials (an exposure overlay, not a
score) - REJECTED on 2026-10-13, see README 3e.

Pass 1: live V7 -> daily returns. Pass 2: each rebalance multiplies the target
weights by s = min(1, target / (std of pass-1 daily returns over the previous
126 sessions * sqrt(252))); the engine trades to them with its own costs.

    python -m v7.swing_v7_volscale
"""
import numpy as np
import pandas as pd

from swing import config as cfgmod
from swing.risk import sizing
from v7.swing_v7 import M30_PORT, PV, V5_OVERRIDES, WF, prepare, run_bt

LIVE = {**M30_PORT, "portfolio.top_n": 30, "risk.stop_loss": 0.10, "risk.regime_filter": True}
TARGETS = (0.15, 0.20)


def stats(r, bench, lo, hi):
    r = r.loc[lo:hi]
    b = bench.reindex(r.index).fillna(0)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    crash = (1 + r.loc["2024-09-27":"2025-03-14"]).prod() - 1
    return (f"CAGR {(eq.iloc[-1] ** (1 / yrs) - 1) * 100:5.1f}%  "
            f"Sharpe {(r.mean() * 252 - 0.065) / (r.std() * np.sqrt(252)):4.2f}  "
            f"MaxDD {(eq / eq.cummax() - 1).min() * 100:6.1f}%  vol {r.std() * np.sqrt(252) * 100:4.1f}%  "
            f"Sep24-Mar25 {crash * 100:+5.1f}%  NIFTY CAGR {((1 + b).prod() ** (1 / yrs) - 1) * 100:4.1f}%")


def main():
    cfgmod.setup_logging()
    x = prepare()
    cfg = cfgmod.load({**V5_OVERRIDES, **LIVE})
    full = (WF[0], PV[1])
    base = run_bt(cfg, x["m200"], x["prices"], x["info200"], x["ctx"], full)
    r1 = base.daily["ret"]
    reb = list(pd.to_datetime(base.targets["date"]))
    out = {"V7 live (no vol scaling)": r1}
    orig = sizing.Exposure.scale
    for tgt in TARGETS:
        vol = r1.rolling(126, min_periods=60).std() * np.sqrt(252)
        s = (tgt / vol).clip(upper=1.0).fillna(1.0)
        it = iter([float(s.loc[:d].iloc[-1]) for d in reb])     # scale() runs once per rebalance, in order

        def scale(self, equity, ctx=None, _it=it):
            return orig(self, equity, ctx) * next(_it, 1.0)
        sizing.Exposure.scale = scale
        try:
            res = run_bt(cfg, x["m200"], x["prices"], x["info200"], x["ctx"], full)
        finally:
            sizing.Exposure.scale = orig
        out[f"V7 + vol target {int(tgt * 100)}%"] = res.daily["ret"]
    for nm, (lo, hi) in (("WALK-FORWARD 2017-01..2025-06", WF), ("PREVIOUSLY VIEWED 2025-07..2026-09", PV)):
        print(f"\n{nm}")
        for k, r in out.items():
            print(f"  {k:26s} {stats(r, x['bench'], lo, hi)}")
    print("\nby year: " + " | ".join(out))
    for y in range(2017, 2027):
        print(f"  {y}: " + " | ".join(f"{((1 + r[r.index.year == y]).prod() - 1) * 100:+6.1f}%" for r in out.values()))


if __name__ == "__main__":
    main()
