"""
V3.3 RESEARCH GRID (formerly V3.2+ trainer): the V3.2 daily ranking model with NSE data (delivery, F&O
OI/PCR, corporate events) and a choice of holding period.

    python -m backtest.ml.research_v33_grid

Grid, chosen on VALIDATION only (test scored once, same date split as
train_v32.py: 60 / 20 / 20 % of trading days):
  horizon  1 = buy 9:45, sell 15:15 same day (intraday costs)
           3 = buy 9:45, hold to the close of the 3rd trading day
           5 = ... 5th trading day   (NSE delivery costs 0.20%/side)
  inputs   "v32"      the original V3.2 features
           "v32+nse"  + delivery, unusual volume, F&O OI/PCR, events
  K        3 or 5 stocks per day
Model: hgb_small (the V3.2 validation winner), target = excess return
over the day's universe average for that horizon.

Portfolio for multi-day horizons: capital is split into h sleeves; each
trading day one sleeve buys that day's top-K for h days (staggered, so
every day's picks are used and the average hold is h days).

Statistics use NON-OVERLAPPING cohorts (every h-th day) for t-stats, so
overlapping holds don't fake significance.

Also printed: each NSE feature's own predictive power (rank IC with the
next-horizon excess return, averaged per day over train+validation) -
the answer to "is this data useful?" independent of the model.
"""
import json
import math
import os

import numpy as np
import pandas as pd

import config
from backtest.ml.train_v32 import BOOKKEEPING, models

HORIZONS = (1, 3, 5)
LABEL_COLS = ["net_h3", "net_h5", "excess_h3", "excess_h5"]


def _t(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def cols(h):
    return ("net_ret", "excess") if h == 1 else (f"net_h{h}", f"excess_h{h}")


def cohort_returns(block, score, k, h):
    net, _ = cols(h)
    d = block[["date", net]].assign(score=score).dropna(subset=[net])
    d = d.sort_values(["date", "score"], ascending=[True, False])
    return d.groupby("date").head(k).groupby("date")[net].mean()


def stats(coh: pd.Series, h: int) -> dict:
    coh = coh.dropna()
    if coh.empty:
        return {"n": 0}
    non_overlap = coh.iloc[::h]
    # staggered sleeves: each day's cohort gets 1/h of capital -> daily P&L ~ cohort/h
    per_day = coh / h
    eq = (1 + per_day).cumprod()
    years = len(per_day) / 245
    return {"cohorts": len(coh), "avg_hold_ret%": round(coh.mean() * 100, 3),
            "per_day%": round(per_day.mean() * 100, 3), "win%": round((coh > 0).mean() * 100, 1),
            "t": round(_t(non_overlap), 2), "cagr%": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1) if years > 0 else 0,
            "maxDD%": round(((eq / eq.cummax()) - 1).min() * 100, 1)}


def fmt(name, s):
    if not s.get("cohorts"):
        return f"  {name:<46} no data"
    return (f"  {name:<46} avg/hold={s['avg_hold_ret%']:>7}%  per-day={s['per_day%']:>7}%  win={s['win%']:>5}%  "
            f"t={s['t']:>6}  CAGR~{s['cagr%']:>6}%  maxDD={s['maxDD%']:>6}%")


def daily_ic(block, score, target):
    d = block[["date", target]].assign(s=score).dropna()
    ic = d.groupby("date").apply(lambda g: g["s"].rank().corr(g[target].rank()) if len(g) > 5 else np.nan,
                                 include_groups=False)
    return ic


def main():
    df = pd.read_csv(config.ML_V33_DATASET_PATH)
    df = df[df["n_day"] >= 20].dropna(subset=["net_ret", "excess"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    exclude = set(BOOKKEEPING) | set(LABEL_COLS)
    all_feats = [c for c in df.columns if c not in exclude]
    nse_feats = [c for c in all_feats if c.startswith(("nse_", "fo_", "fut_", "pcr", "opt_", "call_", "put_",
                                                       "oi_", "ann_", "bm_", "ins_"))]
    base_feats = [c for c in all_feats if c not in nse_feats]
    dates = sorted(df["date"].unique())
    n = len(dates)
    tr_d, va_d, te_d = set(dates[: int(n * .6)]), set(dates[int(n * .6): int(n * .8)]), set(dates[int(n * .8):])
    tr, va, te = df[df.date.isin(tr_d)], df[df.date.isin(va_d)], df[df.date.isin(te_d)]
    print(f"{len(df)} rows | {n} days | base features {len(base_feats)} + NSE features {len(nse_feats)}")
    print(f"TRAIN {min(tr_d)}..{max(tr_d)} | VAL {min(va_d)}..{max(va_d)} | TEST {min(te_d)}..{max(te_d)}\n")

    # ---- is the NSE data useful on its own? (train + validation only) ----
    trva = pd.concat([tr, va])
    print("=" * 112)
    print("NSE FEATURE USEFULNESS - rank IC with next-horizon excess return (train+val days; |t| >= 2 = real signal)")
    print("=" * 112)
    print(f"  {'feature':<24}{'coverage':>9}   " + "   ".join(f"h{h}: IC      t" for h in HORIZONS))
    for f in nse_feats:
        cells = []
        for h in HORIZONS:
            ic = daily_ic(trva, trva[f].to_numpy(float), cols(h)[1])
            cells.append(f"{ic.mean():+.4f} {_t(ic):6.2f}")
        print(f"  {f:<24}{trva[f].notna().mean()*100:8.0f}%   " + "   ".join(cells))

    # ---- grid on validation ----
    print("\n" + "=" * 112)
    print("SELECTION ON VALIDATION (horizon x inputs x K)")
    print("=" * 112)
    factory = models()["hgb_small"]
    best = None
    for h in HORIZONS:
        net, exc = cols(h)
        trh = tr.dropna(subset=[exc])
        lo, hi = trh[exc].quantile([.005, .995])
        print(fmt(f"h{h} equal-weight (no model)", stats(va.groupby("date")[net].mean(), h)))
        for fs_name, fs in (("v32", base_feats), ("v32+nse", base_feats + nse_feats)):
            m = factory().fit(trh[fs], trh[exc].clip(lo, hi))
            pv = m.predict(va[fs])
            ic = daily_ic(va, pv, exc)
            for k in (3, 5):
                s = stats(cohort_returns(va, pv, k, h), h)
                print(fmt(f"h{h} {fs_name:<8} top-{k} (IC {ic.mean():+.3f})", s))
                if s.get("cohorts") and (best is None or s["t"] > best[4]["t"]):
                    best = (h, fs_name, fs, k, s)
    h, fs_name, fs, k, _ = best
    print(f"\nChosen on validation: horizon {h} day(s), inputs {fs_name}, top-{k}")

    # ---- test once ----
    net, exc = cols(h)
    trvah = trva.dropna(subset=[exc])
    lo, hi = trvah[exc].quantile([.005, .995])
    m = factory().fit(trvah[fs], trvah[exc].clip(lo, hi))
    pt = m.predict(te[fs])
    ic = daily_ic(te, pt, exc)
    print("\n" + "=" * 112)
    print(f"OUT-OF-SAMPLE TEST ({min(te_d)} .. {max(te_d)}) - horizon {h}, net of costs")
    print("=" * 112)
    s = stats(cohort_returns(te, pt, k, h), h)
    print(fmt(f"V3.3 {fs_name} top-{k}", s))
    print(fmt(f"V3.3 bottom-{k}", stats(cohort_returns(te, -pt, k, h), h)))
    print(fmt("equal-weight all stocks", stats(te.groupby("date")[net].mean(), h)))
    print(fmt(f"one-line rule top-{k} by 30-min RS vs NIFTY",
              stats(cohort_returns(te, te["rs30_vs_nifty"].fillna(-9).to_numpy(), k, h), h)))
    print(f"\n  Ranking skill (IC) on test: {ic.mean():+.4f}, t {_t(ic):.2f}, positive {100*(ic > 0).mean():.0f}% of days")
    # also show the other horizons on test for the chosen inputs, descriptive only
    print("\n  Descriptive (NOT used for any choice): same inputs/K on test at every horizon")
    for hh in HORIZONS:
        n2, e2 = cols(hh)
        d2 = trva.dropna(subset=[e2])
        lo2, hi2 = d2[e2].quantile([.005, .995])
        m2 = factory().fit(d2[fs], d2[e2].clip(lo2, hi2))
        print(fmt(f"    h{hh} {fs_name} top-{k}", stats(cohort_returns(te, m2.predict(te[fs]), k, hh), hh)))

    verdict = ("POSITIVE and significant on the test block" if s.get("per_day%", 0) > 0 and s.get("t", 0) >= 2
               else "positive but not significant" if s.get("per_day%", 0) > 0 else "NO EDGE on the test block")
    print(f"\nVERDICT: {verdict}")
    import joblib
    full = df.dropna(subset=[exc])
    lo, hi = full[exc].quantile([.005, .995])
    final = factory().fit(full[fs], full[exc].clip(lo, hi))
    path = "backtest/ml/model/research_v33_grid.joblib"
    joblib.dump({"model": final, "features": fs, "horizon": h, "k": k, "inputs": fs_name,
                 "trained_through": dates[-1]}, path)
    with open(os.path.splitext(path)[0] + "_meta.json", "w") as f:
        json.dump({"chosen": {"horizon": h, "inputs": fs_name, "k": k}, "test": s,
                   "test_ic": float(ic.mean()), "test_ic_t": _t(ic), "verdict": verdict}, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
