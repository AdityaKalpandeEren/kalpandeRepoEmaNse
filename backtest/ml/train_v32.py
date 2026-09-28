"""
Trains and evaluates L_ML_V32 - the daily 9:45 stock-ranking model -
from config.ML_V32_DATASET_PATH (backtest/ml/build_dataset_v32.py).

    python -m backtest.ml.train_v32

PROTOCOL (same honesty rules as V2/V3):
  - split BY DATE: first 60% of trading days train, next 20% validation,
    last 20% test
  - the model type AND K (stocks bought per day, config.ML_V32_TOP_K_OPTIONS)
    are chosen on VALIDATION only; the test block is scored once
  - each DAY is one observation: the day's return = average net return of
    the K stocks bought at 9:45 and sold at 15:15 (costs + 2% emergency
    stop already in the label). Day returns are independent, so the
    t-stats here are honest (unlike per-trade t-stats on clustered trades)

The model predicts `excess` - a stock's net return minus the day's
universe average - i.e. it only has to RANK stocks, not call the market.

TEST REPORT compares, on the same days:
  - V3.2 top-K
  - EQUAL-WEIGHT every stock (what "no model" earns)
  - a ONE-LINE RULE: top-K by first-30-min strength vs NIFTY - if V3.2
    can't beat this, the ML adds nothing over a simple scan
  - NIFTY itself 9:45 -> 15:15
plus the ranking skill (information coefficient: rank correlation of
prediction vs realised excess, per day) and a decile table (does the
realised return rise with the predicted score?).
"""
import json
import math
import os

import numpy as np
import pandas as pd

import config

BOOKKEEPING = ["symbol", "date", "sector", "entry", "exit", "outcome", "net_ret",
               "nifty_ret_945_1515", "day_mean", "excess", "n_day"]


def _t(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def day_returns(block: pd.DataFrame, score: np.ndarray, k: int) -> pd.Series:
    """Per day: mean net return of the top-k stocks by `score`."""
    d = block[["date", "net_ret"]].assign(score=score)
    d = d.sort_values(["date", "score"], ascending=[True, False])
    return d.groupby("date").head(k).groupby("date")["net_ret"].mean()


def daily_ic(block: pd.DataFrame, score: np.ndarray) -> pd.Series:
    d = block[["date", "excess"]].assign(score=score)
    return d.groupby("date").apply(
        lambda g: g["score"].rank().corr(g["excess"].rank()) if len(g) > 5 else np.nan,
        include_groups=False)


def summarize(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"days": 0}
    eq = (1 + r).cumprod()
    return {"days": len(r), "avg_day%": round(r.mean() * 100, 3), "days+%": round((r > 0).mean() * 100, 1),
            "t_day": round(_t(r), 2), "total%": round((eq.iloc[-1] - 1) * 100, 1),
            "maxDD%": round(((eq / eq.cummax()) - 1).min() * 100, 1)}


def fmt(name, s) -> str:
    if not s.get("days"):
        return f"  {name:<40} no days"
    return (f"  {name:<40} days={s['days']:<5} avg/day={s['avg_day%']:>7}%  days+={s['days+%']:>5}%  "
            f"t={s['t_day']:>6}  total={s['total%']:>8}%  maxDD={s['maxDD%']:>7}%")


def models():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "hgb_small": lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=300, l2_regularization=5.0,
            max_features=0.5, early_stopping=False, random_state=42),
        "hgb_medium": lambda: HistGradientBoostingRegressor(
            max_depth=5, learning_rate=0.03, max_iter=400, min_samples_leaf=150, l2_regularization=3.0,
            max_features=0.5, early_stopping=False, random_state=42),
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50.0)),
    }


def main():
    df = pd.read_csv(config.ML_V32_DATASET_PATH)
    df = df[df["n_day"] >= 20].copy()          # need a real cross-section to rank
    df = df.dropna(subset=["net_ret", "excess"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    feats = [c for c in df.columns if c not in BOOKKEEPING]
    # Robust target: clip extreme excess returns (circuit moves, data errors)
    # so a handful of outliers don't dominate the regression.
    lo, hi = df["excess"].quantile([0.005, 0.995])
    df["target"] = df["excess"].clip(lo, hi)

    dates = sorted(df["date"].unique())
    n = len(dates)
    tr_d, va_d, te_d = dates[: int(n * 0.6)], dates[int(n * 0.6): int(n * 0.8)], dates[int(n * 0.8):]
    tr, va, te = (df[df["date"].isin(set(x))] for x in (tr_d, va_d, te_d))
    print(f"Rows {len(df)} | {n} trading days | {df['symbol'].nunique()} stocks | {len(feats)} features")
    print(f"  TRAIN {len(tr_d)} days {tr_d[0]} .. {tr_d[-1]}")
    print(f"  VAL   {len(va_d)} days {va_d[0]} .. {va_d[-1]}")
    print(f"  TEST  {len(te_d)} days {te_d[0]} .. {te_d[-1]}\n")

    print("=" * 110)
    print("SELECTION ON VALIDATION (model x K), scored by day-level t of the top-K net return")
    print("=" * 110)
    print(fmt("equal-weight all stocks", summarize(va.groupby("date")["net_ret"].mean())))
    best = None
    for name, factory in models().items():
        m = factory().fit(tr[feats], tr["target"])
        pv = m.predict(va[feats])
        ic = daily_ic(va, pv)
        for k in config.ML_V32_TOP_K_OPTIONS:
            s = summarize(day_returns(va, pv, k))
            print(fmt(f"{name} top-{k} (IC {ic.mean():+.3f}, t {_t(ic):.1f})", s))
            if best is None or s["t_day"] > best[2]["t_day"]:
                best = (name, k, s)
    name, k, _ = best
    print(f"\nChosen on validation: {name}, top-{k}")

    factory = models()[name]
    trva = pd.concat([tr, va])
    m = factory().fit(trva[feats], trva["target"])
    pt = m.predict(te[feats])
    ic = daily_ic(te, pt)
    v32 = day_returns(te, pt, k)
    eqw = te.groupby("date")["net_ret"].mean()
    rule = day_returns(te, te["rs30_vs_nifty"].fillna(-9).to_numpy(), k)
    nifty = te.groupby("date")["nifty_ret_945_1515"].first()
    bottom = day_returns(te, -pt, k)

    print("\n" + "=" * 110)
    print("OUT-OF-SAMPLE TEST (never used for any choice) - one row per day, net of NSE intraday costs")
    print("=" * 110)
    s_v32 = summarize(v32)
    print(fmt(f"V3.2 {name} top-{k}", s_v32))
    print(fmt(f"V3.2 BOTTOM-{k} (should be worst)", summarize(bottom)))
    print(fmt("equal-weight all stocks (no model)", summarize(eqw)))
    print(fmt(f"one-line rule: top-{k} by 30-min RS vs NIFTY", summarize(rule)))
    print(fmt("NIFTY 9:45->15:15 (no costs)", summarize(nifty)))
    print(f"\n  Ranking skill (IC): mean {ic.mean():+.4f}, t {_t(ic):.2f}, positive on "
          f"{(ic > 0).mean()*100:.0f}% of days")
    print(f"  V3.2 minus equal-weight: {(v32 - eqw).mean()*100:+.3f}%/day, t {_t((v32 - eqw).dropna()):.2f}")
    print(f"  V3.2 minus one-line rule: {(v32 - rule).mean()*100:+.3f}%/day, t {_t((v32 - rule).dropna()):.2f}")

    te2 = te.assign(pred=pt)
    te2["decile"] = te2.groupby("date")["pred"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) if len(x) >= 10 else np.nan)
    dec = te2.groupby("decile").agg(avg_net=("net_ret", "mean"), avg_excess=("excess", "mean"), n=("net_ret", "size"))
    print("\n  Decile of predicted score (0 = lowest) -> realised average (should rise left to right):")
    print("  " + "  ".join(f"D{int(i)}:{v*100:+.3f}%" for i, v in dec["avg_excess"].items()))

    by_month = v32.groupby(pd.to_datetime(v32.index).to_period("M")).mean()
    print("\n  V3.2 average day return by month (%): " +
          " ".join(f"{str(p)}:{v*100:+.2f}" for p, v in by_month.items()))

    if s_v32["days"] and s_v32["avg_day%"] > 0 and s_v32["t_day"] >= 2 and _t((v32 - eqw).dropna()) >= 2:
        verdict = "POSITIVE: profitable after costs AND significantly better than equal-weight on the test block."
    elif s_v32["days"] and s_v32["avg_day%"] > 0:
        verdict = "POSITIVE but not statistically decisive on the test block - paper-test before any money."
    elif ic.mean() > 0 and _t(ic) >= 2:
        verdict = ("HAS RANKING SKILL (IC significant) but the top-K still loses after costs - the edge is "
                   "smaller than intraday charges.")
    else:
        verdict = "NO EDGE on the test block."
    print(f"\nVERDICT: {verdict}")

    try:
        from sklearn.inspection import permutation_importance
        m_tr = factory().fit(tr[feats], tr["target"])
        sample = va.sample(min(len(va), 20000), random_state=1)
        imp = permutation_importance(m_tr, sample[feats], sample["target"], n_repeats=3, random_state=1)
        order = np.argsort(-imp.importances_mean)[:15]
        print("\nTop features (permutation importance on validation):")
        for i in order:
            print(f"  {feats[i]:<26}{imp.importances_mean[i]:+.5f}")
    except Exception as e:
        print(f"(importance skipped: {e!r})")

    import joblib
    final = factory().fit(df[feats], df["target"])
    os.makedirs(os.path.dirname(config.ML_V32_MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final, "features": feats, "k": k, "config": name,
                 "trained_through": dates[-1]}, config.ML_V32_MODEL_PATH)
    meta = {"chosen": {"model": name, "k": k}, "test": s_v32, "test_ic_mean": float(ic.mean()),
            "test_ic_t": _t(ic), "equal_weight": summarize(eqw), "rule": summarize(rule),
            "test_range": [te_d[0], te_d[-1]], "verdict": verdict}
    with open(os.path.splitext(config.ML_V32_MODEL_PATH)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nSaved: {config.ML_V32_MODEL_PATH} (final fit on all days)")


if __name__ == "__main__":
    main()
