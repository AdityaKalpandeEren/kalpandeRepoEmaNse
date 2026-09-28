"""
Trains the LIVE V3.3 model (L_ML_V33): same-day V3.2 ranking + NSE data.

    python -m backtest.ml.build_dataset_v33      # if the dataset is stale
    python -m backtest.ml.train_v33

Configuration (fixed; see docs/NSE_RESEARCH_ML.md "V3.3"):
  - horizon: SAME DAY (buy at the 9:45 open, sell at 15:15, 2% stop) - the
    3- and 5-day versions showed no ranking skill (research_v33_grid.py)
  - inputs: the V3.2 features + NSE delivery, F&O OI/PCR, corporate events
  - model: hgb_small regressor on the day's excess return
  - K (stocks per day): 3 or 5, chosen on the VALIDATION block

Reported on validation and test with the EXACT live cost model
(strategy/v33_costs.py: Rs config.V33_BROKERAGE_PER_ORDER per order + GST,
STT, exchange, SEBI, stamp, slippage) at several notional sizes.

NOTE: the test block (last 20% of days) was already looked at during the
V3.2+ research, so its numbers here are a re-confirmation, not a first,
untouched test. Forward paper trading (live_v33.py) is the real test.

The saved model is refit on ALL days so live uses the most recent data.
"""
import json
import math
import os

import numpy as np
import pandas as pd

import config
from backtest.ml.train_v32 import BOOKKEEPING, models
from strategy.v33_costs import net_return

LABELS = ["net_h3", "net_h5", "excess_h3", "excess_h5"]
NSE_PREFIXES = ("nse_", "fo_", "fut_", "pcr", "opt_", "call_", "put_", "oi_", "ann_", "bm_", "ins_")
SIZES = (100_000, 200_000, 500_000, 1_000_000)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in set(BOOKKEEPING) | set(LABELS)]


def _t(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def fit(train, feats):
    lo, hi = train["excess"].quantile([.005, .995])
    return models()["hgb_small"]().fit(train[feats], train["excess"].clip(lo, hi))


def evaluate(block, score, k, notional):
    d = block[["date", "entry", "exit"]].assign(score=score)
    d["net"] = [net_return(e, x, notional) for e, x in zip(d["entry"], d["exit"])]
    d = d.sort_values(["date", "score"], ascending=[True, False])
    day = d.groupby("date").head(k).groupby("date")["net"].mean().dropna()
    eq = (1 + day).cumprod()
    return {"days": len(day), "avg_day%": round(day.mean() * 100, 3), "t": round(_t(day), 2),
            "days+%": round((day > 0).mean() * 100, 1),
            "year%": round((eq.iloc[-1] ** (245 / len(day)) - 1) * 100, 1),
            "maxDD%": round(((eq / eq.cummax()) - 1).min() * 100, 1)}


def daily_ic(block, score):
    d = block[["date", "excess"]].assign(s=score)
    return d.groupby("date").apply(lambda g: g["s"].rank().corr(g["excess"].rank()), include_groups=False)


def main():
    df = pd.read_csv(config.ML_V33_DATASET_PATH)
    df = df[df["n_day"] >= 20].dropna(subset=["net_ret", "excess"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    feats = feature_columns(df)
    dates = sorted(df["date"].unique())
    n = len(dates)
    tr_d, va_d, te_d = set(dates[:int(n * .6)]), set(dates[int(n * .6):int(n * .8)]), set(dates[int(n * .8):])
    tr, va, te = df[df.date.isin(tr_d)], df[df.date.isin(va_d)], df[df.date.isin(te_d)]
    print(f"V3.3: {len(df)} rows, {n} days, {len(feats)} features "
          f"({sum(c.startswith(NSE_PREFIXES) for c in feats)} NSE)")

    m_va = fit(tr, feats)
    pv = m_va.predict(va[feats])
    ic = daily_ic(va, pv)
    notional = config.V33_NOTIONAL_PER_STOCK
    print(f"\nVALIDATION {min(va_d)}..{max(va_d)}  (IC {ic.mean():+.4f}, t {_t(ic):.1f}) at Rs {notional:,.0f}/stock:")
    best = None
    for k in (3, 5):
        s = evaluate(va, pv, k, notional)
        print(f"  top-{k}: {s}")
        if best is None or s["t"] > best[1]["t"]:
            best = (k, s)
    k = best[0]
    print(f"Chosen K on validation: {k}")

    m_te = fit(pd.concat([tr, va]), feats)
    pt = m_te.predict(te[feats])
    ict = daily_ic(te, pt)
    print(f"\nTEST {min(te_d)}..{max(te_d)} (re-confirmation; IC {ict.mean():+.4f}, t {_t(ict):.1f}), top-{k}:")
    res = {}
    for size in SIZES:
        s = evaluate(te, pt, k, size)
        res[size] = s
        print(f"  Rs {size:>9,}/stock: {s}")

    final = fit(df, feats)
    os.makedirs(os.path.dirname(config.ML_V33_MODEL_PATH), exist_ok=True)
    import joblib
    joblib.dump({"model": final, "features": feats, "k": k, "horizon": 1, "trained_through": dates[-1],
                 "name": "L_ML_V33"}, config.ML_V33_MODEL_PATH)
    with open(os.path.splitext(config.ML_V33_MODEL_PATH)[0] + "_meta.json", "w") as f:
        json.dump({"k": k, "validation": best[1], "test_by_size": {str(a): b for a, b in res.items()},
                   "test_ic": float(ict.mean()), "trained_through": dates[-1]}, f, indent=2)
    print(f"\nSaved {config.ML_V33_MODEL_PATH} (trained on all {n} days through {dates[-1]}, K={k})")


if __name__ == "__main__":
    main()
