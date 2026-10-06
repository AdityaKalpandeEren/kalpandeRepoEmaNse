"""
V3.4 research - can V3.3's real ranking skill survive costs by trading less?

    python -m backtest.ml.research_v34

Same data, split, model and exact cost model as train_v33.py (train 60% /
validation 20% / test 20% of days; model for validation fit on train, model
for test fit on train+validation). Pre-registered variants, ONE chosen on
VALIDATION by annualised return (cash idle on skipped days counts as 0%),
with at least 30 traded days:

  K          3 or 5 stocks per day
  notional   Rs 5 lakh or 10 lakh per stock (Rs 20/order brokerage weighs less)
  gate       none            - every day (= V3.3)
             conv50 / conv30 - only days whose conviction (mean predicted excess
                               of the top K minus the day's median prediction)
                               is in the top 50% / 30% of VALIDATION days
             floor           - only days whose top-K mean predicted excess
                               return covers the round-trip cost

NOTE: the test block was already viewed in the V3.2+ / V3.3 research, so it
is a re-confirmation, not an untouched test. Forward paper trading decides.
"""
import math

import numpy as np
import pandas as pd

import config
from backtest.ml.train_v33 import _t, daily_ic, feature_columns, fit
from strategy.v33_costs import net_return


def day_table(block, score, k, notional):
    """Per day: mean net of the top-k picks, conviction, top-k mean prediction."""
    d = block[["date", "entry", "exit"]].assign(score=score)
    d["net"] = [net_return(e, x, notional) for e, x in zip(d["entry"], d["exit"])]
    d = d.sort_values(["date", "score"], ascending=[True, False])
    top = d.groupby("date").head(k).groupby("date").agg(net=("net", "mean"), top_pred=("score", "mean"))
    med = d.groupby("date")["score"].median()
    top["conv"] = top["top_pred"] - med
    return top.dropna(subset=["net"])


def stats(day, traded):
    r = day["net"].where(traded, 0.0)                     # skipped day = cash, 0%
    eq = (1 + r).cumprod()
    t = day.loc[traded, "net"]
    return {"traded_days": int(traded.sum()), "of_days": len(day),
            "avg_traded%": round(t.mean() * 100, 3) if len(t) else np.nan, "t": round(_t(t), 2),
            "win_days%": round((t > 0).mean() * 100, 1) if len(t) else np.nan,
            "year%": round((eq.iloc[-1] ** (245 / len(day)) - 1) * 100, 1),
            "maxDD%": round(((eq / eq.cummax()) - 1).min() * 100, 1)}


def main():
    df = pd.read_csv(config.ML_V33_DATASET_PATH)
    df = df[df["n_day"] >= 20].dropna(subset=["net_ret", "excess"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    feats = feature_columns(df)
    dates = sorted(df["date"].unique())
    n = len(dates)
    tr_d, va_d, te_d = set(dates[:int(n * .6)]), set(dates[int(n * .6):int(n * .8)]), set(dates[int(n * .8):])
    tr, va, te = df[df.date.isin(tr_d)], df[df.date.isin(va_d)], df[df.date.isin(te_d)]
    pv = fit(tr, feats).predict(va[feats])
    pt = fit(pd.concat([tr, va]), feats).predict(te[feats])
    print(f"V3.4 research: {n} days; validation {min(va_d)}..{max(va_d)} (IC {daily_ic(va, pv).mean():+.4f}), "
          f"test {min(te_d)}..{max(te_d)} (IC {daily_ic(te, pt).mean():+.4f})")

    rows = []
    for k in (3, 5):
        for notional in (500_000, 1_000_000):
            dv, dt = day_table(va, pv, k, notional), day_table(te, pt, k, notional)
            # round-trip cost as a return, for the floor gate (typical stock price ~Rs 1000)
            cost = -net_return(1000.0, 1000.0, notional)
            cuts = {"conv50": dv["conv"].quantile(0.5), "conv30": dv["conv"].quantile(0.7)}
            gates = {"none": (pd.Series(True, index=dv.index), pd.Series(True, index=dt.index)),
                     "conv50": (dv["conv"] >= cuts["conv50"], dt["conv"] >= cuts["conv50"]),
                     "conv30": (dv["conv"] >= cuts["conv30"], dt["conv"] >= cuts["conv30"]),
                     "floor": (dv["top_pred"] >= cost, dt["top_pred"] >= cost)}
            for g, (mv, mt) in gates.items():
                sv, st = stats(dv, mv), stats(dt, mt)
                rows.append({"K": k, "notional": notional, "gate": g, "cost%": round(cost * 100, 3),
                             **{f"val_{a}": b for a, b in sv.items()}, **{f"test_{a}": b for a, b in st.items()}})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    show = ["K", "notional", "gate", "cost%", "val_traded_days", "val_avg_traded%", "val_t", "val_year%", "val_maxDD%",
            "test_traded_days", "test_avg_traded%", "test_t", "test_year%", "test_maxDD%"]
    print(t[show].to_string(index=False))
    ok = t[t["val_traded_days"] >= 30]
    best = ok.sort_values("val_year%", ascending=False).iloc[0]
    base = t[(t.K == 5) & (t.notional == 500_000) & (t.gate == "none")].iloc[0]
    print(f"\nChosen on VALIDATION: K={best.K}, Rs {best.notional:,}/stock, gate={best.gate}")
    print(f"  validation year% {best['val_year%']} vs live V3.3 (K5, 5L, none) {base['val_year%']}")
    print(f"  test (re-confirmation) year% {best['test_year%']} (t {best['test_t']}, {best['test_traded_days']} days) "
          f"vs V3.3 {base['test_year%']} (t {base['test_t']})")
    t.to_csv("backtest/ml/results_v34.csv", index=False)


if __name__ == "__main__":
    main()
