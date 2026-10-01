"""
V5.0 research run - the "learn from everything" upgrade of V1-V4.

    python -m v5.run_v5                 # walk-forward + previously-viewed period + report + final model
    python -m v5.run_v5 --set v5.halflife=10

Pre-registered candidates (all smoothed with the same EWMA, all counted in
the Deflated Sharpe together with every V4 trial in swing/results/trials.json):
  v5_ml      ML only: LightGBM (3 seeds) + XGBoost on the blended residual target
  v5_factor  52-week-high proximity + 12-1 momentum (no ML)
  v5_combo   50/50 rank blend of v5_ml and v5_factor      <- primary hypothesis
Ablations reported (also counted as trials): v5_combo without smoothing,
breadth 10 / 30, smoothing half-life 2 / 10.
Selection rule: highest walk-forward net Sharpe among the three candidates.

Honesty: V4 already evaluated 2025-07-01 -> 2026-09-30 once, so that period
is reported here as "PREVIOUSLY VIEWED", not as a clean holdout. The real
test of V5 is forward paper trading (python -m v5.paper).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from swing import config as cfgmod
from swing.backtest import engine
from swing.features import build as fbuild
from swing.models import gbm
from swing.reports import html
from swing.research import (backtest, bench_returns, dataset, load_data, record_trials, shap_importance,
                             summarise)
from swing.validation import metrics, walkforward
from v5 import features as v5f

log = logging.getLogger("v5")
CANDIDATES = ("v5_ml", "v5_factor", "v5_combo")
V5_DEFAULTS = {"halflife": 5, "seeds": [42, 7, 99], "combo_weight_ml": 0.5}


def build_v5_dataset(cfg: dict, data: dict) -> pd.DataFrame:
    cache = cfgmod.path(cfg, "cache")
    f = os.path.join(cache, f"v5_dataset_{data['end']:%Y%m%d}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    df = dataset(cfg, data)
    panel = data["panel"]
    ents = set(df["entity"])
    extra = v5f.extra_stock_features(panel, ents)
    extra = pd.concat([panel.loc[extra.index, ["date", "entity"]], extra], axis=1)
    df = df.merge(extra, on=["date", "entity"], how="left")
    for c in [c for c in extra.columns if c not in ("date", "entity")]:
        df[f"xs_{c}"] = df.groupby("date")[c].rank(pct=True)
    tg = v5f.residual_targets(df, panel)
    df = df.drop(columns=["label_end"]).merge(tg, on=["date", "entity"], how="left")
    # gbm/walkforward read these names: the learning target is the blended residual rank
    df["fwd_rank"] = df["target"]
    df["fwd_ret"] = df["target"]
    df.to_parquet(f, index=False)
    return df


def feature_list(df: pd.DataFrame) -> list[str]:
    drop = {"target", "fwd_5", "fwd_10", "fwd_20", "fwd_excess"}
    return [c for c in fbuild.feature_columns(df) if c not in drop]


def to_rank(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["score"] = s.groupby("date")["score"].rank(pct=True)
    return s


def smooth(s: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Per-entity EWMA of the per-day score rank over the dates it was scored
    (causal: uses scores <= t). halflife <= 0 = no smoothing."""
    s = to_rank(s).sort_values(["entity", "date"])
    if halflife and halflife > 0:
        s["score"] = s.groupby("entity")["score"].transform(lambda x: x.ewm(halflife=halflife).mean())
    return s.sort_values(["date", "entity"]).reset_index(drop=True)


def blend(a: pd.DataFrame, b: pd.DataFrame, wa: float) -> pd.DataFrame:
    m = to_rank(a).merge(to_rank(b), on=["date", "entity"], suffixes=("_a", "_b"))
    m["score"] = wa * m["score_a"] + (1 - wa) * m["score_b"]
    return m[["date", "entity", "score"]]


def fit_ml(train: pd.DataFrame, feats: list[str], cfg: dict, v5: dict) -> dict:
    """LightGBM tuned in-fold (Optuna, swing.models.gbm) then refit with extra
    seeds; XGBoost tuned in-fold. Returns fitted models."""
    models = {}
    m, info = gbm.tune_and_fit("lgbm_reg", train, feats, cfg)
    models["lgbm_42"] = m
    for sd in v5["seeds"][1:]:
        m2 = gbm._make("lgbm_reg", info["params"], sd)
        m2.set_params(n_estimators=max(int(info["n_trees"] * 1.1), 20))
        m2.fit(train[feats], train["fwd_rank"].values)
        models[f"lgbm_{sd}"] = m2
    mx, _ = gbm.tune_and_fit("xgb_reg", train, feats, cfg)
    models["xgb"] = mx
    return models


def predict_ml(models: dict, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    parts = []
    for name, m in models.items():
        p = pd.DataFrame({"date": df["date"].values, "entity": df["entity"].values, "score": m.predict(df[feats])})
        parts.append(to_rank(p).set_index(["date", "entity"])["score"])
    lg = pd.concat([p for n, p in zip(models, parts) if n.startswith("lgbm")], axis=1).mean(axis=1)
    xg = parts[list(models).index("xgb")]
    return pd.concat([lg, xg], axis=1).mean(axis=1).rename("score").reset_index()


def wf_ml_scores(cfg, df, feats, v5, out_dir) -> pd.DataFrame:
    f = os.path.join(out_dir, "wf_scores_v5_ml_raw.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    parts = []
    for fold in walkforward.folds(pd.DatetimeIndex(df["date"].unique()), cfg):
        train, test = walkforward.split(df, fold)
        walkforward.check_no_overlap(train, test)
        if test.empty:
            continue
        models = fit_ml(train, feats, cfg, v5)
        parts.append(predict_ml(models, test, feats))
        log.info("v5_ml fold %s: train %d rows, test %d", fold.test_start.date(), len(train), len(test))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(f, index=False)
    return out


def ic_report(df: pd.DataFrame, s: pd.DataFrame) -> dict:
    m = df[["date", "entity", "fwd_5", "fwd_20", "target"]].merge(s, on=["date", "entity"])
    out = {}
    for col, step in (("fwd_5", 5), ("fwd_20", 20), ("target", 20)):
        ic = gbm.daily_ic(m, m["score"].values, target=col)
        no = ic.iloc[::step]
        out[f"IC_{col}"] = float(ic.mean())
        out[f"t_{col}"] = float(no.mean() / no.std() * np.sqrt(len(no))) if len(no) > 2 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    overrides = {"data.end": "2026-09-30", "label.horizon": 5, "portfolio.top_n": 20, "portfolio.rebalance_days": 5,
                 "portfolio.hold_buffer": 2.5}
    v5 = dict(V5_DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k.startswith("v5."):
            v5[k[3:]] = yaml.safe_load(v)
        else:
            overrides[k] = yaml.safe_load(v)
    cfg = cfgmod.load(overrides)
    cfg["v5"] = v5
    cfgmod.setup_logging()
    np.random.seed(cfg["seed"])
    run = "v5_" + __import__("hashlib").sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    out_dir = cfgmod.path(cfg, "results", run)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1, default=str)

    data = load_data(cfg)
    df = build_v5_dataset(cfg, data)
    feats = feature_list(df)
    log.info("V5 dataset %s, %d features (+%d new vs V4)", df.shape, len(feats),
             len([c for c in feats if any(k in c for k in ("deliv_val", "deliv_pct_20", "vol_shock", "hi_deliv",
                                                           "up_value", "252_high"))]))
    prices = engine.prepare_prices(data["panel"], set(df["entity"]))
    bench = bench_returns(data)
    hs = pd.Timestamp(cfg["validation"]["holdout_start"])
    first = pd.Timestamp(cfg["validation"]["first_test"])
    wf_df = df[df["date"] < hs]
    wf_test = wf_df[wf_df["date"] >= first]

    ml_raw = wf_ml_scores(cfg, wf_df, feats, v5, out_dir)
    fac_raw = pd.DataFrame({"date": wf_test["date"], "entity": wf_test["entity"], "score": v5f.factor_score(wf_test).values})
    raw = {"v5_ml": ml_raw, "v5_factor": fac_raw, "v5_combo": blend(ml_raw, fac_raw, v5["combo_weight_ml"])}
    scores = {k: smooth(v, v5["halflife"]) for k, v in raw.items()}

    rows, rets, results = [], {}, {}
    end_wf = hs - pd.Timedelta(days=1)
    for name, s in scores.items():
        res = backtest(cfg, data, df, prices, s, first, end_wf)
        results[name], rets[name] = res, res.daily["ret"].iloc[1:]
        rows.append({"model": name, **summarise(res, bench, cfg), **ic_report(wf_df, s)})
        res.daily.to_csv(os.path.join(out_dir, f"wf_daily_{name}.csv"))
        res.trades.to_csv(os.path.join(out_dir, f"wf_trades_{name}.csv"), index=False)
    table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    best = table.iloc[0]["model"]

    # ablations of the primary hypothesis (counted as trials too)
    abl = []
    variants = [("combo, no smoothing", {}, 0), ("combo, half-life 2", {}, 2), ("combo, half-life 10", {}, 10),
                ("combo, top 10", {"portfolio.top_n": 10}, None), ("combo, top 30", {"portfolio.top_n": 30}, None),
                ("combo, no drawdown breaker", {"risk.dd_breaker": 9.9}, None)]
    for label, over, hl in variants:
        c2 = cfgmod.load({**overrides, **over})
        s2 = smooth(raw["v5_combo"], v5["halflife"] if hl is None else hl)
        r2 = backtest(c2, data, df, prices, s2, first, end_wf)
        rets[f"abl: {label}"] = r2.daily["ret"].iloc[1:]
        p2 = summarise(r2, bench, c2)
        abl.append({"variant": label, **{k: p2.get(k) for k in ("CAGR", "Sharpe", "MaxDD", "Turnover_ann", "Cost_drag_ann")}})
    abl = pd.DataFrame(abl)
    trial_sr = record_trials(out_dir, cfg, rets)
    dsr = metrics.deflated_sharpe(rets[best], trial_sr)
    table.to_csv(os.path.join(out_dir, "walkforward_summary.csv"), index=False)
    abl.to_csv(os.path.join(out_dir, "ablations.csv"), index=False)
    log.info("V5 walk-forward:\n%s", table[["model", "CAGR", "Sharpe", "MaxDD", "Alpha_ann", "Turnover_ann",
                                             "Cost_drag_ann", "IC_fwd_5", "t_fwd_5", "IC_fwd_20", "t_fwd_20"]].round(3).to_string(index=False))
    log.info("ablations:\n%s", abl.round(3).to_string(index=False))
    log.info("selected %s; DSR %.3f over %d trials (V4 + V5)", best, dsr["DSR"], dsr["N_trials"])

    # previously-viewed period: one final ML fit on everything before it (purged)
    train, test = walkforward.holdout_split(df, cfg)
    walkforward.check_no_overlap(train, test)
    final = fit_ml(train, feats, cfg, v5)
    pv_ml = predict_ml(final, test, feats)
    pv_fac = pd.DataFrame({"date": test["date"], "entity": test["entity"], "score": v5f.factor_score(test).values})
    pv_raw = {"v5_ml": pv_ml, "v5_factor": pv_fac, "v5_combo": blend(pv_ml, pv_fac, v5["combo_weight_ml"])}
    pv_rows, pv_rets = [], {}
    for name, s in pv_raw.items():
        r = backtest(cfg, data, df, prices, smooth(s, v5["halflife"]), hs, df["date"].max())
        pv_rets[name] = r.daily["ret"].iloc[1:]
        pv_rows.append({"model": name, **summarise(r, bench, cfg)})
        r.daily.to_csv(os.path.join(out_dir, f"pv_daily_{name}.csv"))
    pv = pd.DataFrame(pv_rows)
    log.info("previously-viewed period:\n%s", pv[["model", "CAGR", "Sharpe", "MaxDD", "Alpha_ann", "Bench_CAGR"]].round(3).to_string(index=False))

    # live model: refit on ALL labelled data for paper trading
    all_train = df[df["target"].notna()]
    live = fit_ml(all_train, feats, cfg, v5)
    joblib.dump({"models": live, "features": feats, "config": cfg, "selected": best, "v5": v5,
                 "trained_through": str(all_train["label_end"].max().date())},
                os.path.join(cfgmod.path(cfg, "results"), "v5_final_model.joblib"))
    imp = shap_importance(live["lgbm_42"], test[feats].sample(min(3000, len(test)), random_state=cfg["seed"]))
    share = imp / imp.sum()
    share.to_csv(os.path.join(out_dir, "shap_importance.csv"))

    report(cfg, out_dir, table, abl, dsr, rets, best, bench, df, pv, pv_rets, share, v5)


def report(cfg, out_dir, table, abl, dsr, rets, best, bench, df, pv, pv_rets, share, v5):
    rf = cfg["report"]["risk_free"]
    secs = []
    p = metrics.perf(rets[best], bench, rf)
    kpis = "".join(f"<div class='kpi'><span class='muted'>{k}</span><b>{html._fmt(k, p.get(k))}</b></div>"
                   for k in ("CAGR", "Sharpe", "MaxDD", "Calmar", "Bench_CAGR", "Alpha_ann"))
    warn = ("<div class='warn'><b>Suspicious:</b> Sharpe above the bug threshold – check for leakage.</div>"
            if (table["Sharpe"] > cfg["report"]["suspicious_sharpe"]).any() else "")
    secs.append(("", warn + f"<div class='kpis'>{kpis}</div>"))
    secs.append(("What V5 changes vs V1–V4", """<ul>
<li><b>Target</b>: average rank of beta- and industry-neutral residual returns over 5, 10 and 20 days (V4: one raw horizon).</li>
<li><b>Score smoothing</b>: per-stock EWMA of the daily rank (half-life {hl} sessions) – V4's 5-day skill (IC 0.047) died of 21× turnover.</li>
<li><b>Factor blend</b>: 52-week-high proximity + 12-1 momentum (robust Indian anomalies; V4's holdout showed momentum beating ML).</li>
<li><b>New India features</b>: delivered-value accumulation, delivery-% trend, volume shocks with delivery, up-volume share, 52-week-high timing.</li>
<li><b>Ensemble</b>: LightGBM ×3 seeds + XGBoost, Optuna inside each fold; LambdaRank dropped (overfit in V4).</li>
<li><b>Portfolio</b>: top {top}, weekly, hold while rank ≤ {buf}× (no-trade band), inverse-vol, 15% name / 30% sector caps.</li>
<li><b>LLM news</b>: deliberately <i>not</i> in the model – historical LLM backtests carry lookahead bias (the model memorised history); it can only be a live overlay, validated forward.</li>
</ul>""" .format(hl=v5["halflife"], top=cfg["portfolio"]["top_n"], buf=cfg["portfolio"]["hold_buffer"])))
    secs.append(("Walk-forward 2017 → mid-2025 (out of sample, after all costs)",
                 html.table(table[["model", "CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar", "Alpha_ann", "Beta",
                                   "HitRate_trades", "Turnover_ann", "Cost_drag_ann", "IC_fwd_5", "t_fwd_5",
                                   "IC_fwd_20", "t_fwd_20"]].round(4))))
    secs.append(("Ablations of v5_combo", html.table(abl.round(4))))
    secs.append(("Deflated Sharpe (all V4 + V5 trials)", html.table(pd.DataFrame([{"model": best, **dsr}]))))
    curves = {n: rets[n] for n in CANDIDATES}
    curves["NIFTY 50 (price)"] = bench.reindex(rets[best].index).fillna(0)
    secs.append(("Equity vs NIFTY", html.equity_chart(curves, "Walk-forward equity (log)") + html.drawdown_chart(curves)))
    secs.append(("Per year – " + best, html.table(metrics.by_period(rets[best], bench, rf))))
    ctx = df.groupby("date")[["vix_pct_1y", "nifty_dist_ma200"]].first()
    secs.append(("Per regime – " + best, html.table(metrics.by_regime(rets[best], bench, rf, ctx))))
    secs.append(("Monthly – " + best, html.heatmap(rets[best], f"{best} monthly %")))
    pc = dict(pv_rets)
    pc["NIFTY 50 (price)"] = bench.reindex(next(iter(pv_rets.values())).index).fillna(0)
    secs.append(("Jul 2025 → Sep 2026 – PREVIOUSLY VIEWED (not a clean holdout)",
                 "<div class='warn'>V4 already evaluated this period once, and its result (momentum beat ML) is part of "
                 "why V5 blends in momentum. Treat these numbers as optimistic. The real test is forward paper trading.</div>"
                 + html.table(pv[["model", "CAGR", "Sharpe", "MaxDD", "Alpha_ann", "Beta", "Bench_CAGR"]].round(4))
                 + html.equity_chart(pc, "Previously-viewed period")))
    flag = share[share > cfg["report"]["shap_dominance"]]
    secs.append(("SHAP (live model)", ("<div class='warn'>Dominant: " + ", ".join(f"{k} {v:.0%}" for k, v in flag.items())
                                       + "</div>" if len(flag) else "") + html.bar_chart(share.head(25), "mean |SHAP| share")))
    html.render(os.path.join(out_dir, "report.html"), "V5.0 NSE ranking – research report", secs,
                f"Generated {datetime.now():%Y-%m-%d %H:%M} · run {os.path.basename(out_dir)}")
    log.info("report: %s", os.path.join(out_dir, "report.html"))


if __name__ == "__main__":
    main()
