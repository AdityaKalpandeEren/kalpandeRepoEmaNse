"""
V4.0 research pipeline: data -> features -> purged walk-forward for every
candidate model -> realistic backtests -> model selection -> ONE holdout
run -> SHAP -> HTML report.

    python -m v4.run_research                    # walk-forward only (holdout untouched)
    python -m v4.run_research --holdout          # + the one-time holdout evaluation
    python -m v4.run_research --download         # refresh NSE archives first
    python -m v4.run_research --set label.horizon=20 --set portfolio.top_n=15

Honesty mechanics:
  * every configuration's walk-forward Sharpe is appended to
    results/trials.json; the Deflated Sharpe uses ALL trials ever run
  * the holdout writes results/holdout_used.json; running it again needs
    --force-holdout and the report is stamped "REPEATED HOLDOUT"
  * a net Sharpe above report.suspicious_sharpe triggers a leakage warning
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from v4 import config as cfgmod
from v4.backtest import costs as cost_mod
from v4.backtest import engine
from v4.data import macro as macro_mod
from v4.data import nse_daily, panel as panel_mod, sectors
from v4.features import build as fbuild
from v4.models import baselines, gbm
from v4.reports import html
from v4.validation import metrics, walkforward

log = logging.getLogger("v4")


# ─── data + features ─────────────────────────────────────────────────────

def load_data(cfg: dict, download: bool = False) -> dict:
    cache = cfgmod.path(cfg, "cache")
    start = pd.Timestamp(cfg["data"]["start"]).date()
    end = pd.Timestamp(cfg["data"]["end"]).date() if cfg["data"]["end"] else date.today() - timedelta(days=1)
    if download:
        fails = nse_daily.download(cache, start, end, cfg["data"]["download_workers"])
        log.info("download failures: %s", fails)
    macro = macro_mod.fetch(cache, str(start - timedelta(days=400)), refresh=download)
    ind_table = sectors.load(cache)
    panel = panel_mod.build(cfg, cache, start, end, rebuild=download)
    ents = panel_mod.entity_table(panel)
    industry = sectors.assign(ents, ind_table)
    indices = nse_daily.load_kind(cache, "indices", start, end)
    participant = nse_daily.load_kind(cache, "participant", start, end)
    return {"panel": panel, "industry": industry, "indices": indices, "participant": participant,
            "macro": macro, "entities": ents, "start": start, "end": end}


def dataset(cfg: dict, data: dict, rebuild: bool = False) -> pd.DataFrame:
    key = _hash({k: cfg[k] for k in ("universe", "label", "features")}) + f"_{data['end']:%Y%m%d}"
    f = os.path.join(cfgmod.path(cfg, "cache"), f"dataset_{key}.parquet")
    if os.path.exists(f) and not rebuild:
        return pd.read_parquet(f)
    df = fbuild.build(data["panel"], data["industry"], data["indices"], data["participant"], data["macro"], cfg)
    df.to_parquet(f, index=False)
    return df


# ─── models ──────────────────────────────────────────────────────────────

def walk_forward_scores(cfg: dict, df: pd.DataFrame, feats: list[str], name: str,
                        out_dir: str) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-sample scores over all walk-forward test blocks for one model."""
    f = os.path.join(out_dir, f"wf_scores_{name}.parquet")
    finfo = os.path.join(out_dir, f"wf_info_{name}.json")
    if os.path.exists(f):
        return pd.read_parquet(f), json.load(open(finfo)) if os.path.exists(finfo) else []
    folds = walkforward.folds(pd.DatetimeIndex(df["date"].unique()), cfg)
    parts, infos = [], []
    for k, fold in enumerate(folds):
        train, test = walkforward.split(df, fold)
        walkforward.check_no_overlap(train, test)
        if test.empty:
            continue
        if name in baselines.NAMES:
            s = baselines.score(name, test)
            info = {}
        else:
            model, info = gbm.tune_and_fit(name, train, feats, cfg)
            s = pd.Series(gbm.predict(model, test, feats), index=test.index)
        parts.append(pd.DataFrame({"date": test["date"], "entity": test["entity"], "score": s.values}))
        infos.append({"fold": k, "test_start": str(fold.test_start.date()), "train_rows": len(train),
                      "test_rows": len(test), **{kk: v for kk, v in info.items() if kk != "params"},
                      "params": info.get("params")})
        log.info("%s fold %d %s: train %d rows, inner IC %s", name, k, fold.test_start.date(), len(train),
                 info.get("inner_ic"))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(f, index=False)
    json.dump(infos, open(finfo, "w"), indent=1, default=str)
    return out, infos


def ensemble(score_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, s in score_frames.items():
        x = s.copy()
        x["r"] = x.groupby("date")["score"].rank(pct=True)
        parts.append(x.set_index(["date", "entity"])["r"].rename(name))
    m = pd.concat(parts, axis=1).mean(axis=1).rename("score").reset_index()
    return m


def ic_stats(df: pd.DataFrame, scores: pd.DataFrame, horizon: int) -> dict:
    m = df[["date", "entity", "fwd_ret"]].merge(scores, on=["date", "entity"])
    ic = gbm.daily_ic(m, m["score"].values)
    non_overlap = ic.iloc[::horizon]
    t = non_overlap.mean() / non_overlap.std() * np.sqrt(len(non_overlap)) if len(non_overlap) > 2 else np.nan
    # top-decile minus universe average forward return (gross, per H days)
    m["dec"] = m.groupby("date")["score"].rank(pct=True)
    spread = (m[m["dec"] > 0.9].groupby("date")["fwd_ret"].mean() - m.groupby("date")["fwd_ret"].mean()).mean()
    return {"IC_mean": float(ic.mean()), "IC_t_nonoverlap": float(t), "IC_pos_frac": float((ic > 0).mean()),
            "Top10pct_excess_per_H": float(spread)}


# ─── backtest + metrics ──────────────────────────────────────────────────

def backtest(cfg, data, df, prices, scores, start, end):
    info = df[["date", "entity", "vol_63", "industry"]]
    ctx = df.groupby("date")[["vix_pct_1y", "nifty_dist_ma200"]].first()
    res = engine.run(scores, prices, info, ctx, cfg, pd.Timestamp(start), pd.Timestamp(end))
    return res


def bench_returns(data: dict) -> pd.Series:
    from v4.features.market import index_close
    return index_close(data["indices"], "NIFTY 50").pct_change()


def summarise(res: engine.Result, bench: pd.Series, cfg: dict) -> dict:
    d = res.daily
    p = metrics.perf(d["ret"].iloc[1:], bench, cfg["report"]["risk_free"])
    years = len(d) / 252
    avg_eq = d["equity"].mean()
    p["Turnover_ann"] = d["turnover"].sum() / avg_eq / years / 2 if years > 0 else np.nan
    p["Cost_drag_ann"] = d["costs"].sum() / avg_eq / years if years > 0 else np.nan
    p["Avg_gross"] = (d["invested"] / d["equity"]).mean()
    if not res.trades.empty:
        p["HitRate_trades"] = float((res.trades["ret"] > 0).mean())
        p["Trades"] = int(len(res.trades))
    return p


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def record_trials(out_dir: str, cfg: dict, results: dict[str, pd.Series]) -> list[float]:
    """Append this run's per-model daily Sharpes to trials.json (dedup by
    config+model); return every recorded trial Sharpe."""
    f = os.path.join(cfgmod.path(cfg, "results"), "trials.json")
    trials = json.load(open(f)) if os.path.exists(f) else {}
    h = _hash({k: v for k, v in cfg.items() if k not in ("paths",)})
    for name, r in results.items():
        trials[f"{h}:{name}"] = metrics.daily_sharpe(r)
    json.dump(trials, open(f, "w"), indent=1)
    return list(trials.values())


# ─── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the feature dataset")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--force-holdout", action="store_true")
    ap.add_argument("--models", help="comma list (default: config models.candidates)")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override")
    a = ap.parse_args()
    import yaml
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    cfg = cfgmod.load(overrides)
    cfgmod.setup_logging()
    np.random.seed(cfg["seed"])
    run_id = _hash(cfg)
    out_dir = cfgmod.path(cfg, "results", f"run_{run_id}")
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1, default=str)

    data = load_data(cfg, a.download)
    df = dataset(cfg, data, a.rebuild)
    feats = fbuild.feature_columns(df)
    log.info("dataset: %d rows, %d features, %s .. %s", len(df), len(feats), df["date"].min().date(),
             df["date"].max().date())
    prices = engine.prepare_prices(data["panel"], set(df["entity"]))
    bench = bench_returns(data)
    hs = pd.Timestamp(cfg["validation"]["holdout_start"])
    wf_df = df[df["date"] < hs]

    names = a.models.split(",") if a.models else cfg["models"]["candidates"]
    scores, infos = {}, {}
    for name in [n for n in names if n != "ensemble"]:
        scores[name], infos[name] = walk_forward_scores(cfg, wf_df, feats, name, out_dir)
    if "ensemble" in names:
        gb = {n: scores[n] for n in gbm.GBM_NAMES if n in scores}
        if len(gb) >= 2:
            scores["ensemble"] = ensemble(gb)

    first = pd.Timestamp(cfg["validation"]["first_test"])
    end_wf = hs - pd.Timedelta(days=1)
    rows, rets, results = [], {}, {}
    for name, s in scores.items():
        res = backtest(cfg, data, df, prices, s, first, end_wf)
        results[name] = res
        rets[name] = res.daily["ret"].iloc[1:]
        p = summarise(res, bench, cfg)
        p.update(ic_stats(wf_df, s, cfg["label"]["horizon"]))
        rows.append({"model": name, **p})
        res.daily.to_csv(os.path.join(out_dir, f"wf_daily_{name}.csv"))
        res.trades.to_csv(os.path.join(out_dir, f"wf_trades_{name}.csv"), index=False)
    table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    trial_sr = record_trials(out_dir, cfg, rets)
    best = table.iloc[0]["model"]
    dsr = metrics.deflated_sharpe(rets[best], trial_sr)
    table.to_csv(os.path.join(out_dir, "walkforward_summary.csv"), index=False)
    log.info("walk-forward summary:\n%s", table[["model", "CAGR", "Sharpe", "MaxDD", "IC_mean",
                                                  "IC_t_nonoverlap", "Turnover_ann"]].to_string(index=False))
    log.info("selected %s; DSR %.3f over %d trials", best, dsr["DSR"], dsr["N_trials"])

    # sensitivity of the selected model to the risk overlays and costs
    sens = []
    for label, over in (("as configured", {}), ("no drawdown breaker", {"risk.dd_breaker": 9.9}),
                        ("regime filter on", {"risk.regime_filter": True}),
                        ("slippage x2 (24 bps/side)", {"costs.slippage_bps": cfg["costs"]["slippage_bps"] * 2}),
                        ("top 20 names", {"portfolio.top_n": 20})):
        c2 = cfgmod.copy_cfg(cfg)
        for k, v in over.items():
            node = c2
            *ps, leaf = k.split(".")
            for q in ps:
                node = node[q]
            node[leaf] = v
        r2 = backtest(c2, data, df, prices, scores[best], first, end_wf)
        sens.append({"variant": label, **{k: summarise(r2, bench, c2).get(k)
                                          for k in ("CAGR", "Sharpe", "MaxDD", "Calmar", "Avg_gross", "Turnover_ann")}})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(out_dir, "sensitivity.csv"), index=False)
    log.info("sensitivity (%s):\n%s", best, sens.round(3).to_string(index=False))

    holdout = None
    if a.holdout:
        holdout = run_holdout(cfg, data, df, feats, prices, bench, best, out_dir, a.force_holdout)

    make_report(cfg, data, df, feats, bench, table, rets, results, best, dsr, infos, holdout, out_dir, sens)


def run_holdout(cfg, data, df, feats, prices, bench, best, out_dir, force):
    marker = os.path.join(cfgmod.path(cfg, "results"), "holdout_used.json")
    repeated = os.path.exists(marker)
    if repeated and not force:
        raise SystemExit(f"Holdout already used ({open(marker).read().strip()}). "
                         "Re-running it turns it into a validation set; pass --force-holdout to accept that.")
    train, test = walkforward.holdout_split(df, cfg)
    walkforward.check_no_overlap(train, test)
    out = {"repeated": repeated, "models": {}}
    final_models = {}
    gbm_best = best if best in gbm.GBM_NAMES else None
    to_eval = sorted({best, "combo_baseline", "mom_baseline"} | ({"lgbm_rank"} if not gbm_best else set()))
    for name in to_eval:
        if name in baselines.NAMES:
            s = baselines.score(name, test)
        elif name == "ensemble":
            parts = {}
            for n in gbm.GBM_NAMES:
                m, _ = gbm.tune_and_fit(n, train, feats, cfg)
                final_models[n] = m
                parts[n] = pd.DataFrame({"date": test["date"], "entity": test["entity"],
                                         "score": gbm.predict(m, test, feats)})
            sc = ensemble(parts)
            out["models"][name] = sc
            continue
        else:
            m, info = gbm.tune_and_fit(name, train, feats, cfg)
            final_models[name] = m
            s = pd.Series(gbm.predict(m, test, feats), index=test.index)
        out["models"][name] = pd.DataFrame({"date": test["date"], "entity": test["entity"], "score": np.asarray(s)})
    end = df["date"].max()
    out["results"] = {}
    for name, sc in out["models"].items():
        res = backtest(cfg, data, df, prices, sc, pd.Timestamp(cfg["validation"]["holdout_start"]), end)
        out["results"][name] = res
        res.daily.to_csv(os.path.join(out_dir, f"holdout_daily_{name}.csv"))
    json.dump({"used_at": datetime.now().isoformat(timespec="seconds"), "selected": best,
               "config_hash": _hash(cfg)}, open(marker, "w"))
    joblib.dump({"models": final_models, "features": feats, "config": cfg, "selected": best,
                 "trained_through": str(train["label_end"].max().date())},
                os.path.join(cfgmod.path(cfg, "results"), "v4_final_model.joblib"))
    out["final_models"] = final_models
    out["test"] = test
    return out


def shap_importance(model, X: pd.DataFrame) -> pd.Series:
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)


def make_report(cfg, data, df, feats, bench, table, rets, results, best, dsr, infos, holdout, out_dir, sens=None):
    rf = cfg["report"]["risk_free"]
    secs = []
    best_ret = rets[best]
    nifty = bench.reindex(best_ret.index).fillna(0)
    warn = ""
    if (table["Sharpe"] > cfg["report"]["suspicious_sharpe"]).any():
        warn = ("<div class='warn'><b>Suspicious result:</b> a net Sharpe above "
                f"{cfg['report']['suspicious_sharpe']} was produced. Treat as a bug (leakage) until proven otherwise.</div>")
    p = metrics.perf(best_ret, bench, rf)
    kpis = "".join(f"<div class='kpi'><span class='muted'>{k}</span><b>{html._fmt(k, p.get(k))}</b></div>"
                   for k in ("CAGR", "Sharpe", "MaxDD", "Calmar", "Bench_CAGR", "Alpha_ann"))
    secs.append(("", warn + f"<div class='kpis'>{kpis}</div>"))
    secs.append(("Walk-forward results (out of sample, after all costs)",
                 f"<p class='muted'>Test blocks {cfg['validation']['first_test']} → "
                 f"{cfg['validation']['holdout_start']} (exclusive), retrained every "
                 f"{cfg['validation']['refit_months']} months on an expanding, purged + embargoed window. "
                 f"Top {cfg['portfolio']['top_n']} names, rebalanced every {cfg['portfolio']['rebalance_days']} "
                 f"sessions at the next open. Selected by the pre-registered rule <i>highest walk-forward net Sharpe</i>: "
                 f"<b>{best}</b>.</p>"
                 + html.table(table[["model", "CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar", "Alpha_ann", "Beta",
                                     "HitRate_trades", "Turnover_ann", "Cost_drag_ann", "IC_mean", "IC_t_nonoverlap",
                                     "Top10pct_excess_per_H"]].round(4))))
    dsr_df = pd.DataFrame([{"model": best, **dsr}])
    secs.append(("Deflated Sharpe Ratio", "<p class='muted'>Probability that the selected model's true Sharpe "
                 "beats the best Sharpe expected from the same number of unskilled trials (all configurations "
                 "ever run, from trials.json). Above 0.95 is significant.</p>" + html.table(dsr_df)))
    if sens is not None:
        secs.append(("Sensitivity – " + best, "<p class='muted'>Same out-of-sample scores, different risk overlays / "
                     "costs / breadth. Shows how much of the result depends on each choice.</p>" + html.table(sens)))
    curves = {n: rets[n] for n in [best, "combo_baseline", "mom_baseline"] if n in rets}
    curves["NIFTY 50 (price)"] = nifty
    secs.append(("Equity curve vs NIFTY 50", html.equity_chart(curves, "Walk-forward equity (log)")
                 + html.drawdown_chart(curves)))
    secs.append(("Monthly returns – " + best, html.heatmap(best_ret, f"{best} monthly returns (%)")))
    secs.append(("Per year", html.table(metrics.by_period(best_ret, bench, rf))))
    ctx = df.groupby("date")[["vix_pct_1y", "nifty_dist_ma200"]].first()
    secs.append(("Per market regime", html.table(metrics.by_regime(best_ret, bench, rf, ctx))))
    if infos.get(best):
        fi = pd.DataFrame(infos[best]).drop(columns=["params"], errors="ignore")
        secs.append(("Walk-forward folds – " + best, html.table(fi)))

    if holdout:
        hrows = []
        for name, res in holdout["results"].items():
            hp = summarise(res, bench, cfg)
            hrows.append({"model": name, **hp})
        ht = pd.DataFrame(hrows)
        stamp = ("<div class='warn'><b>REPEATED HOLDOUT</b> – this period has been evaluated before, so it is no "
                 "longer an untouched test.</div>" if holdout["repeated"] else
                 "<p class='muted'>First and only evaluation of the untouched holdout.</p>")
        hc = {n: r.daily["ret"].iloc[1:] for n, r in holdout["results"].items()}
        first_h = next(iter(hc.values()))
        hc["NIFTY 50 (price)"] = bench.reindex(first_h.index).fillna(0)
        secs.append((f"Holdout {cfg['validation']['holdout_start']} → {df['date'].max().date()}",
                     stamp + html.table(ht[["model", "CAGR", "Sharpe", "MaxDD", "Alpha_ann", "Beta", "Bench_CAGR",
                                            "HitRate_trades", "Turnover_ann"]].round(4))
                     + html.equity_chart(hc, "Holdout equity")))
        m = holdout["final_models"].get(best) or next(iter(holdout["final_models"].values()), None)
        if m is not None:
            X = holdout["test"][feats].sample(min(3000, len(holdout["test"])), random_state=cfg["seed"])
            imp = shap_importance(m, X)
            share = imp / imp.sum()
            flag = share[share > cfg["report"]["shap_dominance"]]
            note = ("<div class='warn'>Dominant feature(s): " + ", ".join(f"{k} ({v:.0%})" for k, v in flag.items())
                    + " – check for leakage.</div>") if len(flag) else "<p class='muted'>No single feature dominates.</p>"
            secs.append(("SHAP feature importance (final model)", note + html.bar_chart(share.head(25), "mean |SHAP| share")))
            share.to_csv(os.path.join(out_dir, "shap_importance.csv"))

    c = cfg["costs"]
    rt = cost_mod.round_trip_pct(100000, c)
    secs.append(("Assumptions & limitations", f"""<ul>
<li><b>Universe</b>: point-in-time NIFTY-100 <i>proxy</i> (top {cfg['universe']['size']} by trailing median traded value, rebuilt monthly from the bhavcopy). NSE's real historical constituents are not freely available; delisted and renamed stocks are included while they qualified.</li>
<li><b>Prices</b>: NSE bhavcopy, adjusted with NSE's ex-date PREVCLOSE (splits/bonus/rights). Ordinary dividends are not reinvested, so returns are price returns; the benchmark is the NIFTY 50 <i>price</i> index (TRI would add ~1.2–1.5%/yr to both).</li>
<li><b>Execution</b>: signal after the close, fill at the next open ± {c['slippage_bps']} bps; round trip cost ≈ {rt:.2%} on a ₹1 lakh order (STT 0.1% each side dominates). Upper/lower-circuit locks block buys/sells; positions capped at {cfg['risk']['max_adv_frac']:.0%} of 20-day ADV. Fractional shares are allowed (minor at ₹{cfg['portfolio']['capital']:,} capital). Idle cash earns nothing.</li>
<li><b>Macro timing</b>: US/FX/crude data use the previous day's close (known before the NSE close); INDIA VIX same day. GIFT Nifty has no free history — the prior S&P 500 session is the proxy.</li>
<li><b>Not modelled</b>: quarterly fundamentals, F&O ban list / long-short (long-only cash), taxes (STCG), impact beyond the slippage assumption, auction / settlement holidays.</li>
<li><b>Where it will likely fail</b>: abrupt regime breaks (e.g. COVID-style gaps through stops), crowded momentum reversals, very low dispersion markets where ranking skill can't beat ~0.25% round-trip costs, and anything after a change in STT or market structure.</li>
<li><b>SEBI</b>: retail algorithmic trading through broker APIs is governed by SEBI's framework (circular of 4 Feb 2025): orders must go through the broker's registered API with an algo ID; strategies above the order-rate threshold need exchange registration. This project only produces research and paper signals — confirm current rules with your broker before any live use.</li>
</ul>"""))
    path = os.path.join(out_dir, "report.html")
    html.render(path, "V4.0 NSE ML ranking – research report", secs,
                f"Generated {datetime.now():%Y-%m-%d %H:%M} · data {data['start']} → {data['end']} · "
                f"horizon {cfg['label']['horizon']}d · run {os.path.basename(out_dir)}")
    log.info("report: %s", path)


if __name__ == "__main__":
    main()
