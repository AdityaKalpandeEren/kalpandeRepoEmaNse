"""
V3 research step 1: find setups that win BEFORE charges, then check they
still win AFTER charges - before any ML filter is put on top.

Why: on dataset_v3_r2 the 14 V3 setups average about -0.02R before charges
and -0.21R after (charges ~0.19R, because the median stop is only ~0.6%
away). A filter can only drop trades, it cannot create an edge the pool
does not have - so retraining V3 cannot fix that (retrain r2: all three
versions failed out of sample).

What it does, on the V3 dataset (no rebuild needed):
  1. Exits: re-simulates every candidate from the cached 5-minute candles
     (backtest/cache/candles/5m) with wider stops / no target / hold to the
     15:10 flat, so charges are a smaller share of the risk.
  2. Setups: for each exit, each setup (and "ALL") and each input (bottom or
     top fifth, cut points from TRAIN days only), measures R after charges.
     A rule is kept only if it wins on TRAIN (n >= 300, >= 30 days,
     day-t >= 2), then again on VALIDATION (exp > 0, day-t >= 1.5).
     The TEST block is reported once, for the survivors only.
  3. ML: if any rule survives, the V3 model (hgb_shallow) is trained on that
     rule's trades only (train), threshold on validation, test once.

Trades are counted under V3's live selection rules (cooldown, max trades per
symbol, 10 a day) - the same select_trades() the V3 trainer uses - so every
number here compares directly with the retrain r2 baseline (-0.116R).

    python -m backtest.ml.research_v3_gross_edge --dataset backtest/ml/data/dataset_v3_r2.csv

Not modelled: the NIFTY / India VIX shock exit (the re-simulated exits use
only stop, target and the 15:10 flat).
"""
import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

import config
from backtest.candle_cache import _path
from backtest.ml.train_meta_model_v3 import MODEL_COLUMNS, block, day_block, select_trades

COST_PER_SIDE = (config.SLIPPAGE_BPS + config.COMMISSION_BPS) / 10000.0
FLAT_MIN = config.ML_V2_FLAT_MIN                      # 15:10 candle
BOOK = ["symbol", "date", "entry_time", "source_strategy", "direction", "r_multiple", "risk_pct"]

# name -> (minimum stop distance as a fraction of entry, target in R or None)
EXITS = {
    "orig_stop_no_target": (0.0, None),
    "stop1_no_target": (0.010, None),
    "stop1.5_no_target": (0.015, None),
    "stop1_target2R": (0.010, 2.0),
    "stop1.5_target2R": (0.015, 2.0),
}
MIN_TRAIN_N, MIN_TRAIN_DAYS, MIN_TRAIN_T = 300, 30, 2.0
MIN_VAL_T = 1.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="backtest/ml/data/dataset_v3_r2.csv")
    p.add_argument("--val-frac", type=float, default=config.ML_V2_VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=config.ML_V2_TEST_FRAC)
    p.add_argument("--out", default="backtest/results/research_v3_gross_edge.md")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def load_dataset(path):
    """Long-only rows, bookkeeping + model inputs as float32, read in chunks
    (the r2 file is 1.2 GB)."""
    head = pd.read_csv(path, nrows=0).columns
    feats = [c for c in MODEL_COLUMNS if c in head and not c.startswith("src_")]
    dtypes = {c: "float32" for c in feats + ["r_multiple", "risk_pct"]}
    parts = []
    for ch in pd.read_csv(path, usecols=BOOK + feats, dtype=dtypes, chunksize=100_000):
        parts.append(ch[ch["direction"] == "long"])
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["entry_time"], utc=True)
    return df.sort_values("ts", kind="mergesort").reset_index(drop=True), feats


def split_dates(df, val_frac, test_frac):
    """Same chronological split as train_meta_model_v3."""
    dates = sorted(df["date"].unique())
    n = len(dates)
    n_test, n_val = max(1, round(n * test_frac)), max(1, round(n * val_frac))
    tr, va = dates[: n - n_val - n_test], dates[n - n_val - n_test: n - n_test]
    part = pd.Series("test", index=df.index)
    part[df["date"].isin(tr)] = "train"
    part[df["date"].isin(va)] = "val"
    return part, (tr[0], tr[-1], len(tr)), (va[0], va[-1], len(va)), \
        (dates[n - n_test], dates[-1], n_test)


def _symbol_candles(symbol, months):
    frames = []
    for m in months:
        p = _path(symbol, 5, date(int(m[:4]), int(m[5:7]), 1))
        if os.path.exists(p):
            frames.append(pd.read_pickle(p))
    if not frames:
        return None
    c = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    c["date"] = c["timestamp"].dt.strftime("%Y-%m-%d")
    return c


def resimulate_exits(df):
    """Adds gross/net R columns for every EXITS variant. Entry = open of the
    fill candle (as in the dataset build); a stop is hit when a candle's low
    touches it (stop wins ties with the target, as in simulate_forward_v2)."""
    out = {f"{k}_{s}": np.full(len(df), np.nan, dtype="float32")
           for k in EXITS for s in ("gross", "net")}
    found = 0
    for sym, g in df.groupby("symbol", sort=False):
        c = _symbol_candles(sym, sorted({d[:7] for d in g["date"]}))
        if c is None:
            continue
        for day, gd in g.groupby("date", sort=False):
            cd = c[c["date"] == day]
            if cd.empty:
                continue
            ts = cd["timestamp"].values
            o, h, l, cl = (cd[k].to_numpy(float) for k in ("open", "high", "low", "close"))
            mins = ((cd["timestamp"] - cd["timestamp"].dt.normalize()).dt.total_seconds() / 60
                    - 555).to_numpy()
            flat = np.where(mins >= FLAT_MIN)[0]
            for idx, et, risk in zip(gd.index, pd.to_datetime(gd["entry_time"]).values,
                                     gd["risk_pct"].to_numpy(float)):
                k = np.searchsorted(ts, et)
                if k >= len(ts) or ts[k] != et:
                    continue
                end = flat[flat >= k][0] if (flat >= k).any() else len(ts) - 1
                entry = o[k]
                lo, hi = l[k:end + 1], h[k:end + 1]
                found += 1
                for name, (min_stop, tgt_r) in EXITS.items():
                    dist = entry * max(risk, min_stop)
                    stop = entry - dist
                    hit_s = np.flatnonzero(lo <= stop)
                    hit_t = np.flatnonzero(hi >= entry + tgt_r * dist) if tgt_r else np.array([], int)
                    i_s = hit_s[0] if len(hit_s) else 10**9
                    i_t = hit_t[0] if len(hit_t) else 10**9
                    if i_s <= i_t and i_s < 10**9:
                        px = stop
                    elif i_t < 10**9:
                        px = entry + tgt_r * dist
                    else:
                        px = cl[end]
                    out[f"{name}_gross"][idx] = (px - entry) / dist
                    out[f"{name}_net"][idx] = (px * (1 - COST_PER_SIDE)
                                               - entry * (1 + COST_PER_SIDE)) / dist
    for k, v in out.items():
        df[k] = v
    # The dataset's own exit (stop, target, shock, flat): net is recorded,
    # gross adds back the round-trip charge in R.
    df["current_net"] = df["r_multiple"]
    df["current_gross"] = df["r_multiple"] + 2 * COST_PER_SIDE / df["risk_pct"]
    return found


# ═══════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════

def day_t(r, day_codes, n_days):
    """t-stat of the per-day mean R (one observation per day)."""
    cnt = np.bincount(day_codes, minlength=n_days)
    keep = cnt > 0
    if keep.sum() < 2:
        return 0.0, int(keep.sum())
    m = np.bincount(day_codes, weights=r, minlength=n_days)[keep] / cnt[keep]
    sd = m.std(ddof=1)
    return (float(m.mean() / (sd / np.sqrt(len(m)))) if sd > 0 else 0.0), int(keep.sum())


def live(sel, col):
    """Trades actually taken under V3's live rules, scored with column `col`."""
    d = sel.assign(r_multiple=sel[col])
    d = d[d["r_multiple"].notna()]
    return select_trades(d, np.ones(len(d)), 0.0)


def row(name, t):
    b, d = block(t["r_multiple"]), day_block(t)
    return (f"| {name} | {b['n']} | {b['win_pct']} | {b['exp_r']:+.3f} | {b['tot_r']:+.1f} | "
            f"{d['days_pos']}/{d['days']} | {d['day_t']} |")


HDR = "| Rule / exit | Trades | Win % | Avg R | Total R | Days + | Day-t |\n|---|---|---|---|---|---|---|"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    a = parse_args()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    df, feats = load_dataset(a.dataset)
    part, trn, val, tst = split_dates(df, a.val_frac, a.test_frac)
    say(f"# V3 research step 1: gross edge first\n\nRows {len(df)} (long), {df['symbol'].nunique()} "
        f"symbols, {len(feats)} inputs screened.  ")
    say(f"Train {trn[0]}..{trn[1]} ({trn[2]} d), val {val[0]}..{val[1]} ({val[2]} d), "
        f"test {tst[0]}..{tst[1]} ({tst[2]} d).  ")
    found = resimulate_exits(df)
    say(f"Re-simulated {found}/{len(df)} candidates from cached candles.\n")

    exits = ["current"] + list(EXITS)
    # ── 1. Every candidate, every exit: before vs after charges ──
    say("## 1. All candidates, by exit (per candidate, no selection)\n")
    say("| Exit | Train gross | Train net | Val net | Test net |\n|---|---|---|---|---|")
    for e in exits:
        g = df[part == "train"][f"{e}_gross"].mean()
        n = [df[part == p][f"{e}_net"].mean() for p in ("train", "val", "test")]
        say(f"| {e} | {g:+.3f} | {n[0]:+.3f} | {n[1]:+.3f} | {n[2]:+.3f} |")

    say("\n## 2. Each setup, by exit: TRAIN, avg R before / after charges\n")
    tr = df[part == "train"]
    setups = sorted(tr["source_strategy"].unique())
    say("| Setup | N | " + " | ".join(exits) + " |\n|---|---|" + "---|" * len(exits))
    for s in setups:
        x = tr[tr["source_strategy"] == s]
        say(f"| {s} | {len(x)} | " + " | ".join(
            f"{x[e + '_gross'].mean():+.2f} / {x[e + '_net'].mean():+.2f}" for e in exits) + " |")

    # ── 3. Rule screen: exit x setup x input fifth ──
    day_idx = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
    codes = df["date"].map(day_idx).to_numpy()
    n_days = len(day_idx)
    is_tr, is_va = (part == "train").to_numpy(), (part == "val").to_numpy()
    src = df["source_strategy"].to_numpy()
    tested, cands = 0, []
    for e in exits:
        net = df[f"{e}_net"].to_numpy(float)
        for s in ["ALL"] + setups:
            base = np.ones(len(df), bool) if s == "ALL" else (src == s)
            for f in feats:
                v = df[f].to_numpy(float)
                vt = v[base & is_tr]
                vt = vt[~np.isnan(vt)]
                if len(vt) < MIN_TRAIN_N:
                    continue
                q20, q80 = np.quantile(vt, [0.2, 0.8])
                if q20 == q80:
                    continue
                for side, m in (("low", v <= q20), ("high", v >= q80)):
                    tested += 1
                    mt = base & m & is_tr & ~np.isnan(net)
                    if mt.sum() < MIN_TRAIN_N or net[mt].mean() <= 0:
                        continue
                    t_tr, d_tr = day_t(net[mt], codes[mt], n_days)
                    if d_tr < MIN_TRAIN_DAYS or t_tr < MIN_TRAIN_T:
                        continue
                    mv = base & m & is_va & ~np.isnan(net)
                    if mv.sum() < 30:
                        continue
                    t_va, _ = day_t(net[mv], codes[mv], n_days)
                    cands.append((e, s, f, side, q20 if side == "low" else q80,
                                  net[mt].mean(), t_tr, net[mv].mean(), t_va))
    passed = [c for c in cands if c[7] > 0 and c[8] >= MIN_VAL_T]
    say(f"\n## 3. Rule screen\n\nRules tested: {tested}. Won on TRAIN (net > 0, day-t >= "
        f"{MIN_TRAIN_T}, n >= {MIN_TRAIN_N}): {len(cands)}. Also won on VALIDATION (net > 0, "
        f"day-t >= {MIN_VAL_T}): **{len(passed)}**.\n")
    if cands:
        say("Top train winners (train net / day-t -> val net / day-t):\n")
        say("| Exit | Setup | Rule | Train R | Train t | Val R | Val t |\n|---|---|---|---|---|---|---|")
        for c in sorted(cands, key=lambda c: -c[8])[:15]:
            say(f"| {c[0]} | {c[1]} | {c[2]} {'<=' if c[3] == 'low' else '>='} {c[4]:.4g} | "
                f"{c[5]:+.3f} | {c[6]:.2f} | {c[7]:+.3f} | {c[8]:.2f} |")

    # ── 4. Before vs after on the TEST block, live selection rules ──
    te, va = df[part == "test"], df[part == "val"]
    say("\n## 4. Exits under V3's live rules (cooldown, 10 trades a day)\n")
    say("Pick an exit on VALIDATION; the TEST rows only confirm it.\n")
    say(HDR)
    for name, blk in (("VAL", va), ("TEST", te)):
        say(row(f"{name} BEFORE: every candidate, current exit", live(blk, "current_net")))
        for e in EXITS:
            say(row(f"{name} every candidate, {e}", live(blk, f"{e}_net")))
    for c in sorted(passed, key=lambda c: -c[8])[:10]:
        e, s, f, side, cut = c[:5]
        x = te if s == "ALL" else te[te["source_strategy"] == s]
        x = x[x[f] <= cut] if side == "low" else x[x[f] >= cut]
        say(row(f"{s} {f} {'<=' if side == 'low' else '>='} {cut:.4g}, {e}", live(x, f"{e}_net")))

    # ── 5. ML on the best surviving rule ──
    if passed:
        from sklearn.ensemble import HistGradientBoostingClassifier
        e, s, f, side, cut = sorted(passed, key=lambda c: -c[8])[0][:5]
        base = df if s == "ALL" else df[df["source_strategy"] == s]
        pool = base[base[f] <= cut] if side == "low" else base[base[f] >= cut]
        pool = pool[pool[f"{e}_net"].notna()]
        pp = part.loc[pool.index]
        X = pool[feats].to_numpy(np.float32)
        y = (pool[f"{e}_net"] > 0).astype(int).to_numpy()
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           random_state=0).fit(X[pp == "train"], y[pp == "train"])
        pv = pool[pp == "val"].assign(r_multiple=pool[f"{e}_net"])
        prob_v = m.predict_proba(X[pp == "val"])[:, 1]
        best = None
        for th in np.arange(0.30, 0.801, 0.025):
            sel = select_trades(pv, prob_v, th)
            if len(sel) >= 100 and (best is None or sel["r_multiple"].mean() > best[1]):
                best = (round(th, 3), sel["r_multiple"].mean())
        say(f"\n## 5. ML filter on the best rule ({s} {f} {side}, {e})\n")
        if best is None:
            say("No threshold keeps >= 100 validation trades - ML step skipped.")
        else:
            pt = pool[pp == "test"].assign(r_multiple=pool[f"{e}_net"])
            sel = select_trades(pt, m.predict_proba(X[pp == "test"])[:, 1], best[0])
            say(f"Threshold {best[0]} chosen on validation (val avg R {best[1]:+.3f}).\n")
            say(HDR)
            say(row("rule only (test)", live(pt, f"{e}_net")))
            say(row(f"rule + ML p>={best[0]} (test)", sel))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
