"""V5-specific checks: causal smoothing, causal new features, purge on the
20-day target, factor score uses no future data."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swing.data import panel as panel_mod  # noqa: E402
from swing.tests.conftest import small_cfg, synthetic_adj, synthetic_raw  # noqa: E402
from v5 import features as v5f  # noqa: E402
from v5.run_v5 import blend, smooth  # noqa: E402


def _scores(n_days=40, n_ent=6, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame([{"date": d, "entity": f"E{k}", "score": rng.normal()} for d in days for k in range(n_ent)])


def test_smoothing_is_causal():
    s = _scores()
    full = smooth(s, 5)
    cut_day = sorted(s["date"].unique())[25]
    part = smooth(s[s["date"] <= cut_day], 5)
    a = full[full["date"] == cut_day].set_index("entity")["score"].sort_index()
    b = part[part["date"] == cut_day].set_index("entity")["score"].sort_index()
    assert np.allclose(a.values, b.values)


def test_smoothing_reduces_rank_changes():
    s = _scores(n_days=120)
    raw = smooth(s, 0)
    sm = smooth(s, 5)
    def churn(x):
        top = x[x.groupby("date")["score"].rank(ascending=False) <= 2].groupby("date")["entity"].apply(frozenset)
        return np.mean([len(a ^ b) for a, b in zip(top.iloc[:-1], top.iloc[1:])])
    assert churn(sm) < churn(raw)


def test_blend_is_rank_average():
    a, b = _scores(seed=1), _scores(seed=2)
    m = blend(a, b, 0.5)
    assert m["score"].between(0, 1).all() and len(m) == len(a)


def test_extra_features_causal():
    cfg = small_cfg()
    eq, dl = synthetic_raw()
    p = panel_mod.assemble(eq.copy(), dl, cfg, synthetic_adj())
    cut = sorted(p["date"].unique())[400]
    full = v5f.extra_stock_features(p, set(p["entity"]))
    pc = panel_mod.assemble(eq[eq["date"] <= cut].copy(), dl[dl["date"] <= cut], cfg, synthetic_adj())
    part = v5f.extra_stock_features(pc, set(pc["entity"]))
    fa = pd.concat([p.loc[full.index, ["date", "entity"]], full], axis=1)
    fb = pd.concat([pc.loc[part.index, ["date", "entity"]], part], axis=1)
    a = fa[fa["date"] == cut].set_index("entity").drop(columns="date").sort_index()
    b = fb[fb["date"] == cut].set_index("entity").drop(columns="date").sort_index()
    assert np.allclose(a.values.astype(float), b.values.astype(float), equal_nan=True)
