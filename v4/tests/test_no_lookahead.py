"""The core rule: a feature row for date t must not change when data after t
is removed. Also checks label alignment, macro timing and the universe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v4.data import macro as macro_mod
from v4.data import panel as panel_mod
from v4.features import build as fbuild
from v4.features import labels
from v4.data import corporate_actions
from v4.tests.conftest import synthetic_adj


def _features(raw, cfg, market, cutoff=None):
    eq, dl = raw
    idx, part, macro = market
    if cutoff is not None:
        eq, dl = eq[eq["date"] <= cutoff], dl[dl["date"] <= cutoff]
        idx, part = idx[idx["date"] <= cutoff], part[part["date"] <= cutoff]
        macro = macro[macro["date"] <= cutoff + pd.Timedelta(days=10)]   # future macro rows exist but must be ignored
    p = panel_mod.assemble(eq.copy(), dl, cfg, synthetic_adj())
    industry = pd.Series({e: ("A" if i % 2 else "B") for i, e in enumerate(sorted(p["entity"].unique()))})
    return fbuild.build(p, industry, idx, part, macro, cfg)


def test_features_identical_when_future_removed(raw, cfg, market):
    full = _features(raw, cfg, market)
    feats = fbuild.feature_columns(full)
    days = sorted(full["date"].unique())
    for cutoff in (days[len(days) // 2], days[-30]):
        cut = _features(raw, cfg, market, pd.Timestamp(cutoff))
        a = full[full["date"] == cutoff].set_index("entity")[feats].sort_index()
        b = cut[cut["date"] == cutoff].set_index("entity")[feats].sort_index()
        assert list(a.index) == list(b.index), "universe changed when the future was removed"
        diff = ~np.isclose(a.values.astype(float), b.values.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
        bad = sorted({feats[j] for j in np.where(diff)[1]})
        assert not bad, f"lookahead in features: {bad}"


def test_label_is_next_open_to_open(panel, cfg):
    h = cfg["label"]["horizon"]
    lab = labels.forward_returns(panel, h)
    e = panel[panel["entity"] == "S05"].reset_index()
    i = 100
    expect = e.loc[i + 1 + h, "adj_open"] / e.loc[i + 1, "adj_open"] - 1
    got = lab.loc[e.loc[i, "index"], "fwd_ret"]
    assert np.isclose(got, expect)
    assert lab.loc[e.loc[i, "index"], "label_end"] == e.loc[i + 1 + h, "date"]
    assert lab.loc[e.loc[len(e) - 1, "index"], "fwd_ret"] != lab.loc[e.loc[len(e) - 1, "index"], "fwd_ret"]  # NaN at end


def test_split_does_not_create_a_return(panel):
    e = panel[panel["entity"] == "S03"].reset_index(drop=True)
    # raw close halves on the split day, adjusted series must not
    jump_raw = e["close"].iloc[300] / e["close"].iloc[299] - 1
    jump_adj = e["adj_close"].iloc[300] / e["adj_close"].iloc[299] - 1
    assert jump_raw < -0.4 and abs(jump_adj) < 0.15


def test_macro_alignment_timing():
    nse = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    m = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
                      "name": ["india_vix"] * 3 + ["spx"] * 3, "close": [10, 11, 12, 100, 101, 102],
                      "timing": ["same_day"] * 3 + ["prior_day"] * 3})
    a = macro_mod.align(m, nse)
    assert list(a["india_vix"]) == [10, 11, 12]                    # same-day close known at NSE close
    assert np.isnan(a["spx"].iloc[0]) and list(a["spx"].iloc[1:]) == [100, 101]   # US closes after NSE


def test_universe_uses_only_prior_data(raw, cfg):
    eq, dl = raw
    p1 = panel_mod.assemble(eq.copy(), dl, cfg, synthetic_adj())
    month = sorted(p1["date"].dt.to_period("M").unique())[6]
    first_day = p1.loc[p1["date"].dt.to_period("M") == month, "date"].min()
    eq2 = eq.copy()
    boost = (eq2["date"] >= first_day) & (eq2["symbol"] == "S00")      # make S00 hugely liquid from that day on
    eq2.loc[boost, "value"] *= 1000
    p2 = panel_mod.assemble(eq2, dl, cfg, synthetic_adj())
    m1 = set(p1.loc[(p1["date"] == first_day) & p1["in_universe"], "entity"])
    m2 = set(p2.loc[(p2["date"] == first_day) & p2["in_universe"], "entity"])
    assert m1 == m2


def test_corporate_action_parsing():
    assert corporate_actions.parse("Bonus 1:1") == (0.5, False)
    assert corporate_actions.parse("Bonus Issue 3 : 2")[0] == 0.4
    assert corporate_actions.parse("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 1/- Per Share")[0] == 0.1
    assert corporate_actions.parse("Bonus Preference Shares 21:1")[0] == 1.0        # not an equity bonus
    assert corporate_actions.parse("Scheme Of Arrangement - Bonus Debentures 1:1")[0] == 1.0
    assert corporate_actions.parse("Demerger")[1] is True
