"""V7 live paper runner: offline tests (synthetic data, no network)."""
import numpy as np
import pandas as pd

from v7 import live
from v7.momentum import d1_score, m30_score


def _pos(**kw):
    p = {"entity": "A", "isin": "INE000A", "qty": 100, "cost": 10_020.0, "since": "2026-10-01",
         "ref_date": "2026-10-01", "last_px": 100.0, "entry_px": 100.0}
    p.update(kw)
    return p


def test_split_doubles_shares_and_halves_prices():
    # 1:2 split between the reference date and the last date: the panel's
    # adj_close / close factor is 0.5 on the reference date, 1.0 after it
    f = pd.DataFrame({"entity": ["A", "A"], "date": pd.to_datetime(["2026-10-01", "2026-10-09"]), "f": [0.5, 1.0]})
    st = {"positions": {"A": _pos()}}
    notes = live.apply_corporate_actions(st, f, pd.Timestamp("2026-10-09"))
    p = st["positions"]["A"]
    assert p["qty"] == 200 and abs(p["entry_px"] - 50.0) < 1e-9 and abs(p["last_px"] - 50.0) < 1e-9
    assert p["ref_date"] == "2026-10-09" and notes
    # a second run on the same data must not adjust again
    live.apply_corporate_actions(st, f, pd.Timestamp("2026-10-09"))
    assert st["positions"]["A"]["qty"] == 200


def test_no_corporate_action_no_change():
    f = pd.DataFrame({"entity": ["A", "A"], "date": pd.to_datetime(["2026-10-01", "2026-10-09"]), "f": [1.0, 1.0]})
    st = {"positions": {"A": _pos()}}
    assert live.apply_corporate_actions(st, f, pd.Timestamp("2026-10-09")) == []
    assert st["positions"]["A"]["qty"] == 100


def test_split_does_not_fake_a_stop():
    f = pd.DataFrame({"entity": ["A", "A"], "date": pd.to_datetime(["2026-10-01", "2026-10-09"]), "f": [0.5, 1.0]})
    st = {"positions": {"A": _pos()}}
    live.apply_corporate_actions(st, f, pd.Timestamp("2026-10-09"))
    assert live.stops_hit(st, {"A": 49.0}, 0.10) == []          # -2% after the split, not -51%


def test_stop_threshold():
    st = {"positions": {"A": _pos(), "B": _pos(entity="B")}}
    assert live.stops_hit(st, {"A": 90.0, "B": 90.01}, 0.10) == ["A"]
    assert live.stops_hit(st, {"B": 50.0}, 0.10) == ["B"]       # no close for A: not checked


def test_m30_ranks_the_steadiest_uptrend_first():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=300)
    rows = []
    for e, drift in (("UP", 0.002), ("FLAT", 0.0), ("DOWN", -0.002)):
        r = rng.normal(drift, 0.01, len(days))
        px = 100 * np.cumprod(1 + r)
        rows += [{"date": d, "entity": e, "adj_close": c, "ret_cc": x} for d, c, x in zip(days, px, r)]
    panel = pd.DataFrame(rows)
    s = m30_score(panel, pd.DataFrame({"date": days[-1], "entity": ["UP", "FLAT", "DOWN"]})).set_index("entity")["score"]
    assert s.idxmax() == "UP" and s.idxmin() == "DOWN"


def _panel(deliv=None):
    """Five stocks over 300 sessions; UP_A and UP_B have identical prices. deliv:
    entity -> (delivery %, delivery % over the last 20 sessions); others stay at 45%."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=300)
    up = rng.normal(0.002, 0.01, len(days))
    rows = []
    for e, r in (("UP_A", up), ("UP_B", up), ("MID", rng.normal(0.001, 0.01, len(days))),
                 ("FLAT", rng.normal(0.0, 0.01, len(days))), ("DOWN", rng.normal(-0.002, 0.01, len(days)))):
        px = 100 * np.cumprod(1 + r)
        old, new = (deliv or {}).get(e, (45.0, 45.0))
        dp = np.where(np.arange(len(days)) >= len(days) - 20, new, old)
        rows += [{"date": d, "entity": e, "adj_close": c, "ret_cc": x, "deliv_pct": v}
                 for d, c, x, v in zip(days, px, r, dp)]
    return pd.DataFrame(rows), days


def test_d1_rising_delivery_breaks_a_momentum_tie():
    panel, days = _panel({"UP_B": (45.0, 60.0)})
    members = pd.DataFrame({"date": days[-1], "entity": panel["entity"].unique()})
    m30 = m30_score(panel, members).set_index("entity")["score"]
    d1 = d1_score(panel, members).set_index("entity")["score"]
    assert m30["UP_A"] == m30["UP_B"]
    assert d1["UP_B"] > d1["UP_A"]
    assert d1.drop("UP_B").idxmax() == "UP_A" and d1.idxmin() == "DOWN"


def test_d1_without_delivery_signal_keeps_the_momentum_order():
    panel, days = _panel()                                   # delivery flat at 45% everywhere
    members = pd.DataFrame({"date": days[-1], "entity": panel["entity"].unique()})
    m30 = m30_score(panel, members).set_index("entity")["score"].sort_index()
    for p in (panel, panel.drop(columns="deliv_pct"), panel.assign(deliv_pct=np.nan)):
        d1 = d1_score(p, members).set_index("entity")["score"].sort_index()
        assert (d1.rank() == m30.rank()).all()


def test_d1_matches_the_research_tilt():
    from v7.swing_v7c import variants
    rng = np.random.default_rng(1)
    panel, days = _panel()
    panel["deliv_pct"] = rng.uniform(20, 70, len(panel))
    panel.loc[rng.random(len(panel)) < 0.05, "deliv_pct"] = np.nan     # missing delivery days
    panel.loc[panel["entity"] == "MID", "deliv_pct"] = np.nan           # a stock with no delivery data
    members = panel.loc[panel["date"] >= days[-30], ["date", "entity"]]
    research = variants({"m200": m30_score(panel, members), "u200": members, "panel": panel})[0]["D1_tilt20"]
    live = d1_score(panel, members)
    j = research.merge(live, on=["date", "entity"], suffixes=("_research", "_live"))
    assert len(j) == len(research) == len(live) == 150
    assert np.allclose(j["score_research"], j["score_live"], rtol=0, atol=1e-12)
