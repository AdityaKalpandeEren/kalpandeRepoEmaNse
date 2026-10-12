"""V7 live paper runner: offline tests (synthetic data, no network)."""
import numpy as np
import pandas as pd

from v7 import live
from v7.momentum import m30_score


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
