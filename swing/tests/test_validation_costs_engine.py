"""Purging/embargo, cost model, execution rules and risk caps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing.backtest import costs as cost_mod
from swing.backtest import engine
from swing.features import labels
from swing.risk import sizing
from swing.validation import metrics, walkforward


# ─── validation ─────────────────────────────────────────────────────────

def _labelled(panel, cfg):
    lab = labels.forward_returns(panel, cfg["label"]["horizon"])
    return pd.concat([panel[["date", "entity"]], lab], axis=1)


def test_walkforward_purge_and_embargo(panel, cfg):
    df = _labelled(panel, cfg)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    fs = walkforward.folds(dates, cfg)
    assert len(fs) >= 2
    for f in fs:
        tr, te = walkforward.split(df, f)
        walkforward.check_no_overlap(tr, te)
        first_test = te["date"].min()
        gap = dates.searchsorted(first_test) - dates.searchsorted(tr["label_end"].max())
        assert gap >= cfg["validation"]["embargo_days"]


def test_holdout_split_untouched(panel, cfg):
    df = _labelled(panel, cfg)
    tr, te = walkforward.holdout_split(df, cfg)
    walkforward.check_no_overlap(tr, te)
    assert te["date"].min() >= pd.Timestamp(cfg["validation"]["holdout_start"])


def test_overlap_check_catches_leak(panel, cfg):
    df = _labelled(panel, cfg)
    leak = df[df["date"] < "2014-01-01"].copy()
    leak["label_end"] = pd.Timestamp("2014-02-01")
    with pytest.raises(AssertionError):
        walkforward.check_no_overlap(leak, df[df["date"] >= "2014-01-01"])


# ─── costs ──────────────────────────────────────────────────────────────

def test_delivery_charges(cfg):
    c = cfg["costs"]
    buy = cost_mod.order_charges(100000, "buy", c)
    sell = cost_mod.order_charges(100000, "sell", c)
    # STT 100 + stamp 15 + exch 2.97 + sebi 0.1 + GST on (exch+sebi)
    assert buy == pytest.approx(100 + 15 + 2.97 + 0.1 + 0.18 * 3.07, rel=1e-6)
    assert sell == pytest.approx(100 + 2.97 + 0.1 + 0.18 * 3.07 + c["dp_per_sell"], rel=1e-6)
    rt = cost_mod.round_trip_pct(100000, c)
    assert 0.0035 < rt < 0.006            # ~0.24% charges + 2 x 12 bps slippage


def test_brokerage_cap(cfg):
    c = {**cfg["costs"], "brokerage_pct": 0.0005, "brokerage_cap": 20.0}
    small = cost_mod.order_charges(10000, "buy", c) - cost_mod.order_charges(10000, "buy", cfg["costs"])
    big = cost_mod.order_charges(1e7, "buy", c) - cost_mod.order_charges(1e7, "buy", cfg["costs"])
    assert small == pytest.approx(5 * 1.18) and big == pytest.approx(20 * 1.18)


# ─── engine ─────────────────────────────────────────────────────────────

def _toy_prices(locked_up_day=None, delist_after=None):
    days = pd.bdate_range("2020-01-01", periods=12)
    rows = []
    for e, base in (("A", 100.0), ("B", 50.0)):
        for i, d in enumerate(days):
            if e == "B" and delist_after is not None and i > delist_after:
                continue
            c = base * (1 + 0.01 * i)
            o, h, lo = c, c * 1.01, c * 0.99         # a normal candle with a range
            if e == "A" and i == 6:
                c = c * 1.10                  # big move at the CLOSE of the signal day
            if e == "A" and locked_up_day is not None and i == locked_up_day:
                o = h = lo = c = base * 1.2
            rows.append({"date": d, "entity": e, "symbol": e, "open": o, "high": max(h, c), "low": min(lo, c),
                         "close": c, "prevclose": base * (1 + 0.01 * (i - 1)) if i else c, "adj_open": o,
                         "adj_close": c, "value": 1e9})
    return engine.prepare_prices(pd.DataFrame(rows)), days


def _run(prices, days, signal_day, cfg, names=("A",)):
    sc = pd.DataFrame({"date": days[signal_day], "entity": list(names), "score": 1.0})
    info = pd.DataFrame({"date": days[signal_day], "entity": list(names), "vol_63": 0.02, "industry": "X"})
    c2 = {**cfg, "portfolio": {**cfg["portfolio"], "top_n": len(names), "rebalance_days": 1}}
    return engine.run(sc, prices, info, pd.DataFrame(), c2, days[0], days[-1])


def test_executes_at_next_open_not_signal_close(cfg):
    prices, days = _toy_prices()
    res = _run(prices, days, 6, cfg)
    d = res.daily
    assert d.loc[days[6], "invested"] == 0                      # nothing bought at the signal-day close
    assert d.loc[days[7], "invested"] > 0                       # bought at the next open
    # the +10% jump at the signal-day close must not be in the P&L
    assert d["equity"].max() < cfg["portfolio"]["capital"] * 1.05


def test_upper_circuit_blocks_buy(cfg):
    prices, days = _toy_prices(locked_up_day=7)
    res = _run(prices, days, 6, cfg)
    assert res.daily.loc[days[7], "invested"] == 0


def test_delisted_position_is_closed(cfg):
    prices, days = _toy_prices(delist_after=8)
    res = _run(prices, days, 5, cfg, names=("B",))
    assert res.daily.loc[days[7], "n_pos"] == 1
    assert res.daily.loc[days[-1], "n_pos"] == 0 and len(res.trades) == 1


def test_weight_caps(cfg):
    names = list("ABCDEFGH")
    vol = pd.Series([0.001] + [0.03] * 7, index=names)            # A would get ~30% unconstrained
    sector = pd.Series(["S1"] * 5 + ["S2", "S3", "S4"], index=names)
    w = sizing.weights(names, vol, sector, cfg)
    assert w.max() <= cfg["risk"]["max_weight"] + 1e-9
    assert w.groupby(sector).sum().max() <= cfg["risk"]["max_sector_weight"] + 1e-9


def test_hold_buffer_keeps_holdings():
    s = pd.Series(np.arange(20, 0, -1), index=[f"E{i}" for i in range(20)])   # E0 best
    picked = sizing.select(s, held={"E7"}, top_n=5, buffer=2.0)
    assert "E7" in picked and len(picked) == 5


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    few = metrics.deflated_sharpe(r, [0.01, 0.02])
    many = metrics.deflated_sharpe(r, list(rng.normal(0, 0.03, 200)))
    assert many["DSR"] < few["DSR"]


def test_dd_breaker_trips_and_resets(cfg):
    ex = sizing.Exposure(cfg)
    assert ex.scale(100.0) == 1.0
    assert ex.scale(80.0) == 0.5                    # -20% from peak -> half exposure
    for _ in range(ex.window + 1):                  # a year later the old peak has rolled off
        s = ex.scale(80.0)
    assert s == 1.0
