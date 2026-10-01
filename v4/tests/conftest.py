"""Synthetic NSE-like data so every test runs offline in seconds."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v4 import config as cfgmod  # noqa: E402
from v4.data import panel as panel_mod  # noqa: E402


def small_cfg() -> dict:
    return cfgmod.load({"universe.size": 12, "universe.value_window": 20, "universe.min_history": 60,
                        "universe.min_price": 1.0, "label.horizon": 5,
                        "features.windows": [5, 10, 21, 63], "validation.first_test": "2013-10-01",
                        "validation.holdout_start": "2014-04-01", "validation.refit_months": 3,
                        "portfolio.top_n": 3, "portfolio.capital": 1_000_000})


def synthetic_raw(n_ent: int = 25, n_days: int = 520, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2012-06-01", periods=n_days)
    rows = []
    for k in range(n_ent):
        px = 100.0 * (1 + k / 10)
        vol = 0.01 + 0.02 * rng.random()
        liq = 1e7 * (1 + 3 * rng.random())
        split_day = 300 if k == 3 else None          # 1:2 split on day 300 for entity 3
        for i, d in enumerate(days):
            prev = px
            reported_prev = px                       # NSE: PREVCLOSE NOT adjusted on the ex-date
            if split_day is not None and i == split_day:
                prev = px / 2
                px = px / 2
            r = rng.normal(0.0003, vol)
            o = prev * (1 + rng.normal(0, vol / 3))
            c = prev * (1 + r)
            h, lo = max(o, c) * (1 + abs(rng.normal(0, vol / 2))), min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
            value = liq * np.exp(rng.normal(0, 0.3))
            rows.append({"date": d, "symbol": f"S{k:02d}", "isin": f"INE{k:05d}", "open": o, "high": h, "low": lo,
                         "close": c, "prevclose": reported_prev, "volume": value / c, "value": value, "trades": 1000.0})
            px = c
    eq = pd.DataFrame(rows)
    eq["entity"] = eq["symbol"]
    dl = eq[["date", "symbol"]].copy()
    dl["deliv_pct"] = 40 + 20 * rng.random(len(dl))
    return eq, dl


def synthetic_adj() -> pd.DataFrame:
    """The corporate-action record for the synthetic 1:2 split."""
    days = pd.bdate_range("2012-06-01", periods=520)
    return pd.DataFrame({"isin": ["INE00003"], "date": [days[300]], "symbol": ["S03"], "factor": [0.5],
                         "demerger": [False]})


def synthetic_market(days: pd.DatetimeIndex, seed: int = 1):
    rng = np.random.default_rng(seed)
    n = len(days)
    nifty = 10000 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    idx = pd.concat([pd.DataFrame({"date": days, "index": name, "close": nifty * (1 + 0.1 * j)})
                     for j, name in enumerate(["NIFTY 50", "NIFTY BANK", "NIFTY MIDCAP 50"])])
    part = pd.concat([pd.DataFrame({"date": days, "participant": who,
                                    "fut_idx_long": rng.integers(1e4, 1e5, n), "fut_idx_short": rng.integers(1e4, 1e5, n),
                                    "opt_idx_call_long": rng.integers(1e4, 1e5, n), "opt_idx_put_long": rng.integers(1e4, 1e5, n),
                                    "opt_idx_call_short": rng.integers(1e4, 1e5, n), "opt_idx_put_short": rng.integers(1e4, 1e5, n)})
                      for who in ("FII", "DII", "Client", "Pro")])
    cal = pd.bdate_range(days[0] - pd.Timedelta(days=30), days[-1] + pd.Timedelta(days=5))
    macro = pd.concat([pd.DataFrame({"date": cal, "name": name, "close": 50 * np.cumprod(1 + rng.normal(0, 0.01, len(cal))),
                                     "timing": timing})
                       for name, timing in (("india_vix", "same_day"), ("usdinr", "prior_day"), ("brent", "prior_day"),
                                            ("us10y", "prior_day"), ("spx", "prior_day"))])
    return idx, part, macro


@pytest.fixture(scope="session")
def cfg():
    return small_cfg()


@pytest.fixture(scope="session")
def raw():
    return synthetic_raw()


@pytest.fixture(scope="session")
def panel(raw, cfg):
    eq, dl = raw
    return panel_mod.assemble(eq.copy(), dl, cfg, synthetic_adj())


@pytest.fixture(scope="session")
def market(panel):
    return synthetic_market(pd.DatetimeIndex(sorted(panel["date"].unique())))
