"""python -m pytest nse_scr/tests -q  - synthetic candles, no network."""
import numpy as np
import pandas as pd

from nse_scr import strategy as S


def _day(seed=0):
    """09:15-15:25 IST 5-min candles: a volume shock + run from 10:00 with pullbacks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-07 09:15", "2026-10-07 15:25", freq="5min", tz="Asia/Kolkata")
    n = len(idx)
    steps = rng.normal(0, 0.001, n)
    vol = rng.integers(20_000, 40_000, n).astype(float)
    k = 9
    while k + 9 < n:
        steps[k:k + 6] += 0.008
        steps[k + 6:k + 9] -= 0.004
        vol[k:k + 6] *= 30
        vol[k + 6:k + 9] *= 6
        k += 9
    px = 200 * np.cumprod(1 + steps)
    o = np.r_[px[0], px[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, px) * 1.001, "low": np.minimum(o, px) * 0.999,
                         "close": px, "volume": vol}, index=idx)


def _sig(x):
    return S.setups(x, S.bar_features(x, 195.0, 200_000))


def test_setups_are_causal():
    x = _day()
    full = _sig(x)
    assert len(full) > 0
    for cut in (40, 55, 70):
        part = x.iloc[:cut]
        assert list(_sig(part)) == [t for t in full if t <= part.index[-1]]


def test_live_matches_research():
    x = _day(seed=2)
    sig = _sig(x)[0]
    done = S.simulate("T", x, sig)
    i = x.index.get_loc(sig)
    for cut in range(i + 2, len(x) + 1):
        tr = S.simulate("T", x.iloc[:cut], sig, final=False)
        if tr.outcome != "OPEN":
            assert (tr.exit_ts, round(tr.exit, 6), tr.outcome) == (done.exit_ts, round(done.exit, 6), done.outcome)
            break


def test_no_fill_on_locked_candle_and_charges_included():
    x = _day(seed=4)
    sig = _sig(x)[0]
    i = x.index.get_loc(sig)
    y = x.copy()
    v = float(y.iloc[i + 1]["open"])
    y.iloc[i + 1, :4] = [v, v, v, v]                              # upper-circuit lock: open == high == low == close
    assert S.simulate("T", y, sig) is None
    tr = S.simulate("T", x, sig)
    gross = (tr.exit / tr.entry - 1) * 100
    assert tr.ret_pct < gross                                     # NSE charges deducted
    assert 1 - tr.stop0 / tr.entry <= S.STOP_PCT + 1e-9
