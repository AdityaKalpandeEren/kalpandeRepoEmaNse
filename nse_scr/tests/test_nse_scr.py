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


def _mirror(x):
    """The same day flipped upside down (a falling shocker): price -> 400 - price."""
    y = x.copy()
    y["open"], y["close"] = 400 - x["open"], 400 - x["close"]
    y["high"], y["low"] = 400 - x["low"], 400 - x["high"]
    return y


def _short_sig(x, mode):
    return S.short_setups(x, S.bar_features(x, 205.0, 200_000), mode)


def test_short_setups_causal_and_live_matches_research():
    x = _mirror(_day(seed=2))
    full = _short_sig(x, "lod")
    assert len(full) > 0
    for cut in (40, 55, 70):
        part = x.iloc[:cut]
        assert list(_short_sig(part, "lod")) == [t for t in full if t <= part.index[-1]]
    sig = full[0]
    done = S.simulate_short("T", x, sig, mode="lod")
    assert done.stop0 > done.entry and done.stop0 / done.entry - 1 <= S.SHORT_STOP_PCT + 1e-9
    for cut in range(x.index.get_loc(sig) + 2, len(x) + 1):
        tr = S.simulate_short("T", x.iloc[:cut], sig, final=False, mode="lod")
        if tr.outcome != "OPEN":
            assert (tr.exit_ts, round(tr.exit, 6), tr.outcome) == (done.exit_ts, round(done.exit, 6), done.outcome)
            break


def test_short_pnl_sign_charges_and_lower_circuit():
    x = _mirror(_day(seed=4))
    sig = _short_sig(x, "lod")[0]
    tr = S.simulate_short("T", x, sig, mode="lod")
    gross = (1 - tr.exit / tr.entry) * 100                        # a short gains when the price falls
    assert tr.ret_pct < gross
    i = x.index.get_loc(sig)
    y = x.copy()
    v = float(y.iloc[i + 1]["open"])
    y.iloc[i + 1, :4] = [v, v, v, v]                              # lower-circuit lock: can't sell
    assert S.simulate_short("T", y, sig, mode="lod") is None
