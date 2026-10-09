"""python -m pytest nse_lcr/tests -q - synthetic candles, no network."""
import numpy as np
import pandas as pd

from nse_lcr import strategy as S

EMAS = {"ema10": 98.0, "ema20": 97.0, "ema30": 96.0, "ema40": 95.0, "ema60": 93.0, "ema180": 88.0}


def _day(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-08 09:15", "2026-10-08 15:25", freq="5min", tz="Asia/Kolkata")
    n = len(idx)
    steps = rng.normal(0, 0.0004, n)
    vol = rng.integers(30_000, 50_000, n).astype(float)
    k = 3
    while k + 9 < n:
        steps[k:k + 6] += 0.003
        steps[k + 6:k + 9] -= 0.0035              # ~1% pullback (rule needs >= 0.8%)
        vol[k:k + 6] *= 8
        vol[k + 6:k + 9] *= 2
        k += 9
    px = 100 * np.cumprod(1 + steps)
    o = np.r_[px[0], px[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, px) * 1.0004, "low": np.minimum(o, px) * 0.9996,
                         "close": px, "volume": vol}, index=idx)


def _sig(x, emas=EMAS):
    return S.setups(x, S.bar_features(x, 98.0, 2_000_000), emas)


def test_causal_and_live_matches_research():
    x = _day()
    full = _sig(x)
    assert len(full) > 0
    for cut in (25, 40, 55):
        part = x.iloc[:cut]
        assert list(_sig(part)) == [t for t in full if t <= part.index[-1]]
    sig = full[0]
    done = S.simulate("T", x, sig)
    i = x.index.get_loc(sig)
    for cut in range(i + 2, len(x) + 1):
        tr = S.simulate("T", x.iloc[:cut], sig, final=False)
        if tr.outcome != "OPEN":
            assert (tr.exit_ts, round(tr.exit, 6), tr.outcome) == (done.exit_ts, round(done.exit, 6), done.outcome)
            break


def test_ema_rule_and_perfect():
    assert S.ema_ok(100, EMAS) and S.is_perfect(100, EMAS)
    long_down = {**EMAS, "ema60": 104.0, "ema180": 110.0}
    assert S.ema_ok(100, long_down) and not S.is_perfect(100, long_down)
    assert not S.ema_ok(100, {**EMAS, "ema20": 101.0})


def test_locked_candle_no_fill_and_charges():
    x = _day(seed=3)
    sig = _sig(x)[0]
    i = x.index.get_loc(sig)
    y = x.copy()
    v = float(y.iloc[i + 1]["open"])
    y.iloc[i + 1, :4] = [v, v, v, v]
    assert S.simulate("T", y, sig) is None
    tr = S.simulate("T", x, sig)
    assert tr.ret_pct < (tr.exit / tr.entry - 1) * 100
