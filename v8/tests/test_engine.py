"""v8 engine: fills, stops, partial sale, trailing exits, no look-ahead (synthetic one-stock data)."""
import numpy as np
import pandas as pd

from v8 import engine


class NoCost:
    buy_rate = 0.0

    def slip(self, kind):
        return 0.0

    def buy(self, money):
        return 0.0

    def sell(self, money):
        return 0.0


def _px(o, h, l, c):
    dates = pd.bdate_range("2020-01-01", periods=len(c))
    a = lambda x: np.array(x, dtype=float).reshape(-1, 1)
    C = a(c)
    return {"dates": dates, "entities": ["X"], "O": a(o), "H": a(h), "L": a(l), "C": C, "CF": C.copy(),
            "last_valid": np.array([len(c) - 1])}


def _ind(n, **sig):
    z = np.zeros((n, 1))
    ind = {k: np.zeros((n, 1), bool) for k in ("bo", "ep", "ep_fresh", "ath", "ath_504")}
    ind.update({k: z.copy() for k in ("pivot", "adr", "rs", "vratio", "atr42")})
    ind["sma10"] = np.full((n, 1), 0.0)
    ind["sma20"] = np.full((n, 1), 0.0)
    for k, v in sig.items():
        ind[k] = v
    return ind


CFG = {"start_equity": 100_000, "risk_pct": 0.01, "max_pos": 10, "max_weight": 0.20, "partial_day": 3,
       "partial_frac": 1 / 3, "bo_stop_cap": 0.08, "regime": False, "trail_ma": 10, "setups": ["BO"]}


def test_bo_buy_stop_fills_next_day_at_level_and_stops_out():
    n = 6
    o = [100, 100, 101, 104, 103, 99]
    h = [101, 101, 105, 105, 104, 100]
    l = [99, 99, 100.5, 103, 101, 95]
    c = [100, 100, 104, 104, 102, 96]
    bo = np.zeros((n, 1), bool); bo[1, 0] = True             # signal at the close of day 1
    piv = np.full((n, 1), 102.0); adr = np.full((n, 1), 0.03)
    res = engine.run(_px(o, h, l, c), _ind(n, bo=bo, pivot=piv, adr=adr), CFG, NoCost(), "2020-01-01", "2020-12-31")
    t = res.trades.iloc[0]
    assert t["entry_date"] == res.daily.index[1]               # day 2 (index 1 of the run, which starts at day 1)
    assert abs(t["entry_px"] - 102.0) < 1e-9                   # buy-stop at the level, not the open (101) or close
    assert t["reason"] == "STOP"                               # day 5 low 95 < stop 102 * 0.97
    assert abs(t["exit_px"] - 102 * 0.97) < 1e-9
    assert abs(t["R"] + 1) < 1e-6                              # lost exactly the risked amount (no costs)


def test_signal_on_last_day_is_not_traded_same_day():
    n = 4
    bo = np.zeros((n, 1), bool); bo[3, 0] = True               # signal on the final close: nothing to fill
    res = engine.run(_px([1] * n, [2] * n, [0.5] * n, [1] * n), _ind(n, bo=bo, pivot=np.full((n, 1), 1.5),
                     adr=np.full((n, 1), 0.03)), CFG, NoCost(), "2020-01-01", "2020-12-31")
    assert res.trades.empty and res.daily["n_pos"].max() == 0


def test_partial_then_trailing_exit():
    n = 9
    c = [100, 100, 103, 106, 109, 112, 111, 108, 107]
    o = [100, 100, 101] + c[2:-1]
    h = [x + 1 for x in c]; l = [x - 1 for x in c]; l[2] = 100.5
    bo = np.zeros((n, 1), bool); bo[1, 0] = True
    sma10 = np.full((n, 1), 0.0); sma10[7, 0] = 109.0          # close 108 < 109 on day 7 -> exit day 8 open
    res = engine.run(_px(o, h, l, c), _ind(n, bo=bo, pivot=np.full((n, 1), 102.0), adr=np.full((n, 1), 0.03),
                     sma10=sma10), CFG, NoCost(), "2020-01-01", "2020-12-31")
    t = res.trades.iloc[0]
    assert t["partial"] and t["reason"] == "TRAIL"
    assert t["exit_date"] == pd.bdate_range("2020-01-01", periods=n)[8]
    # 1/3 sold at day 6 open (112), 2/3 at day 8 open (108): average exit
    assert abs(t["exit_px"] - (112 / 3 + 108 * 2 / 3)) < 1e-6


def test_ath_trailing_stop_uses_highest_close_and_exits_next_open():
    n = 8
    c = [10, 10, 11, 12, 13, 12.9, 11.0, 11.5]
    o = [10, 10, 10.5, 11.5, 12.5, 13, 12.5, 10.8]
    h = [x + 0.2 for x in c]; l = [x - 0.2 for x in c]
    ath = np.zeros((n, 1), bool); ath[1, 0] = True
    atr = np.full((n, 1), 0.15)                                # 10 x ATR = 1.5 below the highest close
    cfg = {**CFG, "setups": ["ATH"], "max_weight": 0.10, "ath_k": 10}
    res = engine.run(_px(o, h, l, c), _ind(n, ath=ath, atr42=atr), cfg, NoCost(), "2020-01-01", "2020-12-31")
    t = res.trades.iloc[0]
    assert abs(t["entry_px"] - 10.5) < 1e-9                    # next open after the signal
    assert t["reason"] == "ATR_TRAIL"                           # close 11.0 < 13 - 1.5 on day 6
    assert abs(t["exit_px"] - 10.8) < 1e-9                     # sold at day 7's open


def test_cash_and_bond_etfs_excluded_but_not_companies():
    from v8.setups import cash_like
    out = dict(zip(["LIQUIDBEES", "LIQUID1", "GROWWLIQID", "LTGILTBEES", "EBBETF0431", "SDL24BEES", "CASHIETF",
                    "GOLDBEES", "NIFTYBEES", "PNBGILTS", "CHEMBOND", "MONEYBOXX", "RELIANCE"],
                   cash_like(["LIQUIDBEES", "LIQUID1", "GROWWLIQID", "LTGILTBEES", "EBBETF0431", "SDL24BEES", "CASHIETF",
                              "GOLDBEES", "NIFTYBEES", "PNBGILTS", "CHEMBOND", "MONEYBOXX", "RELIANCE"])))
    assert all(out[s] for s in ("LIQUIDBEES", "LIQUID1", "GROWWLIQID", "LTGILTBEES", "EBBETF0431", "SDL24BEES", "CASHIETF"))
    assert not any(out[s] for s in ("GOLDBEES", "NIFTYBEES", "PNBGILTS", "CHEMBOND", "MONEYBOXX", "RELIANCE"))
