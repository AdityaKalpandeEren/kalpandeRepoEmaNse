"""Wide scan of the ML filters (config.LIVE_RESEARCH_WIDE_STRATEGIES): offline tests with stub
models - which strategy groups see which candles, the per-group marks, alert tagging, the
end-of-day pass and the daily universe cache."""
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import config
import strategy.strategies as strategies
from paper_trading import research_live as rl

DAY = "2026-10-13"
NOW = pd.Timestamp(f"{DAY} 12:00", tz=rl.MARKET_TZ).to_pydatetime()


def _candles(n=30):
    ts = pd.date_range(f"{DAY} 09:15", periods=n, freq="5min", tz=rl.MARKET_TZ)
    return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000})


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls, fire = [], {}

    def evaluate_all(symbol, sub, models, directions):
        ts = sub["timestamp"].iat[-1]
        calls.append((symbol, ts, tuple(sorted(models))))
        return [s for name, s in fire.get(ts, []) if name in models]

    monkeypatch.setattr(config, "LIVE_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LIVE_RESEARCH_WIDE_STRATEGIES", ["ML"])
    monkeypatch.setattr(rl, "_selected_models", lambda: {"BASE": None, "ML": None})
    monkeypatch.setattr(rl, "_enrich", lambda df: df)
    monkeypatch.setattr(strategies, "evaluate_all", evaluate_all)
    rp = rl.ResearchPass(NOW)
    rp.state = rl._new_state(DAY)
    return SimpleNamespace(rp=rp, calls=calls, fire=fire, df=_candles())


def _evaluated(calls, model):
    return [ts for _, ts, models in calls if model in models]


def test_wide_only_stock_is_seen_by_the_wide_strategies_only(env):
    env.rp.process_symbol("X", env.df, wide_tag="MID · off-index")
    assert {m for _, _, m in env.calls} == {("ML",)}
    assert len(env.calls) == 30 - config.MIN_WARMUP_CANDLES
    assert "X" not in env.rp.state["last_ts"]
    assert env.rp.state["last_ts_wide"]["X"] == str(env.df["timestamp"].iat[-1])


def test_stock_joining_the_watchlist_scan_is_not_re_evaluated_by_ml(env):
    env.rp.process_symbol("X", env.df.iloc[:20], wide_tag="MID")       # wide passes up to bar 19
    env.calls.clear()
    env.rp.process_symbol("X", env.df)                                  # now in the watchlist scan
    first_new = env.df["timestamp"].iat[20]
    assert all(ts >= first_new for ts in _evaluated(env.calls, "ML"))
    assert len(_evaluated(env.calls, "ML")) == 10
    assert len(_evaluated(env.calls, "BASE")) == 30 - config.MIN_WARMUP_CANDLES   # its backlog, as before
    assert env.rp.state["last_ts"]["X"] == env.rp.state["last_ts_wide"]["X"]


def test_switching_on_mid_day_creates_no_ml_backlog(env):
    env.rp.state["last_ts"]["X"] = str(env.df["timestamp"].iat[19])   # full scans before the wide scan existed
    env.rp.process_symbol("X", env.df)
    assert len(_evaluated(env.calls, "ML")) == 10 and len(_evaluated(env.calls, "BASE")) == 10
    env.calls.clear()
    env.rp.process_symbol("Y", env.df.iloc[:20])                        # Y full-scanned, then drops out ...
    env.rp.state["last_ts_wide"].pop("Y")                               # ... with no wide mark (old state)
    env.calls.clear()
    env.rp.process_symbol("Y", env.df, wide_tag="LARGE")
    assert len(_evaluated(env.calls, "ML")) == 10 and not _evaluated(env.calls, "BASE")


def test_wide_scan_off_is_the_old_behaviour(env, monkeypatch):
    monkeypatch.setattr(config, "LIVE_RESEARCH_WIDE_STRATEGIES", [])
    env.rp.process_symbol("X", env.df)
    assert {m for _, _, m in env.calls} == {("BASE", "ML")}
    env.calls.clear()
    env.rp.process_symbol("Z", env.df, wide_tag="MID")
    assert env.calls == []


def test_wide_trade_is_tagged_and_alert_says_so(env):
    ts = env.df["timestamp"].iat[25]
    sig = SimpleNamespace(strategy="ML", direction="long", regime="weak_bull/low", trend_strength=0.5, score=56.0,
                          reason="meta-filter: A_BREAKOUT p=0.56", candle_time=ts, entry=100.0, stop_loss=99.0,
                          target=101.5, exit_mode="standard")
    env.fire[ts] = [("ML", sig)]
    env.rp.process_symbol("X", env.df, wide_tag="MID · off-index")
    (t,) = env.rp.state["trades"]
    assert t["scope"] == "MID · off-index" and t["reason"].endswith("[wide scan: MID · off-index]")
    assert "wide scan" not in rl._why(t) and "win probability 56%" in rl._why(t)
    assert "Wide scan: MID · off-index · not on the watchlist" in rl._format_entries([t])


def test_end_of_day_pass_keeps_wide_stocks_to_the_wide_strategies(env, monkeypatch):
    monkeypatch.setattr(rl, "send_eod_report", lambda *a, **k: None)
    monkeypatch.setattr(rl, "send_text", lambda *a, **k: True)
    env.rp.state["trades"].append({"id": "W|ML|long|x", "symbol": "W", "strategy": "ML", "direction": "long",
                                   "scope": "LARGE", "signal_time": "not-a-candle", "status": "PENDING"})
    env.rp.state["last_ts"]["X"] = str(env.df["timestamp"].iat[19])
    env.rp.finish_day(lambda s: env.df)
    assert {m for s, _, m in env.calls if s == "W"} == {("ML",)}
    assert {m for s, _, m in env.calls if s == "X"} == {("BASE", "ML")}


def test_universe_daily_base_plus_same_day_joiners(env, monkeypatch):
    import os

    import data.instruments as ins
    import nse_lcr.live as lcr
    base = {f"S{i}": "MID" for i in range(150)}
    monkeypatch.setattr(lcr, "universe", lambda: base)
    assert rl.wide_universe() == (base, set())

    def boom():
        raise AssertionError("rebuilt")
    monkeypatch.setattr(lcr, "universe", boom)
    monkeypatch.setattr(lcr, "off_index_large_caps", lambda: {"S1": ("MID", 30000), "NEWCO": ("MID", 20500)})
    monkeypatch.setattr(ins, "get_instrument_key", lambda s: f"NSE_EQ|{s}")
    assert rl.wide_universe() == (base, set())                          # inside the refresh interval
    monkeypatch.setattr(config, "LIVE_RESEARCH_WIDE_REFRESH_MIN", 0)
    tags, joined = rl.wide_universe()
    assert joined == {"NEWCO"} and len(tags) == 151 and tags["NEWCO"].startswith("MID · off-index · joined ")
    assert tags["S1"] == "MID"                                          # already in the base list
    day = datetime.now(rl.MARKET_TZ).strftime("%Y%m%d")
    os.remove(os.path.join(config.LIVE_STATE_DIR, f"wide_universe_{day}.json"))
    monkeypatch.setattr(lcr, "universe", lambda: {"A": "MID"})
    assert rl.wide_universe() == ({"A": "MID"}, set())                 # short list: used, not kept
    assert not any(n.startswith("wide_universe_") for n in os.listdir(config.LIVE_STATE_DIR))


def test_trending_now_filters_and_tags(env, monkeypatch):
    import nse_scr.data as scr_data
    rows = [dict(symbol="RUN", ltp=300.0, pct=8.4, vol_x_week=5.2, turnover=1e8, sources="gainer+shocker"),
            dict(symbol="DROP", ltp=300.0, pct=-6.0, vol_x_week=4.0, turnover=1e8, sources="loser"),
            dict(symbol="FLAT", ltp=300.0, pct=1.0, vol_x_week=None, turnover=1e8, sources="active_value"),
            dict(symbol="THIN", ltp=300.0, pct=5.0, vol_x_week=None, turnover=1e7, sources="gainer"),
            dict(symbol="PENNY", ltp=10.0, pct=9.0, vol_x_week=None, turnover=1e8, sources="gainer"),
            dict(symbol="BIG", ltp=900.0, pct=3.1, vol_x_week=None, turnover=5e8, sources="active_value"),
            dict(symbol="T2T", ltp=300.0, pct=7.0, vol_x_week=None, turnover=1e8, sources="gainer")]
    monkeypatch.setattr(scr_data, "watchlist", lambda: pd.DataFrame(rows))
    monkeypatch.setattr(rl, "_eq_series", lambda: {"RUN", "DROP", "FLAT", "THIN", "PENNY", "BIG"})
    out = rl.trending_now({"BIG": "LARGE"})
    assert out == {"RUN": "SMALL · trending +8.4%, volume 5.2x", "BIG": "LARGE · trending +3.1%"}
    assert list(out) == ["RUN", "BIG"]
    monkeypatch.setattr(scr_data, "watchlist", lambda: pd.DataFrame(columns=["symbol"]))
    assert rl.trending_now({}) == {}


def test_stock_added_during_the_day_starts_at_the_candle_it_appears_on(env):
    ts = env.df["timestamp"]
    env.rp.process_symbol("T", env.df.iloc[:25], wide_tag="SMALL · trending +5.0%", join_now=True)
    assert _evaluated(env.calls, "ML") == [ts.iat[24]]                  # no candles from before it was seen
    env.calls.clear()
    env.rp.process_symbol("T", env.df.iloc[:26], wide_tag="SMALL · trending +5.2%", join_now=True)
    assert _evaluated(env.calls, "ML") == [ts.iat[25]]                  # next run: carries on
    env.calls.clear()
    env.rp.process_symbol("T", env.df, wide_tag="SMALL · trending +6.0%", join_now=True)
    assert _evaluated(env.calls, "ML") == [ts.iat[29]]                  # back after 20 min: from "now" again
    env.calls.clear()
    env.rp.process_symbol("U", env.df, wide_tag="MID")                  # a base-list stock keeps its backlog
    assert len(_evaluated(env.calls, "ML")) == 30 - config.MIN_WARMUP_CANDLES
