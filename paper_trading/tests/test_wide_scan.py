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
    assert "Wide scan: MID · off-index stock" in rl._format_entries([t])


def test_end_of_day_pass_keeps_wide_stocks_to_the_wide_strategies(env, monkeypatch):
    monkeypatch.setattr(rl, "send_eod_report", lambda *a, **k: None)
    monkeypatch.setattr(rl, "send_text", lambda *a, **k: True)
    env.rp.state["trades"].append({"id": "W|ML|long|x", "symbol": "W", "strategy": "ML", "direction": "long",
                                   "scope": "LARGE", "signal_time": "not-a-candle", "status": "PENDING"})
    env.rp.state["last_ts"]["X"] = str(env.df["timestamp"].iat[19])
    env.rp.finish_day(lambda s: env.df)
    assert {m for s, _, m in env.calls if s == "W"} == {("ML",)}
    assert {m for s, _, m in env.calls if s == "X"} == {("BASE", "ML")}


def test_universe_is_built_once_a_day_and_short_lists_are_not_kept(env, monkeypatch):
    import nse_lcr.live as lcr
    big = {f"S{i}": "MID" for i in range(150)}
    monkeypatch.setattr(lcr, "universe", lambda: big)
    assert rl.wide_universe() == big

    def boom():
        raise AssertionError("rebuilt")
    monkeypatch.setattr(lcr, "universe", boom)
    assert rl.wide_universe() == big                                    # today's file
    day = datetime.now(rl.MARKET_TZ).strftime("%Y%m%d")
    import os
    os.remove(os.path.join(config.LIVE_STATE_DIR, f"wide_universe_{day}.json"))
    monkeypatch.setattr(lcr, "universe", lambda: {"A": "MID"})
    assert rl.wide_universe() == {"A": "MID"}
    assert not any(n.startswith("wide_universe_") for n in os.listdir(config.LIVE_STATE_DIR))
