"""Loss-cut variants SCORE_ENGINE_LC / J_VWAP_LC: offline tests (synthetic candles, no network)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402
from strategy import strategies as S  # noqa: E402
from strategy.regime import Regime  # noqa: E402

BULL = Regime(direction="strong_bull", volatility="normal", trend_strength=0.8)


def _df(hhmm="11:00", close=100.0, low=99.9, atr=0.2, z=-2.5):
    """One enriched candle: stretched below VWAP and turning up; the candle low sits inside
    both ATR stops, so the stop is set by the ATR multiple alone."""
    ts = pd.Timestamp(f"2026-10-13 {hhmm}", tz=config.MARKET_TIMEZONE)
    return pd.DataFrame([{"timestamp": ts, "open": close - 0.5, "high": close + 0.1, "low": low, "close": close,
                          "vwap": close + 3, "vwap_z": z, "rvol": 1.5, f"atr_{config.ATR_PERIOD}": atr}])


def test_lc_models_are_live_but_not_ml_candidates():
    for name in ("SCORE_ENGINE_LC", "J_VWAP_LC"):
        assert name in S.ENTRY_MODELS
        assert name not in S.BASE_MODELS                  # the ML filters loop over BASE_MODELS only
        assert name in config.LIVE_RESEARCH_STRATEGIES
    assert "SCORE_ENGINE" in S.BASE_MODELS and "J_VWAP_BAND_REVERSION" in S.BASE_MODELS
    assert "SCORE_ENGINE_LC" not in config.LIVE_RESEARCH_SILENT_STRATEGIES     # LC alerts on Telegram


def test_vwap_lc_stop_is_twice_as_wide_and_original_unchanged():
    df = _df()
    orig = S.model_j_vwap_band_reversion("X", df, "long", BULL)
    lc = S.model_j_vwap_lc("X", df, "long", BULL)
    assert orig.strategy == "J_VWAP_BAND_REVERSION" and lc.strategy == "J_VWAP_LC"
    assert orig.stop_loss == pytest.approx(100 - 1.5 * 0.2)             # unchanged: 1.5 x ATR
    assert lc.stop_loss == pytest.approx(100 - 3.0 * 0.2)               # LC: 3 x ATR
    assert lc.target - 100 == pytest.approx(2 * (orig.target - 100))   # same 1.5R target, twice the distance


@pytest.mark.parametrize("hhmm,fires", [("10:00", True), ("11:50", True), ("11:55", False), ("13:30", False)])
def test_vwap_lc_only_fills_before_noon(hhmm, fires):
    # signal candle 11:50 fills at the 11:55 open (fires); 11:55 fills at 12:00 (blocked)
    df = _df(hhmm)
    assert (S.model_j_vwap_lc("X", df, "long", BULL) is not None) is fires
    assert S.model_j_vwap_band_reversion("X", df, "long", BULL) is not None     # original: any time


def test_vwap_lc_long_only():
    assert S.model_j_vwap_lc("X", _df(), "short", BULL) is None


@pytest.mark.parametrize("level,orig_fires,lc_fires", [(0.50, False, False), (0.60, True, False), (0.70, True, True)])
def test_score_engine_lc_needs_65(monkeypatch, level, orig_fires, lc_fires):
    # every component at `level` -> total = 100 x level (the weights sum to 1)
    monkeypatch.setattr(S, "_component_scores", lambda row, df, d: {k: level for k in config.SCORE_WEIGHTS})
    df = _df()
    assert (S.model_score_engine("X", df, "long", BULL) is not None) is orig_fires
    lc = S.model_score_engine_lc("X", df, "long", BULL)
    assert (lc is not None) is lc_fires
    if lc is not None:
        orig = S.model_score_engine("X", df, "long", BULL)
        assert lc.strategy == "SCORE_ENGINE_LC" and lc.score == pytest.approx(70.0)
        assert (lc.entry - lc.stop_loss) == pytest.approx(2 * (orig.entry - orig.stop_loss))


def test_lc_skips_when_wide_stop_breaches_max_risk(monkeypatch):
    # ATR 0.6 on a Rs 100 stock: 1.5 x ATR = 0.9% (allowed), 3 x ATR = 1.8% > MAX_RISK_PCT 1.5% (skipped)
    monkeypatch.setattr(S, "_component_scores", lambda row, df, d: {k: 0.70 for k in config.SCORE_WEIGHTS})
    df = _df(atr=0.6, low=99.9)
    assert S.model_score_engine("X", df, "long", BULL) is not None
    assert S.model_score_engine_lc("X", df, "long", BULL) is None


def test_alert_text_names_the_right_bar():
    from paper_trading.research_live import _why, _STRATEGY_INFO
    t = {"strategy": "SCORE_ENGINE_LC", "reason": "trend=0.70 vwap=0.70", "score": 70}
    assert "needs 65" in _why(t)
    assert "needs 55" in _why({**t, "strategy": "SCORE_ENGINE"})
    assert "VWAP z-score" in _why({"strategy": "J_VWAP_LC", "reason": "vwap_z -2.40 (>= 2.0 SD stretch), turning up"})
    assert "LC" in _STRATEGY_INFO["SCORE_ENGINE_LC"] and "LC" in _STRATEGY_INFO["J_VWAP_LC"]
