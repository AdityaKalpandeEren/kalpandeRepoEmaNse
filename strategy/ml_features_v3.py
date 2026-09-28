"""
Features for model L_ML_META_V3 (NSE): every V2 feature, plus sector
tape, overnight macro, dynamic sector/stock sensitivities, breakout
structure, sector one-hots and the two V3-only candidate sources.
Shared by backtest/ml/build_dataset_v3.py and the live model.
"""
import math

import numpy as np

from strategy.ml_features_v2 import FEATURE_COLUMNS_V2, extract_features_v2
from strategy.sector_map import SECTOR_CODES
from strategy.v3_candidates import V3_MODELS

SECTOR_COLUMNS = [
    "sector_ret_day", "sector_ret_30m", "sector_ret_since_open", "sector_range_pos",
    "rs_vs_sector_day", "rs_vs_sector_30m", "sector_vs_nifty_day",
]
MACRO_COLUMNS = [
    "brent_ret_1d", "brent_ret_5d", "usdinr_ret_1d", "spx_ret_1d", "us10y_chg_1d", "gold_ret_1d",
    "sector_crude_beta", "crude_impulse", "sector_inr_beta", "inr_impulse", "sector_us_beta", "us_impulse",
    "stock_crude_beta", "stock_crude_impulse", "stock_inr_beta", "stock_inr_impulse",
    "stock_us_beta", "stock_us_impulse",
]
BREAKOUT_COLUMNS = ["dist_day_high", "new_day_high", "weak_mkt_rs"]
ALIGN_V3_COLUMNS = [
    "align_sector_day", "align_rs_sector", "align_crude_impulse", "align_inr_impulse",
    "align_us_impulse", "align_stock_crude_impulse", "align_stock_inr_impulse", "align_stock_us_impulse",
]
SECTOR_ONEHOT = [f"sec_{c}" for c in SECTOR_CODES]
V3_SOURCE_COLUMNS = [f"src_{n}" for n in V3_MODELS]

V3_ONLY_COLUMNS = (SECTOR_COLUMNS + MACRO_COLUMNS + BREAKOUT_COLUMNS + ALIGN_V3_COLUMNS
                   + SECTOR_ONEHOT + V3_SOURCE_COLUMNS)
FEATURE_COLUMNS_V3 = FEATURE_COLUMNS_V2 + V3_ONLY_COLUMNS


def _num(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if np.isfinite(v) else math.nan


def extract_features_v3(row, df, direction, regime, comps, source_strategy,
                        context_feats_v3: dict, signal) -> dict:
    """`context_feats_v3` = MarketContextV3.features_v3(); `signal` must be
    the V3-geometry signal (so risk_pct/risk_atr describe the real trade)."""
    feats = extract_features_v2(row, df, direction, regime, comps, source_strategy,
                                context_feats_v3, signal)
    for c in V3_ONLY_COLUMNS:
        if c in context_feats_v3:
            feats[c] = context_feats_v3[c]
    for name in V3_MODELS:
        feats[f"src_{name}"] = 1.0 if source_strategy == name else 0.0
    return {c: _num(feats.get(c)) for c in FEATURE_COLUMNS_V3}
