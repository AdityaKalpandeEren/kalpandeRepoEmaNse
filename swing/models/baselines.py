"""
Rule baselines the ML models must beat. Each returns a score per row
(higher = buy first); no fitting, so no leakage is possible.

  mom_baseline    12-1 month momentum (classic cross-sectional momentum)
  rev_baseline    1-week reversal: last week's losers
  combo_baseline  average of the two per-day ranks
"""
from __future__ import annotations

import pandas as pd


def score(name: str, df: pd.DataFrame) -> pd.Series:
    if name == "mom_baseline":
        return df["mom_12_1"]
    if name == "rev_baseline":
        return -df["ret_5"]
    if name == "combo_baseline":
        r1 = df.groupby("date")["mom_12_1"].rank(pct=True)
        r2 = (-df["ret_5"]).groupby(df["date"]).rank(pct=True)
        return (r1 + r2) / 2
    raise ValueError(name)


NAMES = ("mom_baseline", "rev_baseline", "combo_baseline")
