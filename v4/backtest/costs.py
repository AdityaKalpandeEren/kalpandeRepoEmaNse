"""
Indian equity DELIVERY (CNC) charges, per order. All rates in config.costs.

  brokerage   min(brokerage_pct * value, brokerage_cap) per order (0 at many brokers)
  STT         0.1% of buy AND sell value (delivery)
  exchange    NSE transaction charge on turnover
  SEBI        Rs 10 per crore of turnover
  GST         18% on brokerage + exchange + SEBI
  stamp duty  0.015% of the buy value
  DP charge   flat per scrip on each sell day (depository)
  slippage    applied to the fill price: buy at open*(1+s), sell at open*(1-s)
"""
from __future__ import annotations


def order_charges(value: float, side: str, c: dict) -> float:
    """Explicit charges (Rs) for one order of `value` Rs; excludes slippage."""
    if value <= 0:
        return 0.0
    brokerage = min(c["brokerage_pct"] * value, c["brokerage_cap"]) if c["brokerage_pct"] > 0 else 0.0
    exch = c["exchange"] * value
    sebi = c["sebi"] * value
    gst = c["gst"] * (brokerage + exch + sebi)
    stt = c["stt"] * value
    if side == "buy":
        return brokerage + exch + sebi + gst + stt + c["stamp_buy"] * value
    return brokerage + exch + sebi + gst + stt + c["dp_per_sell"]


def fill_price(price: float, side: str, c: dict) -> float:
    s = c["slippage_bps"] / 1e4
    return price * (1 + s) if side == "buy" else price * (1 - s)


def round_trip_pct(value: float, c: dict) -> float:
    """Total cost of buying and later selling `value` Rs, as a fraction (incl. slippage)."""
    s = c["slippage_bps"] / 1e4
    return (order_charges(value, "buy", c) + order_charges(value, "sell", c)) / value + 2 * s
