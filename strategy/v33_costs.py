"""
Exact NSE intraday (MIS) charges for one paper round trip - used by the
V3.3 trainer's report and by the live V3.3 runner, so both price trades
the same way.

  brokerage   config.V33_BROKERAGE_PER_ORDER per order (buy + sell), + 18% GST
  STT         0.025% of the SELL value
  exchange    0.00297% of turnover (both sides), + 18% GST
  SEBI fee    Rs 10 per crore (0.0001%) of turnover, + 18% GST
  stamp duty  0.003% of the BUY value
  slippage    config.V33_SLIPPAGE_BPS per side, applied to the fill prices
"""
import config

STT_SELL = 0.00025
EXCHANGE = 0.0000297
SEBI = 0.000001
STAMP_BUY = 0.00003
GST = 0.18


def round_trip(entry: float, exit_: float, qty: int, slippage_bps: float = None) -> dict:
    s = (config.V33_SLIPPAGE_BPS if slippage_bps is None else slippage_bps) / 1e4
    buy_px, sell_px = entry * (1 + s), exit_ * (1 - s)
    buy_val, sell_val = buy_px * qty, sell_px * qty
    turnover = buy_val + sell_val
    brokerage = 2 * config.V33_BROKERAGE_PER_ORDER
    exch, sebi = EXCHANGE * turnover, SEBI * turnover
    charges = (brokerage + exch + sebi) * (1 + GST) + STT_SELL * sell_val + STAMP_BUY * buy_val
    gross = sell_val - buy_val
    net = gross - charges
    return {"buy_px": buy_px, "sell_px": sell_px, "gross": gross, "charges": charges, "net": net,
            "net_pct": net / (entry * qty) if qty and entry else 0.0}


def net_return(entry: float, exit_: float, notional: float, slippage_bps: float = None) -> float:
    """Net return fraction of one round trip at a given notional (whole shares)."""
    qty = int(notional // entry) if entry > 0 else 0
    return round_trip(entry, exit_, qty, slippage_bps)["net_pct"] if qty > 0 else float("nan")
