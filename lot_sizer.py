# lot_sizer.py
# Live-aware position sizing for MT5 "zero/commission" accounts.
# - Reads contract size, lot filters, leverage, and free margin from MT5
# - Mirrors backtest risk model: risk_per_trade, max_risk_ratio, min_stop_usd
# - Caps by broker max lot and config lot_cap
# - Uses real margin via order_calc_margin, backing off to fit free margin

from __future__ import annotations
from typing import Optional, Tuple
import MetaTrader5 as mt5
from config import CONFIG

SAFETY_MARGIN_FRACTION = 0.95  # leave some buffer instead of using all free margin

def _ensure_symbol(symbol: str) -> None:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"MT5: cannot select symbol {symbol}")

def _get_symbol_specs(symbol: str) -> dict:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"MT5: symbol_info({symbol}) returned None")
    return {
        "digits": info.digits,
        "point": info.point,
        "contract_size": float(getattr(info, "trade_contract_size", CONFIG.get("dollar_per_unit_per_lot", 100.0))),
        "min_lot": float(getattr(info, "volume_min", CONFIG.get("min_lot_size", 0.01))),
        "lot_step": float(getattr(info, "volume_step", CONFIG.get("lot_size_increment", 0.01))),
        "max_lot_broker": float(getattr(info, "volume_max", CONFIG.get("max_lots", 100.0))),
        "spread_points": int(getattr(info, "spread", 0)),
    }

def _get_account_specs() -> dict:
    ai = mt5.account_info()
    if ai is None:
        raise RuntimeError("MT5: account_info() returned None")
    return {
        "leverage": float(getattr(ai, "leverage", CONFIG.get("leverage", 500.0))),
        "equity": float(getattr(ai, "equity", CONFIG.get("start_balance", 10000.0))),
        "free_margin": float(getattr(ai, "margin_free", getattr(ai, "equity", CONFIG.get("start_balance", 10000.0)))),
        "currency": getattr(ai, "currency", "USD"),
    }

def _quantize_lots(lots: float, min_lot: float, step: float) -> float:
    if lots < min_lot:
        return 0.0
    steps = round(lots / step)
    q = steps * step
    # fix floating artifacts
    return round(max(q, min_lot), 3)

def _fit_margin(symbol: str, lots: float, price: float, free_margin: float, step: float) -> float:
    """
    Reduce lots until required margin <= free_margin * SAFETY_MARGIN_FRACTION.
    Uses BUY for margin calc; for spot gold BUY/SELL margin is typically identical.
    """
    if lots <= 0:
        return 0.0
    target = free_margin * SAFETY_MARGIN_FRACTION
    vol = lots
    while vol >= step:
        # mt5.order_calc_margin(order_type, symbol, volume, price) -> margin in account currency
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, vol, price)
        # Some brokers return None/negative on calc errors; guard and back off.
        if margin is None or margin < 0:
            vol -= step
            continue
        if margin <= target:
            return vol
        vol -= step
    return 0.0

def calculate_position_size(stop_distance_usd: float, price_ref: float) -> float:
    """
    Live-aware lot sizing:
      - stop_distance_usd: stop distance in USD per 1 oz (for XAUUSD) i.e., price units.
      - price_ref: reference price for margin calc (use current bid/ask).
    Returns a broker-quantized volume (lots). Returns 0.0 if no margin headroom.
    """
    symbol = CONFIG["symbol"]
    _ensure_symbol(symbol)

    specs = _get_symbol_specs(symbol)
    acct  = _get_account_specs()

    # Risk model (USD)
    min_stop = float(CONFIG.get("min_stop_usd", 0.0))
    stop_d = max(float(stop_distance_usd), min_stop)

    equity = acct["equity"]
    risk_pct = float(CONFIG.get("risk_per_trade", 0.02))
    max_rr  = float(CONFIG.get("max_risk_ratio", 0.03))
    dollar_risk = min(risk_pct * equity, max_rr * equity)

    # $ per $1 move per lot = contract_size (for XAUUSD typically 100 oz/lot)
    dollar_per_unit_per_lot = float(specs["contract_size"] or CONFIG.get("dollar_per_unit_per_lot", 100.0))
    risk_per_lot = stop_d * dollar_per_unit_per_lot
    pre_lots = dollar_risk / risk_per_lot if risk_per_lot > 0 else 0.0

    # Apply caps/floors
    min_lot = specs["min_lot"]
    step    = specs["lot_step"]
    broker_max = specs["max_lot_broker"]
    cfg_cap = float(CONFIG.get("max_lots", broker_max))
    hard_cap = min(broker_max, cfg_cap)
    pre_lots = min(max(pre_lots, 0.0), hard_cap)

    # Quantize to broker step
    lots_q = _quantize_lots(pre_lots, min_lot, step)
    if lots_q <= 0.0:
        return 0.0

    # Margin-fit using real MT5 calculation and free margin
    # Use BUY for calc; difference vs SELL is negligible for XAUUSD
    lots_fit = _fit_margin(symbol, lots_q, float(price_ref), acct["free_margin"], step)
    if lots_fit <= 0.0:
        return 0.0

    # Respect hard cap again post-fit and quantize once more
    lots_fit = min(lots_fit, hard_cap)
    lots_fit = _quantize_lots(lots_fit, min_lot, step)
    return lots_fit
