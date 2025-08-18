# entry.py
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from math import isfinite
from config import CONFIG
from lot_sizer import calculate_position_size

def fetch_tick(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("Failed to fetch live tick")
    # return bid, ask
    return float(tick.bid), float(tick.ask)

def _band_for(direction: str, lo: float, hi: float) -> tuple[float, float, float]:
    """Return (band_low, band_high, mid)."""
    mid = (lo + hi) / 2.0
    if direction == "BUY":
        side = CONFIG.get("band_side_buy", "upper")
        return (mid, hi, mid) if side == "upper" else (lo, mid, mid)
    else:
        side = CONFIG.get("band_side_sell", "lower")
        return (lo, mid, mid) if side == "lower" else (mid, hi, mid)

def band_trigger_price(direction: str, lo: float, hi: float) -> float:
    bl, bh, _ = _band_for(direction, lo, hi)
    bw = bh - bl
    if bw <= 0:
        return float("nan")
    thr = float(CONFIG.get("band_entry_threshold", 0.0))
    thr = 0.0 if thr < 0.0 else 1.0 if thr > 1.0 else thr
    return (bl + thr * bw) if direction == "BUY" else (bh - thr * bw)

def should_trigger_trade(direction: str, entry_low: float, entry_high: float,
                         bid: float, ask: float) -> tuple[bool, float]:
    """
    Returns (triggered, trigger_price). Direction-sensitive, band-aware.
    BUY triggers when ask >= trigger; SELL when bid <= trigger.
    """
    trig = band_trigger_price(direction, entry_low, entry_high)
    if not isfinite(trig):
        return (False, trig)
    if direction == "BUY":
        return (ask >= trig, trig)
    else:
        return (bid <= trig, trig)

def _quantize_to_step(vol: float, symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    step = float(getattr(info, "volume_step", CONFIG.get("lot_size_increment", 0.01)))
    vmin = float(getattr(info, "volume_min", CONFIG.get("min_lot_size", 0.01)))
    if vol <= 0:
        return 0.0
    q = round(vol / step) * step
    return round(max(q, vmin), 3)

def _fit_to_margin(symbol: str, lots: float, price: float) -> float:
    """Reduce lots until order_calc_margin fits current free margin (95% buffer)."""
    ai = mt5.account_info()
    if ai is None:
        return 0.0
    free = float(getattr(ai, "margin_free", getattr(ai, "equity", 0.0)))
    target = free * 0.95
    info = mt5.symbol_info(symbol)
    step = float(getattr(info, "volume_step", CONFIG.get("lot_size_increment", 0.01)))
    v = lots
    while v >= step:
        m = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, v, price)
        if m is not None and m >= 0 and m <= target:
            return _quantize_to_step(v, symbol)
        v = round(v - step, 3)
    return 0.0

def send_market_order(direction: str, entry_price_ref: float, stop_dist: float,
                      *, trigger_price: float | None = None) -> dict:
    """
    Sends a live market order. Opens total lots = 1.5 × base (main + runner),
    matching the backtest exposure split.
    """
    symbol = CONFIG["symbol"]

    # 1) Base lots per risk model
    base_lots = calculate_position_size(stop_dist, entry_price_ref)
    if base_lots <= 0:
        raise RuntimeError("Lot sizing returned zero — check margin/equity/config")

    # 2) Total exposure = 1.5 × base (main 1.0 + runner 0.5)
    total_lots = _quantize_to_step(base_lots * 1.5, symbol)

    # 3) Fit to margin conservatively
    bid, ask = fetch_tick(symbol)
    side_price = ask if direction == "BUY" else bid
    total_lots = _fit_to_margin(symbol, total_lots, side_price)
    if total_lots <= 0:
        raise RuntimeError("Not enough free margin for requested volume")

    # 4) Send market
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price = side_price

    comment_bits = [ "live sweep-OB entry",
                     f"sd={stop_dist:.2f}" ]
    if trigger_price is not None:
        comment_bits.append(f"trig={trigger_price:.2f}")
    comment = " ".join(comment_bits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": total_lots,
        "type": order_type,
        "price": price,
        "deviation": CONFIG["deviation"],
        "magic": CONFIG["magic"],
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    return {
        "requested_lots": total_lots,
        "requested_price": price,
        "entry_time": datetime.utcnow(),
        "entry_result": result
    }
