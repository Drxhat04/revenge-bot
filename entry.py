import time
import MetaTrader5 as mt5
from datetime import datetime
from math import isfinite
from config import CONFIG
from lot_sizer import calculate_position_size

# ───────── ticks ─────────
def fetch_tick(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("Failed to fetch live tick")
    return float(tick.bid), float(tick.ask)

# ───────── bands (trigger logic) ─────────
def _band_for(direction: str, lo: float, hi: float) -> tuple[float, float, float]:
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
    trig = band_trigger_price(direction, entry_low, entry_high)
    if not isfinite(trig):
        return (False, trig)
    return ((ask >= trig), trig) if direction == "BUY" else ((bid <= trig), trig)

# ───────── helpers ─────────
def _normalize_price(symbol: str, price: float) -> float:
    """
    Align price to the symbol's trade_tick_size (preferred) or point, then round to digits.
    Prevents TRADE_RETCODE_INVALID_PRICE (10009).
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return float(price)
    digits  = int(getattr(info, "digits", 2))
    tick_sz = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 0.0)
    p = float(price)
    if tick_sz > 0:
        p = round(round(p / tick_sz) * tick_sz, digits)
    else:
        p = round(p, digits)
    return p

def _quantize_to_step(vol: float, symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    step = float(getattr(info, "volume_step", CONFIG.get("lot_size_increment", 0.01)))
    vmin = float(getattr(info, "volume_min", CONFIG.get("min_lot_size", 0.01)))
    if vol <= 0:
        return 0.0
    q = round(vol / step) * step
    return round(max(q, vmin), 3)

def _fit_to_margin(symbol: str, lots: float, price: float) -> float:
    ai = mt5.account_info()
    if ai is None:
        return 0.0
    free = float(getattr(ai, "margin_free", getattr(ai, "equity", 0.0)))
    target = free * 0.95
    step = float(getattr(mt5.symbol_info(symbol), "volume_step", CONFIG.get("lot_size_increment", 0.01)))
    v = lots
    while v >= step:
        m = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, v, price)
        if m is not None and m >= 0 and m <= target:
            return _quantize_to_step(v, symbol)
        v = round(v - step, 3)
    return 0.0

def _sanitize_comment(s: str, max_len: int = 24) -> str:
    return s.encode("ascii", "ignore").decode("ascii")[:max_len]

def _build_request(direction: str, symbol: str, price: float, lots: float, comment: str) -> dict:
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": _normalize_price(symbol, price),
        "deviation": int(CONFIG.get("deviation", 300)),
        "magic": int(CONFIG.get("magic", 234000)),
        "comment": _sanitize_comment(comment, 24),
        "type_time": mt5.ORDER_TIME_GTC,
        "sl": 0.0,
        "tp": 0.0,
        # "type_filling" set per attempt
    }

def _supported_fillings(symbol: str):
    """
    Exness metals: restrict to IOC first, FOK second. Optional override via config to force IOC or FOK.
    """
    ov = str(CONFIG.get("filling_override", "") or "").upper().strip()
    if ov == "IOC":
        return [(mt5.ORDER_FILLING_IOC, "IOC")]
    if ov == "FOK":
        return [(mt5.ORDER_FILLING_FOK, "FOK")]
    return [(mt5.ORDER_FILLING_IOC, "IOC"), (mt5.ORDER_FILLING_FOK, "FOK")]

# ───────── SL/TP policy helpers (NEW) ─────────
def _compute_levels(direction: str, entry_price: float, stop_dist: float):
    tp1_mult = float(CONFIG.get("tp1_mult", 0.68))
    tp2_mult = float(CONFIG.get("tp2_mult", 1.16))
    if direction == "BUY":
        sl  = entry_price - stop_dist
        tp1 = entry_price + tp1_mult * stop_dist
        tp2 = entry_price + tp2_mult * stop_dist
    else:
        sl  = entry_price + stop_dist
        tp1 = entry_price - tp1_mult * stop_dist
        tp2 = entry_price - tp2_mult * stop_dist
    return sl, tp1, tp2

def _apply_sltp_policy(symbol: str, req: dict, direction: str,
                       entry_price: float, stop_dist: float) -> None:
    """
    Set broker SL/TP on the request based on configuration flags:
      - place_sl_on_entry: bool
      - place_tp_on_entry: "none" | "tp1" | "tp2"
    """
    place_sl = bool(CONFIG.get("place_sl_on_entry", True))
    tp_choice = str(CONFIG.get("place_tp_on_entry", "none")).strip().lower()

    if not place_sl and tp_choice == "none":
        return

    sl, tp1, tp2 = _compute_levels(direction, entry_price, stop_dist)
    if place_sl:
        req["sl"] = _normalize_price(symbol, sl)
    if tp_choice == "tp1":
        req["tp"] = _normalize_price(symbol, tp1)
    elif tp_choice == "tp2":
        req["tp"] = _normalize_price(symbol, tp2)
    # else leave TP=0.0 (managed by code later)

# ───────── entry (log each attempt) ─────────
def send_market_order(direction: str, entry_price_ref: float, stop_dist: float,
                      *, trigger_price: float | None = None) -> dict:
    symbol = CONFIG["symbol"]

    # 1) Size
    base_lots = calculate_position_size(stop_dist, entry_price_ref)
    if base_lots <= 0:
        raise RuntimeError("Lot sizing returned zero — check margin/equity/config")
    total_lots = _quantize_to_step(base_lots * 1.5, symbol)

    # 2) Fit to margin
    bid, ask = fetch_tick(symbol)
    side_price = ask if direction == "BUY" else bid
    total_lots = _fit_to_margin(symbol, total_lots, side_price)
    if total_lots <= 0:
        raise RuntimeError("Not enough free margin for requested volume")

    # 3) Build comment
    bits = [f"sd={stop_dist:.2f}"]
    if trigger_price is not None:
        bits.append(f"trig={trigger_price:.2f}")
    comment = " ".join(bits)

    attempts = []
    last_error = None
    price_used = None

    for fill_const, fill_label in _supported_fillings(symbol):
        for _ in range(6):  # retry per filling with fresh price
            bid, ask = fetch_tick(symbol)
            price_used = _normalize_price(symbol, ask if direction == "BUY" else bid)

            req = _build_request(direction, symbol, price_used, total_lots, comment)
            req["type_filling"] = fill_const

            # NEW: set broker-side SL/TP according to config
            _apply_sltp_policy(symbol, req, direction, price_used, stop_dist)

            # pre-check
            check = mt5.order_check(req)
            check_rc = getattr(check, "retcode", None)
            check_cmt = getattr(check, "comment", "")

            if check_rc not in (mt5.TRADE_RETCODE_DONE, 0):
                attempts.append({
                    "filling": fill_label, "price": price_used,
                    "check_retcode": check_rc, "check_comment": check_cmt,
                    "send_retcode": None, "send_comment": None,
                })
                break  # no point in retrying this filling if precheck fails

            # send
            res = mt5.order_send(req)
            send_rc = getattr(res, "retcode", None) if res is not None else None
            send_cmt = getattr(res, "comment", "") if res is not None else None

            attempts.append({
                "filling": fill_label, "price": price_used,
                "check_retcode": check_rc, "check_comment": check_cmt,
                "send_retcode": send_rc, "send_comment": send_cmt,
            })

            if res is not None and send_rc in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                return {
                    "requested_lots": total_lots,
                    "requested_price": price_used,
                    "entry_time": datetime.utcnow(),
                    "entry_result": res,
                    "attempts": attempts,
                    "comment_used": _sanitize_comment(comment, 24),
                }

            try:
                last_error = mt5.last_error()
            except Exception:
                last_error = None

            time.sleep(0.02)

    if last_error is None:
        try:
            last_error = mt5.last_error()
        except Exception:
            last_error = None

    return {
        "requested_lots": total_lots,
        "requested_price": price_used,
        "entry_time": datetime.utcnow(),
        "entry_result": None,
        "attempts": attempts,
        "last_error": last_error,
        "comment_used": _sanitize_comment(comment, 24),
    }
