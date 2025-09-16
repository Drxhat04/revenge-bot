# trade_manager.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import re
import MetaTrader5 as mt5

from config import CONFIG
from utils import fmt_time, is_near_swap_cutoff, now_utc

# ─────────────────────────── Helpers ───────────────────────────

@dataclass
class SymbolFilters:
    min_lot: float
    lot_step: float
    digits: int

def _pos_time_dt(pos) -> datetime:
    """
    Coerce MT5 position time to naive UTC datetime.
    Some brokers expose seconds-since-epoch; others a datetime.
    """
    t = getattr(pos, "time", None)
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(int(t), tz=timezone.utc).replace(tzinfo=None)
    if isinstance(t, datetime):
        return t if t.tzinfo is None else t.astimezone(timezone.utc).replace(tzinfo=None)
    return now_utc()

def _get_symbol_filters(symbol: str) -> SymbolFilters:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info({symbol}) returned None")
    return SymbolFilters(
        min_lot=float(getattr(info, "volume_min", 0.01)),
        lot_step=float(getattr(info, "volume_step", 0.01)),
        digits=int(getattr(info, "digits", 2)),
    )

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

def _stops_min_distance(symbol: str) -> float:
    """
    Convert broker stops level (points) to price distance.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0
    level_pts = int(getattr(info, "trade_stops_level", 0) or 0)
    # Prefer trade_tick_size if present, else point
    tick = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 0.0)
    return float(level_pts) * float(tick)

def _respect_stops_level(symbol: str, raw_sl: Optional[float], raw_tp: Optional[float],
                         direction: str, ref_bid: float, ref_ask: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Ensure SL/TP are at least broker min distance from current market.
    Adds a one-tick buffer if needed.
    """
    if raw_sl is None and raw_tp is None:
        return None, None

    info = mt5.symbol_info(symbol)
    if info is None:
        return raw_sl, raw_tp

    min_dist = _stops_min_distance(symbol)
    tick = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 0.0)
    buf = tick  # 1 extra tick for safety

    sl = raw_sl
    tp = raw_tp

    if direction == "BUY":
        # Closing side for SL is bid; for TP we check ask.
        if sl is not None and (ref_bid - sl) < (min_dist + buf):
            sl = ref_bid - (min_dist + buf)
        if tp is not None and (tp - ref_ask) < (min_dist + buf):
            tp = ref_ask + (min_dist + buf)
    else:
        # SELL
        if sl is not None and (sl - ref_ask) < (min_dist + buf):
            sl = ref_ask + (min_dist + buf)
        if tp is not None and (ref_bid - tp) < (min_dist + buf):
            tp = ref_bid - (min_dist + buf)

    # Normalize to tradable increments
    sl = None if sl is None else _normalize_price(symbol, sl)
    tp = None if tp is None else _normalize_price(symbol, tp)
    return sl, tp

def _quantize_volume(vol: float, f: SymbolFilters) -> float:
    if vol <= 0:
        return 0.0
    steps = round(vol / f.lot_step)
    q = steps * f.lot_step
    return round(max(q, f.min_lot), 3)

def _position_price_side(pos) -> Tuple[str, str]:
    """Return ('BUY'/'SELL', 'bid'/'ask' close-side label)."""
    if pos.type == mt5.ORDER_TYPE_BUY:
        return "BUY", "bid"   # closing a BUY uses bid
    else:
        return "SELL", "ask"  # closing a SELL uses ask

def _current_prices(symbol: str) -> Tuple[float, float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("No tick data")
    return float(tick.bid), float(tick.ask)

def _close_filling():
    """
    Respect filling_override for closes; default to IOC.
    """
    ov = str(CONFIG.get("filling_override", "") or "").upper().strip()
    if ov == "FOK":
        return mt5.ORDER_FILLING_FOK
    # Default (and Exness-friendly)
    return mt5.ORDER_FILLING_IOC

# ─────────────────────────── MT5 actions ───────────────────────

def _send_close_market(symbol: str, volume: float, close_side: str, reason: str) -> dict:
    bid, ask = _current_prices(symbol)
    order_type = mt5.ORDER_TYPE_SELL if close_side == "bid" else mt5.ORDER_TYPE_BUY
    raw_price = bid if close_side == "bid" else ask
    price = _normalize_price(symbol, raw_price)

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "deviation": int(CONFIG.get("deviation", 20)),
        "magic": int(CONFIG.get("magic", 234000)),
        "comment": f"exit: {reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _close_filling(),
    }

    check = mt5.order_check(req)
    if check is None or getattr(check, "retcode", 0) != mt5.TRADE_RETCODE_DONE:
        return {
            "ok": False,
            "stage": "order_check",
            "retcode": None if check is None else check.retcode,
            "comment": None if check is None else getattr(check, "comment", ""),
            "request": req,
        }

    res = mt5.order_send(req)
    rc = getattr(res, "retcode", 0) if res is not None else None
    ok = rc in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
    return {
        "ok": ok,
        "stage": "order_send",
        "retcode": rc,
        "comment": getattr(res, "comment", "") if res is not None else None,
        "result": res._asdict() if hasattr(res, "_asdict") else res,
        "price": price,
    }

def _modify_position_sl_tp(symbol: str, ticket: int, *,
                           sl: Optional[float] = None, tp: Optional[float] = None) -> dict:
    """
    Modify SL/TP on an existing position, normalizing and respecting min stop distance.
    """
    # Get current prices + direction to apply min distance properly
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return {"ok": False, "stage": "lookup", "error": "position_not_found", "ticket": ticket}

    pos = pos_list[0]
    direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
    bid, ask = _current_prices(symbol)

    sl_adj, tp_adj = _respect_stops_level(symbol, sl, tp, direction, bid, ask)

    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": int(ticket),
    }
    if sl_adj is not None:
        req["sl"] = float(sl_adj)
    if tp_adj is not None:
        req["tp"] = float(tp_adj)

    res = mt5.order_send(req)
    rc = getattr(res, "retcode", 0) if res is not None else None
    ok = rc in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
    return {
        "ok": ok,
        "stage": "modify",
        "retcode": rc,
        "comment": getattr(res, "comment", "") if res is not None else None,
        "requested": {"sl": sl, "tp": tp},
        "applied": {"sl": sl_adj, "tp": tp_adj},
        "result": res._asdict() if hasattr(res, "_asdict") else res,
    }

# ─────────────────────────── Core logic ───────────────────────────

RUNNER_FRACTION_OF_TOTAL = 1.0 / 3.0   # runner = 1/3 of total (base 2/3 + runner 1/3)
MAIN_FRACTION_OF_TOTAL   = 1.0 - RUNNER_FRACTION_OF_TOTAL  # 2/3

def get_open_positions(symbol: str, magic: int):
    """
    Fetch all live positions for the bot.
    """
    all_positions = mt5.positions_get(symbol=symbol)
    if all_positions is None:
        return []
    return [p for p in all_positions if int(getattr(p, "magic", 0)) == int(magic)]

def _parse_stop_from_comment(comment: str) -> Optional[float]:
    """
    Optional: parse 'sd=3.50' style stop distance from entry comment.
    """
    if not comment:
        return None
    m = re.search(r"\bsd\s*=\s*([0-9]*\.?[0-9]+)", str(comment))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def _infer_stop_distance(pos) -> float:
    """
    Determine stop distance for level calc:
      1) If SL is set on the position -> use |entry - sl|.
      2) Else, parse 'sd=' from comment if present.
      3) Else, fallback: max(min_stop_usd, risk_mult * zone_width_usd).
    """
    entry = float(pos.price_open)
    if getattr(pos, "sl", 0) not in (None, 0.0):
        return abs(entry - float(pos.sl))

    sd = _parse_stop_from_comment(getattr(pos, "comment", ""))
    if sd is not None and sd > 0:
        return sd

    structural = float(CONFIG.get("risk_mult", 2.3)) * float(CONFIG.get("zone_width_usd", 5.0))
    return max(float(CONFIG.get("min_stop_usd", 1.0)), structural)

def calculate_levels(entry_price: float, direction: str, stop_distance: float) -> dict:
    tp1_mult = float(CONFIG["tp1_mult"])
    tp2_mult = float(CONFIG["tp2_mult"])

    if direction == "BUY":
        return {
            "tp1": entry_price + tp1_mult * stop_distance,
            "tp2": entry_price + tp2_mult * stop_distance,
            "sl":  entry_price - stop_distance,
            "be":  entry_price,
        }
    else:
        return {
            "tp1": entry_price - tp1_mult * stop_distance,
            "tp2": entry_price - tp2_mult * stop_distance,
            "sl":  entry_price + stop_distance,
            "be":  entry_price,
        }

def check_exit_conditions(pos, levels: dict) -> str | None:
    """
    Check if any exit condition (TP1, TP2, SL, force-close, swap-avoidance) is met.
    Uses the same side-of-book convention as the backtest:
      • BUY: SL uses bid<=SL; TP uses ask>=TP
      • SELL: SL uses ask>=SL; TP uses bid<=TP
    """
    now = now_utc()
    entry_time = _pos_time_dt(pos)
    elapsed = now - entry_time

    # Force-close window
    if elapsed > timedelta(hours=float(CONFIG["force_close_hours"])):
        return "force_close"

    # Swap avoidance window
    if CONFIG.get("swap_avoidance", False) and is_near_swap_cutoff(now):
        return "swap_cutoff"

    # Live prices
    bid, ask = _current_prices(pos.symbol)
    direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"

    # Stop-loss check
    if direction == "BUY":
        if bid <= levels["sl"]:
            return "sl"
    else:
        if ask >= levels["sl"]:
            return "sl"

    # Take profit checks (TP2 first, then TP1)
    if direction == "BUY":
        if ask >= levels["tp2"]:
            return "tp2"
        if ask >= levels["tp1"]:
            return "tp1"
    else:
        if bid <= levels["tp2"]:
            return "tp2"
        if bid <= levels["tp1"]:
            return "tp1"

    return None

def close_position(pos, reason: str) -> dict:
    """
    Close FULL position at market and return metadata.
    """
    symbol = pos.symbol
    direction, close_side = _position_price_side(pos)
    volume = float(pos.volume)
    filters = _get_symbol_filters(symbol)
    vol_q = _quantize_volume(volume, filters)

    if vol_q <= 0.0:
        return {"error": "volume_too_small", "requested": volume}

    balance_before = getattr(mt5.account_info(), "balance", None)
    res = _send_close_market(symbol, vol_q, close_side, reason)
    balance_after = getattr(mt5.account_info(), "balance", None)

    return {
        "exit_time": now_utc(),
        "exit_reason": reason,
        "exit_ok": res.get("ok", False),
        "exit_side": close_side,
        "exit_price": res.get("price"),
        "mt5": res,
        "balance_before": balance_before,
        "balance_after": balance_after,
    }

def partial_close_and_move_be(pos, main_fraction: float, be_price: float, reason: str = "tp1") -> dict:
    """
    Close 'main_fraction' of the position and move SL of the remainder to BE.
    """
    symbol = pos.symbol
    filters = _get_symbol_filters(symbol)
    direction, close_side = _position_price_side(pos)

    total = float(pos.volume)
    close_vol = _quantize_volume(total * main_fraction, filters)

    # If close amount is smaller than min lot, skip partial but still try BE move
    if close_vol >= filters.min_lot and close_vol <= total:
        close_result = _send_close_market(symbol, close_vol, close_side, reason)
    else:
        close_result = {"ok": False, "stage": "partial_skip", "reason": "below_min_lot"}

    # After partial, move SL of remaining position to BE (respect stops level)
    # Re-query position because volume may have changed
    pos_after = mt5.positions_get(ticket=pos.ticket)
    current_pos = pos if not pos_after else pos_after[0]

    be_norm = _normalize_price(symbol, be_price)
    mod = _modify_position_sl_tp(symbol, int(current_pos.ticket), sl=be_norm, tp=None)

    return {
        "partial_ok": bool(close_result.get("ok", False)),
        "partial_result": close_result,
        "move_be_ok": bool(mod.get("ok", False)),
        "move_be_result": mod,
    }

def manage_all_trades() -> List[dict]:
    """
    Runs exit checks on all open bot trades. Closes or partially closes if needed.
    """
    symbol = CONFIG["symbol"]
    magic = int(CONFIG["magic"])

    open_trades = get_open_positions(symbol, magic)
    closed_logs: List[dict] = []

    for pos in open_trades:
        direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        entry_price = float(pos.price_open)

        # Stop distance inference (per-position)
        stop_dist = _infer_stop_distance(pos)

        # Compute levels
        levels = calculate_levels(entry_price, direction, stop_dist)

        # If already at BE (SL ~= entry), skip TP1 partial logic in future
        at_be = (getattr(pos, "sl", 0) not in (None, 0.0)) and (abs(float(pos.sl) - entry_price) < 1e-6)

        reason = check_exit_conditions(pos, levels)
        if not reason:
            continue

        if reason in ("force_close", "swap_cutoff", "sl", "tp2"):
            # Close entire position
            res = close_position(pos, reason)
            closed_logs.append({
                "ticket": pos.ticket,
                "symbol": symbol,
                "direction": direction,
                "volume": float(pos.volume),
                "entry_time": fmt_time(_pos_time_dt(pos)),
                "entry_price": entry_price,
                **res,
            })
            continue

        if reason == "tp1" and not at_be:
            # Partial close main, move remaining SL to BE
            part = partial_close_and_move_be(pos, MAIN_FRACTION_OF_TOTAL, be_price=entry_price, reason="tp1")
            closed_logs.append({
                "ticket": pos.ticket,
                "symbol": symbol,
                "direction": direction,
                "volume_before": float(pos.volume),
                "entry_time": fmt_time(pos.time),
                "entry_price": entry_price,
                "action": "tp1_partial_and_BE",
                **part,
            })
            continue

        # If TP1 hit but already at BE (runner stage), do nothing here;
        # runner will be managed by tp2 or be (SL) conditions in subsequent cycles.

    return closed_logs
