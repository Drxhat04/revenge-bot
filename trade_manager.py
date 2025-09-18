# trade_manager.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import re
import MetaTrader5 as mt5

from config import CONFIG
from utils import fmt_time, is_near_swap_cutoff, now_utc

# ─────────────────────────── Config flags (single-target + overtime) ───────────────────────────

ST_ON  = bool(CONFIG.get("force_single_entry", False))
ST_TP  = CONFIG.get("single_entry_target", None)
BE_AFTER_TP1 = bool(CONFIG.get("be_after_tp1", True))

TP3_MULT = float(CONFIG.get("tp3_mult", 1.80))
TP4_MULT = float(CONFIG.get("tp4_mult", 2.40))
USE_TP3  = bool(CONFIG.get("use_tp3", False)) or (ST_ON and ST_TP == 3)
USE_TP4  = bool(CONFIG.get("use_tp4", False)) or (ST_ON and ST_TP == 4)

FC_HOURS   = float(CONFIG.get("force_close_hours", 24))
GRACE_HRS  = float(CONFIG.get("grace_hours", 0))
FC_POLICY  = str(CONFIG.get("force_close_policy", "final_close")).lower()  # "final_close"|"tp2"|"tp1"
OT_SL_MODE = str(CONFIG.get("overtime_sl_mode", "none")).lower()
MAX_EXT    = CONFIG.get("max_extension_hours", None)  # None or hours

# ─────────────────────────── Helpers ───────────────────────────

@dataclass
class SymbolFilters:
    min_lot: float
    lot_step: float
    digits: int

def _pos_time_dt(pos) -> datetime:
    """
    Coerce MT5 position time to naive UTC datetime.
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
    Align price to trade_tick_size/point; round to digits.
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
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0
    level_pts = int(getattr(info, "trade_stops_level", 0) or 0)
    tick = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 0.0)
    return float(level_pts) * float(tick)

def _respect_stops_level(symbol: str, raw_sl: Optional[float], raw_tp: Optional[float],
                         direction: str, ref_bid: float, ref_ask: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Ensure SL/TP are at least broker min distance from current market.
    """
    if raw_sl is None and raw_tp is None:
        return None, None

    info = mt5.symbol_info(symbol)
    if info is None:
        return raw_sl, raw_tp

    min_dist = _stops_min_distance(symbol)
    tick = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 0.0)
    buf = tick  # safety tick

    sl = raw_sl
    tp = raw_tp

    if direction == "BUY":
        if sl is not None and (ref_bid - sl) < (min_dist + buf):
            sl = ref_bid - (min_dist + buf)
        if tp is not None and (tp - ref_ask) < (min_dist + buf):
            tp = ref_ask + (min_dist + buf)
    else:
        if sl is not None and (sl - ref_ask) < (min_dist + buf):
            sl = ref_ask + (min_dist + buf)
        if tp is not None and (ref_bid - tp) < (min_dist + buf):
            tp = ref_bid - (min_dist + buf)

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
        return "BUY", "bid"
    else:
        return "SELL", "ask"

def _current_prices(symbol: str) -> Tuple[float, float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("No tick data")
    return float(tick.bid), float(tick.ask)

def _close_filling():
    ov = str(CONFIG.get("filling_override", "") or "").upper().strip()
    if ov == "FOK":
        return mt5.ORDER_FILLING_FOK
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

# ─────────────────────────── Level math + overtime ───────────────────────────

RUNNER_FRACTION_OF_TOTAL = 1.0 / 3.0   # legacy partial sizing
MAIN_FRACTION_OF_TOTAL   = 1.0 - RUNNER_FRACTION_OF_TOTAL

def _parse_stop_from_comment(comment: str) -> Optional[float]:
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
    1) If broker SL is set → return |entry - sl| (must be > 0 and finite)
    2) Else parse sd= from comment (e.g., 'sd=9.2'); if valid and > 0, use it
    3) Else fallback: max(min_stop_usd, risk_mult * zone_width_usd)
    Always returns a positive float.
    """
    import math

    # 1) From broker SL
    try:
        entry = float(getattr(pos, "price_open", 0.0) or 0.0)
        sl_raw = getattr(pos, "sl", None)
        sl_val = None if sl_raw in (None, 0.0) else float(sl_raw)
        if sl_val is not None and math.isfinite(entry) and math.isfinite(sl_val):
            d = abs(entry - sl_val)
            if d > 0:
                return float(d)
    except Exception:
        pass  # fall through

    # 2) From comment "sd=<number>"
    try:
        sd = _parse_stop_from_comment(getattr(pos, "comment", "") or "")
        if isinstance(sd, (int, float)) and sd > 0 and math.isfinite(sd):
            return float(sd)
    except Exception:
        pass  # fall through

    # 3) Structural fallback (guaranteed numeric)
    try:
        structural = float(CONFIG.get("risk_mult", 2.3)) * float(CONFIG.get("zone_width_usd", 5.0))
        floor = float(CONFIG.get("min_stop_usd", 1.0))
        fallback = max(floor, structural)
        if not math.isfinite(fallback) or fallback <= 0:
            fallback = 1.0
        return float(fallback)
    except Exception:
        # absolute last resort
        return 1.0

def calculate_levels(entry_price: float, direction: str, stop_distance: float) -> dict:
    """
    Compute SL/TP levels from entry, direction, and stop distance.
    - Accepts messy inputs and always returns a valid dict.
    - TP3/TP4 are included only if enabled by config flags (USE_TP3/USE_TP4).
    """
    import math

    # 1) Sanitize entry price
    try:
        ep = float(entry_price)
        if not math.isfinite(ep):
            raise ValueError
    except Exception:
        ep = 0.0  # last-resort fallback; downstream logic should still be safe

    # 2) Sanitize/repair stop distance (must be > 0)
    try:
        sd = float(stop_distance)
        if not (math.isfinite(sd) and sd > 0):
            raise ValueError
    except Exception:
        try:
            structural = float(CONFIG.get("risk_mult", 2.3)) * float(CONFIG.get("zone_width_usd", 5.0))
            floor = float(CONFIG.get("min_stop_usd", 1.0))
            sd = max(floor, structural)
            if not (math.isfinite(sd) and sd > 0):
                sd = 1.0
        except Exception:
            sd = 1.0

    # 3) Multipliers (tolerant)
    def _safe_mult(key: str, default: float) -> float:
        try:
            v = float(CONFIG.get(key, default))
            return v if math.isfinite(v) else default
        except Exception:
            return default

    tp1_mult = _safe_mult("tp1_mult", 0.68)
    tp2_mult = _safe_mult("tp2_mult", 1.16)

    # 4) Direction (tolerant)
    d = (direction or "").upper()
    if d not in ("BUY", "SELL"):
        d = "BUY"

    # 5) Levels
    if d == "BUY":
        sl = ep - sd
        lvls = {
            "tp1": ep + tp1_mult * sd,
            "tp2": ep + tp2_mult * sd,
            "tp3": (ep + TP3_MULT * sd) if USE_TP3 else None,
            "tp4": (ep + TP4_MULT * sd) if USE_TP4 else None,
            "sl":  sl,
            "be":  ep,
        }
    else:
        sl = ep + sd
        lvls = {
            "tp1": ep - tp1_mult * sd,
            "tp2": ep - tp2_mult * sd,
            "tp3": (ep - TP3_MULT * sd) if USE_TP3 else None,
            "tp4": (ep - TP4_MULT * sd) if USE_TP4 else None,
            "sl":  sl,
            "be":  ep,
        }

    return lvls

def _overtime_state(entry_time: datetime, now: datetime, entry_price: float, sl: float,
                    stop_distance: float, direction: str):
    """
    Returns: (in_overtime: bool, adjust_to: "tp1"/"tp2"/None, new_sl: float, ext_deadline: datetime|None)
    """
    hardcut = entry_time + timedelta(hours=FC_HOURS)
    grace_end = hardcut + timedelta(hours=GRACE_HRS)
    ext_deadline = None if MAX_EXT is None else (grace_end + timedelta(hours=float(MAX_EXT)))

    # default: not in overtime, unchanged SL
    new_sl = sl
    adjust = None  # "tp1"/"tp2"/None

    if now <= hardcut:
        return False, None, new_sl, ext_deadline

    if FC_POLICY in ("tp1", "tp2") and now > grace_end:
        adjust = FC_POLICY
        if OT_SL_MODE == "to_entry":
            new_sl = entry_price
        elif OT_SL_MODE == "tighten_to_half":
            if direction == "BUY":
                new_sl = max(sl, entry_price - 0.5 * stop_distance)
            else:
                new_sl = min(sl, entry_price + 0.5 * stop_distance)

    return True, adjust, new_sl, ext_deadline

def check_exit_conditions(pos, levels: dict) -> str | None:
    """
    Returns an exit reason or None.
    Reasons include: "sl","tp1","tp1_be","tp2","tp3","tp4","force_close",
    "final_close_timeout","force_tp1","force_tp2","swap_cutoff".
    """
    now = now_utc()
    entry_time = _pos_time_dt(pos)
    direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"

    # --- Guarantee numeric SL/stop_distance even when SL is missing ---
    be_val = float(levels.get("be", float(pos.price_open)))
    sl_val = levels.get("sl")
    if sl_val is None:
        sd = _infer_stop_distance(pos)
        sl_val = be_val - sd if direction == "BUY" else be_val + sd
        levels["sl"] = _normalize_price(pos.symbol, sl_val)

    stop_distance = abs(be_val - float(levels["sl"]))

    # --- Swap-avoidance (preempts everything) ---
    if CONFIG.get("swap_avoidance", False) and is_near_swap_cutoff(now):
        return "swap_cutoff"

    # --- Overtime state & deadlines ---
    in_ot, adjust, sl_adj, ext_deadline = _overtime_state(
        entry_time, now, be_val, float(levels["sl"]), stop_distance, direction
    )

    # Hard kill if final_close and past hardcut
    if FC_POLICY == "final_close" and in_ot:
        return "force_close"

    # Absolute extension cap
    if ext_deadline is not None and now > ext_deadline:
        return "final_close_timeout"

    # --- Prices (side-of-book checks) ---
    bid, ask = _current_prices(pos.symbol)

    # SL with (possibly) adjusted SL
    if direction == "BUY":
        if bid <= sl_adj:
            return "sl"
    else:
        if ask >= sl_adj:
            return "sl"

    def reached(p: Optional[float]) -> bool:
        if p is None:
            return False
        return (ask >= p) if direction == "BUY" else (bid <= p)

    # --- Single-target mode ---
    if ST_ON:
        if ST_TP == 3 and reached(levels.get("tp3")): return "tp3"
        if ST_TP == 4 and reached(levels.get("tp4")): return "tp4"
        if ST_TP == 1 and reached(levels.get("tp1")): return "tp1"
        if ST_TP == 2 and reached(levels.get("tp2")): return "tp2"
        # After grace, collapse remaining to TP2/TP1
        if adjust == "tp2" and reached(levels.get("tp2")): return "force_tp2"
        if adjust == "tp1" and reached(levels.get("tp1")): return "force_tp1"
        return None

    # --- Multi-target legacy ---
    if reached(levels.get("tp4")): return "tp4"
    if reached(levels.get("tp3")): return "tp3"
    if reached(levels.get("tp2")): return "tp2"
    if reached(levels.get("tp1")):
        return "tp1_be" if BE_AFTER_TP1 else "tp1"
    if adjust == "tp2" and reached(levels.get("tp2")): return "force_tp2"
    if adjust == "tp1" and reached(levels.get("tp1")): return "force_tp1"
    return None

# ─────────────────────────── Position closing orchestration ───────────────────────────

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
    (Used only in legacy multi-target mode with BE-after-TP1.)
    """
    symbol = pos.symbol
    filters = _get_symbol_filters(symbol)
    direction, close_side = _position_price_side(pos)

    total = float(pos.volume)
    close_vol = _quantize_volume(total * main_fraction, filters)

    if close_vol >= filters.min_lot and close_vol <= total:
        close_result = _send_close_market(symbol, close_vol, close_side, reason)
    else:
        close_result = {"ok": False, "stage": "partial_skip", "reason": "below_min_lot"}

    # Re-query position and move SL to BE
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

def get_open_positions(symbol: str, magic: int):
    """
    Fetch all live positions for the bot (filtered by magic).
    """
    all_positions = mt5.positions_get(symbol=symbol)
    if all_positions is None:
        return []
    return [p for p in all_positions if int(getattr(p, "magic", 0)) == int(magic)]

def manage_all_trades(notifier=None) -> List[dict]:
    """
    Runs exit checks on all open bot trades. Closes or partially closes if needed.
    If a notifier is provided, it updates the linked Telegram message.
    """
    symbol = CONFIG["symbol"]
    magic = int(CONFIG["magic"])

    open_trades = get_open_positions(symbol, magic)
    closed_logs: List[dict] = []

    for pos in open_trades:
        direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        entry_price = float(pos.price_open)

        stop_dist = _infer_stop_distance(pos)
        levels = calculate_levels(entry_price, direction, stop_dist)

        reason = check_exit_conditions(pos, levels)
        if not reason:
            continue

        # Terminal events → full close
        if reason in ("sl","tp2","tp3","tp4","force_close","final_close_timeout","swap_cutoff","force_tp1","force_tp2"):
            # Hit notifications before sending close, to show which TP/SL was reached
            try:
                if notifier:
                    if reason in ("tp2","force_tp2"): notifier.mark_tp_hit(int(pos.ticket), tp_idx=2, hit_price=float(levels.get("tp2") or entry_price))
                    elif reason == "tp3": notifier.mark_tp_hit(int(pos.ticket), tp_idx=3, hit_price=float(levels.get("tp3") or entry_price))
                    elif reason == "tp4": notifier.mark_tp_hit(int(pos.ticket), tp_idx=4, hit_price=float(levels.get("tp4") or entry_price))
                    elif reason == "sl": notifier.mark_sl_hit(int(pos.ticket), hit_price=float(levels.get("sl") or entry_price))
            except Exception:
                pass

            res = close_position(pos, reason)
            try:
                if notifier and res.get("exit_ok", False):
                    notifier.mark_closed(int(pos.ticket), reason=reason, exit_price=float(res.get("exit_price") or 0.0))
            except Exception:
                pass

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

        # Legacy TP1 partial with BE
        if reason == "tp1" and (not ST_ON) and BE_AFTER_TP1:
            part = partial_close_and_move_be(pos, MAIN_FRACTION_OF_TOTAL, be_price=entry_price, reason="tp1")
            try:
                if notifier and part.get("partial_ok", False):
                    notifier.mark_partial(int(pos.ticket), lots_closed=float(pos.volume) * MAIN_FRACTION_OF_TOTAL, exit_price=float(part.get("partial_result", {}).get("price") or 0.0))
                    notifier.mark_tp_hit(int(pos.ticket), tp_idx=1, hit_price=float(levels.get("tp1") or entry_price))
            except Exception:
                pass

            closed_logs.append({
                "ticket": pos.ticket,
                "symbol": symbol,
                "direction": direction,
                "volume_before": float(pos.volume),
                "entry_time": fmt_time(_pos_time_dt(pos)),
                "entry_price": entry_price,
                "action": "tp1_partial_and_BE",
                **part,
            })
            continue

        if reason == "tp1_be":
            mod = _modify_position_sl_tp(symbol, int(pos.ticket), sl=_normalize_price(symbol, entry_price), tp=None)
            try:
                if notifier:
                    notifier.mark_tp_hit(int(pos.ticket), tp_idx=1, hit_price=float(levels.get("tp1") or entry_price))
            except Exception:
                pass
            closed_logs.append({
                "ticket": pos.ticket,
                "symbol": symbol,
                "direction": direction,
                "entry_time": fmt_time(_pos_time_dt(pos)),
                "entry_price": entry_price,
                "action": "tp1_BE_only",
                "move_be_ok": bool(mod.get("ok", False)),
                "move_be_result": mod,
            })
            continue

    return closed_logs
