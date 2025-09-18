#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────
# backtest_engine.py · v10.5
#  • Band-trigger entries + configurable depth + fill-price policy
#  • Enforce minimum delay after signal (no same-minute fills)
#  • Robust stops: max(structural, ATR) when dynamic_stops is on
#  • Proper cost accounting and audit fields
#  • Margin stop — stop the whole BT when balance ≤ floor
#  • Configurable BE-after-TP1 + TP3/TP4 with extra units
#  • NEW: Single-target mode (route 100% to TP1/2/3/4; one-entry intent)
# ────────────────────────────────────────────────────────────────

from __future__ import annotations
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# ────────────────────────── defaults (only if caller forgets) ──
DEFAULTS = {
    # Risk & target multipliers
    "risk_mult": 2.3,
    "tp1_mult": 0.68,
    "tp2_mult": 1.16,
    # NEW: extra targets (off by default)
    "tp3_mult": 1.80,
    "tp4_mult": 2.40,
    "use_tp3": False,
    "use_tp4": False,
    "extra_split_across_tp3_tp4": False,   # if True and TP4 enabled: half extra at TP3, half at TP4
    "risk_per_trade": 0.02,
    "max_risk_ratio": 0.03,

    # Units (exposure weights in “units”, not lots)
    "main_units": 1.0,           # TP1
    "runner_units": 0.5,         # TP2
    "extra_runner_units": 0.5,   # TP3/TP4

    # Management
    "be_after_tp1": True,        # original behavior; set False to keep SL where it was

    # Volatility stops
    "dynamic_stops": True,
    "atr_multiplier": 3.0,
    "min_stop_usd": 1.0,        # absolute hard floor for stops used in sizing

    # Timeouts
    "no_touch_timeout_hours": 3,
    "force_close_hours": 24,

    "force_close_policy": "final_close",  # "final_close" | "tp2" | "tp1"
    "max_extension_hours": None,          # None = unlimited; or an int cap (e.g., 72)
    "grace_hours": 0,                     # hours after hardcut to keep aiming for the original TP
    "overtime_sl_mode": "none",           # "none" | "to_entry" | "tighten_to_half"


    # Lot sizing
    "min_lot_size": 0.01,
    "max_lots": 10.0,
    "lot_size_increment": 0.01,

    # Dealing costs & execution
    "tick_size": 0.0,                     # zero-spread account
    "slip_usd_side_lot": 5.0,
    "use_csv_spread": False,
    "commission_per_lot_per_side": 5.5,

    # Slippage scaling
    "slippage_scale_threshold_lots": 2.0,
    "slippage_scale_factor": 0.5,

    # Swap management
    "swap_avoidance": False,
    "swap_cutoff_hour": 20,
    "swap_buffer_hours": 3,
    "wednesday_multiplier": 3,
    "swap_fee_long": 0.0,
    "swap_fee_short": 0.0,

    # Instrument
    "dollar_per_unit_per_lot": 100.0,

    # Daily drawdown
    "day_loss_limit_pct": 0.05,

    # Leverage / margin
    "leverage": 500.0,

    # Equity
    "start_balance": 10_000.0,

    # ENTRY LOGIC
    "entry_trigger": "band",      # "band" | "mid"
    "band_side_buy": "upper",
    "band_side_sell": "lower",
    "band_entry_threshold": 0.00, # 0..1 inside chosen band
    "band_fill_price": "threshold",  # "threshold" | "touch"
    "entry_min_delay_minutes": 1,

    # ── Margin stop controls
    "stop_on_negative_balance": True,
    "min_equity_usd": 0.0,

    # ── Single-target mode
    "force_single_entry": False,   # when True, route 100% to one TP
    "single_entry_target": None,   # None or 1..4
}

# ╭─ CONFIG LOAD (used only if caller passes nothing) ───────────╮
def load_cfg(path: str | Path = "config.yaml") -> dict:
    cfg = DEFAULTS.copy()
    if Path(path).exists():
        user_cfg = yaml.safe_load(open(path)) or {}
        cfg.update(user_cfg)
    cfg["lot_cap"] = cfg.get("max_lots", DEFAULTS["max_lots"])
    return cfg
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ PRICE HELPERS ─────────────────────────────────────────────╮
ask_price = lambda bar: bar.close + (getattr(bar, 'spread', 0) / 2)
bid_price = lambda bar: bar.close - (getattr(bar, 'spread', 0) / 2)
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ SWAP MANAGEMENT ────────────────────────────────────────────╮
def calculate_swap_cost(entry_time: datetime, exit_time: datetime,
                        direction: str, lots: float, cfg: dict) -> float:
    swap_cost = 0.0
    current = entry_time.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < exit_time:
        swap_time = current.replace(hour=cfg['swap_cutoff_hour'], minute=0,
                                    second=0, microsecond=0)
        if entry_time <= swap_time < exit_time:
            mult = cfg['wednesday_multiplier'] if swap_time.weekday() == 2 else 1
            fee = cfg['swap_fee_long'] if direction.upper() == "BUY" else cfg['swap_fee_short']
            swap_cost += fee * lots * mult
        current += timedelta(days=1)
    return swap_cost


def is_near_swap_cutoff(signal_time: datetime, cfg: dict) -> bool:
    swap_time = signal_time.replace(hour=cfg['swap_cutoff_hour'], minute=0)
    if signal_time.hour >= cfg['swap_cutoff_hour']:
        swap_time += timedelta(days=1)  # Next day's swap time
    time_diff = (swap_time - signal_time)
    buffer = timedelta(hours=cfg['swap_buffer_hours'])
    return timedelta(0) <= time_diff < buffer
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ RISK-BASED LOT SIZING w/ leverage cap ──────────────────────╮
def calculate_position_size(cfg: dict, balance: float,
                             stop_distance: float, price: float) -> float:
    if balance <= 0:
        return 0.0

    stop_distance = max(stop_distance, cfg.get("min_stop_usd", 0.0))
    dollar_risk = cfg["risk_per_trade"] * balance
    dollar_risk = min(dollar_risk, cfg["max_risk_ratio"] * balance)
    if dollar_risk <= 0 or stop_distance <= 0:
        return 0.0

    risk_per_lot = stop_distance * cfg["dollar_per_unit_per_lot"]
    lots = dollar_risk / risk_per_lot if risk_per_lot > 0 else 0.0

    # enforce leverage cap
    margin_per_lot = price * cfg["dollar_per_unit_per_lot"] / cfg["leverage"]
    if margin_per_lot > 0:
        max_lots_by_margin = balance / margin_per_lot
        lots = min(lots, max_lots_by_margin)

    # cap, floor, quantize
    lots = min(lots, cfg["lot_cap"])
    lots = max(lots, cfg["min_lot_size"]) if lots > 0 else 0.0
    inc = cfg["lot_size_increment"]
    return round(lots / inc) * inc if lots > 0 else 0.0
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ DEAL COSTS + commission ───────────────────────────────────╮
def pts_to_price(points: float, tick: float) -> float:
    return points * tick

def deal_cost(bars: pd.DataFrame, entry_ts, exit_ts,
              lots: float, cfg: dict) -> float:
    usd_spread_slip = 0.0
    if cfg.get("use_csv_spread", False) and "spread" in bars.columns:
        spr_in = pts_to_price(bars.loc[bars.time >= entry_ts, "spread"].iloc[0],
                              cfg["tick_size"])
        spr_out = pts_to_price(
            bars.loc[bars.time == exit_ts, "spread"].iloc[0]
            if (bars.time == exit_ts).any() else spr_in,
            cfg["tick_size"]
        )
        usd_spread = (spr_in + spr_out) * cfg["dollar_per_unit_per_lot"] * lots
        base_slip = cfg["slip_usd_side_lot"]
        threshold = cfg["slippage_scale_threshold_lots"]
        extra = max(0.0, lots - threshold) * cfg["slippage_scale_factor"]
        usd_slip = (base_slip + extra) * 2 * lots
        usd_spread_slip = usd_spread + usd_slip
    else:
        usd_spread_slip = cfg["slip_usd_side_lot"] * 2 * lots

    comm = cfg["commission_per_lot_per_side"] * 2 * lots
    return usd_spread_slip + comm
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ BAND HELPERS ───────────────────────────────────────────────╮
def derive_band(sig, cfg: dict) -> tuple[float, float]:
    bl = getattr(sig, "entry_band_low", np.nan)
    bh = getattr(sig, "entry_band_high", np.nan)
    if pd.notna(bl) and pd.notna(bh):
        return float(bl), float(bh)

    mid = float(sig.entry_mid)
    lo  = float(sig.entry_low)
    hi  = float(sig.entry_high)

    if sig.direction == "BUY":
        return (mid, hi) if cfg.get("band_side_buy", "upper") == "upper" else (lo, mid)
    else:
        return (lo, mid) if cfg.get("band_side_sell", "lower") == "lower" else (mid, hi)


def band_trigger_price(sig, cfg: dict) -> tuple[float, float, float]:
    band_low, band_high = derive_band(sig, cfg)
    band_width = band_high - band_low
    if band_width <= 0:
        return band_low, band_high, np.nan

    thr = float(cfg.get("band_entry_threshold", 0.0))
    thr = 0.0 if thr < 0.0 else 1.0 if thr > 1.0 else thr

    if sig.direction == "BUY":
        trigger = band_low + thr * band_width
    else:
        trigger = band_high - thr * band_width
    return band_low, band_high, trigger
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ TRADE SIMULATOR ────────────────────────────────────────────╮
def simulate_trade(sig: pd.Series, bars: pd.DataFrame,
                   cfg: dict, stop_distance: float) -> dict:
    if any(pd.isna(getattr(sig, k, np.nan)) for k in ("entry_low", "entry_high", "entry_mid", "zone_width")):
        return {"filled": False, "reason": "invalid_signal"}

    if cfg.get("swap_avoidance", False) and is_near_swap_cutoff(sig.datetime, cfg):
        return {"filled": False, "reason": "swap_avoidance"}

    # Window (enforce minimum entry delay)
    touch_to  = sig.datetime + timedelta(hours=cfg["no_touch_timeout_hours"])
    hardcut   = sig.datetime + timedelta(hours=cfg["force_close_hours"])
    start_time = sig.datetime + timedelta(minutes=int(cfg.get("entry_min_delay_minutes", 0)))

    # Overtime policy and limits
    policy = str(cfg.get("force_close_policy", "final_close")).lower()  # "final_close" | "tp2" | "tp1"
    grace_hours = float(cfg.get("grace_hours", 0.0) or 0.0)
    # end of grace (we still aim for original targets up to here)
    grace_end = hardcut + timedelta(hours=grace_hours)

    # overall end time bound (grace + max extension) for performance control
    max_ext = cfg.get("max_extension_hours", None)
    if policy in ("tp1", "tp2"):
        if max_ext is None:
            # no hard cap → we still bound by data end for performance
            end_time = bars["time"].max()
        else:
            end_time = grace_end + timedelta(hours=float(max_ext))
        # bounded scan
        look = bars[(bars.time >= start_time) & (bars.time <= end_time)]
    else:
        # legacy bounded scan
        look = bars[(bars.time >= start_time) & (bars.time <= hardcut)]

    if look.empty:
        return {"filled": False, "reason": "no_entry_data"}

    # Trigger
    mode = cfg.get("entry_trigger", "band")
    if mode == "mid":
        band_low, band_high, trigger_price = float(sig.entry_mid), float(sig.entry_mid), float(sig.entry_mid)
    else:
        band_low, band_high, trigger_price = band_trigger_price(sig, cfg)
        if pd.isna(trigger_price):
            return {"filled": False, "reason": "bad_band"}

    # Wait for trigger or timeout
    fill_idx = None
    fill_bar = None
    for i, b in enumerate(look.itertuples()):
        if b.time >= touch_to:
            break
        if mode == "mid":
            reached = (b.low <= trigger_price <= b.high)
        else:
            reached = (b.high >= trigger_price) if sig.direction == "BUY" else (b.low <= trigger_price)
        if reached:
            fill_idx = i
            fill_bar = b
            break

    if fill_idx is None or fill_bar.time >= touch_to:
        return {"filled": False, "reason": "timeout"}

    # Entry price policy
    if cfg.get("band_fill_price", "threshold") == "threshold":
        entry_price = float(trigger_price)
    else:
        entry_price = ask_price(fill_bar) if sig.direction == "BUY" else bid_price(fill_bar)

    entry_time = fill_bar.time

    # Stop & targets
    stop_distance = max(stop_distance, cfg.get("min_stop_usd", 0.0))

    umain  = float(cfg.get("main_units", 1.0))
    urun   = float(cfg.get("runner_units", 0.5))
    uextra = float(cfg.get("extra_runner_units", 0.0))
    use_tp3 = bool(cfg.get("use_tp3", False))
    use_tp4 = bool(cfg.get("use_tp4", False))

    # Pre-compute target prices + helpers
    if sig.direction == "BUY":
        sl0 = entry_price - stop_distance  # original SL snapshot for overtime math
        sl  = sl0
        tp1 = entry_price + cfg["tp1_mult"] * stop_distance
        tp2 = entry_price + cfg["tp2_mult"] * stop_distance
        tp3 = entry_price + (cfg.get("tp3_mult", 0.0) * stop_distance if use_tp3 else 0.0)
        tp4 = entry_price + (cfg.get("tp4_mult", 0.0) * stop_distance if use_tp4 else 0.0)
        price_ge = lambda br, p: ask_price(br) >= p
        hit_sl   = lambda br: bid_price(br) <= sl
        tighten_half = lambda curr_sl: max(curr_sl, entry_price - (stop_distance * 0.5))
        sign = 1
        mk_px = lambda br: bid_price(br)  # MTM on exit for BUY
    else:
        sl0 = entry_price + stop_distance
        sl  = sl0
        tp1 = entry_price - cfg["tp1_mult"] * stop_distance
        tp2 = entry_price - cfg["tp2_mult"] * stop_distance
        tp3 = entry_price - (cfg.get("tp3_mult", 0.0) * stop_distance if use_tp3 else 0.0)
        tp4 = entry_price - (cfg.get("tp4_mult", 0.0) * stop_distance if use_tp4 else 0.0)
        price_ge = lambda br, p: bid_price(br) <= p   # inverted for SELL
        hit_sl   = lambda br: ask_price(br) >= sl
        tighten_half = lambda curr_sl: min(curr_sl, entry_price + (stop_distance * 0.5))
        sign = -1
        mk_px = lambda br: ask_price(br)  # MTM on exit for SELL

    hit_tp1 = (lambda br: price_ge(br, tp1))
    hit_tp2 = (lambda br: price_ge(br, tp2))
    hit_tp3 = (lambda br: price_ge(br, tp3)) if use_tp3 else (lambda br: False)
    hit_tp4 = (lambda br: price_ge(br, tp4)) if use_tp4 else (lambda br: False)

    # Manage trade
    seq = look.iloc[fill_idx:]
    pnl_raw = 0.0
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    tp4_hit = False
    exit_time = None
    exit_reason = None

    be_after_tp1 = bool(cfg.get("be_after_tp1", True))

    # remaining units
    r_main  = umain
    r_run   = urun
    r_ex3   = uextra if use_tp3 and not use_tp4 else (0.0 if not use_tp3 else uextra)
    r_ex4   = uextra if use_tp4 else 0.0
    if use_tp3 and use_tp4:
        pass  # keep caller's split behavior if any

    # Overtime state
    adjust_active = False     # when True, targets collapsed to adj_tp
    adjust_to = None          # "tp1" or "tp2"
    adj_tp = None
    ext_deadline = None
    if policy in ("tp1", "tp2"):
        # Absolute deadline after grace (if capped)
        if max_ext is not None:
            ext_deadline = grace_end + timedelta(hours=float(max_ext))

    for b in seq.itertuples():
        # Legacy policy: respect hardcut and exit at market
        if policy == "final_close" and b.time > hardcut:
            px = mk_px(b)
            remaining_units = r_main + r_run + r_ex3 + r_ex4
            pnl_raw += sign * (px - entry_price) * remaining_units
            exit_time, exit_reason = b.time, "force_close"
            break

        # If we do have a cap and it's exceeded → final_close_timeout
        if ext_deadline is not None and b.time > ext_deadline:
            px = mk_px(b)
            remaining_units = r_main + r_run + r_ex3 + r_ex4
            pnl_raw += sign * (px - entry_price) * remaining_units
            exit_time, exit_reason = b.time, "final_close_timeout"
            break

        # PRIORITY: TPs fire before SL
        if (r_main > 0) and hit_tp1(b):
            tp1_hit = True
            pnl_raw += sign * (tp1 - entry_price) * r_main
            r_main = 0.0
            if be_after_tp1:
                sl = entry_price  # BE hop

        if (r_run > 0) and hit_tp2(b):
            tp2_hit = True
            pnl_raw += sign * (tp2 - entry_price) * r_run
            r_run = 0.0
            if (r_ex3 + r_ex4) == 0.0 and not adjust_active:
                exit_time, exit_reason = b.time, "tp2"
                break

        if (r_ex3 > 0) and hit_tp3(b):
            tp3_hit = True
            pnl_raw += sign * (tp3 - entry_price) * r_ex3
            r_ex3 = 0.0
            if r_ex4 == 0.0 and not adjust_active:
                exit_time, exit_reason = b.time, "tp3"
                break

        if (r_ex4 > 0) and hit_tp4(b):
            tp4_hit = True
            pnl_raw += sign * (tp4 - entry_price) * r_ex4
            r_ex4 = 0.0
            exit_time, exit_reason = b.time, "tp4"
            break

        # Activate "adjusted overtime" AFTER grace window expires
        if (policy in ("tp1","tp2")) and (not adjust_active) and (b.time > grace_end):
            adjust_active = True
            adjust_to = policy
            adj_tp = tp1 if policy == "tp1" else tp2

            # SL tightening on switch
            mode = str(cfg.get("overtime_sl_mode", "none")).lower()
            if mode == "to_entry":
                sl = entry_price
            elif mode == "tighten_to_half":
                sl = tighten_half(sl)
            else:
                pass  # "none": leave SL untouched

        # If we are in adjusted overtime: one composite TP for all remaining units
        if adjust_active:
            remaining_units = r_main + r_run + r_ex3 + r_ex4
            if remaining_units > 0 and price_ge(b, adj_tp):
                pnl_raw += sign * (adj_tp - entry_price) * remaining_units
                r_main = r_run = r_ex3 = r_ex4 = 0.0
                exit_time, exit_reason = b.time, f"force_{adjust_to}"
                break

        # SL logic (always active)
        if hit_sl(b):
            remaining_units = r_main + r_run + r_ex3 + r_ex4
            pnl_raw += sign * (sl - entry_price) * remaining_units
            exit_time, exit_reason = b.time, ("be" if (be_after_tp1 and tp1_hit and sl == entry_price) else "sl")
            break

    if exit_time is None:
        # Ran out of bars: MTM
        last = seq.iloc[-1]
        px = mk_px(last)
        remaining_units = r_main + r_run + r_ex3 + r_ex4
        pnl_raw += sign * (px - entry_price) * remaining_units
        exit_time = last.time
        exit_reason = "final_close" if not adjust_active else f"force_{adjust_to}_eod"

    return {
        "filled": True,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "pnl_raw": pnl_raw,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "tp3_hit": tp3_hit,
        "tp4_hit": tp4_hit,
        "entry_price_used": entry_price,
        "trigger_price": trigger_price,
        "band_low": band_low,
        "band_high": band_high,
    }
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ PORTFOLIO LOOP ─────────────────────────────────────────────╮
def run_backtest(signals: pd.DataFrame, bars: pd.DataFrame,
                 cfg: dict|None=None, *,
                 start_balance: float|None=None,
                 trial_no: int|None=None) -> pd.DataFrame:
    cfg = load_cfg() if cfg is None else cfg
    balance = start_balance if start_balance is not None else cfg["start_balance"]
    equity_floor = max(float(cfg.get("min_equity_usd", 0.0)), 0.0)
    stop_on_neg = bool(cfg.get("stop_on_negative_balance", True))

    # ── Single-target override (applied once at start)
    st_on = bool(cfg.get("force_single_entry", False))
    st_tp = cfg.get("single_entry_target", None)
    if st_on and st_tp in (1, 2, 3, 4):
        # Zero all first
        cfg["main_units"] = 0.0
        cfg["runner_units"] = 0.0
        cfg["extra_runner_units"] = 0.0
        # Disable BE-after-TP1 since TP1 may be unused
        cfg["be_after_tp1"] = False
        # Disable extra targets by default, then enable exactly one
        cfg["use_tp3"] = False
        cfg["use_tp4"] = False
        if st_tp == 1:
            cfg["main_units"] = 1.0
        elif st_tp == 2:
            cfg["runner_units"] = 1.0
        elif st_tp == 3:
            cfg["extra_runner_units"] = 1.0
            cfg["use_tp3"] = True
        elif st_tp == 4:
            cfg["extra_runner_units"] = 1.0
            cfg["use_tp4"] = True

    dollar_pt = cfg["dollar_per_unit_per_lot"]
    rows = []
    current_day = None
    start_bal_day = balance
    skip = False
    early_stop = False

    # cache units for portfolio-level lot allocation (after overrides)
    umain  = float(cfg.get("main_units", 1.0))
    urun   = float(cfg.get("runner_units", 0.5))
    uextra = float(cfg.get("extra_runner_units", 0.0))

    for sig in signals.itertuples():
        # Equity hard stop BEFORE considering the next trade
        if stop_on_neg and balance <= equity_floor:
            early_stop = True
            break

        day = sig.datetime.date()
        if day != current_day:
            current_day = day
            start_bal_day = balance
            skip = False

        if skip:
            rows.append({
                "signal_time": sig.datetime,
                "direction": sig.direction,
                "entry_mid": sig.entry_mid,
                "balance_before": balance,
                "exit_time": sig.datetime,
                "exit_reason": "daily_loss_limit",
                "filled": False,
                "lot_main": 0.0,
                "lot_runner": 0.0,
                "lot_extra": 0.0,
                "pnl": 0.0,
                "swap_cost": 0.0,
                "commission": 0.0,
                "slippage_spread": 0.0,
                "total_cost": 0.0,
                "balance_after": balance,
            })
            continue

        # Robust stop: max(structural, ATR*mult) if available
        struct_stop = cfg["risk_mult"] * sig.zone_width
        atr_stop = None
        if cfg.get("dynamic_stops", False) and getattr(sig, "atr", 0) and not pd.isna(sig.atr):
            atr_stop = float(sig.atr) * cfg["atr_multiplier"]
        stop = max(struct_stop, atr_stop) if atr_stop is not None else struct_stop

        res = simulate_trade(sig, bars, cfg, stop)
        row = {
            "signal_time": sig.datetime,
            "direction": sig.direction,
            "entry_mid": sig.entry_mid,
            "balance_before": balance,
            **{k: res.get(k) for k in (
                "exit_time","exit_reason","tp1_hit","tp2_hit","tp3_hit","tp4_hit","filled",
                "entry_price_used","trigger_price"
            )}
        }

        if res.get("filled", False):
            # lot sizing at actual fill price — lot size per 1.0 “unit”
            price_for_sizing = res["entry_price_used"]
            unit_lot = calculate_position_size(cfg, balance, stop, price_for_sizing)
            if unit_lot <= 0:
                row.update(
                    filled=False,
                    lot_main=0.0,
                    lot_runner=0.0,
                    lot_extra=0.0,
                    pnl=0.0,
                    swap_cost=0.0,
                    commission=0.0,
                    slippage_spread=0.0,
                    total_cost=0.0,
                    exit_reason="insufficient_margin",
                )
                row["balance_after"] = balance
                rows.append(row)
                continue

            lot_main   = round(unit_lot * umain, 3)
            lot_runner = round(unit_lot * urun, 3)
            lot_extra  = round(unit_lot * uextra, 3)
            total_lots = lot_main + lot_runner + lot_extra

            # costs
            total_cost = deal_cost(bars, res["entry_time"], res["exit_time"], total_lots, cfg)
            commission = cfg["commission_per_lot_per_side"] * 2 * total_lots
            slippage_spread = total_cost - commission

            swap = calculate_swap_cost(res["entry_time"], res["exit_time"], sig.direction, total_lots, cfg)

            # pnl_raw is USD-per-unit across realized exposures; multiply by unit lot
            pnl = res["pnl_raw"] * dollar_pt * unit_lot - total_cost - swap
            balance += pnl

            # daily loss check
            if balance - start_bal_day < -cfg["day_loss_limit_pct"] * start_bal_day:
                skip = True

            # margin stop AFTER trade
            if stop_on_neg and balance <= equity_floor:
                row["exit_reason"] = "margin_stop"

            row.update(
                lot_main=lot_main,
                lot_runner=lot_runner,
                lot_extra=lot_extra,
                pnl=pnl,
                swap_cost=swap,
                commission=commission,
                slippage_spread=slippage_spread,
                total_cost=total_cost,
            )
        else:
            row.update(
                lot_main=0.0,
                lot_runner=0.0,
                lot_extra=0.0,
                pnl=0.0,
                swap_cost=0.0,
                commission=0.0,
                slippage_spread=0.0,
                total_cost=0.0,
            )

        row["balance_after"] = balance
        rows.append(row)

        # if margin stop triggered by this trade — stop processing further signals
        if stop_on_neg and balance <= equity_floor:
            early_stop = True
            break

    out = pd.DataFrame(rows)
    if len(out) and early_stop:
        out.attrs["early_stop_reason"] = "margin_stop"
    return out
# ╰──────────────────────────────────────────────────────────────╯
