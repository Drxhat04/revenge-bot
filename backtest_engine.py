#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────
# backtest_engine.py · v10.1
#  • Band-trigger entries + configurable depth + fill-price policy
#  • Enforce minimum delay after signal (no same-minute fills)
#  • Robust stops: max(structural, ATR) when dynamic_stops is on
#  • Proper cost accounting and audit fields
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
    "risk_per_trade": 0.02,
    "max_risk_ratio": 0.03,

    # Volatility stops
    "dynamic_stops": True,
    "atr_multiplier": 3.0,
    "min_stop_usd": 1.0,

    # Timeouts
    "no_touch_timeout_hours": 3,
    "force_close_hours": 24,

    # Lot sizing
    "min_lot_size": 0.01,
    "max_lots": 10.0,
    "lot_size_increment": 0.01,

    # Dealing costs & execution
    "tick_size": 0.0,                     # zero-spread account
    "slip_usd_side_lot": 5.0,
    "use_csv_spread": False,
    "commission_per_lot_per_side": 5.5,  # override via config (e.g., 5.5)

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
    "leverage": 500.0,  # e.g. 1:500 for zero account

    # Equity
    "start_balance": 10_000.0,

    # ───────── Entry logic ─────────
    # Trigger type: "band" uses half-zone entry bands, "mid" requires midpoint touch
    "entry_trigger": "band",
    # Band sides: BUY default band [mid, high]; SELL default band [low, mid]
    "band_side_buy": "upper",  # "upper" -> [mid, high], "lower" -> [low, mid]
    "band_side_sell": "lower",  # "lower" -> [low, mid], "upper" -> [mid, high]
    # Depth into band to trigger a fill: 0.00 = band edge at mid, 0.50 = halfway, 1.00 = far edge
    "band_entry_threshold": 0.00,
    # Fill price policy: "threshold" (deterministic) or "touch" (bar's tradeable side)
    "band_fill_price": "threshold",

    # Realism: prevent same-minute fills after the signal timestamp
    "entry_min_delay_minutes": 1,
}

# ╭─ CONFIG LOAD (used only if caller passes nothing) ───────────╮
def load_cfg(path: str | Path = "config.yaml") -> dict:
    cfg = DEFAULTS.copy()
    if Path(path).exists():
        user_cfg = yaml.safe_load(open(path)) or {}
        cfg.update(user_cfg)
    # alias for lot cap
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
    stop_distance = max(stop_distance, cfg.get("min_stop_usd", 0))
    dollar_risk = cfg["risk_per_trade"] * balance
    dollar_risk = min(dollar_risk, cfg["max_risk_ratio"] * balance)
    risk_per_lot = stop_distance * cfg["dollar_per_unit_per_lot"]
    lots = dollar_risk / risk_per_lot if risk_per_lot > 0 else 0.0

    # enforce leverage cap
    margin_per_lot = price * cfg["dollar_per_unit_per_lot"] / cfg["leverage"]
    max_lots_by_margin = balance / margin_per_lot if margin_per_lot > 0 else lots
    lots = min(lots, max_lots_by_margin)

    # cap, floor, quantize
    lots = min(lots, cfg["lot_cap"])
    lots = max(lots, cfg["min_lot_size"])
    inc = cfg["lot_size_increment"]
    return round(lots / inc) * inc
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ DEAL COSTS + commission ───────────────────────────────────╮
def pts_to_price(points: float, tick: float) -> float:
    return points * tick

def deal_cost(bars: pd.DataFrame, entry_ts, exit_ts,
              lots: float, cfg: dict) -> float:
    # spread & slippage
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
        # zero spread → just slippage
        usd_spread_slip = cfg["slip_usd_side_lot"] * 2 * lots

    # commission (entry + exit)
    comm = cfg["commission_per_lot_per_side"] * 2 * lots

    return usd_spread_slip + comm
# ╰──────────────────────────────────────────────────────────────╯

# ╭─ BAND HELPERS (new) ────────────────────────────────────────╮
def derive_band(sig, cfg: dict) -> tuple[float, float]:
    """
    Return (band_low, band_high) for this signal from precomputed columns
    or by splitting the zone at its midpoint.
    """
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
    """
    Compute the trigger price within the chosen band based on band_entry_threshold.
    Returns (band_low, band_high, trigger_price).
    """
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
    touch_to = sig.datetime + timedelta(hours=cfg["no_touch_timeout_hours"])
    hardcut  = sig.datetime + timedelta(hours=cfg["force_close_hours"])
    start_time = sig.datetime + timedelta(minutes=int(cfg.get("entry_min_delay_minutes", 0)))
    look = bars[(bars.time >= start_time) & (bars.time <= hardcut)]
    if look.empty:
        return {"filled": False, "reason": "no_entry_data"}

    # Decide trigger
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
            if sig.direction == "BUY":
                reached = (b.high >= trigger_price)  # hit or exceed price
            else:
                reached = (b.low <= trigger_price)

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

    # Stop distance and targets
    stop_distance = max(stop_distance, cfg.get("min_stop_usd", 0.0))
    if sig.direction == "BUY":
        sl  = entry_price - stop_distance
        tp1 = entry_price + cfg["tp1_mult"] * stop_distance
        tp2 = entry_price + cfg["tp2_mult"] * stop_distance
        hit_sl  = lambda br: bid_price(br) <= sl
        hit_tp1 = lambda br: ask_price(br) >= tp1
        hit_tp2 = lambda br: ask_price(br) >= tp2
        sign = 1
    else:
        sl  = entry_price + stop_distance
        tp1 = entry_price - cfg["tp1_mult"] * stop_distance
        tp2 = entry_price - cfg["tp2_mult"] * stop_distance
        hit_sl  = lambda br: ask_price(br) >= sl
        hit_tp1 = lambda br: bid_price(br) <= tp1
        hit_tp2 = lambda br: bid_price(br) <= tp2
        sign = -1

    # Manage trade
    seq = look.iloc[fill_idx:]
    pnl_raw = 0.0
    tp1_hit = False
    tp2_hit = False
    exit_time = None
    exit_reason = None
    main_u, run_u = 1.0, 0.5  # base + runner = 1.5x base units

    for b in seq.itertuples():
        if b.time > hardcut:
            px = bid_price(b) if sig.direction == "BUY" else ask_price(b)
            pnl_raw += sign * (px - entry_price) * (main_u + run_u)
            exit_time, exit_reason = b.time, "force_close"
            break

        if not tp1_hit and hit_tp1(b):
            tp1_hit = True
            pnl_raw += sign * (tp1 - entry_price) * main_u
            main_u = 0.0
            sl = entry_price  # BE after TP1

        if tp1_hit and hit_tp2(b):
            tp2_hit = True
            pnl_raw += sign * (tp2 - entry_price) * run_u
            exit_time, exit_reason = b.time, "tp2"
            break

        if not tp1_hit and hit_sl(b):
            pnl_raw += sign * (sl - entry_price) * (main_u + run_u)
            exit_time, exit_reason = b.time, "sl"
            break

        if tp1_hit and hit_sl(b):
            exit_time, exit_reason = b.time, "be"
            break

    if exit_time is None:
        last = seq.iloc[-1]
        px = bid_price(last) if sig.direction == "BUY" else ask_price(last)
        pnl_raw += sign * (px - entry_price) * (main_u + run_u)
        exit_time, exit_reason = last.time, "final_close"

    return {
        "filled": True,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "pnl_raw": pnl_raw,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
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
    dollar_pt = cfg["dollar_per_unit_per_lot"]
    rows = []
    current_day = None
    start_bal_day = balance
    skip = False

    for sig in signals.itertuples():
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
                "pnl": 0.0,
                "swap_cost": 0.0,
                "commission": 0.0,
                "balance_after": balance,
            })
            continue

        # Robust stop: take the larger of structural and ATR (if available)
        struct_stop = cfg["risk_mult"] * sig.zone_width
        atr_stop = None
        if cfg.get("dynamic_stops", False) and getattr(sig, "atr", 0) > 0:
            atr_stop = sig.atr * cfg["atr_multiplier"]
        stop = max(struct_stop, atr_stop) if atr_stop is not None else struct_stop

        res = simulate_trade(sig, bars, cfg, stop)
        row = {
            "signal_time": sig.datetime,
            "direction": sig.direction,
            "entry_mid": sig.entry_mid,
            "balance_before": balance,
            **{k: res.get(k) for k in ("exit_time", "exit_reason", "tp1_hit", "tp2_hit", "filled",
                                       "entry_price_used", "trigger_price")}
        }

        if res.get("filled", False):
            # lot sizing at actual fill price
            price_for_sizing = res["entry_price_used"]
            base_lot = calculate_position_size(cfg, balance, stop, price_for_sizing)
            lot_runner = round(base_lot * 0.5, 3)
            total_lots = base_lot + lot_runner

            # costs
            total_cost = deal_cost(bars, res["entry_time"], res["exit_time"], total_lots, cfg)
            commission = cfg["commission_per_lot_per_side"] * 2 * total_lots
            slippage_spread = total_cost - commission

            swap = calculate_swap_cost(res["entry_time"], res["exit_time"], sig.direction, total_lots, cfg)

            # pnl_raw is USD-per-unit across exposures (1.5x); multiply by base_lot to size
            pnl = res["pnl_raw"] * dollar_pt * base_lot - total_cost - swap
            balance += pnl

            # daily loss check
            if balance - start_bal_day < -cfg["day_loss_limit_pct"] * start_bal_day:
                skip = True

            row.update(
                lot_main=base_lot,
                lot_runner=lot_runner,
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
                pnl=0.0,
                swap_cost=0.0,
                commission=0.0,
            )

        row["balance_after"] = balance
        rows.append(row)

    return pd.DataFrame(rows)
# ╰──────────────────────────────────────────────────────────────╯
