# logger.py

import os
import csv
from datetime import datetime
from config import CONFIG

LOG_FILE = "executed_trades.csv"

# Stable header (optional fields are simply left blank when absent)
FIELDNAMES = [
    "ticket", "symbol", "direction", "lots",
    "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
    "balance_before", "balance_after",
    "slippage_cost", "commission_cost", "swap_cost", "gross_pnl", "net_pnl",
    # Optional technicals (written only if provided)
    "stop_distance", "tp1", "tp2", "tp3", "tp4", "sl",
    # Optional overtime metadata (if you choose to pass them)
    "overtime", "adjust_to", "overtime_sl_mode",
]

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _fmt_time_any(t) -> str:
    """Accepts datetime or string; returns 'YYYY-mm-dd HH:MM:SS'."""
    if isinstance(t, datetime):
        return t.strftime("%Y-%m-%d %H:%M:%S")
    # assume it is already a string
    return str(t)

def write_trade_to_csv(trade: dict):
    """
    Appends a closed trade entry to CSV with cost breakdown and balance tracking.
    Works with legacy fields; tolerates missing balances and adds optional TP3/TP4/OT columns if present.
    """
    file_exists = os.path.isfile(LOG_FILE)

    entry_price = float(trade["entry_price"])
    exit_price  = float(trade["exit_price"])
    direction   = trade["direction"]
    lots        = float(trade.get("requested_lots", trade.get("volume", 0.0)))

    # Cost calculations (record-only; matches your BT/live cost model)
    slip_cost = CONFIG["slip_usd_side_lot"] * 2 * lots
    comm_cost = CONFIG["commission_per_lot_per_side"] * 2 * lots
    swap_cost = CONFIG["swap_fee_long"] * lots if direction == "BUY" else CONFIG["swap_fee_short"] * lots

    # PnL in account currency
    gross_pnl = (exit_price - entry_price) * lots * CONFIG["dollar_per_unit_per_lot"]
    if direction == "SELL":
        gross_pnl *= -1
    net_pnl = gross_pnl - slip_cost - comm_cost - swap_cost

    row = {
        "ticket": trade["ticket"],
        "symbol": trade["symbol"],
        "direction": direction,
        "lots": round(lots, 2),

        "entry_time": _fmt_time_any(trade["entry_time"]),
        "entry_price": round(entry_price, 2),
        "exit_time": _fmt_time_any(trade["exit_time"]),
        "exit_price": round(exit_price, 2),
        "exit_reason": trade["exit_reason"],

        "balance_before": round(_safe_float(trade.get("balance_before")), 2),
        "balance_after":  round(_safe_float(trade.get("balance_after")), 2),

        "slippage_cost": round(slip_cost, 2),
        "commission_cost": round(comm_cost, 2),
        "swap_cost": round(swap_cost, 2),

        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
    }

    # Optional tech fields if provided by trade_manager
    for f in ("stop_distance", "tp1", "tp2", "tp3", "tp4", "sl"):
        if f in trade and trade[f] is not None:
            row[f] = round(float(trade[f]), 2)

    # Optional overtime metadata (write if present)
    if "overtime" in trade:
        row["overtime"] = bool(trade["overtime"])
    if "adjust_to" in trade:
        row["adjust_to"] = str(trade["adjust_to"])
    if "overtime_sl_mode" in trade:
        row["overtime_sl_mode"] = str(trade["overtime_sl_mode"])

    # Write with stable header
    write_header = (not file_exists)
    if file_exists:
        # If an older file exists with different header, we still force the stable header now.
        # New columns will be added; old readers can ignore extras.
        try:
            with open(LOG_FILE, "r", newline="") as f:
                reader = csv.DictReader(f)
                write_header = (reader.fieldnames != FIELDNAMES)
        except Exception:
            write_header = True

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
