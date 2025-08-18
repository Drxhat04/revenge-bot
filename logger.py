# logger.py

import os
import csv
from config import CONFIG

LOG_FILE = "executed_trades.csv"

def write_trade_to_csv(trade: dict):
    """
    Appends a closed trade entry to CSV with cost breakdown and balance tracking.
    """
    file_exists = os.path.isfile(LOG_FILE)

    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    direction = trade["direction"]
    lots = float(trade.get("requested_lots", trade["volume"]))

    # Cost calculations (record-only)
    slip_cost = CONFIG["slip_usd_side_lot"] * 2 * lots
    comm_cost = CONFIG["commission_per_lot_per_side"] * 2 * lots
    swap_cost = CONFIG["swap_fee_long"] * lots if direction == "BUY" else CONFIG["swap_fee_short"] * lots

    # PnL
    gross_pnl = (exit_price - entry_price) * lots * CONFIG["dollar_per_unit_per_lot"]
    if direction == "SELL":
        gross_pnl *= -1
    net_pnl = gross_pnl - slip_cost - comm_cost - swap_cost

    row = {
        "ticket": trade["ticket"],
        "symbol": trade["symbol"],
        "direction": direction,
        "lots": round(lots, 2),
        "entry_time": trade["entry_time"],
        "entry_price": round(entry_price, 2),
        "exit_time": trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "exit_price": round(exit_price, 2),
        "exit_reason": trade["exit_reason"],
        "balance_before": round(float(trade["balance_before"]), 2),
        "balance_after": round(float(trade["balance_after"]), 2),
        "slippage_cost": round(slip_cost, 2),
        "commission_cost": round(comm_cost, 2),
        "swap_cost": round(swap_cost, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
    }

    # record SL/TP/Stop info if available
    for field in ("stop_distance", "tp1", "tp2", "sl"):
        if field in trade:
            row[field] = round(float(trade[field]), 2)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
