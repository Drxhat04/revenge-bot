# main.py
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5

from config import CONFIG
from utils import init_mt5, within_trading_session, now_utc, is_near_swap_cutoff
from scanner import get_today_zones
from entry import fetch_tick, should_trigger_trade, send_market_order
from trade_manager import manage_all_trades
from logger import write_trade_to_csv


def _struct_stop(width: float) -> float:
    return max(CONFIG["min_stop_usd"], CONFIG["risk_mult"] * width)


def _robust_stop(zone_row) -> float:
    # ATR may be NaN in live; fall back to structural if so.
    struct = _struct_stop(float(zone_row.get("zone_width", CONFIG["zone_width_usd"])))
    atr = zone_row.get("atr", float("nan"))
    if atr == atr and atr > 0:  # finite
        return max(struct, atr * CONFIG["atr_multiplier"])
    return struct


def _m1_gap_excessive(symbol: str, ref_price: float) -> bool:
    """Return True if current ref_price is an excessive M1 gap vs previous close."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2)
    if rates is None or len(rates) < 2:
        return False
    prev_close = float(rates[-2]["close"])
    if prev_close <= 0:
        return False
    gap = abs(ref_price - prev_close) / prev_close
    return gap > float(CONFIG.get("gap_filter", 0.005))


def main_loop():
    init_mt5()
    print("Starting live session loop...")

    zones = get_today_zones()
    if zones.empty:
        print("No zones for today — sleeping 5 minutes.")
        time.sleep(300)
        zones = get_today_zones()

    triggered = set()
    min_delay = int(CONFIG.get("entry_min_delay_minutes", 0))

    # Daily loss-limit tracking (UTC)
    day_date = None
    day_start_equity = None
    loss_cut = float(CONFIG.get("day_loss_limit_pct", 0.05))

    while True:
        now = now_utc()

        # — Refresh daily equity baseline and zones at UTC date change
        if day_date != now.date():
            ai = mt5.account_info()
            day_start_equity = float(getattr(ai, "equity", 0.0)) if ai else None
            day_date = now.date()
            triggered.clear()
            zones = get_today_zones()
            print(f"[{now}] New UTC day → equity baseline set to {day_start_equity}, refreshed zones: {len(zones)}")

        # Compute gating flags for *new entries* (we still always manage exits)
        in_session = within_trading_session(now)
        near_swap = bool(CONFIG.get("swap_avoidance", False)) and is_near_swap_cutoff(now, respect_flag=False)

        ai = mt5.account_info()
        eq = float(getattr(ai, "equity", 0.0)) if ai else None
        daily_hit = bool(day_start_equity) and bool(eq) and (eq <= day_start_equity * (1.0 - loss_cut))

        allow_entries = in_session and (not near_swap) and (not daily_hit)

        # Take one tick snapshot only if we may place entries
        bid = ask = None
        if allow_entries:
            try:
                bid, ask = fetch_tick(CONFIG["symbol"])
            except Exception as e:
                print(f"{now} → Tick fetch failed: {e}")

        # Try new entries only when allowed
        if allow_entries and bid is not None and ask is not None:
            for idx, z in zones.iterrows():
                if idx in triggered:
                    continue

                # honor scanner timestamp + extra live delay
                sig_time = z.datetime.to_pydatetime() if hasattr(z.datetime, "to_pydatetime") else z.datetime
                if now < (sig_time + timedelta(minutes=min_delay)):
                    continue

                # band-aware trigger check (returns (ok, trigger_price))
                ok, trig = should_trigger_trade(
                    z.direction,
                    float(z.entry_low),
                    float(z.entry_high),
                    bid,
                    ask
                )
                if not ok:
                    continue

                # Gap filter (match scanner hygiene)
                ref_px = ask if z.direction == "BUY" else bid
                if _m1_gap_excessive(CONFIG["symbol"], ref_px):
                    print(f"{now} → Gap filter blocked {z.direction} trade (ref {ref_px:.2f}).")
                    continue

                stop_dist = _robust_stop(z)

                try:
                    order = send_market_order(
                        z.direction,
                        ask if z.direction == "BUY" else bid,
                        stop_dist,
                        trigger_price=trig
                    )
                except Exception as e:
                    print(f"{now} → Order send failed: {e}")
                    continue

                order.update({
                    "ticket": getattr(order["entry_result"], "order", None),
                    "symbol": CONFIG["symbol"],
                    "direction": z.direction,
                    "volume": order["requested_lots"],
                    "entry_price": order["requested_price"],
                })

                triggered.add(idx)
                print(f"{now} → Order triggered @ {order['entry_price']:.2f} for {order['volume']} lots "
                      f"(trig {trig:.2f}, sd {stop_dist:.2f})")

        else:
            # Explain why entries are paused (once per loop, lightweight)
            if not in_session:
                pass  # normal quiet period
            elif near_swap:
                print(f"[{now}] Entries paused near swap cutoff.")
            elif daily_hit:
                print(f"[{now}] Daily loss limit hit — entries paused until next UTC day "
                      f"(start={day_start_equity:.2f}, now={eq:.2f}).")

        # Always manage exits & log closed trades
        closed_trades = manage_all_trades()
        for trade in closed_trades:
            write_trade_to_csv(trade)
            print(f"{trade['exit_time']} → Trade closed: {trade['exit_reason']}")

        # Gentle pacing
        sleep_s = 60 if (not in_session or near_swap or daily_hit) else 20
        time.sleep(sleep_s)


if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        print("CRITICAL ERROR:", e)
    finally:
        mt5.shutdown()
