# main.py
import time
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5

from config import CONFIG
from utils import init_mt5, within_trading_session, now_utc, is_near_swap_cutoff
from scanner import get_today_zones
from entry import fetch_tick, should_trigger_trade, send_market_order
from trade_manager import manage_all_trades
from logger import write_trade_to_csv


def _struct_stop(width: float) -> float:
    return max(float(CONFIG.get("min_stop_usd", 1.0)),
               float(CONFIG.get("risk_mult", 2.3)) * float(width))


def _robust_stop(zone_row) -> float:
    struct = _struct_stop(float(zone_row.get("zone_width", float(CONFIG.get("zone_width_usd", 5.0)))))
    atr = zone_row.get("atr", float("nan"))
    if atr == atr and atr > 0:
        return max(struct, atr * float(CONFIG.get("atr_multiplier", 3.0)))
    return struct


def _m1_gap_excessive(symbol: str, ref_price: float) -> bool:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2)
    if rates is None or len(rates) < 2:
        return False
    prev_close = float(rates[-2]["close"])
    if prev_close <= 0:
        return False
    gap = abs(ref_price - prev_close) / prev_close
    return gap > float(CONFIG.get("gap_filter", 0.005))


def _print_last_error(prefix: str, now: datetime):
    try:
        code, msg = mt5.last_error()
        print(f"{now} → {prefix} last_error: code={code} msg='{msg}'")
    except Exception:
        pass


def _dump_attempts(prefix: str, order: dict, now: datetime):
    """
    Pretty-print the attempt log returned by entry.send_market_order.
    Each attempt has: filling, price, check_retcode, check_comment, send_retcode, send_comment
    """
    attempts = order.get("attempts", []) or []
    if not attempts:
        print(f"{now} → {prefix} attempts: <none>")
        return
    print(f"{now} → {prefix} attempts ({len(attempts)}):")
    for i, att in enumerate(attempts, 1):
        print(
            f"  #{i} fill={att.get('filling')} "
            f"price={att.get('price')} "
            f"check={att.get('check_retcode')}('{att.get('check_comment')}') "
            f"send={att.get('send_retcode')}('{att.get('send_comment')}')"
        )


def _print_env_caps(prefix: str):
    """Dump terminal/account/symbol trading capability flags for debugging."""
    try:
        ti = mt5.terminal_info()
        ai = mt5.account_info()
        si = mt5.symbol_info(CONFIG["symbol"])
        print(f"{prefix} term.trade_allowed={getattr(ti,'trade_allowed',None)}  "
              f"acct.trade_allowed={getattr(ai,'trade_allowed',None)}  "
              f"acct.margin_mode={getattr(ai,'margin_mode',None)}  "
              f"acct.company='{getattr(ai,'company',None)}'")
        print(f"{prefix} symbol.trade_mode={getattr(si,'trade_mode',None)}  "
              f"symbol.filling_mode={getattr(si,'filling_mode',None)}  "
              f"symbol.expiration_mode={getattr(si,'expiration_mode',None)}  "
              f"digits={getattr(si,'digits',None)}  step={getattr(si,'volume_step',None)}  "
              f"min_vol={getattr(si,'volume_min',None)}  max_vol={getattr(si,'volume_max',None)}")
    except Exception as e:
        print(f"{prefix} env-caps error: {e}")


def _force_test_entry_if_requested(now: datetime):
    if not bool(CONFIG.get("force_test_entry", False)):
        return
    try:
        symbol = str(CONFIG.get("symbol", "XAUUSD"))
        _, ask = fetch_tick(symbol)
        sd = _struct_stop(float(CONFIG.get("zone_width_usd", 5.0)))
        order = send_market_order("BUY", ask, sd, trigger_price=ask)

        _dump_attempts("TEST", order, now)

        res = order.get("entry_result")
        if res is None:
            print(f"{now} → TEST order FAILED: entry_result=None  last_error={order.get('last_error')}")
            _print_env_caps("[test]")
            return

        rc = getattr(res, "retcode", None)
        ticket = getattr(res, "order", None)
        ok = rc in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)

        if ok:
            ok_att = next(
                (a for a in order.get("attempts", [])
                 if a.get("send_retcode") in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)),
                None
            )
            ok_fill = ok_att.get("filling") if ok_att else "?"
            print(f"{now} → TEST order OK  retcode={rc} ticket={ticket} "
                  f"price={order['requested_price']:.2f} lots={order['requested_lots']} "
                  f"filling={ok_fill} comment='{order.get('comment_used')}'")
        else:
            print(f"{now} → TEST order FAILED  retcode={rc} (expected DONE/PLACED)")
            _dump_attempts("TEST", order, now)
            _print_last_error("TEST", now)
    except Exception as e:
        print(f"{now} → TEST order EXCEPTION: {e}")
        _print_env_caps("[test]")
    finally:
        CONFIG["force_test_entry"] = False


def main_loop():
    # --- Login/init MT5 ---
    login    = int(CONFIG.get("login", 0)) or None
    password = CONFIG.get("password")
    server   = CONFIG.get("server")
    init_mt5(login=login, password=password, server=server)
    print("Starting live session loop...")
    _print_env_caps("[env]")

    # --- Initial zone load ---
    try:
        zones = get_today_zones()
        zones = pd.DataFrame() if zones is None else zones.reset_index(drop=True)
    except Exception as e:
        print(f"[{now_utc()}] Initial zone load failed: {e}")
        zones = pd.DataFrame()

    # Intraday refresh cadence
    last_zone_refresh = now_utc()
    refresh_secs = 15 * 60

    # Trigger tracking
    triggered_keys: set[tuple[datetime, str]] = set()
    min_delay = int(CONFIG.get("entry_min_delay_minutes", 0))

    # Daily loss-limit
    day_date = None
    day_start_equity = None
    loss_cut = float(CONFIG.get("day_loss_limit_pct", 0.05))

    while True:
        now = now_utc()

        # One-shot plumbing check
        _force_test_entry_if_requested(now)

        # New UTC day refresh
        if day_date != now.date():
            ai = mt5.account_info()
            day_start_equity = float(getattr(ai, "equity", 0.0)) if ai else None
            day_date = now.date()
            triggered_keys.clear()
            try:
                zones = get_today_zones().reset_index(drop=True)
            except Exception as e:
                print(f"[{now}] Zone refresh (new UTC day) failed: {e}")
                zones = pd.DataFrame()
            print(f"[{now}] New UTC day → equity baseline={day_start_equity}, zones={len(zones)}")
            last_zone_refresh = now

        # Intraday zone refresh
        if (now - last_zone_refresh).total_seconds() >= refresh_secs:
            try:
                new = get_today_zones().reset_index(drop=True)
                if not new.empty:
                    zones = new
                    print(f"[{now}] Refreshed zones intraday: {len(zones)}")
                else:
                    print(f"[{now}] Intraday refresh: no zones.")
            except Exception as e:
                print(f"[{now}] Intraday zone refresh failed: {e}")
            last_zone_refresh = now

        # Gating
        in_session = within_trading_session(now)
        near_swap = bool(CONFIG.get("swap_avoidance", False)) and is_near_swap_cutoff(now, respect_flag=False)

        ai = mt5.account_info()
        eq = float(getattr(ai, "equity", 0.0)) if ai else None
        daily_hit = bool(day_start_equity) and bool(eq) and (eq <= day_start_equity * (1.0 - loss_cut))

        allow_entries = in_session and (not near_swap) and (not daily_hit)

        # Tick snapshot
        bid = ask = None
        if allow_entries:
            try:
                bid, ask = fetch_tick(CONFIG["symbol"])
            except Exception as e:
                print(f"{now} → Tick fetch failed: {e}")
                _print_last_error("Tick", now)

        # Try entries
        if allow_entries and bid is not None and ask is not None and not zones.empty:
            for _, z in zones.iterrows():
                sig_time = z.datetime.to_pydatetime() if hasattr(z.datetime, "to_pydatetime") else z.datetime
                key = (pd.Timestamp(sig_time).to_pydatetime(), str(z.direction))

                if key in triggered_keys:
                    continue
                if now < (sig_time + timedelta(minutes=min_delay)):
                    continue

                ok, trig = should_trigger_trade(
                    z.direction,
                    float(z.entry_low),
                    float(z.entry_high),
                    bid,
                    ask
                )
                if not ok:
                    continue

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
                    print(f"{now} → Order send EXCEPTION: {e}")
                    _print_last_error("LIVE order", now)
                    continue

                res = order.get("entry_result")
                if res is None:
                    print(f"{now} → LIVE order FAILED: entry_result=None  last_error={order.get('last_error')}")
                    _dump_attempts("LIVE order", order, now)
                    _print_last_error("LIVE order", now)
                    continue

                rc  = getattr(res, "retcode", None)
                ticket = getattr(res, "order", None)

                ok_att = next((a for a in order.get("attempts", []) if a.get("send_retcode") in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)), None)
                ok_fill = ok_att.get("filling") if ok_att else "?"

                if rc in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                    print(f"{now} → LIVE order OK  retcode={rc} ticket={ticket} "
                          f"price={order['requested_price']:.2f} lots={order['requested_lots']} "
                          f"filling={ok_fill} comment='{order.get('comment_used')}'")
                else:
                    # Rare branch; res exists but not OK → log attempts anyway
                    print(f"{now} → LIVE order NOT OK  retcode={rc}")
                    _dump_attempts("LIVE order", order, now)
                    _print_last_error("LIVE order", now)
                    continue

                order.update({
                    "ticket": ticket,
                    "symbol": CONFIG["symbol"],
                    "direction": z.direction,
                    "volume": order["requested_lots"],
                    "entry_price": order["requested_price"],
                })

                triggered_keys.add(key)

        else:
            if not in_session:
                pass
            elif near_swap:
                print(f"[{now}] Entries paused near swap cutoff.")
            elif daily_hit:
                print(f"[{now}] Daily loss limit hit — entries paused until next UTC day "
                      f"(start={day_start_equity:.2f}, now={eq:.2f}).")

        # Manage exits & log
        try:
            closed_trades = manage_all_trades()
            for trade in closed_trades:
                write_trade_to_csv(trade)
                print(f"{trade['exit_time']} → Trade closed: {trade['exit_reason']}")
        except Exception as e:
            print(f"{now} → Trade management error: {e}")

        # Pace
        sleep_s = 60 if (not in_session or near_swap or daily_hit) else 20
        time.sleep(sleep_s)


if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        print("CRITICAL ERROR:", e)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
