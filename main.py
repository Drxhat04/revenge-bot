#!/usr/bin/env python
# main.py — live orchestrator (freshness + trigger gating)
# ------------------------------------------------------------------
# Adds:
#   • is_fresh(ts_utc): gate stale signals on startup and per loop
#   • is_near_trigger(direction, bid, ask, trigger, band_lo, band_hi)
#   • Telegram posts respect send_stale_on_startup and freshness window
#   • Order placement requires both is_fresh and is_near_trigger
#   • Backward-compatible with older scanner columns
# ------------------------------------------------------------------

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

# Telegram notifier bridge
from notify_bridge import Notifier, _sig_key


# ───────────────────────── Freshness / trigger helpers ─────────────────────────

def _coerce_ts_utc(ts) -> datetime:
    """Coerce any pandas/python datetime (aware or naive) to a naive UTC datetime."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    else:
        # Treat naive as UTC already
        t = t.tz_localize(None)
    return t.to_pydatetime()

def is_fresh(ts_utc) -> bool:
    """
    True if the signal time is within freshness window.
    CONFIG:
      freshness_minutes: int, default 45
    """
    fresh_min = int(CONFIG.get("freshness_minutes", 45))
    if fresh_min <= 0:
        return True
    try:
        sig = _coerce_ts_utc(ts_utc)
    except Exception:
        return False
    delta = now_utc() - sig
    return delta <= timedelta(minutes=fresh_min)

def is_near_trigger(direction: str,
                    bid: float,
                    ask: float,
                    trigger: float,
                    band_lo: float,
                    band_hi: float) -> bool:
    """
    True if the current market is near the trigger and inside (or grazing) the band.
    CONFIG:
      trigger_tolerance_usd: float, default 0.02
      allow_outside_band_tolerance: float, default 0.00
    """
    tol = float(CONFIG.get("trigger_tolerance_usd", 0.02))
    band_tol = float(CONFIG.get("allow_outside_band_tolerance", 0.00))

    price = float(ask if str(direction).upper() == "BUY" else bid)
    t = float(trigger)
    lo = float(min(band_lo, band_hi))
    hi = float(max(band_lo, band_hi))

    # Must be within band (with small tolerance if allowed)
    inside = (lo - band_tol) <= price <= (hi + band_tol)
    if not inside:
        return False

    # Must be near trigger
    return abs(price - t) <= tol


# ───────────────────────── Sizing/price helpers ─────────────────────────

def _struct_stop(width: float) -> float:
    return max(float(CONFIG.get("min_stop_usd", 1.0)),
               float(CONFIG.get("risk_mult", 2.3)) * float(width))

def _robust_stop(zone_row) -> float:
    struct = _struct_stop(float(zone_row.get("zone_width", float(CONFIG.get("zone_width_usd", 5.0)))))
    atr = zone_row.get("atr", float("nan"))
    if atr == atr and atr > 0:
        return max(struct, atr * float(CONFIG.get("atr_multiplier", 3.0)))
    return struct

def _effective_entry_price(direction: str, trigger_price: float, bid: float, ask: float) -> float:
    fp = str(CONFIG.get("band_fill_price", "touch")).lower().strip()
    if fp == "threshold":
        return float(trigger_price)
    return float(ask if direction == "BUY" else bid)


# ───────────────────────── Diagnostics helpers ─────────────────────────

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

def _dump_attempts(prefix: str, order, now: datetime):
    if not isinstance(order, dict):
        print(f"{now} → {prefix} attempts: <no order/unknown type>")
        return
    attempts = order.get("attempts", []) or []
    if not attempts:
        print(f"{now} → {prefix} attempts: <none>")
        return
    print(f"{now} → {prefix} attempts ({len(attempts)}):")
    for i, att in enumerate(attempts, 1):
        try:
            print(
                f"  #{i} fill={att.get('filling')} "
                f"price={att.get('price')} "
                f"check={att.get('check_retcode')}('{att.get('check_comment')}') "
                f"send={att.get('send_retcode')}('{att.get('send_comment')}')"
            )
        except Exception as e:
            print(f"  #{i} <print-error: {e}>  raw={att}")

def _print_env_caps(prefix: str):
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


# ───────────────────────── Order wrapper ─────────────────────────

def _send_order_safe(direction: str, eff_px: float, stop_dist: float, trig: float,
                     *, notifier: Notifier | None = None, pending_key: str | None = None):
    try:
        out = send_market_order(direction, eff_px, stop_dist,
                                trigger_price=trig,
                                notifier=notifier,
                                pending_key=pending_key)
        if out is None:
            le = None
            try:
                le = mt5.last_error()
            except Exception:
                pass
            return {
                "entry_result": None,
                "attempts": [],
                "last_error": le or ("send_market_order_returned_none",),
                "requested_lots": 0.0,
                "requested_price": eff_px,
                "comment_used": "",
            }
        if not isinstance(out, dict):
            return {
                "entry_result": None,
                "attempts": [],
                "last_error": ("bad_return_type", type(out).__name__),
                "requested_lots": 0.0,
                "requested_price": eff_px,
                "comment_used": "",
            }
        out.setdefault("attempts", [])
        out.setdefault("requested_price", eff_px)
        out.setdefault("requested_lots", 0.0)
        out.setdefault("comment_used", "")
        return out
    except Exception as e:
        return {
            "entry_result": None,
            "attempts": [],
            "last_error": ("exception", repr(e)),
            "requested_lots": 0.0,
            "requested_price": eff_px,
            "comment_used": "",
        }


# ───────────────────────── Formatting helpers ─────────────────────────

def _fmt_levels_for_post(direction: str, entry_mid: float, stop_dist: float):
    tp1_mult = float(CONFIG.get("tp1_mult", 0.68))
    tp2_mult = float(CONFIG.get("tp2_mult", 1.16))
    tp3_mult = float(CONFIG.get("tp3_mult", 1.80))
    use_tp3   = bool(CONFIG.get("use_tp3", True))
    d = direction.upper()
    if d == "BUY":
        sl  = entry_mid - stop_dist
        tps = [entry_mid + tp1_mult*stop_dist, entry_mid + tp2_mult*stop_dist]
        if use_tp3: tps.append(entry_mid + tp3_mult*stop_dist)
    else:
        sl  = entry_mid + stop_dist
        tps = [entry_mid - tp1_mult*stop_dist, entry_mid - tp2_mult*stop_dist]
        if use_tp3: tps.append(entry_mid - tp3_mult*stop_dist)
    return float(sl), [float(x) for x in tps]


# ───────────────────────── Optional test hook ─────────────────────────

def _force_test_entry_if_requested(now: datetime, tg: Notifier | None):
    if not bool(CONFIG.get("force_test_entry", False)):
        return
    try:
        symbol = str(CONFIG.get("symbol", "XAUUSD"))
        bid, ask = fetch_tick(symbol)
        sd = _struct_stop(float(CONFIG.get("zone_width_usd", 5.0)))
        eff = _effective_entry_price("BUY", ask, bid, ask)
        pending_key = _sig_key(symbol, "BUY", ask, ask, now_utc())
        order = _send_order_safe("BUY", eff, sd, trig=ask, notifier=tg, pending_key=pending_key)
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
            print(f"{now} → TEST order OK  retcode={rc} ticket={ticket} "
                  f"price={order['requested_price']:.2f} lots={order['requested_lots']}")
        else:
            print(f"{now} → TEST order FAILED  retcode={rc} (expected DONE/PLACED)")
            _print_last_error("TEST", now)
    except Exception as e:
        print(f"{now} → TEST order EXCEPTION: {e}")
        _print_env_caps("[test]")
    finally:
        CONFIG["force_test_entry"] = False


# ───────────────────────── Main loop ─────────────────────────

def main_loop():
    # Login/init MT5
    login    = int(CONFIG.get("login", 0)) or None
    password = CONFIG.get("password")
    server   = CONFIG.get("server")
    init_mt5(login=login, password=password, server=server)
    print("Starting live session loop...")
    _print_env_caps("[env]")

    # Telegram notifier
    tg = Notifier()
    if tg.enabled:
        tgc = CONFIG.get("telegram", {})
        print("[tg] enabled=True")
        print("     chat_id =", tgc.get("chat_id"))
        try:
            tg.client.client.send_message("✅ Bot online. I will post signals here.")
        except Exception as e:
            print("[tg] start message failed:", repr(e))
    else:
        tgc = CONFIG.get("telegram", {})
        print("[tg] enabled=False  (bot won’t post)")
        print("     reason:", tg.reason)
        print("     bot_token set? ", bool(tgc.get("bot_token")))
        print("     chat_id set?   ", bool(tgc.get("chat_id")))
        print("     If creds are set, check: bot is admin in the private channel & server can reach api.telegram.org.")

    # Track already posted signal cards
    posted_keys: set[str] = set()

    # Where is send_market_order sourced from
    try:
        print(f"[env] entry.send_market_order from module: {getattr(send_market_order, '__module__', '?')}")
    except Exception:
        pass

    # Initial zone load
    try:
        z = get_today_zones()
        zones = pd.DataFrame() if z is None else z.reset_index(drop=True)
    except Exception as e:
        print(f"[{now_utc()}] Initial zone load failed: {e}")
        zones = pd.DataFrame()

    # Post signals on startup, but only fresh if configured
    startup = True
    if tg.enabled and not zones.empty:
        _post_signals_to_telegram(zones, tg, posted_keys, startup=startup)

    # Intraday refresh cadence
    last_zone_refresh = now_utc()
    refresh_secs = 15 * 60

    # Trigger tracking
    triggered_keys: set[tuple[datetime, str]] = set()
    min_delay = int(CONFIG.get("entry_min_delay_minutes", 0))

    # Daily throughput control
    day_date = None
    day_start_equity = None
    loss_cut = float(CONFIG.get("day_loss_limit_pct", 0.05))
    max_per_day = int(CONFIG.get("max_signals_per_day", 1))
    both_sides = bool(CONFIG.get("both_sides_per_day", True))
    day_fills = 0
    did_buy = False
    did_sell = False

    while True:
        now = now_utc()
        startup = False  # after first loop tick we are no longer at startup

        # One-shot plumbing check
        _force_test_entry_if_requested(now, tg if tg.enabled else None)

        # New UTC day refresh
        if day_date != now.date():
            ai = mt5.account_info()
            day_start_equity = float(getattr(ai, "equity", 0.0)) if ai else None
            day_date = now.date()
            triggered_keys.clear()
            posted_keys.clear()
            day_fills = 0
            did_buy = did_sell = False
            try:
                z = get_today_zones()
                zones = pd.DataFrame() if z is None else z.reset_index(drop=True)
            except Exception as e:
                print(f"[{now}] Zone refresh (new UTC day) failed: {e}")
                zones = pd.DataFrame()
            print(f"[{now}] New UTC day → equity baseline={day_start_equity}, zones={len(zones)}")
            last_zone_refresh = now

            if tg.enabled and not zones.empty:
                _post_signals_to_telegram(zones, tg, posted_keys, startup=True)

        # Intraday zone refresh
        if (now - last_zone_refresh).total_seconds() >= refresh_secs:
            try:
                new = get_today_zones()
                if new is not None and not new.empty:
                    zones = new.reset_index(drop=True)
                    print(f"[{now}] Refreshed zones intraday: {len(zones)}")
                    if tg.enabled:
                        _post_signals_to_telegram(zones, tg, posted_keys, startup=False)
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
                # Backward compatibility: prefer new columns, fall back to legacy
                ts = z.get("ts_utc", z.get("datetime"))
                blo = float(z.get("band_lo", z.get("entry_low")))
                bhi = float(z.get("band_hi", z.get("entry_high")))
                trig = float(z.get("trigger", (blo + bhi) / 2.0))
                direction = str(z.get("direction"))

                # Freshness gate
                if not is_fresh(ts):
                    continue

                # Near-trigger gate
                if not is_near_trigger(direction, bid, ask, trig, blo, bhi):
                    continue

                # Throughput cap: stop after max_per_day
                if day_fills >= max_per_day:
                    break
                # Respect one-side-per-day if both_sides is False
                if (direction == "BUY" and did_buy and not both_sides) or \
                   (direction == "SELL" and did_sell and not both_sides):
                    continue

                sig_time = _coerce_ts_utc(ts)
                key = (sig_time, direction)
                if key in triggered_keys:
                    continue
                if now < (sig_time + timedelta(minutes=min_delay)):
                    continue

                # Legacy microstructure check preserved
                ok, legacy_trig = should_trigger_trade(
                    direction,
                    float(z.get("entry_low", blo)),
                    float(z.get("entry_high", bhi)),
                    bid,
                    ask
                )
                if not ok:
                    continue

                ref_px = ask if direction == "BUY" else bid
                if _m1_gap_excessive(CONFIG["symbol"], ref_px):
                    print(f"{now} → Gap filter blocked {direction} trade (ref {ref_px:.2f}).")
                    continue

                stop_dist = _robust_stop(z)
                eff_px = _effective_entry_price(direction, trig, bid, ask)

                # Link to posted signal via deterministic key
                pending_key = _sig_key(
                    CONFIG["symbol"],
                    direction,
                    blo,
                    bhi,
                    _coerce_ts_utc(ts),
                )

                order = _send_order_safe(
                    direction,
                    eff_px,
                    stop_dist,
                    trig=trig,
                    notifier=(tg if tg.enabled else None),
                    pending_key=pending_key
                )

                _dump_attempts("LIVE order", order, now)

                res = order.get("entry_result")
                if res is None:
                    print(f"{now} → LIVE order FAILED: entry_result=None  last_error={order.get('last_error')}")
                    _print_last_error("LIVE order", now)
                    continue

                rc  = getattr(res, "retcode", None)
                ticket = getattr(res, "order", None)
                if rc in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                    print(f"{now} → LIVE order OK  retcode={rc} ticket={ticket} "
                          f"price={order['requested_price']:.2f} lots={order['requested_lots']}")
                else:
                    print(f"{now} → LIVE order NOT OK  retcode={rc}")
                    _print_last_error("LIVE order", now)
                    continue

                order.update({
                    "ticket": ticket,
                    "symbol": CONFIG["symbol"],
                    "direction": direction,
                    "volume": order["requested_lots"],
                    "entry_price": order["requested_price"],
                })

                triggered_keys.add(key)
                day_fills += 1
                if direction == "BUY":
                    did_buy = True
                else:
                    did_sell = True

        else:
            if near_swap:
                print(f"[{now}] Entries paused near swap cutoff.")
            elif daily_hit:
                print(f"[{now}] Daily loss limit hit — entries paused until next UTC day "
                      f"(start={day_start_equity:.2f}, now={eq:.2f}).")

        # Manage exits & log
        try:
            closed_trades = manage_all_trades(notifier=(tg if tg.enabled else None))
            for trade in closed_trades:
                try:
                    write_trade_to_csv(trade)
                except Exception as e:
                    print(f"{now} → CSV log error: {e}  row={trade}")
                print(f"{trade['exit_time']} → Trade closed: {trade['exit_reason']}")
        except Exception as e:
            print(f"{now} → Trade management error: {e}")

        # Pace
        sleep_s = 60 if (not in_session or near_swap or daily_hit) else 20
        time.sleep(sleep_s)


def _post_signals_to_telegram(zones: pd.DataFrame,
                              tg: Notifier,
                              posted_keys: set[str],
                              *,
                              startup: bool):
    """
    Post one Telegram 'signal card' per zone once per day.
    Only fresh signals are posted. On startup, stale announcements are skipped
    if CONFIG['send_stale_on_startup'] is False.
    """
    symbol = CONFIG["symbol"]
    allow_stale_on_start = bool(CONFIG.get("send_stale_on_startup", False))

    for _, z in zones.iterrows():
        ts = z.get("ts_utc", z.get("datetime"))
        blo = float(z.get("band_lo", z.get("entry_low")))
        bhi = float(z.get("band_hi", z.get("entry_high")))
        direction = str(z.get("direction"))

        # Freshness gating for announcements
        fresh = is_fresh(ts)
        if startup and not allow_stale_on_start and not fresh:
            continue
        if not fresh:
            # Even outside startup, do not spam stale signals
            continue

        key = str(_sig_key(
            symbol,
            direction,
            blo,
            bhi,
            _coerce_ts_utc(ts),
        ))
        if key in posted_keys:
            continue

        # Compose levels
        zone_width = float((bhi - blo) if "zone_width" not in z else z["zone_width"])
        entry_mid = float(((blo + bhi) / 2.0) if "entry_mid" not in z else z["entry_mid"])
        # If the scanner provided "atr" attach it; not required here though
        stop_dist = _robust_stop({
            "zone_width": zone_width,
            "atr": z.get("atr", float("nan"))
        })
        sl, tps = _fmt_levels_for_post(direction, entry_mid, stop_dist)

        try:
            res = tg.post_signal(
                symbol=symbol,
                side=direction,
                entry_zone=(blo, bhi),
                sl=float(sl),
                tps=tps,
                when_utc=_coerce_ts_utc(ts),
            )
            if res:
                posted_keys.add(key)
        except Exception as e:
            print("Telegram post failed:", e)


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
