# utils.py

from __future__ import annotations
from typing import Iterable, Tuple, Optional
import MetaTrader5 as mt5
from datetime import datetime, timedelta, time as dtime, timezone
from config import CONFIG


# ─────────────────────────── Time helpers (UTC, naive) ───────────────────────────

def now_utc() -> datetime:
    """Naive UTC timestamp to match the rest of the codebase."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

def _parse_session(s: str) -> Tuple[dtime, dtime]:
    """'HH:MM-HH:MM' → (start_time, end_time)"""
    start_s, end_s = s.split("-")
    h1, m1 = map(int, start_s.split(":"))
    h2, m2 = map(int, end_s.split(":"))
    return dtime(h1, m1), dtime(h2, m2)

def within_trading_session(
    now: Optional[datetime] = None,
    sessions: Optional[Iterable[str]] = None
) -> bool:
    """
    Check if the current time (or provided 'now') is within any of the trading sessions.
    """
    ts = now or now_utc()
    t = ts.time()
    sess = list(sessions) if sessions is not None else list(CONFIG.get("sessions", []))
    if not sess:
        return True  # no filter configured → allow
    for s in sess:
        start_t, end_t = _parse_session(s)
        if start_t <= end_t:
            # same-day window
            if start_t <= t < end_t:
                return True
        else:
            # overnight window
            if t >= start_t or t < end_t:
                return True
    return False


def is_near_swap_cutoff(ts: datetime, *, respect_flag: bool = True) -> bool:
    """
    True if 'ts' (UTC) is within the pre-specified buffer window ahead of the swap cutoff.
    If respect_flag is True, returns False immediately when CONFIG['swap_avoidance'] is False.
    """
    if respect_flag and not CONFIG.get("swap_avoidance", False):
        return False
    cutoff_hour = int(CONFIG.get("swap_cutoff_hour", 19))
    buffer_hours = int(CONFIG.get("swap_buffer_hours", 3))
    cutoff = ts.replace(hour=cutoff_hour, minute=0, second=0, microsecond=0)
    if ts.hour >= cutoff_hour:
        cutoff += timedelta(days=1)
    return timedelta(0) <= (cutoff - ts) < timedelta(hours=buffer_hours)


def is_weekend(dt: datetime) -> bool:
    """Saturday (5) or Sunday (6) in UTC."""
    return dt.weekday() in (5, 6)


def fmt_time(dt):
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(int(dt), tz=timezone.utc)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────── MT5 helpers ───────────────────────────

def init_mt5(*, login: int | None = None, password: str | None = None,
             server: str | None = None, path: str | None = None) -> None:
    """
    Initialize MT5, optionally logging into a specific account/server.
    Selects the configured symbol.
    """
    if not mt5.initialize(path) if path else not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    # Optional login (if you run multiple accounts)
    if login is not None and password is not None and server is not None:
        if not mt5.login(login=login, password=password, server=server):
            mt5.shutdown()
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    symbol = CONFIG["symbol"]
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Failed to select symbol: {symbol}")

    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise RuntimeError(f"symbol_info({symbol}) returned None")

    # Basic tradability check (trade_mode != disabled)
    # Some brokers set 0=disabled, 1=longonly, 2=shortonly, 3=closeonly, 4=full
    if getattr(info, "trade_mode", 4) == 0:
        mt5.shutdown()
        raise RuntimeError(f"Symbol {symbol} trade mode is disabled")

    print(f"MT5 initialized; symbol '{symbol}' ready (digits={info.digits}, contract={getattr(info,'trade_contract_size', 'n/a')}).")


def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass
