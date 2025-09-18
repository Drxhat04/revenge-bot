# scanner.py

import pandas as pd
from datetime import datetime, timedelta, timezone

from config import CONFIG
import zone_scanner as zs


def _load_m1_from_mt5(symbol: str, days: int = 3) -> pd.DataFrame:
    """
    Pull recent M1 directly from MT5 (last `days`) with UTC timestamps.
    """
    import MetaTrader5 as mt5
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M1,
        start.replace(tzinfo=None),
        end.replace(tzinfo=None),
    )
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "spread"])

    m1 = pd.DataFrame(rates)
    # MT5 returns epoch seconds in 'time'
    m1["time"] = pd.to_datetime(m1["time"], unit="s", utc=True)
    if "spread" not in m1.columns:
        m1["spread"] = 0

    # Ensure strictly sorted & deduped by time
    m1 = (
        m1.sort_values("time")
           .drop_duplicates(subset="time", keep="last")
           .reset_index(drop=True)
    )
    return m1


def _ensure_tr_column(m1: pd.DataFrame) -> pd.DataFrame:
    """
    zone_scanner expects an M1 'tr' column (True Range vs previous close).
    Add it here for live data pulled from MT5.
    """
    if "tr" in m1.columns:
        return m1
    if m1.empty:
        m1["tr"] = []
        return m1
    prev_close = m1["close"].shift(1)
    hi_lo = (m1["high"] - m1["low"]).abs()
    hi_pc = (m1["high"] - prev_close).abs()
    lo_pc = (m1["low"]  - prev_close).abs()
    m1["tr"] = pd.concat([hi_lo, hi_pc, lo_pc], axis=1).max(axis=1)
    return m1


def _zones_from_today(zones: pd.DataFrame) -> pd.DataFrame:
    """
    Filter scan results to *today's* UTC signals and compute helper columns.
    """
    if zones is None or zones.empty:
        return pd.DataFrame(columns=["datetime", "direction", "entry_low", "entry_high",
                                     "entry_mid", "zone_width"])

    # Make sure datetime is tz-aware UTC
    zdt = pd.to_datetime(zones["datetime"], utc=True)
    zones = zones.copy()
    zones["datetime"] = zdt

    today = datetime.utcnow().date()
    zones = zones[zones["datetime"].dt.tz_convert("UTC").dt.date == today].copy()

    if zones.empty:
        return pd.DataFrame(columns=["datetime", "direction", "entry_low", "entry_high",
                                     "entry_mid", "zone_width"])

    zones["entry_mid"] = (zones["entry_low"] + zones["entry_high"]) / 2.0
    zones["zone_width"] = (zones["entry_high"] - zones["entry_low"]).abs()

    return zones.sort_values("datetime").reset_index(drop=True)


# ---------- Public API (LIVE ONLY) ----------
def get_today_zones() -> pd.DataFrame:
    """
    Live-only: pull M1 from MT5, precompute into zone_scanner, run scan(),
    and return today's UTC zones with entry_mid/zone_width.
    """
    symbol = str(CONFIG.get("symbol", "XAUUSD"))
    try:
        m1 = _load_m1_from_mt5(symbol, days=int(CONFIG.get("live_days_back", 3)))
    except Exception as e:
        print(f"scanner:get_today_zones MT5 load failed: {e}")
        return pd.DataFrame(columns=["datetime", "direction", "entry_low", "entry_high",
                                     "entry_mid", "zone_width"])

    if m1.empty:
        return pd.DataFrame(columns=["datetime", "direction", "entry_low", "entry_high",
                                     "entry_mid", "zone_width"])

    # Add TR column expected by zone_scanner, then prime its globals
    m1 = _ensure_tr_column(m1)
    try:
        zs._precompute(m1)   # set M1/M15/H1/ATR15/EMA_H1 inside zone_scanner
        zones = zs.scan()    # new signature: takes no args
    except Exception as e:
        print(f"scanner:get_today_zones scan failed: {e}")
        return pd.DataFrame(columns=["datetime", "direction", "entry_low", "entry_high",
                                     "entry_mid", "zone_width"])

    return _zones_from_today(zones)
