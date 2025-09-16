# scanner.py

import pandas as pd
from datetime import datetime, timedelta, timezone

from config import CONFIG
from zone_scanner import scan as run_scan


def _zones_from_m1_today(m1: pd.DataFrame) -> pd.DataFrame:
    """
    Run the scanner on provided M1 and return ONLY today's UTC signals
    with entry_mid and zone_width computed, plus an optional post-gap filter.
    """
    zones = run_scan(m1)
    if zones.empty:
        return zones

    zones["entry_mid"] = (zones.entry_low + zones.entry_high) / 2
    zones["zone_width"] = (zones.entry_high - zones.entry_low).abs()

    # keep only today's UTC signals
    today = datetime.utcnow().date()
    zones = zones[zones["datetime"].dt.tz_convert("UTC").dt.date == today].copy()

    # Optional post-gap filter using the exact signal minute
    filtered = []
    for _, row in zones.iterrows():
        match = m1[m1.time == row.datetime]
        if match.empty:
            filtered.append(row)
            continue
        prev_idx = m1.index.get_loc(match.index[0]) - 1
        if prev_idx >= 0:
            prev_close = m1.iloc[prev_idx].close
            bar_open = match.open.values[0]
            gap = abs(bar_open - prev_close) / (prev_close or 1.0)
            if gap <= float(CONFIG.get("gap_filter", 0.005)):
                filtered.append(row)
        else:
            filtered.append(row)

    return pd.DataFrame(filtered).sort_values("datetime").reset_index(drop=True)


def _load_m1_from_mt5(symbol: str, days: int = 3) -> pd.DataFrame:
    """
    Pull recent M1 directly from MT5 (last `days`) with UTC timestamps.
    This covers yesterday 18:00 → today 09:00 Asia window reliably.
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
    return (
        m1.sort_values("time")
          .drop_duplicates(subset="time", keep="last")
          .reset_index(drop=True)
    )


# ---------- Public API (LIVE ONLY) ----------
def get_today_zones() -> pd.DataFrame:
    """
    Live-only: pull M1 from MT5, scan, and return today's UTC zones.
    No CSV fallback.
    """
    symbol = str(CONFIG.get("symbol", "XAUUSD"))
    try:
        m1 = _load_m1_from_mt5(symbol, days=3)
    except Exception as e:
        print(f"scanner:get_today_zones MT5 load failed: {e}")
        return pd.DataFrame(columns=["datetime", "direction", "entry_low", "entry_high",
                                     "entry_mid", "zone_width"])

    if m1.empty:
        return pd.DataFrame(columns=["datetime", "direction", "entry_low", "entry_high",
                                     "entry_mid", "zone_width"])

    return _zones_from_m1_today(m1)
