#!/usr/bin/env python
"""
Enhanced XAUUSD OB / Liquidity-Sweep Scanner (v2.5, UTC-native, no look-ahead)
------------------------------------------------------------------------------
• All timestamps in UTC
• Confirms pierce/FVG on 15m bar 'c' and timestamps signal at next 15m open
• Writes ob_time and confirm_time for audits
• Skips Saturday / Sunday data
"""

from __future__ import annotations
import pathlib, warnings, yaml, pandas as pd
import numpy as np
from datetime import timedelta

# ── Config ───────────────────────────────────────────────────────────────
CFG            = yaml.safe_load(open("config.yaml"))
ZONE_W         = CFG["zone_width_usd"]
BUFFER_USD     = CFG.get("sweep_buffer_usd", 0.05)
ASIA_START     = CFG.get("asia_start", "18:00")  # interpreted as UTC
ASIA_END       = CFG.get("asia_end",   "09:00")  # interpreted as UTC
TOKYO_ONLY     = CFG.get("tokyo_only", False)
REQUIRE_PIERCE = CFG.get("require_pierce", True)
REQUIRE_FVG    = CFG.get("require_fvg",    True)
DEBUG_DAYS     = CFG.get("debug_days",     0)
GAP_FILTER     = CFG.get("gap_filter", 0.005)

# ── Helpers ───────────────────────────────────────────────────────────────
def _td(val: str | int | float) -> pd.Timedelta:
    """HH[:MM[:SS]] → Timedelta converter"""
    if isinstance(val, (int, float)):
        val = str(int(val))
    parts = str(val).strip().split(":")
    if len(parts) == 1:
        val = f"{parts[0]}:00:00"
    elif len(parts) == 2:
        val = f"{parts[0]}:{parts[1]}:00"
    return pd.to_timedelta(val)

START_TD, END_TD = _td(ASIA_START), _td(ASIA_END)


def load_m1(folder: str = "data") -> pd.DataFrame:
    """
    Load raw M1 CSVs, keep UTC tz, drop weekends, compute True Range.
    Expect columns: time_utc, open, high, low, close [, spread]
    """
    files = sorted(pathlib.Path(folder).glob("XAUUSD_M1_*.csv"))
    if not files:
        raise FileNotFoundError("No XAUUSD_M1_*.csv in /data")

    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["time_utc"]).rename(columns={"time_utc": "time"})
        # ensure UTC tz-aware
        if df.time.dt.tz is None:
            df.time = df.time.dt.tz_localize("UTC")
        dfs.append(df)

    m1 = pd.concat(dfs, ignore_index=True).sort_values("time")
    # ── REMOVE DUPLICATE TIMESTAMPS (keep last)
    before = len(m1)
    m1 = m1.drop_duplicates(subset="time", keep="last")
    dup_cnt = before - len(m1)
    if dup_cnt:
        print(f"Deduplicated {dup_cnt:,} overlapping M1 rows")

    # ── Weekend filter (UTC weekdays 5=Sat,6=Sun)
    m1 = m1[~m1.time.dt.weekday.isin([5, 6])]

    # ── Pre-compute True Range (for optional diagnostics)
    m1["prev_close"] = m1["close"].shift(1)
    m1["tr"] = np.maximum.reduce([
        m1["high"] - m1["low"],
        (m1["high"] - m1["prev_close"]).abs(),
        (m1["low"]  - m1["prev_close"]).abs()
    ])
    return m1


def fvg_ok(prev: pd.Series, cur: pd.Series, side: str) -> bool:
    """Simple 2-candle FVG test on 15m bars."""
    if not REQUIRE_FVG:
        return True
    return (cur.low > prev.high) if side == "BUY" else (prev.low > cur.high)


def last_opposite(df15: pd.DataFrame, idx: int, side: str) -> pd.Series:
    """Find the most recent opposite-color candle before index idx."""
    mask = (df15.close < df15.open) if side == "BUY" else (df15.close > df15.open)
    sub  = df15.loc[: idx - 1][mask]
    return sub.iloc[-1] if not sub.empty else df15.iloc[idx - 1]

# ── Core scan ─────────────────────────────────────────────────────────────

def scan(m1: pd.DataFrame) -> pd.DataFrame:
    """Generate OB / sweep signals using UTC timestamps, no look-ahead."""
    # Build 15-min bars in UTC
    m15 = (m1.set_index("time")
              .resample("15min")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
              .dropna()
              .reset_index())

    signals = []
    dbg_left = DEBUG_DAYS
    boxes = pierces = 0

    for date, day15 in m15.groupby(m15.time.dt.date):
        # skip weekends
        if pd.Timestamp(date, tz="UTC").weekday() >= 5:
            continue

        # Determine Asia box window in UTC
        d0 = pd.Timestamp(date, tz="UTC")
        if TOKYO_ONLY:
            win_start = d0 + pd.Timedelta(hours=1)   # 01-07 UTC
            win_end   = d0 + pd.Timedelta(hours=7)
        else:
            win_start = d0 - pd.Timedelta(days=1) + START_TD
            win_end   = d0 + END_TD

        asia = m1[(m1.time >= win_start) & (m1.time < win_end)]
        if asia.empty:
            continue

        boxes += 1
        hi, lo = asia.high.max(), asia.low.min()
        pierced_today = False
        if dbg_left:
            print(f"{date}  Asia UTC hi={hi:.2f}  lo={lo:.2f}  rows={len(asia)}")

        for i in range(1, len(day15)):
            p, c = day15.iloc[i - 1], day15.iloc[i]

            # gap filter on 15m close-to-open change
            denom = p.close if p.close != 0 else 1.0
            if abs(c.open - p.close) / abs(denom) > GAP_FILTER:
                if dbg_left:
                    print("    gap filter hit")
                continue

            # BUY side
            buy_pierce = c.low < lo - BUFFER_USD
            if (buy_pierce or not REQUIRE_PIERCE) and fvg_ok(p, c, "BUY"):
                ob = last_opposite(day15, i, "BUY")
                ob_lo, ob_hi = ob.low, min(ob.high, ob.low + ZONE_W)
                if ob_hi > ob_lo:
                    confirm_time = c.time + pd.Timedelta(minutes=15)  # next 15m open
                    signals.append([
                        confirm_time, "BUY", ob_lo, ob_hi, np.nan,  # atr left NaN (runner attaches proper ATR)
                        ob.time, c.time
                    ])
                    pierced_today |= buy_pierce
                    break

            # SELL side
            sell_pierce = c.high > hi + BUFFER_USD
            if (sell_pierce or not REQUIRE_PIERCE) and fvg_ok(p, c, "SELL"):
                ob = last_opposite(day15, i, "SELL")
                ob_hi, ob_lo = ob.high, max(ob.low, ob.high - ZONE_W)
                if ob_hi > ob_lo:
                    confirm_time = c.time + pd.Timedelta(minutes=15)  # next 15m open
                    signals.append([
                        confirm_time, "SELL", ob_lo, ob_hi, np.nan,
                        ob.time, c.time
                    ])
                    pierced_today |= sell_pierce
                    break

        if pierced_today:
            pierces += 1
        if dbg_left:
            print(f"    pierced={pierced_today}")
            dbg_left -= 1

    print(f"Days with Asia box: {boxes},  pierced: {pierces}")
    return pd.DataFrame(
        signals,
        columns=[
            "datetime", "direction", "entry_low", "entry_high", "atr",
            "ob_time", "confirm_time"
        ],
    )

# ── CLI wrapper ─────────────────────────────────────────────────────────────

def main() -> None:
    warnings.filterwarnings("ignore")
    print("Loading M1 data (UTC)")
    m1 = load_m1()
    print(f"Loaded rows: {len(m1):,}")

    print("Scanning for sweep-OB zones in UTC (no look-ahead)")
    zones = scan(m1)
    if zones.empty:
        print("No zones found with current config.")
        return

    zones["zone_width"] = zones.entry_high - zones.entry_low
    zones["entry_mid"]  = (zones.entry_low + zones.entry_high) / 2

    # Post-scan gap filter (UTC) — ensure the signal minute itself isn't a large gap from prior minute
    valid = []
    for _, row in zones.iterrows():
        bar = m1[m1.time == row.datetime]
        if bar.empty:
            valid.append(row)
            continue
        prev_idx = m1.index.get_loc(bar.index[0]) - 1
        if prev_idx >= 0:
            prev_close = m1.iloc[prev_idx].close
            denom = prev_close if prev_close != 0 else 1.0
            if abs(bar.open.values[0] - prev_close) / abs(denom) <= GAP_FILTER:
                valid.append(row)
        else:
            valid.append(row)

    zones = pd.DataFrame(valid).sort_values("datetime")
    zones.to_csv("signals_long_utc.csv", index=False)
    print(f"Generated {len(zones)} signals → signals_long_utc.csv")

if __name__ == "__main__":
    main()
