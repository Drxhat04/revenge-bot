#!/usr/bin/env python
"""
Enhanced XAUUSD OB / Liquidity-Sweep Scanner (multi-profile, unbiased)
----------------------------------------------------------------------
• UTC-native, no look-ahead
• Confirms pierce/FVG on 15m bar 'c' and timestamps signal at next 15m open
• Writes ob_time and confirm_time for audits
• Skips Saturday / Sunday data
• Runs multiple setting profiles and saves one CSV per profile
• Adds `min_conditions` and neutral side selection to remove BUY-only bias
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import pathlib, warnings, yaml, pandas as pd
import numpy as np
from typing import List
from datetime import timedelta

# ── Load base config ──────────────────────────────────────────────────────
CFG = yaml.safe_load(open("config.yaml"))

# Convenience access (kept for defaults)
BASE_ZONE_W      = float(CFG.get("zone_width_usd", 5.0))
BASE_BUFFER_USD  = float(CFG.get("sweep_buffer_usd", 0.1))
BASE_ASIA_START  = str(CFG.get("asia_start", "18:00"))  # UTC
BASE_ASIA_END    = str(CFG.get("asia_end",   "09:00"))  # UTC
BASE_TOKYO_ONLY  = bool(CFG.get("tokyo_only", False))
BASE_REQUIRE_P   = bool(CFG.get("require_pierce", True))
BASE_REQUIRE_FVG = bool(CFG.get("require_fvg",   True))
BASE_GAP_FILTER  = float(CFG.get("gap_filter",   0.005))
BASE_DEBUG_DAYS  = int(CFG.get("debug_days",     0))
BASE_MIN_COND    = int(CFG.get("min_conditions", 1))  # NEW

# ── Parameters for one scan run ───────────────────────────────────────────
@dataclass
class ScanParams:
    name: str
    zone_width_usd: float = BASE_ZONE_W
    sweep_buffer_usd: float = BASE_BUFFER_USD
    asia_start: str = BASE_ASIA_START
    asia_end: str = BASE_ASIA_END
    tokyo_only: bool = BASE_TOKYO_ONLY
    require_pierce: bool = BASE_REQUIRE_P
    require_fvg: bool = BASE_REQUIRE_FVG
    min_conditions: int = BASE_MIN_COND   # NEW: how many enabled conditions must actually hit (0/1/2)
    gap_filter: float = BASE_GAP_FILTER
    debug_days: int = BASE_DEBUG_DAYS

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
        if df.time.dt.tz is None:
            df.time = df.time.dt.tz_localize("UTC")
        dfs.append(df)

    m1 = pd.concat(dfs, ignore_index=True).sort_values("time")

    # Deduplicate timestamps (keep last)
    before = len(m1)
    m1 = m1.drop_duplicates(subset="time", keep="last")
    if before - len(m1):
        print(f"Deduplicated {before - len(m1):,} overlapping M1 rows")

    # Drop weekends (UTC 5=Sat,6=Sun)
    m1 = m1[~m1.time.dt.weekday.isin([5, 6])]

    # True Range (optional diagnostics)
    m1["prev_close"] = m1["close"].shift(1)
    m1["tr"] = np.maximum.reduce([
        m1["high"] - m1["low"],
        (m1["high"] - m1["prev_close"]).abs(),
        (m1["low"]  - m1["prev_close"]).abs()
    ])
    return m1

def fvg_hit(prev: pd.Series, cur: pd.Series, side: str) -> bool:
    """Actual 2-candle FVG event (boolean) irrespective of 'require_fvg'."""
    return (cur.low > prev.high) if side == "BUY" else (prev.low > cur.high)

def last_opposite(df15: pd.DataFrame, idx: int, side: str) -> pd.Series:
    """Find the most recent opposite-color candle before index idx."""
    mask = (df15.close < df15.open) if side == "BUY" else (df15.close > df15.open)
    sub  = df15.loc[: idx - 1][mask]
    return sub.iloc[-1] if not sub.empty else df15.iloc[idx - 1]

# ── Core scan for one profile ─────────────────────────────────────────────
def scan(m1: pd.DataFrame, P: ScanParams) -> pd.DataFrame:
    """Generate OB / sweep signals using UTC timestamps, no look-ahead, for one profile."""
    # Build 15-min bars in UTC
    m15 = (m1.set_index("time")
              .resample("15min")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
              .dropna()
              .reset_index())

    signals = []
    dbg_left = P.debug_days
    boxes = pierces = 0

    start_td, end_td = _td(P.asia_start), _td(P.asia_end)

    for date, day15 in m15.groupby(m15.time.dt.date):
        # skip weekends
        if pd.Timestamp(date, tz="UTC").weekday() >= 5:
            continue

        # Determine Asia box window in UTC
        d0 = pd.Timestamp(date, tz="UTC")
        if P.tokyo_only:
            # 01–07 UTC
            win_start = d0 + pd.Timedelta(hours=1)
            win_end   = d0 + pd.Timedelta(hours=7)
        else:
            win_start = d0 - pd.Timedelta(days=1) + start_td
            win_end   = d0 + end_td

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
            if abs(c.open - p.close) / abs(denom) > P.gap_filter:
                if dbg_left:
                    print("    gap filter hit")
                continue

            # Evaluate "hits" (events that actually happened)
            buy_pierce_hit  = (c.low  < lo - P.sweep_buffer_usd)
            sell_pierce_hit = (c.high > hi + P.sweep_buffer_usd)
            buy_fvg_h       = fvg_hit(p, c, "BUY")
            sell_fvg_h      = fvg_hit(p, c, "SELL")

            # Count only ENABLED conditions toward min_conditions
            buy_hits  = int(P.require_pierce and buy_pierce_hit) + int(P.require_fvg and buy_fvg_h)
            sell_hits = int(P.require_pierce and sell_pierce_hit) + int(P.require_fvg and sell_fvg_h)
            enabled_cnt = int(P.require_pierce) + int(P.require_fvg)

            # Effective need: if nothing is enabled, require at least 1 real thing anyway
            need = P.min_conditions if enabled_cnt > 0 else max(1, P.min_conditions)

            buy_ok  = (buy_hits  >= min(need, max(1, enabled_cnt)))
            sell_ok = (sell_hits >= min(need, max(1, enabled_cnt)))

            # Neutral side selection (avoid BUY-first bias):
            # prefer side that pierced; else side with FVG; else skip (tie/none)
            choose = None
            if buy_ok or sell_ok:
                if buy_pierce_hit and not sell_pierce_hit:
                    choose = "BUY"
                elif sell_pierce_hit and not buy_pierce_hit:
                    choose = "SELL"
                elif buy_fvg_h and not sell_fvg_h:
                    choose = "BUY"
                elif sell_fvg_h and not buy_fvg_h:
                    choose = "SELL"
                else:
                    # both/neither → skip this 15m to avoid arbitrary bias
                    choose = None

            if choose == "BUY":
                ob = last_opposite(day15, i, "BUY")
                ob_lo, ob_hi = ob.low, min(ob.high, ob.low + P.zone_width_usd)
                if ob_hi > ob_lo:
                    confirm_time = c.time + pd.Timedelta(minutes=15)  # next 15m open
                    signals.append([
                        confirm_time, "BUY", ob_lo, ob_hi, np.nan,
                        ob.time, c.time
                    ])
                    pierced_today |= buy_pierce_hit
                    break

            elif choose == "SELL":
                ob = last_opposite(day15, i, "SELL")
                ob_hi, ob_lo = ob.high, max(ob.low, ob.high - P.zone_width_usd)
                if ob_hi > ob_lo:
                    confirm_time = c.time + pd.Timedelta(minutes=15)  # next 15m open
                    signals.append([
                        confirm_time, "SELL", ob_lo, ob_hi, np.nan,
                        ob.time, c.time
                    ])
                    pierced_today |= sell_pierce_hit
                    break

        if pierced_today:
            pierces += 1
        if dbg_left:
            print(f"    pierced={pierced_today}")
            dbg_left -= 1

    print(f"[{P.name}] Days with Asia box: {boxes},  pierced: {pierces}")
    out = pd.DataFrame(
        signals,
        columns=[
            "datetime", "direction", "entry_low", "entry_high", "atr",
            "ob_time", "confirm_time"
        ],
    )
    if not out.empty:
        out["zone_width"] = out.entry_high - out.entry_low
        out["entry_mid"]  = (out.entry_low + out.entry_high) / 2
    return out.sort_values("datetime")

def post_gap_filter(zones: pd.DataFrame, m1: pd.DataFrame, gap_filter: float) -> pd.DataFrame:
    """Ensure the signal minute itself isn't a large gap from prior minute."""
    if zones.empty:
        return zones
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
            if abs(bar.open.values[0] - prev_close) / abs(denom) <= gap_filter:
                valid.append(row)
        else:
            valid.append(row)
    return pd.DataFrame(valid).sort_values("datetime")

# ── Profile construction ──────────────────────────────────────────────────
def build_profiles_from_config() -> List[ScanParams]:
    """
    If config.yaml defines:
      scan_profiles:
        - name: base
          require_pierce: false
          require_fvg: true
          min_conditions: 1
          gap_filter: 0.005
          tokyo_only: false
          zone_width_usd: 5.0
          sweep_buffer_usd: 0.10
          asia_start: "18:00"
          asia_end: "09:00"
    ...those keys override the BASE_* values.
    If absent, we generate a default matrix including strict/loose combos.
    """
    profiles_cfg = CFG.get("scan_profiles")
    base = ScanParams(name="base")
    profiles: List[ScanParams] = []

    if isinstance(profiles_cfg, list) and profiles_cfg:
        for pcfg in profiles_cfg:
            p = replace(
                base,
                name             = str(pcfg.get("name", "noname")),
                zone_width_usd   = float(pcfg.get("zone_width_usd",   base.zone_width_usd)),
                sweep_buffer_usd = float(pcfg.get("sweep_buffer_usd", base.sweep_buffer_usd)),
                asia_start       = str(pcfg.get("asia_start",        base.asia_start)),
                asia_end         = str(pcfg.get("asia_end",          base.asia_end)),
                tokyo_only       = bool(pcfg.get("tokyo_only",       base.tokyo_only)),
                require_pierce   = bool(pcfg.get("require_pierce",   base.require_pierce)),
                require_fvg      = bool(pcfg.get("require_fvg",      base.require_fvg)),
                min_conditions   = int(pcfg.get("min_conditions",    base.min_conditions)),
                gap_filter       = float(pcfg.get("gap_filter",      base.gap_filter)),
                debug_days       = int(pcfg.get("debug_days",        base.debug_days)),
            )
            profiles.append(p)
        return profiles

    # Default matrix if none provided
    profiles = [
        base,  # as per config.yaml
        # stricter & clearer gating
        replace(base, name="strict_both_min2", require_pierce=True,  require_fvg=True,  min_conditions=2),
        replace(base, name="strict_any_one",   require_pierce=True,  require_fvg=True,  min_conditions=1),
        replace(base, name="pierce_only",      require_pierce=True,  require_fvg=False, min_conditions=1),
        replace(base, name="fvg_only",         require_pierce=False, require_fvg=True,  min_conditions=1),
        # variations
        replace(base, name="tokyo_only",       tokyo_only=True),
        replace(base, name="loose_gap_10bp",   gap_filter=0.010),
        replace(base, name="tight_gap_3bp",    gap_filter=0.003),
        replace(base, name="wide_zone_7",      zone_width_usd=7.0),
    ]
    return profiles

# ── CLI wrapper ───────────────────────────────────────────────────────────
def main() -> None:
    warnings.filterwarnings("ignore")
    out_dir = pathlib.Path("signals")
    out_dir.mkdir(exist_ok=True)

    print("Loading M1 data (UTC)")
    m1 = load_m1()
    print(f"Loaded rows: {len(m1):,}")

    profiles = build_profiles_from_config()
    print(f"Running {len(profiles)} profile(s): {[p.name for p in profiles]}")

    for P in profiles:
        print(f"\n=== Profile: {P.name} ===")
        zones = scan(m1, P)
        zones = post_gap_filter(zones, m1, P.gap_filter)

        out_name = f"signals_long_utc__{P.name}.csv"
        zones.to_csv(out_dir / out_name, index=False)

        print(f"[{P.name}] Generated {len(zones)} signals → {out_dir/out_name}")
        if not zones.empty:
            dmin, dmax = zones.datetime.min(), zones.datetime.max()
            side_counts = zones.direction.value_counts().to_dict()
            print(f"[{P.name}] Date range: {dmin} → {dmax} | sides: {side_counts}")

if __name__ == "__main__":
    main()
