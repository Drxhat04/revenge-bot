"""
Advanced XAUUSD OB / Liquidity‑Sweep scanner  (debug‑ready, switchable rules)
-----------------------------------------------------------------------------
• Reads all tunables from config.yaml
• If   require_pierce = true  → need sweep of Asia range ± buffer (old logic)
• If   require_pierce = false → first qualifying OB/FVG of the day is accepted
• tokyo_only = true           → ignore asia_start / asia_end; use 00‑06 UTC
• Writes   signals_long.csv   for the back‑test engine
"""

from __future__ import annotations

import pathlib, warnings, yaml, pandas as pd

# ── Config ────────────────────────────────────────────────────────────────
CFG            = yaml.safe_load(open("config.yaml"))
ZONE_W         = CFG["zone_width_usd"]
BUFFER_USD     = CFG.get("sweep_buffer_usd", 0.05)
ASIA_START     = CFG.get("asia_start", "18:00")
ASIA_END       = CFG.get("asia_end",   "09:00")
TOKYO_ONLY     = CFG.get("tokyo_only", False)
REQUIRE_PIERCE = CFG.get("require_pierce", True)
REQUIRE_FVG    = CFG.get("require_fvg",    True)
DEBUG_DAYS     = CFG.get("debug_days",     0)

BERLIN = "Europe/Berlin"

# ── Helpers ───────────────────────────────────────────────────────────────
def _td(val: str | int | float) -> pd.Timedelta:
    """
    Robust ‘HH[:MM[:SS]]’ → pandas Timedelta.
    Accepts 18, '18', '18:00', '18:00:00'.
    """
    if isinstance(val, (int, float)):
        val = str(int(val))
    val = str(val).strip()
    parts = val.split(":")
    if len(parts) == 1:
        val = f"{parts[0]}:00:00"
    elif len(parts) == 2:
        val = f"{parts[0]}:{parts[1]}:00"
    return pd.to_timedelta(val)

START_TD = _td(ASIA_START)
END_TD   = _td(ASIA_END)

def load_m1(folder: str = "data") -> pd.DataFrame:
    files = sorted(pathlib.Path(folder).glob("XAUUSD_M1_*.csv"))
    if not files:
        raise FileNotFoundError("No XAUUSD_M1_*.csv in /data")
    dfs = [pd.read_csv(f, parse_dates=["time_utc"]).rename(columns={"time_utc": "time"})
           for f in files]
    m1 = pd.concat(dfs, ignore_index=True).sort_values("time")
    if m1.time.dt.tz is None:
        m1.time = m1.time.dt.tz_localize("UTC")
    m1.time = m1.time.dt.tz_convert(BERLIN).dt.tz_localize(None)
    return m1

def fvg_ok(prev: pd.Series, cur: pd.Series, side: str) -> bool:
    if not REQUIRE_FVG:
        return True
    return (cur.low > prev.high) if side == "BUY" else (prev.low > cur.high)

def last_opposite(df15: pd.DataFrame, idx: int, side: str) -> pd.Series:
    mask = (df15.close < df15.open) if side == "BUY" else (df15.close > df15.open)
    sub  = df15.loc[: idx-1][mask]
    return sub.iloc[-1] if not sub.empty else df15.iloc[idx-1]

# ── Core scan ─────────────────────────────────────────────────────────────
def scan(m1: pd.DataFrame) -> pd.DataFrame:
    m15 = (m1.set_index("time")
              .resample("15min")
              .agg({"open":"first", "high":"max", "low":"min", "close":"last"})
              .dropna()
              .reset_index())

    signals   : list[list] = []
    dbg_left  = DEBUG_DAYS
    boxes     = pierces = 0

    for date, day15 in m15.groupby(m15.time.dt.date):

        # ---- 1. Determine the Asia window for this trading day -------------
        d0 = pd.Timestamp(date)

        if TOKYO_ONLY:
            win_start = d0 + pd.Timedelta(hours=1)   # 00‑06 UTC → 01‑07 Berlin
            win_end   = d0 + pd.Timedelta(hours=7)
        else:
            win_start = d0 - pd.Timedelta("1D") + START_TD
            win_end   = d0 + END_TD

        asia = m1[(m1.time >= win_start) & (m1.time < win_end)]
        if asia.empty:
            continue

        boxes += 1
        hi, lo = asia.high.max(), asia.low.min()
        pierced_today = False

        if dbg_left:
            print(f"{date}  Asia hi={hi:.2f}  lo={lo:.2f}  bars={len(asia)}")

        # ---- 2. Walk forward through the day’s 15‑minute bars --------------
        for i in range(1, len(day15)):
            p, c = day15.iloc[i-1], day15.iloc[i]

            # -------- BUY scenario ------------------------------------------
            buy_pierce = c.low < lo - BUFFER_USD
            if (buy_pierce or not REQUIRE_PIERCE) and fvg_ok(p, c, "BUY"):
                ob = last_opposite(day15, i, "BUY")
                lo_, hi_ = ob.low, min(ob.high, ob.low + ZONE_W)
                if hi_ > lo_:
                    signals.append([ob.time, "BUY", lo_, hi_])
                    pierced_today |= buy_pierce
                    break

            # -------- SELL scenario -----------------------------------------
            sell_pierce = c.high > hi + BUFFER_USD
            if (sell_pierce or not REQUIRE_PIERCE) and fvg_ok(p, c, "SELL"):
                ob = last_opposite(day15, i, "SELL")
                hi_, lo_ = ob.high, max(ob.low, ob.high - ZONE_W)
                if hi_ > lo_:
                    signals.append([ob.time, "SELL", lo_, hi_])
                    pierced_today |= sell_pierce
                    break

        if pierced_today:
            pierces += 1
        if dbg_left:
            print(f"    pierced={pierced_today}")
            dbg_left -= 1

    print(f"Days with Asia box: {boxes},  with pierce: {pierces}")
    return pd.DataFrame(signals,
                        columns=["datetime", "direction", "entry_low", "entry_high"])

# ── CLI wrapper ───────────────────────────────────────────────────────────
def main() -> None:
    warnings.filterwarnings("ignore")
    print("Loading M1 data …")
    m1 = load_m1()
    print(f"M1 rows: {len(m1):,}")

    print("Scanning for sweep‑OB zones …")
    zones = scan(m1)

    if zones.empty:
        print("⚠️  No zones found with current config.")
        return

    zones["zone_width"] = zones.entry_high - zones.entry_low
    zones["entry_mid"]  = (zones.entry_low + zones.entry_high) / 2
    zones.sort_values("datetime").to_csv("signals_long.csv", index=False)
    print(f"Generated {len(zones)} signals → signals_long.csv")

if __name__ == "__main__":
    main()
