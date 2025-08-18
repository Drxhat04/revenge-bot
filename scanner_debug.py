"""
scanner_debug.py  –  XAUUSD M1 → signals_long.csv (+diagnostics)
-----------------------------------------------------------------
Adds per‑day fields:
    • asia_hi / asia_lo          – daily Asia range
    • pierce_hi / pierce_lo      – did price break either side ≥ buffer?
    • fvg_pass                   – bar satisfied FVG rule?
"""

import pathlib, yaml, pandas as pd, warnings

CFG = yaml.safe_load(open("config.yaml"))
ZONE_W      = CFG["zone_width_usd"]
BUF_USD     = CFG.get("sweep_buffer_usd", 0.05)
ASIA_START  = CFG.get("asia_start", "18:00")
ASIA_END    = CFG.get("asia_end",   "09:00")
REQ_FVG     = CFG.get("require_fvg", True)
BERLIN      = "Europe/Berlin"

# ── helpers ────────────────────────────────────────────────────────────────
def _td(hms: str) -> pd.Timedelta:
    """Return pd.Timedelta for 'HH', 'HH:MM' or 'HH:MM:SS'."""
    parts = hms.split(":")
    if len(parts) == 1:                 # "18"
        hms = f"{parts[0]}:00:00"
    elif len(parts) == 2:               # "18:30"
        hms = f"{parts[0]}:{parts[1]}:00"
    return pd.to_timedelta(hms)

START_TD = _td(ASIA_START)
END_TD   = _td(ASIA_END)

def load_m1(folder="data") -> pd.DataFrame:
    dfs = [pd.read_csv(f, parse_dates=["time_utc"]).rename(columns={"time_utc": "time"})
           for f in sorted(pathlib.Path(folder).glob("XAUUSD_M1_*.csv"))]
    m1  = pd.concat(dfs, ignore_index=True).sort_values("time")
    if m1.time.dt.tz is None:
        m1.time = m1.time.dt.tz_localize("UTC")
    m1.time = m1.time.dt.tz_convert(BERLIN).dt.tz_localize(None)
    return m1

def fvg(prev, cur, side):                # fair‑value‑gap rule
    if not REQ_FVG:
        return True
    return cur.low > prev.high if side == "BUY" else prev.low > cur.high

def last_OB(df, i, side):                # last opposite‑colour M15 candle
    mask = (df.close < df.open) if side == "BUY" else (df.close > df.open)
    sub  = df.loc[: i-1][mask]
    return sub.iloc[-1] if not sub.empty else df.iloc[i-1]

# ── main scan ──────────────────────────────────────────────────────────────
def scan(m1: pd.DataFrame) -> pd.DataFrame:
    m15 = (m1.set_index("time").resample("15min")
           .agg({"open":"first","high":"max","low":"min","close":"last"})
           .dropna().reset_index())

    out = []
    for date, m15day in m15.groupby(m15.time.dt.date):
        base = pd.Timestamp(date)
        win  = (base - pd.Timedelta("1D") + START_TD,
                base + END_TD)

        asia = m1[(m1.time >= win[0]) & (m1.time < win[1])]
        if asia.empty:
            out.append({"date": date, "reason": "no_asia_data"})
            continue

        hi, lo = asia.high.max(), asia.low.min()
        pierced = {"hi": False, "lo": False}

        for i in range(1, len(m15day)):
            p, c = m15day.iloc[i-1], m15day.iloc[i]

            # ---- BUY sweep ------------------------------------------------
            if c.low < lo - BUF_USD:
                pierced["lo"] = True
                if fvg(p, c, "BUY"):
                    ob = last_OB(m15day, i, "BUY")
                    lo_, hi_ = ob.low, min(ob.high, ob.low + ZONE_W)
                    if hi_ > lo_:
                        out.append(dict(date=date, direction="BUY",
                                        datetime=ob.time, entry_low=lo_,
                                        entry_high=hi_, zone_width=hi_-lo_,
                                        asia_hi=hi, asia_lo=lo,
                                        pierce_lo=True, fvg_pass=True))
                    break

            # ---- SELL sweep ----------------------------------------------
            if c.high > hi + BUF_USD:
                pierced["hi"] = True
                if fvg(p, c, "SELL"):
                    ob = last_OB(m15day, i, "SELL")
                    hi_, lo_ = ob.high, max(ob.low, ob.high - ZONE_W)
                    if hi_ > lo_:
                        out.append(dict(date=date, direction="SELL",
                                        datetime=ob.time, entry_low=lo_,
                                        entry_high=hi_, zone_width=hi_-lo_,
                                        asia_hi=hi, asia_lo=lo,
                                        pierce_hi=True, fvg_pass=True))
                    break
        else:
            out.append(dict(date=date, asia_hi=hi, asia_lo=lo,
                            pierce_hi=pierced["hi"], pierce_lo=pierced["lo"],
                            reason="no_pierce_or_fvg"))

    df = pd.DataFrame(out)
    sigs = df.dropna(subset=["datetime"])
    if not sigs.empty:
        sigs["entry_mid"] = (sigs.entry_low + sigs.entry_high) / 2
        sigs[["datetime","direction","entry_low","entry_high",
               "zone_width","entry_mid"]].to_csv("signals_long.csv", index=False)
        print(f"signals_long.csv written  ({len(sigs)} signals)")
    df.to_csv("scan_debug_daily.csv", index=False)
    print("Per‑day diagnostics → scan_debug_daily.csv")
    return df

# run
warnings.filterwarnings("ignore")
scan(load_m1())
