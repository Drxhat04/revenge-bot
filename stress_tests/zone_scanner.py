#!/usr/bin/env python
"""
Enhanced XAUUSD OB / Liquidity-Sweep Scanner (v3.1.3, UTC-native, no look-ahead)
------------------------------------------------------------------------------
• All timestamps in UTC
• Confirms pierce/FVG on 15m bar 'c' and timestamps signal at next 15m open
• Strict OB option: displacement-validated order block selection
• Dynamic sweep buffer: USD / pct of session range / ATR / auto
• Optional M1 micro BOS confirmation into the band
• Optional H1 EMA bias gate
• Skips Saturday / Sunday data
• Allows multiple signals per day and both sides per day via config:
    - max_signals_per_day: int (default 3, clamped to >=1)
    - both_sides_per_day:  bool (default True)
• NEW: dedup at emit-time to prevent duplicates across overlapping sessions
"""

from __future__ import annotations
import pathlib, warnings, yaml, pandas as pd
import numpy as np
import datetime as _dt

# ── Config ───────────────────────────────────────────────────────────────
CFG                 = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
ZONE_W              = float(CFG["zone_width_usd"])
BUFFER_USD          = float(CFG.get("sweep_buffer_usd", 0.05))
ASIA_START          = CFG.get("asia_start", "18:00")
ASIA_END            = CFG.get("asia_end",   "09:00")
TOKYO_ONLY          = bool(CFG.get("tokyo_only", False))
REQUIRE_PIERCE      = bool(CFG.get("require_pierce", True))
REQUIRE_FVG         = bool(CFG.get("require_fvg",    True))
DEBUG_DAYS          = int(CFG.get("debug_days",     0))
GAP_FILTER          = float(CFG.get("gap_filter", 0.005))
MAX_PER_DAY         = max(1, int(CFG.get("max_signals_per_day", 3)))
ALLOW_BOTH_SIDES    = bool(CFG.get("both_sides_per_day", True))

# Dynamic buffer
BUFFER_MODE         = str(CFG.get("buffer_mode", "usd"))        # "usd" | "pct" | "atr" | "auto"
BUFFER_PCT          = float(CFG.get("buffer_pct", 0.015))
BUFFER_ATR_MULT     = float(CFG.get("buffer_atr_mult", 0.35))
MIN_BUFFER_USD      = float(CFG.get("min_buffer_usd", 0.05))
MAX_BUFFER_USD      = float(CFG.get("max_buffer_usd", 0.60))

# Strict OB selection
STRICT_OB           = bool(CFG.get("strict_ob", True))
OB_LOOKBACK_BARS    = int(CFG.get("ob_lookback_bars", 20))
DISP_MULT_ATR       = float(CFG.get("disp_mult_atr", 1.2))

# Microstructure confirmation (M1)
USE_M1_BOS          = bool(CFG.get("use_m1_bos", False))
M1_BOS_LOOKBACK_MIN = int(CFG.get("m1_bos_lookback_min", 10))

# Optional H1 bias
USE_H1_BIAS         = bool(CFG.get("use_h1_bias", False))
H1_EMA_LEN          = int(CFG.get("h1_ema_len", 50))

# ── Helpers ───────────────────────────────────────────────────────────────
def _td(val) -> pd.Timedelta:
    if isinstance(val, pd.Timedelta): return val
    if isinstance(val, _dt.timedelta): return pd.to_timedelta(val)
    if isinstance(val, _dt.time): return pd.to_timedelta(f"{val.hour:02d}:{val.minute:02d}:{val.second:02d}")
    if isinstance(val, (int, float)): return pd.to_timedelta(f"{int(val):02d}:00:00")
    s = str(val).strip(); parts = s.split(":")
    if len(parts) == 1: s = f"{parts[0]}:00:00"
    elif len(parts) == 2: s = f"{parts[0]}:{parts[1]}:00"
    return pd.to_timedelta(s)

def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

START_TD, END_TD = _td(ASIA_START), _td(ASIA_END)

def load_m1(folder: str = "data") -> pd.DataFrame:
    files = sorted(pathlib.Path(folder).glob("XAUUSD_M1_*.csv"))
    if not files:
        raise FileNotFoundError("No XAUUSD_M1_*.csv in /data")
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["time_utc"]).rename(columns={"time_utc":"time"})
        if df.time.dt.tz is None:
            df.time = df.time.dt.tz_localize("UTC")
        dfs.append(df)
    m1 = pd.concat(dfs, ignore_index=True).sort_values("time")
    before = len(m1)
    m1 = m1.drop_duplicates(subset="time", keep="last")
    if before - len(m1):
        print(f"Deduplicated {before - len(m1):,} overlapping M1 rows")
    m1 = m1[~m1.time.dt.weekday.isin([5,6])].reset_index(drop=True)
    m1["prev_close"] = m1["close"].shift(1)
    m1["tr"] = np.maximum.reduce([
        m1["high"] - m1["low"],
        (m1["high"] - m1["prev_close"]).abs(),
        (m1["low"]  - m1["prev_close"]).abs()
    ])
    return m1

def fvg_ok(prev: pd.Series, cur: pd.Series, side: str) -> bool:
    if not REQUIRE_FVG: return True
    return (cur.low > prev.high) if side == "BUY" else (prev.low > cur.high)

def last_opposite(df15: pd.DataFrame, idx: int, side: str) -> pd.Series:
    mask = (df15.close < df15.open) if side == "BUY" else (df15.close > df15.open)
    sub  = df15.loc[: idx - 1][mask]
    return sub.iloc[-1] if not sub.empty else df15.iloc[idx - 1]

def _session_windows_for(date_utc: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    boxes = CFG.get("session_boxes", [])
    out = []
    for b in boxes:
        s = _td(b["start"]); e = _td(b["end"])
        start = (date_utc - pd.Timedelta(days=1) + s) if s > e else (date_utc + s)
        end   = date_utc + e
        out.append((b.get("name","BOX"), start, end))
    if not out:
        start = date_utc - pd.Timedelta(days=1) + START_TD
        end   = date_utc + END_TD
        out = [("ASIA", start, end)]
    return out

def _dyn_buffer(sess_df_m1: pd.DataFrame) -> float:
    rng = float(sess_df_m1.high.max() - sess_df_m1.low.min()) if not sess_df_m1.empty else 0.0
    tr15 = (sess_df_m1.set_index("time")["tr"].resample("15min").mean())
    atr15_series = tr15.rolling(14).mean()
    atr15 = float(atr15_series.iloc[-1]) if len(atr15_series) and not np.isnan(atr15_series.iloc[-1]) else 0.0
    if BUFFER_MODE == "usd": buf = BUFFER_USD
    elif BUFFER_MODE == "pct": buf = BUFFER_PCT * rng
    elif BUFFER_MODE == "atr": buf = BUFFER_ATR_MULT * atr15
    else: buf = max(BUFFER_USD, BUFFER_PCT * rng, BUFFER_ATR_MULT * atr15)  # auto
    return float(max(MIN_BUFFER_USD, min(buf, MAX_BUFFER_USD)))

def _strict_ob(df15: pd.DataFrame, idx: int, side: str) -> pd.Series:
    if not STRICT_OB:
        return last_opposite(df15, idx, side)
    look = OB_LOOKBACK_BARS
    start = max(0, idx - look)
    seg = df15.iloc[start:idx+1]
    if seg.empty or len(seg) < 3:
        return last_opposite(df15, idx, side)
    tr_df = pd.concat([
        (seg["high"] - seg["low"]),
        (seg["high"] - seg["close"].shift(1)).abs(),
        (seg["low"]  - seg["close"].shift(1)).abs()
    ], axis=1)
    tr = tr_df.max(axis=1)
    atr15 = tr.rolling(14, min_periods=1).mean()
    disp_idx = None
    for j in range(len(seg)-2, 0, -1):
        bar = seg.iloc[j]
        body = float(abs(bar.close - bar.open))
        atrv = float(atr15.iloc[j])
        if atrv <= 0 or np.isnan(atrv): continue
        if side == "BUY" and bar.close > bar.open and body >= DISP_MULT_ATR * atrv: disp_idx = j; break
        if side == "SELL" and bar.close < bar.open and body >= DISP_MULT_ATR * atrv: disp_idx = j; break
    if disp_idx is None or disp_idx == 0:
        return last_opposite(df15, idx, side)
    opp_mask = (seg.close < seg.open) if side == "BUY" else (seg.close > seg.open)
    opp = seg.iloc[:disp_idx][opp_mask]
    return opp.iloc[-1] if not opp.empty else last_opposite(df15, idx, side)

def _m1_bos_ok(m1: pd.DataFrame, signal_time: pd.Timestamp, side: str) -> bool:
    if not USE_M1_BOS: return True
    start = signal_time - pd.Timedelta(minutes=M1_BOS_LOOKBACK_MIN)
    win = m1[(m1.time >= start) & (m1.time <= signal_time)]
    if win.empty: return True
    if side == "BUY":
        return bool(win.high.max() > (win.high.iloc[-2] if len(win) >= 2 else win.high.iloc[-1]))
    else:
        return bool(win.low.min() < (win.low.iloc[-2] if len(win) >= 2 else win.low.iloc[-1]))

def _h1_bias_allows(m1: pd.DataFrame, confirm_time: pd.Timestamp, side: str) -> bool:
    if not USE_H1_BIAS: return True
    h1 = (m1.set_index("time").resample("1H")
            .agg({"open":"first","high":"max","low":"min","close":"last"}).dropna())
    if h1.empty or len(h1) < H1_EMA_LEN + 1: return True
    ema = h1["close"].ewm(span=H1_EMA_LEN, adjust=False).mean()
    try:
        idxer = h1.index.get_indexer([confirm_time], method="pad")
        if len(idxer) == 0 or idxer[0] == -1: return True
        bar_idx = idxer[0]
    except Exception:
        return True
    price = h1["close"].iloc[bar_idx]
    return bool((price > ema.iloc[bar_idx]) if side == "BUY" else (price < ema.iloc[bar_idx]))

def _sig_key(ts: pd.Timestamp, direction: str, lo: float, hi: float) -> tuple:
    """Key used to dedup across overlapping sessions; round to cents."""
    return (pd.Timestamp(ts).tz_convert("UTC"), direction, round(lo, 2), round(hi, 2))

# ── Core scan ─────────────────────────────────────────────────────────────
def scan(m1: pd.DataFrame) -> pd.DataFrame:
    m15 = (m1.set_index("time")
              .resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"})
              .dropna().reset_index())

    signals = []
    emitted = set()  # NEW: prevent duplicates across sessions
    dbg_left = DEBUG_DAYS

    for date, day15 in m15.groupby(m15.time.dt.date):
        if pd.Timestamp(date, tz="UTC").weekday() >= 5:  # skip Sat/Sun
            continue

        d0 = pd.Timestamp(date, tz="UTC")
        sessions = _session_windows_for(d0)

        made_today = 0
        did_buy = False
        did_sell = False

        for sess_name, win_start, win_end in sessions:
            if made_today >= MAX_PER_DAY:
                break

            sess = m1[(m1.time >= win_start) & (m1.time < win_end)]
            if sess.empty:
                continue

            hi, lo = float(sess.high.max()), float(sess.low.min())
            BUF = _dyn_buffer(sess)

            if dbg_left:
                print(f"{date} [{sess_name}] hi={hi:.2f} lo={lo:.2f} buf={BUF:.2f} rows={len(sess)}")

            for i in range(1, len(day15)):
                if made_today >= MAX_PER_DAY:
                    break

                p, c = day15.iloc[i - 1], day15.iloc[i]

                denom = p.close if p.close != 0 else 1.0
                if abs(c.open - p.close) / abs(denom) > GAP_FILTER:
                    if dbg_left: print("    gap filter hit")
                    continue

                # BUY
                buy_pierce = bool(c.low < lo - BUF)
                if ((not did_buy) or ALLOW_BOTH_SIDES) and ((buy_pierce or not REQUIRE_PIERCE) and fvg_ok(p, c, "BUY")):
                    ob = _strict_ob(day15, i, "BUY") if STRICT_OB else last_opposite(day15, i, "BUY")
                    ob_lo, ob_hi = float(ob.low), float(min(ob.high, ob.low + ZONE_W))
                    if ob_hi > ob_lo:
                        confirm_time = _utc(c.time) + pd.Timedelta(minutes=15)
                        k = _sig_key(confirm_time, "BUY", ob_lo, ob_hi)
                        if k not in emitted and _m1_bos_ok(m1, confirm_time, "BUY") and _h1_bias_allows(m1, confirm_time, "BUY"):
                            signals.append([confirm_time, "BUY", ob_lo, ob_hi, np.nan, ob.time, c.time])
                            emitted.add(k)
                            did_buy = True
                            made_today += 1
                            # allow SELL in same bar if ALLOW_BOTH_SIDES

                # SELL
                sell_pierce = bool(c.high > hi + BUF)
                if ((not did_sell) or ALLOW_BOTH_SIDES) and ((sell_pierce or not REQUIRE_PIERCE) and fvg_ok(p, c, "SELL")):
                    ob = _strict_ob(day15, i, "SELL") if STRICT_OB else last_opposite(day15, i, "SELL")
                    ob_hi, ob_lo = float(ob.high), float(max(ob.low, ob.high - ZONE_W))
                    if ob_hi > ob_lo:
                        confirm_time = _utc(c.time) + pd.Timedelta(minutes=15)
                        k = _sig_key(confirm_time, "SELL", ob_lo, ob_hi)
                        if k not in emitted and _m1_bos_ok(m1, confirm_time, "SELL") and _h1_bias_allows(m1, confirm_time, "SELL"):
                            signals.append([confirm_time, "SELL", ob_lo, ob_hi, np.nan, ob.time, c.time])
                            emitted.add(k)
                            did_sell = True
                            made_today += 1

        if dbg_left:
            print(f"    made_today={made_today} did_buy={did_buy} did_sell={did_sell}")
            dbg_left -= 1

    zones = pd.DataFrame(
        signals,
        columns=["datetime","direction","entry_low","entry_high","atr","ob_time","confirm_time"],
    )
    if not zones.empty:
        zones["datetime"] = pd.to_datetime(zones["datetime"], utc=True)
        # Final belt-and-suspenders: drop any exact dup rows
        zones = zones.drop_duplicates(subset=["datetime","direction","entry_low","entry_high"], keep="first")
    return zones

# ── CLI wrapper ───────────────────────────────────────────────────────────
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

    # Post-scan minute gap filter
    valid = []
    m1i = m1.set_index("time")

    def _match_bar(ts: pd.Timestamp) -> pd.DataFrame:
        try:
            exact = m1i.loc[ts:ts]
            if not exact.empty: return exact.iloc[0:1]
        except Exception:
            pass
        win = m1i.loc[ts - pd.Timedelta(seconds=60): ts + pd.Timedelta(seconds=60)]
        if win.empty: return win
        try:
            pos = win.index.get_indexer([ts], method="nearest")[0]
        except Exception:
            delta_ns = (win.index - ts).asi8
            pos = int(np.argmin(np.abs(delta_ns)))
        return win.iloc[[pos]]

    for _, row in zones.iterrows():
        bar = _match_bar(row.datetime)
        if bar.empty:
            valid.append(row); continue
        prev_idx = m1i.index.get_indexer([bar.index[0]], method="pad")[0] - 1
        if prev_idx >= 0:
            prev_close = float(m1i.iloc[prev_idx].close)
            denom = prev_close if prev_close else 1.0
            gap_ok = abs(float(bar["open"].values[0]) - prev_close) / abs(denom) <= GAP_FILTER
            if gap_ok: valid.append(row)
        else:
            valid.append(row)

    zones = pd.DataFrame(valid).drop_duplicates(subset=["datetime","direction","entry_low","entry_high"], keep="first")
    zones = zones.sort_values("datetime")
    zones.to_csv("signals_long_utc.csv", index=False)
    print(f"Generated {len(zones)} signals → signals_long_utc.csv")

if __name__ == "__main__":
    main()
