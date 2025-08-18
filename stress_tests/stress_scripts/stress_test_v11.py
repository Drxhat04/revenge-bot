#!/usr/bin/env python
"""
──────────────────────────────────────────────────────────────────────────────
stress_test_v11.py · “Reality‑Biter” edition  —  real‑engine port
2025‑07‑30  • maintenance patch 2
    • CLI flags for dynamic‑stops (‑‑dyn_stops) & TP‑before‑SL (‑‑tp_first)
    • Grid sweep no longer includes the zero‑friction row
    • Monte‑Carlo spread‑uplift floored at 1 %
    • No other logic changes — still a drop‑in replacement for v11
Data & signal files:  ./data  +  signals_long.csv
"""
from __future__ import annotations

import argparse, itertools, gc, pathlib, warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_engine import run_backtest      # ← live trade simulator (v6)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ╭─ GLOBAL CONSTANTS ───────────────────────────────────────────────────╮
N_RANDOM_TRIALS             = 500
SNAPSHOT_EQUITY_FLOOR_RATIO = 0.30
SNAPSHOT_DIR                = pathlib.Path("stress_deaths_v11")
SNAPSHOT_DIR.mkdir(exist_ok=True, parents=True)

# --- stress parameter grids -------------------------------------------
SPREAD_UPLIFT_SET       = (0.00, 0.03, 0.10)
BURST_SPREAD_MULT_SET   = (1.0, 1.5, 2.0)
LAT_RANGE_MINUTES_SET   = (
    (0, 0),     # ideal routing
    (0, 2),     # routine congestion
    (2, 5),     # worst acceptable (<14 min veto)
)
SKIP_PROB_SET           = (0.00, 0.02, 0.05)
FLASH_GAP_SET           = (0.00, 0.005, 0.015)
STOPRUN_ATR_SET         = (0.00, 0.15, 0.30)
ATR_MULT_SET            = (1.5, 2.0, 2.5)

# latency / slip veto thresholds ---------------------------------------
LATENCY_VETO_SEC = 14 * 60     # veto if lag ≥14 min
MAX_SLIP_PIPS    = 10

# partial‑fill parameters ----------------------------------------------
PARTIAL_FILL_PROB  = 0.50
PARTIAL_MIN_RATIO  = 0.10
PARTIAL_MAX_RATIO  = 0.80

rng = np.random.default_rng(42)

# ╭────────────────────────── I/O HELPERS ──────────────────────────────╮
def load_signals() -> pd.DataFrame:
    df = pd.read_csv("signals_long.csv", parse_dates=["datetime"])
    df["entry_mid"]  = (df.entry_low + df.entry_high) / 2
    df["zone_width"] = (df.entry_high - df.entry_low).abs()
    return df.sort_values("datetime", ignore_index=True)

def load_bars() -> pd.DataFrame:
    files = sorted(pathlib.Path("data").glob("XAUUSD_M1_*.csv"))
    if not files:
        raise FileNotFoundError("No XAUUSD_M1_*.csv files found in ./data")
    dfs = [pd.read_csv(f, parse_dates=["time_utc"]).rename(columns={"time_utc": "time"})
           for f in files]
    bars = (pd.concat(dfs, ignore_index=True)
              .sort_values("time", ignore_index=True))
    if bars.time.dt.tz is None:
        bars.time = bars.time.dt.tz_localize("UTC")
    bars.time = bars.time.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return bars

# ╭────────────────────── STRESS MUTATORS ──────────────────────────────╮
def inject_flat_gap(b: pd.DataFrame, pct: float) -> None:
    if pct == 0: return
    monday_open = ((b.time.dt.weekday == 0) &
                   (b.time.dt.hour == 0) &
                   (b.time.dt.minute == 0))
    b.loc[monday_open, "open"] *= 1 + rng.choice([-pct, pct])

def inject_burst_spread(b: pd.DataFrame, mult: float, prob: float = 0.02) -> None:
    if mult <= 1 or prob == 0: return
    picks = b.sample(frac=prob, random_state=rng.integers(1<<32)).index
    b.loc[picks, "spread"] *= mult

def inject_stoprun_wick(b: pd.DataFrame, atr: pd.Series,
                        mult: float, prob: float = 0.15) -> None:
    if mult == 0 or prob == 0: return
    picks = b.sample(frac=prob, random_state=rng.integers(1<<32)).index
    for i in picks:
        spike = mult * atr.iloc[i]
        if rng.random() < 0.5:
            b.at[i, "low"]  -= spike
        else:
            b.at[i, "high"] += spike

def compute_true_atr(b: pd.DataFrame, window: int = 90 * 1440) -> pd.Series:
    hl = b.high - b.low
    hc = (b.high - b.close.shift()).abs()
    lc = (b.low  - b.close.shift()).abs()
    tr = np.maximum.reduce([hl.values, hc.values, lc.values])
    return (pd.Series(tr, index=b.index)
              .rolling(window, min_periods=window)
              .mean())

def apply_latency(s: pd.DataFrame, b: pd.DataFrame,
                  rng_: Tuple[int, int]) -> pd.DataFrame:
    lo, hi = rng_
    if lo == hi == 0: return s
    out = s.copy()
    out["datetime_orig"]  = out.datetime
    out["entry_mid_orig"] = out.entry_mid
    for i, row in out.iterrows():
        lag = rng.integers(lo, hi + 1)
        ts  = row.datetime + pd.Timedelta(minutes=int(lag))
        bar = b[b.time >= ts].iloc[0]
        out.at[i, "datetime"]  = bar.time
        out.at[i, "entry_mid"] = (bar.open + bar.close) / 2
    return out

# ╭──────────────────── PARTIAL‑FILL HELPER ────────────────────────────╮
def apply_partial_fill(bt: pd.DataFrame, spr_uplift: float) -> pd.DataFrame:
    if spr_uplift < 0.15: return bt
    factor = rng.uniform(PARTIAL_MIN_RATIO, 1.0)
    bt = bt.copy()
    bt["pnl"] *= factor
    bt["balance_after"] = bt["balance_before"] + bt["pnl"].cumsum()
    return bt

# ╭──────────────────────────── SINGLE TRIAL ───────────────────────────╮
def run_single_trial(sig0: pd.DataFrame, bars0: pd.DataFrame,
                     cfg0: dict, tid: str,
                     params: Dict[str, float]) -> Dict[str, float] | None:

    spr_uplift, burst_mult, lat_rng, skip_p, gap_pct, wick_mult, atr_mult = (
        params["spread_uplift"], params["burst_mult"], params["lat_range"],
        params["skip_prob"],     params["flash_gap_pct"], params["stoprun_mult"],
        params["atr_mult"],
    )

    sigs = sig0.copy()
    bars = bars0.copy(deep=False)
    cfg  = cfg0.copy()
    cfg["atr_multiplier"] = atr_mult

    # spread uplift & burst
    if "spread" not in bars.columns:
        bars["spread"] = 8 * cfg["tick_size"]
    bars["spread"] *= 1 + spr_uplift
    inject_burst_spread(bars, burst_mult)
    cfg["slippage_usd_side_lot"] *= 1 + spr_uplift

    # gaps & stop‑runs
    inject_flat_gap(bars, gap_pct)
    true_atr = compute_true_atr(bars).bfill()
    inject_stoprun_wick(bars, true_atr, wick_mult)

    # shuffle → skip → latency
    sigs = sigs.sample(frac=1.0, random_state=rng.integers(1<<32)).reset_index(drop=True)
    sigs = sigs[rng.random(len(sigs)) > skip_p].reset_index(drop=True)
    sigs = apply_latency(sigs, bars, lat_rng)

    # veto on lag/slip
    lag_sec   = (sigs.datetime - sigs.get("datetime_orig", sigs.datetime)).dt.total_seconds().abs()
    slip_pips = (sigs.entry_mid - sigs.get("entry_mid_orig", sigs.entry_mid)).abs() / cfg["tick_size"]
    sigs = sigs[(lag_sec <= LATENCY_VETO_SEC) & (slip_pips <= MAX_SLIP_PIPS)].reset_index(drop=True)
    if sigs.empty: return None

    # rolling 3‑month windows
    rows, equity = [], cfg["start_balance"]
    start, end = sigs.datetime.min().to_period("M").to_timestamp(), sigs.datetime.max()

    while start < end and equity > 0:
        end_slice = start + pd.DateOffset(months=3)
        slc = sigs[(sigs.datetime >= start) & (sigs.datetime < end_slice)]
        start = end_slice
        if slc.empty: continue

        bt = run_backtest(slc, bars, cfg, start_balance=equity, trial_no=tid)
        if bt.empty: break

        bt = apply_partial_fill(bt, spr_uplift)
        equity = bt.balance_after.iloc[-1]
        rows.append(bt)

        if equity < cfg["start_balance"] * SNAPSHOT_EQUITY_FLOOR_RATIO:
            bt.tail(100).to_parquet(SNAPSHOT_DIR / f"{tid}_snapshot.parquet")
            break

    if not rows: return None

    full = pd.concat(rows, ignore_index=True)
    max_dd = ((full.balance_after.cummax() - full.balance_after) /
              full.balance_after.cummax()).max() * 100
    return {**params, "trial_id": tid,
            "equity_final": full.balance_after.iloc[-1],
            "max_dd_pct": max_dd, "status": "OK"}

# ╭────────────────────────────── MAIN ────────────────────────────────╮
def main(grid_mode: bool, only_lat0: bool, risk_override: float | None,
         equity_brake: bool, dyn_stops: bool, tp_first: bool) -> None:

    sig_base, bars_base = load_signals(), load_bars()

    # --- baseline cfg ------------------------------------------------------
    cfg = dict(
        risk_mult = 2.3, tp1_mult = 0.68, tp2_mult = 1.16,
        tick_size = 0.01, slippage_usd_side_lot = 3.52,
        dollar_per_unit_per_lot = 100.0, risk_per_trade = 0.02,
        max_risk_ratio = 0.03, min_stop_usd = 1.0, min_lot_size = 0.01,
        lot_cap = 10.0, start_balance = 10_000.0,
        dynamic_stops = dyn_stops, equity_brake = equity_brake,
        no_touch_timeout_hours = 3, force_close_hours = 5,
        tp_before_sl_same_bar = tp_first, atr_multiplier = 2.0,
    )
    if risk_override is not None:
        cfg["risk_mult"] = float(risk_override)

    lat_set = ((0, 0),) if only_lat0 else LAT_RANGE_MINUTES_SET
    trials: List[Dict[str, float]] = []

    # --- GRID or MC SWEEP --------------------------------------------------
    if grid_mode:
        grid = itertools.product(SPREAD_UPLIFT_SET[1:], BURST_SPREAD_MULT_SET,
                                 lat_set, SKIP_PROB_SET, FLASH_GAP_SET,
                                 STOPRUN_ATR_SET, ATR_MULT_SET)
        grid = list(grid)    # realise the generator for len()
        print(f"Running grid sweep: {len(grid)} scenarios …\n")
        for n, combo in enumerate(grid, 1):
            params = dict(zip(
                ("spread_uplift","burst_mult","lat_range","skip_prob",
                 "flash_gap_pct","stoprun_mult","atr_mult"), combo))
            tid = f"G{n:03d}"
            res = run_single_trial(sig_base, bars_base, cfg, tid, params)
            if res:
                trials.append(res)
                print(f"{tid} → ${res['equity_final']:>11,.0f}  DD {res['max_dd_pct']:5.1f}%")
            gc.collect()
    else:
        print(f"Running {N_RANDOM_TRIALS} Monte‑Carlo trials …\n")
        for n in range(1, N_RANDOM_TRIALS + 1):
            lat_rng = (0, 0) if only_lat0 else tuple(sorted(rng.integers(0, 20, size=2)))
            if lat_rng[0] == lat_rng[1]:
                lat_rng = (lat_rng[0], lat_rng[0] + 1)

            params = dict(
                spread_uplift = rng.uniform(0.01, 0.30),   # floor at 1 %
                burst_mult    = rng.choice(BURST_SPREAD_MULT_SET),
                lat_range     = lat_rng,
                skip_prob     = rng.uniform(0, 0.30),
                flash_gap_pct = rng.choice(FLASH_GAP_SET),
                stoprun_mult  = rng.choice(STOPRUN_ATR_SET),
                atr_mult      = rng.choice(ATR_MULT_SET),
            )
            tid = f"MC{n:03d}"
            res = run_single_trial(sig_base, bars_base, cfg, tid, params)
            if res:
                trials.append(res)
                print(f"{tid} → ${res['equity_final']:>11,.0f}  "
                      f"DD {res['max_dd_pct']:5.1f}%  lat_range={lat_rng}")
            gc.collect()

    # --- WRAP‑UP -----------------------------------------------------------
    if not trials:
        print("No successful trials – check data / parameters.")
        return

    out = pd.DataFrame(trials)
    out.to_csv("stress_results_v11.csv", index=False)

    print("\n=== V11 summary (real engine) ===")
    print(out["equity_final"].describe(percentiles=[0.5, 0.9]))
    print("\nFull details → stress_results_v11.csv")
    if any(SNAPSHOT_DIR.iterdir()):
        print(f"Snapshots of failed runs in  {SNAPSHOT_DIR}/")

# ╭──────────────────────── CLI ENTRY ─────────────────────────╮
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc",           action="store_true",
                    help="Monte‑Carlo mode (default = grid sweep)")
    ap.add_argument("--lat0",         action="store_true",
                    help="Force latency‑range (0,0) only")
    ap.add_argument("--risk",         type=float, default=None,
                    help="Override risk_mult (e.g. 1.5)")
    ap.add_argument("--equity_brake", action="store_true",
                    help="Enable soft equity brake (lot‑cap only)")
    ap.add_argument("--dyn_stops",    action="store_true",
                    help="Enable ATR‑based dynamic stops")
    ap.add_argument("--tp_first",     action="store_true",
                    help="Require TP to be hit before SL in same bar")
    args = ap.parse_args()

    main(grid_mode=not args.mc,
         only_lat0=args.lat0,
         risk_override=args.risk,
         equity_brake=args.equity_brake,
         dyn_stops=args.dyn_stops,
         tp_first=args.tp_first)
