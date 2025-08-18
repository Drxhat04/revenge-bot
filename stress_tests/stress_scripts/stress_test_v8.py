#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────
# stress_test_v8.py · “Failure‑Finder”  – *windowed* edition
#   · deterministic grid + rich logging + death snapshots
#   · NEW: 3‑month rolling windows (no overlapping replays)
# ────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse, gc, itertools, pathlib, warnings, yaml
from typing import Dict, List

import numpy as np
import pandas as pd
from backtest_engine import run_backtest, CFG as ENGINE_DEFAULTS

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ╭─ GLOBAL CONSTANTS ────────────────────────────────────────────╮
N_RANDOM_TRIALS             = 500        # when --mc flag is used
SNAPSHOT_EQUITY_FLOOR_RATIO = 0.30       # dump if equity < 30 % of start
SNAPSHOT_DIR                = pathlib.Path("stress_deaths_v8")
SNAPSHOT_DIR.mkdir(exist_ok=True)

SPREAD_UPLIFT_SET     = (0.00, 0.15, 0.30)           # 0 %, 15 %, 30 %
LAT_RANGE_MINUTES_SET = ((0, 0), (5, 7), (15, 20))   # artificial latency
SKIP_PROB_SET         = (0.00, 0.15, 0.30)
FLASH_GAP_SET         = (0.00, 0.03, 0.06)           # Monday gap ±3 %, 6 %
STOPRUN_ATR_SET       = (0.00, 0.30, 0.60)           # fake stop‑run wicks
# ╰──────────────────────────────────────────────────────────────╯

rng = np.random.default_rng(42)   # single RNG → repeatable tests

# ─────────────────────────── I/O HELPERS ───────────────────────
def load_signals() -> pd.DataFrame:
    return (pd.read_csv("signals_long.csv", parse_dates=["datetime"])
              .sort_values("datetime", ignore_index=True))

def load_bars() -> pd.DataFrame:
    files = sorted(pathlib.Path("data").glob("XAUUSD_M1_*.csv"))
    dfs   = [pd.read_csv(f, parse_dates=["time_utc"])
               .rename(columns={"time_utc": "time"}) for f in files]
    bars  = (pd.concat(dfs, ignore_index=True)
               .sort_values("time", ignore_index=True))
    if bars.time.dt.tz is None:
        bars.time = bars.time.dt.tz_localize("UTC")
    bars.time = bars.time.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return bars
# ────────────────────────────────────────────────────────────────
#   Stress primitives (unchanged) …
#   • inject_flat_gap
#   • inject_stoprun_wick
#   • compute_true_atr
#   • apply_latency
# ────────────────────────────────────────────────────────────────
def inject_flat_gap(b: pd.DataFrame, pct: float) -> None:
    if pct == 0: return
    monday_open = (b.time.dt.weekday == 0) & (b.time.dt.hour == 0) & (b.time.dt.minute == 0)
    b.loc[monday_open, "open"] *= (1 + rng.choice([-pct, pct]))

def inject_stoprun_wick(b: pd.DataFrame, atr: pd.Series,
                        mult: float, prob: float = 0.15) -> None:
    if mult == 0.0 or prob == 0.0: return
    picks = b.sample(frac=prob, random_state=rng.integers(0, 2**32 - 1)).index
    for i in picks:
        spike = mult * atr.iloc[i]
        if rng.random() < .5:  b.at[i, "low"]  -= spike
        else:                  b.at[i, "high"] += spike

def compute_true_atr(b: pd.DataFrame, window: int = 90*1440) -> pd.Series:
    hl = b.high - b.low
    hc = (b.high - b.close.shift()).abs()
    lc = (b.low  - b.close.shift()).abs()
    tr = np.maximum.reduce([hl.values, hc.values, lc.values])
    return pd.Series(tr, index=b.index).rolling(window, min_periods=window).mean()

def apply_latency(s: pd.DataFrame, b: pd.DataFrame,
                  atr: pd.Series, rng_: tuple[int,int]) -> pd.DataFrame:
    lo, hi = rng_
    if lo == hi == 0: return s
    out = s.copy()
    for i, row in out.iterrows():
        lag = rng.integers(lo, hi + 1)
        ts  = row.datetime + pd.Timedelta(minutes=lag)
        bar = b[b.time >= ts].iloc[0]
        out.at[i, "datetime"]  = bar.time
        out.at[i, "entry_mid"] = (bar.open + bar.close) / 2
    return out
# ────────────────────────────────────────────────────────────────
def run_single_trial(sig0: pd.DataFrame, bars0: pd.DataFrame,
                     cfg0: dict, tid: str,
                     params: Dict[str, float]) -> Dict[str, float] | None:

    # unpack
    spr_uplift, lat_rng, skip_p, gap_pct, wick_mult = (
        params["spread_uplift"], params["lat_range"], params["skip_prob"],
        params["flash_gap_pct"], params["stoprun_mult"]
    )

    sigs = sig0.copy()
    bars = bars0.copy(deep=False)
    cfg  = cfg0.copy()

    # ensure spread column
    if "spread" not in bars.columns:
        bars["spread"] = 8 * cfg["tick_size"]
    # apply stresses
    bars["spread"] *= (1 + spr_uplift)
    cfg["slippage_usd_side_lot"] *= (1 + spr_uplift)
    inject_flat_gap(bars, gap_pct)

    atr = compute_true_atr(bars).bfill()
    inject_stoprun_wick(bars, atr, wick_mult)

    sigs = sigs.sample(frac=1.0, random_state=rng.integers(0, 2**32-1)).reset_index(drop=True)
    sigs = sigs[rng.random(len(sigs)) > skip_p].reset_index(drop=True)
    sigs = apply_latency(sigs, bars, atr, lat_rng)

    # ── 3‑month *window* loop (no overlap) ──────────────────────
    rows, equity = [], cfg["start_balance"]
    start = sigs.datetime.min().to_period("M").to_timestamp()
    end   = sigs.datetime.max()

    while start < end and equity > 0:
        end_slice  = start + pd.DateOffset(months=3)
        slice_sigs = sigs[(sigs.datetime >= start) & (sigs.datetime < end_slice)]
        if slice_sigs.empty:
            start = end_slice;  continue

        bt = run_backtest(slice_sigs, bars, cfg,
                          start_balance=equity,
                          trial_no=tid)
        if bt.empty:
            break

        rows.append(bt)
        equity = bt.balance_after.iloc[-1]

        # margin call / death snapshot
        if equity < cfg["start_balance"] * SNAPSHOT_EQUITY_FLOOR_RATIO:
            bt.tail(100).to_parquet(SNAPSHOT_DIR / f"{tid}_snapshot.parquet")
            break

        start = end_slice   # advance by exactly 3 months

    if not rows:  return None

    full = pd.concat(rows, ignore_index=True)
    max_dd = ((full.balance_after.cummax() - full.balance_after)
              / full.balance_after.cummax()).max() * 100

    return {
        **params,
        "trial_id"     : tid,
        "equity_final" : full.balance_after.iloc[-1],
        "max_dd_pct"   : max_dd,
        "win_rate"     : (full.pnl > 0).mean(),
        "profit_factor": full.pnl[full.pnl > 0].sum() /
                         max(1, abs(full.pnl[full.pnl < 0].sum())),
        "status"       : "OK",
    }
# ───────────────────────────── MAIN ────────────────────────────
def main(grid_mode: bool = True) -> None:
    sig_base  = load_signals()
    bars_base = load_bars()

    usr = yaml.safe_load(open("config.yaml"))
    cfg = ENGINE_DEFAULTS.copy()
    cfg.update({k: usr[k] for k in (
        "risk_mult", "tp1_mult", "tp2_mult",
        "no_touch_timeout_hours", "force_close_hours",
        "tick_size", "slip_usd_side_lot", "start_balance")})
    cfg["lot_table"] = [tuple(r) for r in usr["lot_table"]]

    trials: List[Dict[str, float]] = []
    if grid_mode:
        grid = list(itertools.product(SPREAD_UPLIFT_SET,
                                      LAT_RANGE_MINUTES_SET,
                                      SKIP_PROB_SET,
                                      FLASH_GAP_SET,
                                      STOPRUN_ATR_SET))
        print(f"Running grid sweep: {len(grid)} scenarios …\n")
        for n, (spr, lat, skp, gap, wick) in enumerate(grid, 1):
            params = dict(spread_uplift=spr, lat_range=lat,
                          skip_prob=skp, flash_gap_pct=gap,
                          stoprun_mult=wick)
            tid  = f"G{n:03d}"
            res  = run_single_trial(sig_base, bars_base, cfg, tid, params)
            if res:
                trials.append(res)
                print(f"{tid} → ${res['equity_final']:>11,.0f}  "
                    f"DD {res['max_dd_pct']:5.1f}%  "
                    f"lat_range={params['lat_range']}")

            gc.collect()
    else:
        print(f"Running {N_RANDOM_TRIALS} Monte‑Carlo trials …\n")
        for n in range(1, N_RANDOM_TRIALS + 1):
            lo = rng.integers(0, 14)
            hi = rng.integers(lo + 1, 16)
            params = dict(
                spread_uplift=rng.uniform(0, .30),
                lat_range=(lo, hi),
                skip_prob=rng.uniform(0, .30),
                flash_gap_pct=rng.choice([0, .03, .06]),
                stoprun_mult=rng.choice([0, .3, .6]),
            )
            tid  = f"MC{n:03d}"
            res  = run_single_trial(sig_base, bars_base, cfg, tid, params)
            if res:
                trials.append(res)
                print(f"{tid} → ${res['equity_final']:>11,.0f}  "
                    f"DD {res['max_dd_pct']:5.1f}%  "
                    f"lat_range={params['lat_range']}")
            gc.collect()


    if not trials:
        print("No successful trials – check data / parameters.");  return
    out = pd.DataFrame(trials)
    out.to_csv("stress_results_v8.csv", index=False)

    print("\n=== V8 summary (windowed) ===")
    print(out.groupby("status")["equity_final"].describe(percentiles=[.5,.9]))
    print("\nFull details → stress_results_v8.csv")
    if any(SNAPSHOT_DIR.iterdir()):
        print(f"Snapshots of failed runs in  {SNAPSHOT_DIR}/")
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", action="store_true",
        help="Monte‑Carlo mode instead of deterministic grid")
    args = ap.parse_args()
    main(grid_mode=not args.mc)
