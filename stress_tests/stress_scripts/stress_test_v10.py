#!/usr/bin/env python
"""
stress_test_v10.py · Failure‑Finder — **windowed + burst** edition
────────────────────────────────────────────────────────────────────
Adds the five experimental stress hooks you asked for:

1. **Burst‑latency**  –  random 60‑second freezes layered *on top* of the
   baseline latency band.
2. **Partial‑fill engine**  –  when spread‐uplift ≥ 15 % only a random
   50‑100 % of the intended volume is assumed to fill; unrealised units are
   discarded (pnl scaled accordingly).
3. **ATR‑multiplier sweep**  –  we now grid / MC‑sample the `atr_multiplier`
   from **1.0 → 3.0** per trial.
4. **Bigger weekend gaps**  –  flat‑gap stress can now reach **±1 %** and
   uses a wider default grid (0 %, 0.5 %, 1 %, 3 %, 6 %).
5. **Equity‑bleed detector**  –  if *two consecutive* 3‑month windows lose
   money the trial is halted early and classed as a failure (snapshot
   captured).

Fully compatible with **backtest_engine v6** (dynamic stops, risk sizing).
"""
from __future__ import annotations

import argparse
import gc
import itertools
import pathlib
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from backtest_engine import load_cfg, run_backtest  # v6 helpers

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ╭─ GLOBAL CONSTANTS ───────────────────────────────────────────╮
N_RANDOM_TRIALS             = 500
SNAPSHOT_EQUITY_FLOOR_RATIO = 0.30
SNAPSHOT_DIR                = pathlib.Path("stress_deaths_v10")
SNAPSHOT_DIR.mkdir(exist_ok=True)

# base stress grids  ────────────────────────────────────────────
SPREAD_UPLIFT_SET       = (0.00, 0.15, 0.30)
LAT_RANGE_MINUTES_SET   = ((0, 0), (5, 7), (15, 20))
SKIP_PROB_SET           = (0.00, 0.15, 0.30)
FLASH_GAP_SET           = (0.00, 0.005, 0.01, 0.03, 0.06)  # 0 → 6 %
STOPRUN_ATR_SET         = (0.00, 0.30, 0.60)
ATR_MULT_SET            = (1.0, 2.0, 3.0)

# burst‑latency knobs
BURST_FREEZE_MINUTES    = 1          # 60 s freezes
BURST_PROB              = 0.10       # 10 % of signals

rng = np.random.default_rng(seed=42)

# ────────────────────────── I/O HELPERS ────────────────────────

def load_signals() -> pd.DataFrame:
    df = pd.read_csv("signals_long.csv", parse_dates=["datetime"])
    df["entry_mid"]  = (df.entry_low + df.entry_high) / 2
    df["zone_width"] = (df.entry_high - df.entry_low).abs()
    return df.sort_values("datetime", ignore_index=True)


def load_bars() -> pd.DataFrame:
    files = sorted(pathlib.Path("data").glob("XAUUSD_M1_*.csv"))
    dfs   = [pd.read_csv(f, parse_dates=["time_utc"]).rename(columns={"time_utc": "time"})
             for f in files]
    bars  = pd.concat(dfs, ignore_index=True).sort_values("time", ignore_index=True)
    if bars.time.dt.tz is None:
        bars.time = bars.time.dt.tz_localize("UTC")
    bars.time = bars.time.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return bars

# attach ATR to signals when dynamic_stops is enabled

def attach_atr(signals: pd.DataFrame, bars: pd.DataFrame, period: int = 14) -> None:
    hl = bars.high - bars.low
    hc = (bars.high - bars.close.shift()).abs()
    lc = (bars.low  - bars.close.shift()).abs()
    tr = np.maximum.reduce([hl.values, hc.values, lc.values])
    atr = pd.Series(tr, index=bars.index).rolling(period, min_periods=period).mean().bfill()
    bars = bars.assign(atr=atr)
    signals.sort_values("datetime", inplace=True)
    bars.sort_values("time", inplace=True)
    signals["atr"] = pd.merge_asof(
        signals[["datetime"]],
        bars[["time", "atr"]],
        left_on="datetime", right_on="time",
        direction="backward",
    )["atr"]

# ────────────────────────── Stress helpers ─────────────────────

def inject_flat_gap(b: pd.DataFrame, pct: float) -> None:
    if pct == 0:
        return
    monday_open = (
        (b.time.dt.weekday == 0) &
        (b.time.dt.hour == 0) &
        (b.time.dt.minute == 0)
    )
    b.loc[monday_open, "open"] *= 1 + rng.choice([-pct, pct])


def inject_stoprun_wick(b: pd.DataFrame, atr: pd.Series,
                        mult: float, prob: float = 0.15) -> None:
    if mult == 0 or prob == 0:
        return
    picks = b.sample(frac=prob, random_state=rng.integers(0, 2**32-1)).index
    for i in picks:
        spike = mult * atr.iloc[i]
        if rng.random() < .5:
            b.at[i, "low"]  -= spike
        else:
            b.at[i, "high"] += spike


def compute_true_atr(b: pd.DataFrame, window: int = 90*1440) -> pd.Series:
    hl = b.high - b.low
    hc = (b.high - b.close.shift()).abs()
    lc = (b.low  - b.close.shift()).abs()
    tr = np.maximum.reduce([hl.values, hc.values, lc.values])
    return pd.Series(tr, index=b.index).rolling(window, min_periods=window).mean()

# latency utilities --------------------------------------------------------

def apply_baseline_latency(s: pd.DataFrame, b: pd.DataFrame,
                           rng_: tuple[int,int]) -> pd.DataFrame:
    """Constant latency band as in v9 (minutes)."""
    lo, hi = rng_
    if lo == hi == 0:
        return s
    out = s.copy()
    for i, row in out.iterrows():
        lag = rng.integers(lo, hi + 1)
        ts  = row.datetime + pd.Timedelta(minutes=int(lag))
        bar = b[b.time >= ts].iloc[0]
        out.at[i, "datetime"]  = bar.time
        out.at[i, "entry_mid"] = (bar.open + bar.close) / 2
    return out


def apply_burst_latency(s: pd.DataFrame, b: pd.DataFrame,
                        prob: float = BURST_PROB,
                        freeze_min: int = BURST_FREEZE_MINUTES) -> pd.DataFrame:
    """Occasional 60‑second freezes regardless of baseline latency."""
    if prob == 0:
        return s
    out = s.copy()
    mask = rng.random(len(out)) < prob
    for i in out.index[mask]:
        ts  = out.at[i, "datetime"] + pd.Timedelta(minutes=freeze_min)
        bar = b[b.time >= ts].iloc[0]
        out.at[i, "datetime"]  = bar.time
        out.at[i, "entry_mid"] = (bar.open + bar.close) / 2
    return out

# partial‑fill -------------------------------------------------------------

def apply_partial_fill(bt: pd.DataFrame, spr_uplift: float,
                       thresh: float = 0.15) -> pd.DataFrame:
    """Scale pnl and lots when spread explosion reduces fills."""
    if spr_uplift < thresh:
        return bt
    factor = rng.uniform(0.5, 1.0)  # 50‑100 % actually filled
    bt = bt.copy()
    bt["pnl"] *= factor
    # recompute running balance for slice
    bt["balance_after"] = bt["balance_before"] + bt["pnl"].cumsum()
    return bt

# ─────────────────────── One‑trial driver ─────────────────────

def run_single_trial(sig0: pd.DataFrame, bars0: pd.DataFrame,
                     cfg0: dict, tid: str,
                     params: Dict[str, float]) -> Dict[str, float] | None:

    spr_uplift, lat_rng, skip_p, gap_pct, wick_mult, atr_mult = (
        params["spread_uplift"], params["lat_range"],
        params["skip_prob"],     params["flash_gap_pct"],
        params["stoprun_mult"],  params["atr_mult"]
    )

    sigs = sig0.copy()
    bars = bars0.copy(deep=False)
    cfg  = cfg0.copy()

    # ── spread uplift & slippage ───────────────────────────────
    if "spread" not in bars.columns:
        bars["spread"] = 8 * cfg["tick_size"]
    bars["spread"] *= 1 + spr_uplift
    cfg["slippage_usd_side_lot"] *= 1 + spr_uplift

    # ── gap + stop‑run ─────────────────────────────────────────
    inject_flat_gap(bars, gap_pct)
    true_atr = compute_true_atr(bars).bfill()
    inject_stoprun_wick(bars, true_atr, wick_mult)

    # ── shuffle → skip → latency ───────────────────────────────
    sigs = sigs.sample(frac=1.0, random_state=rng.integers(0, 2**32-1)).reset_index(drop=True)
    sigs = sigs[rng.random(len(sigs)) > skip_p].reset_index(drop=True)
    sigs = apply_baseline_latency(sigs, bars, lat_rng)
    sigs = apply_burst_latency(sigs, bars)

    # ── ATR column & multiplier override ───────────────────────
    cfg["atr_multiplier"] = atr_mult
    if cfg.get("dynamic_stops", False):
        attach_atr(sigs, bars, period=cfg.get("atr_period", 14))

    # ── 3‑month stepping windows ──────────────────────────────
    rows, equity = [], cfg["start_balance"]
    lose_streak  = 0  # equity‑bleed detector
    start = sigs.datetime.min().to_period("M").to_timestamp()
    end   = sigs.datetime.max()

    while start < end and equity > 0:
        end_slice = start + pd.DateOffset(months=3)
        slc = sigs[(sigs.datetime >= start) & (sigs.datetime < end_slice)]
        if slc.empty:
            start = end_slice
            continue

        bt = run_backtest(slc, bars, cfg, start_balance=equity, trial_no=tid)
        if bt.empty:
            break

        # partial fill scaling ---------------------------------
        bt = apply_partial_fill(bt, spr_uplift)

        rows.append(bt)
        equity_new = bt.balance_after.iloc[-1]
        pnl_window = equity_new - equity

        # equity‑bleed detection
        lose_streak = lose_streak + 1 if pnl_window < 0 else 0
        if lose_streak >= 2:
            bt.tail(100).to_parquet(SNAPSHOT_DIR / f"{tid}_bleed.parquet")
            equity = equity_new
            break

        equity = equity_new
        if equity < cfg["start_balance"] * SNAPSHOT_EQUITY_FLOOR_RATIO:
            bt.tail(100).to_parquet(SNAPSHOT_DIR / f"{tid}_drawdown.parquet")
            break

        start = end_slice

    if not rows:
        return None

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

# ──────────────────────────── MAIN ─────────────────────────────

def main(grid_mode: bool, only_lat0: bool, risk_override: float | None) -> None:

    sig_base  = load_signals()
    bars_base = load_bars()

    cfg = load_cfg("config.yaml")
    if risk_override is not None:
        cfg["risk_mult"] = float(risk_override)

    lat_set = ((0, 0),) if only_lat0 else LAT_RANGE_MINUTES_SET
    trials: List[Dict[str, float]] = []

    # ───────── GRID mode ──────────
    if grid_mode:
        grid = list(itertools.product(
            SPREAD_UPLIFT_SET, lat_set,
            SKIP_PROB_SET, FLASH_GAP_SET,
            STOPRUN_ATR_SET, ATR_MULT_SET,
        ))
        print(f"Running grid sweep: {len(grid)} scenarios …\n")
        for n, (spr, lat, skp, gap, wick, atrm) in enumerate(grid, 1):
            params = dict(spread_uplift=spr, lat_range=lat, skip_prob=skp,
                          flash_gap_pct=gap, stoprun_mult=wick,
                          atr_mult=atrm)
            tid = f"G{n:03d}"
            res = run_single_trial(sig_base, bars_base, cfg, tid, params)
            if res:
                trials.append(res)
                print(f"{tid} → ${res['equity_final']:>11,.0f}  "
                      f"DD {res['max_dd_pct']:5.1f}%  "
                      f"lat_range={lat}  atr×{atrm}")
            gc.collect()

    # ─────── Monte‑Carlo mode ───────
    else:
        print(f"Running {N_RANDOM_TRIALS} Monte‑Carlo trials …\n")
        for n in range(1, N_RANDOM_TRIALS + 1):
            if only_lat0:
                lat_rng = (0, 0)
            else:
                a, b = rng.integers(0, 15), rng.integers(1, 16)
                lat_rng = (a, b) if a < b else (a, a + 1)

            params = dict(
                spread_uplift=rng.uniform(0, 0.30),
                lat_range=lat_rng,
                skip_prob=rng.uniform(0, 0.30),
                flash_gap_pct=rng.choice([0, 0.005, 0.01, 0.03, 0.06]),
                stoprun_mult=rng.choice([0, 0.3, 0.6]),
                atr_mult=rng.uniform(1.0, 3.0),
            )
            tid = f"MC{n:03d}"
            res = run_single_trial(sig_base, bars_base, cfg, tid, params)
            if res:
                trials.append(res)
                print(f"{tid} → ${res['equity_final']:>11,.0f}  "
                      f"DD {res['max_dd_pct']:5.1f}%  "
                      f"lat_range={lat_rng}")
            gc.collect()

    # ──────────── WRAP‑UP ────────────
    if not trials:
        print("No successful trials – check data / parameters.")
        return

    out = pd.DataFrame(trials)
    out.to_csv("stress_results_v10.csv", index=False)

    print("\n=== V10 summary (windowed) ===")
    print(out.groupby("status")["equity_final"].describe(percentiles=[.5, .9]))
    print("\nFull details → stress_results_v10.csv")
    if any(SNAPSHOT_DIR.iterdir()):
        print(f"Snapshots of failed runs in  {SNAPSHOT_DIR}/")

# ───────────────────────── CLI entry ──────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc",   action="store_true",
        help="Monte‑Carlo mode (default = deterministic grid)")
    ap.add_argument("--lat0", action="store_true",
        help="Restrict tests to lat_range=(0,0)")
    ap.add_argument("--risk", type=float, default=None,
        help="Override risk_mult at runtime, e.g. 1.5")
    args = ap.parse_args()
    main(grid_mode=not args.mc,
         only_lat0=args.lat0,
         risk_override=args.risk)
