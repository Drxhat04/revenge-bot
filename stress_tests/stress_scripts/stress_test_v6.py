# ────────────────────────────────────────────────────────────────
# stress_test_v6.py · Monte‑Carlo torture‑test  “nuclear‑winter”
# harsher costs · flash‑crash bar · dynamic leverage cut
# ────────────────────────────────────────────────────────────────
from __future__ import annotations
import gc, pathlib, warnings, yaml, random
from typing import List

import numpy as np
import pandas as pd
from backtest_engine import run_backtest, CFG as ENGINE_DEFAULTS

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ╭─ GLOBAL KNOBS ───────────────────────────────────────────────╮
TRIALS                      = 500
SHUFFLE_SIGNALS             = True

# market friction (nastier)
SPREAD_UPLIFT_RANGE         = (0.00, 0.50)     # +0 … +50 % broker spread
SLIPPAGE_UPLIFT_RANGE       = (0.00, 0.80)     # +0 … +80 % slippage cost
BASE_SPREAD_TICKS           = 8
TICK_SIZE                   = 0.01

ATR_WINDOW_MIN              = 90 * 1440        # 90‑day ATR
MAX_ATR_SPREAD_MULT         = 2.0

# execution quirks
ENTRY_SLIDE_RANGE_PCT       = (0.00, 0.40)
LAT_RANGE_MINUTES           = (5, 7)
PARTIAL_FILL_PROB           = 0.05
PARTIAL_FILL_FRACTION       = 0.50
REJECT_FILL_PROB            = 0.02
SKIP_PROBABILITY            = 0.15

STOPRUN_PROB                = 0.15
STOPRUN_MULT_ATR            = 0.30

# flash‑crash
FLASH_CRASH_PROB            = 0.05            # 5 % of trials
FLASH_CRASH_PCT_RANGE       = (-0.01, -0.005) # −1 % … −0.5 %

# walk‑forward & risk
TRAIN_MONTHS                = 9
OOS_WINDOW_MONTHS           = 3
REGIME_SWITCH_YEARS         = [(2019, 2020), (2022, 2023)]
MARGIN_CALL_EQUITY_RATIO    = 0.70

assert 0 <= PARTIAL_FILL_PROB <= 1
assert 0 <= REJECT_FILL_PROB  <= 1
assert 0 <= STOPRUN_PROB      <= 1
# ╰──────────────────────────────────────────────────────────────╯


rng = np.random.default_rng(seed=42)

# ─────────────────────────── IO HELPERS ─────────────────────────
def load_signals() -> pd.DataFrame:
    return (pd.read_csv("signals_long.csv", parse_dates=["datetime"])
              .sort_values("datetime", ignore_index=True))

def load_bars() -> pd.DataFrame:
    files = sorted(pathlib.Path("data").glob("XAUUSD_M1_*.csv"))
    dfs   = [pd.read_csv(f, parse_dates=["time_utc"])
               .rename(columns={"time_utc": "time"}) for f in files]
    bars  = pd.concat(dfs, ignore_index=True).sort_values("time", ignore_index=True)
    if bars.time.dt.tz is None:
        bars.time = bars.time.dt.tz_localize("UTC")
    bars.time = bars.time.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return bars

# ───────────────────── market‑friction helpers ──────────────────
def _ensure_spread_col(b: pd.DataFrame) -> None:
    if "spread" not in b.columns:
        b["spread"] = BASE_SPREAD_TICKS * TICK_SIZE

def _true_atr(b: pd.DataFrame) -> pd.Series:
    hl = b.high - b.low
    hc = (b.high - b.close.shift()).abs()
    lc = (b.low  - b.close.shift()).abs()
    tr = np.maximum.reduce([hl.values, hc.values, lc.values])
    return pd.Series(tr, index=b.index).rolling(ATR_WINDOW_MIN,
                                                min_periods=ATR_WINDOW_MIN).mean()

def apply_flat_uplift(b: pd.DataFrame, pct: float) -> None:
    _ensure_spread_col(b)
    b["spread"] *= (1 + pct)

def apply_vol_scaled_spread(b: pd.DataFrame, atr: pd.Series) -> None:
    _ensure_spread_col(b)
    vol_mult = (atr / atr.median()).clip(0, 1) * MAX_ATR_SPREAD_MULT
    b["spread"] *= (1 + vol_mult)

# ───────────────────── execution helpers ────────────────────────
def slide_entries(s: pd.DataFrame, pct: float) -> None:
    d = pct * s.zone_width
    s.entry_mid += np.where(s.direction == "BUY", -d, +d)

def skip_signals(s: pd.DataFrame, p: float) -> pd.DataFrame:
    keep = rng.random(len(s)) > p
    return s.loc[keep].reset_index(drop=True)

def delay_and_fill(s: pd.DataFrame, b: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    atr_norm = (atr / atr.max()).bfill()
    out = s.copy()
    for i, row in out.iterrows():
        lag = int(np.interp(atr_norm[b.time >= row.datetime].iloc[0],
                            [0, 1], LAT_RANGE_MINUTES))
        ts  = row.datetime + pd.Timedelta(minutes=lag)
        bar = b.loc[b.time >= ts].iloc[0]

        ask = bar.open + 0.5 * bar.spread
        bid = bar.open - 0.5 * bar.spread
        out.at[i, "datetime"]  = bar.time
        out.at[i, "entry_mid"] = ask if row.direction == "BUY" else bid
    return out

def partial_or_reject(s: pd.DataFrame) -> pd.DataFrame:
    rnd  = rng.random(len(s))
    mult = np.where(rnd < PARTIAL_FILL_PROB, PARTIAL_FILL_FRACTION, 1.0)
    out  = s.loc[rnd >= REJECT_FILL_PROB].reset_index(drop=True)
    out["risk_mult"] = mult[rnd >= REJECT_FILL_PROB]
    return out

def inject_weekend_gaps(b: pd.DataFrame) -> None:
    monday_open = (b.time.dt.weekday == 0) & (b.time.dt.hour == 0) & (b.time.dt.minute == 0)
    b.loc[monday_open, "open"] *= (1 + rng.uniform(-0.0025, 0.0025, monday_open.sum()))

def inject_flash_crash(b: pd.DataFrame) -> None:
    """Randomly nuke one bar {‑1 % … ‑0 .5 %} to mimic a mini flash‑crash."""
    if rng.random() > FLASH_CRASH_PROB:
        return

    idx = rng.integers(ATR_WINDOW_MIN, len(b) - 2)       # keep away from the head
    pct = rng.uniform(*FLASH_CRASH_PCT_RANGE)            # negative %

    # scale the whole OHLC line
    b.loc[idx, ["open", "high", "low", "close"]] *= (1 + pct)

    # re‑sanity‑check OHLC ordering
    b.at[idx, "low"]  = b.loc[idx, ["open", "low", "close"]].min()
    b.at[idx, "high"] = b.loc[idx, ["open", "high", "close"]].max()

def stop_run_wicks(tr: pd.DataFrame, b: pd.DataFrame, atr: pd.Series) -> None:
    if tr.empty or STOPRUN_PROB == 0:
        return
    picks = tr.sample(frac=STOPRUN_PROB,
                      random_state=rng.integers(0, 2**32 - 1))
    for _, sig in picks.iterrows():
        idx   = b[b.time >= sig.datetime].index[0]
        spike = STOPRUN_MULT_ATR * atr.iloc[idx]
        if sig.direction.upper() == "BUY":
            b.at[idx, "low"]  = min(b.at[idx, "low"],  sig.entry_mid - spike)
        else:
            b.at[idx, "high"] = max(b.at[idx, "high"], sig.entry_mid + spike)

# ───────────────────── Monte‑Carlo SINGLE TRIAL ─────────────────
def run_single_trial(sig_base: pd.DataFrame,
                     bars_base: pd.DataFrame,
                     cfg_base : dict,
                     n: int) -> dict | None:

    sigs = (sig_base.sample(frac=1.0,
                            random_state=rng.integers(0, 2**32-1),
                            ignore_index=True)
            if SHUFFLE_SIGNALS else sig_base.copy())

    bars = bars_base.copy(deep=False)
    cfg  = cfg_base.copy()

    # market‑friction
    apply_flat_uplift(bars, rng.uniform(*SPREAD_UPLIFT_RANGE))
    cfg["slippage_usd_side_lot"] *= (1 + rng.uniform(*SLIPPAGE_UPLIFT_RANGE))

    atr_full = _true_atr(bars).bfill()
    apply_vol_scaled_spread(bars, atr_full)
    inject_weekend_gaps(bars)
    inject_flash_crash(bars)

    # execution quirks
    slide_entries(sigs, rng.uniform(*ENTRY_SLIDE_RANGE_PCT))
    sigs = skip_signals(sigs, SKIP_PROBABILITY)
    sigs = delay_and_fill(sigs, bars, atr_full)
    sigs = partial_or_reject(sigs)

    # walk‑forward
    rows, equity = [], cfg["start_balance"]
    regime_idx   = 0
    start        = sigs.datetime.min().to_period("M").to_timestamp()
    end          = sigs.datetime.max()

    while start < end and equity > 0:
        train_end = start + pd.DateOffset(months=TRAIN_MONTHS)
        test_end  = train_end + pd.DateOffset(months=OOS_WINDOW_MONTHS)

        regime_idx ^= 1
        yr_lo, yr_hi = REGIME_SWITCH_YEARS[regime_idx]
        mask = (sigs.datetime >= train_end) & (sigs.datetime < test_end) \
             & sigs.datetime.dt.year.between(yr_lo, yr_hi)
        slice_sigs = sigs[mask]
        if slice_sigs.empty:
            start += pd.DateOffset(months=OOS_WINDOW_MONTHS)
            continue

        stop_run_wicks(slice_sigs, bars, atr_full)

        # dynamic de‑leverage
        cfg_dyn = cfg.copy()
        cfg_dyn["risk_mult"] = max(0.1,
                                   min(cfg["risk_mult"] *
                                       (equity / cfg["start_balance"]),
                                       cfg["risk_mult"]))

        bt = run_backtest(slice_sigs, bars, cfg_dyn,
                          start_balance=equity,
                          trial_no=n)
        if bt.empty:
            start += pd.DateOffset(months=OOS_WINDOW_MONTHS)
            continue

        rows.append(bt)
        equity = bt.balance_after.iloc[-1]

        if bt.balance_after.min() < equity * (1 - MARGIN_CALL_EQUITY_RATIO):
            equity *= MARGIN_CALL_EQUITY_RATIO
            break

        start += pd.DateOffset(months=OOS_WINDOW_MONTHS)

    if not rows:
        return None

    full = pd.concat(rows, ignore_index=True)
    peak = full.balance_after.cummax()
    max_dd_pct = ((peak - full.balance_after) / peak * 100).max()

    return {
        "trial"        : n,
        "equity_final" : full.balance_after.iloc[-1],
        "equity_min"   : full.balance_after.min(),
        "max_dd_pct"   : max_dd_pct,
        "profit_factor": full.pnl[full.pnl > 0].sum() /
                         max(1, abs(full.pnl[full.pnl < 0].sum())),
        "win_rate"     : (full.pnl > 0).mean()
    }

# ─────────────────────────── MAIN LOOP ──────────────────────────
def main() -> None:
    sig_base  = load_signals()
    bars_base = load_bars()

    usr = yaml.safe_load(open("config.yaml"))
    cfg = ENGINE_DEFAULTS.copy()
    cfg.update({
        "risk_mult"             : usr["risk_mult"],
        "tp1_mult"              : usr["tp1_mult"],
        "tp2_mult"              : usr["tp2_mult"],
        "no_touch_timeout_hours": usr["no_touch_timeout_hours"],
        "force_close_hours"     : usr["force_close_hours"],
        "tick_size"             : usr["tick_size"],
        "slippage_usd_side_lot" : usr["slip_usd_side_lot"],
        "lot_table"             : [tuple(r) for r in usr["lot_table"]],
        "start_balance"         : usr["start_balance"],
    })

    results: List[dict] = []
    for n in range(1, TRIALS + 1):
        res = run_single_trial(sig_base, bars_base, cfg, n)
        if res:
            results.append(res)
            print(f"Trial {n:3d}/{TRIALS} → "
                  f"${res['equity_final']:>10,.0f}  "
                  f"DD {res['max_dd_pct']:6.2f}%")
        gc.collect()

    if not results:
        print("No successful trials – check data / parameters.")
        return

    out = pd.DataFrame(results)
    out.to_csv("stress_results_v6.csv", index=False)

    summary = (
        f"\n=== Monte‑Carlo v6 summary "
        f"({len(out)}/{TRIALS} runs) ===\n"
        f"equity_final : ${out.equity_final.min():,.0f} – "
        f"{out.equity_final.mean():,.0f} – ${out.equity_final.max():,.0f}\n"
        f"max_dd_pct   : {out.max_dd_pct.min():.2f}% – "
        f"{out.max_dd_pct.mean():.2f}% – {out.max_dd_pct.max():.2f}%\n"
        f"win‑rate     : {out.win_rate.min():.2%} – "
        f"{out.win_rate.mean():.2%} – {out.win_rate.max():.2%}\n"
        f"profit fact. : {out.profit_factor.min():.2f} – "
        f"{out.profit_factor.mean():.2f} – {out.profit_factor.max():.2f}\n"
    )
    pathlib.Path("stress_summary_v6.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
