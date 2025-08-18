# ────────────────────────────────────────────────────────────────
# stress_test_v5.py · Monte‑Carlo torture‑test  v5.3 “meltdown++”
# fully consolidated version – single RNG, proper latency fills,
# better DD capture, faster stop‑run, lower mem‑footprint
# ────────────────────────────────────────────────────────────────
from __future__ import annotations
import gc, pathlib, warnings, yaml
from typing import List

import numpy as np
import pandas as pd
from backtest_engine import run_backtest, CFG as ENGINE_DEFAULTS

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ╭─ GLOBAL KNOBS ───────────────────────────────────────────────╮
TRIALS                      = 500          # deeper Monte‑Carlo sweep
SHUFFLE_SIGNALS             = True

# ── market friction ────────────────────────────────────────────
SPREAD_UPLIFT_RANGE         = (0.00, 0.30)   # up to +30 % broker spread
SLIPPAGE_UPLIFT_RANGE       = (0.00, 0.50)   # up to +50 % slippage cost
BASE_SPREAD_TICKS           = 8              # baseline 8 ticks
TICK_SIZE                   = 0.01           # $0.01 tick size

ATR_WINDOW_MIN              = 90 * 1440      # 90‑day ATR window
MAX_ATR_SPREAD_MULT         = 2.0            # vol‑scaled spread cap

# ── execution quirks ───────────────────────────────────────────
ENTRY_SLIDE_RANGE_PCT       = (0.00, 0.40)   # adverse entry slide
LAT_RANGE_MINUTES           = (5, 7)         # 5‑7 min artificial latency
PARTIAL_FILL_PROB           = 0.05           # 5 % partial fills
PARTIAL_FILL_FRACTION       = 0.50           # …half‑size when partial
REJECT_FILL_PROB            = 0.02           # 2 % outright rejects
SKIP_PROBABILITY            = 0.15           # 15 % signals skipped

STOPRUN_PROB                = 0.15           # 15 % of trades get a spike
STOPRUN_MULT_ATR            = 0.30           # spike size = 0.30 × ATR

# ── walk‑forward & risk ────────────────────────────────────────
TRAIN_MONTHS                = 9              # 9 M IS  + 3 M OOS
OOS_WINDOW_MONTHS           = 3
REGIME_SWITCH_YEARS         = [(2019, 2020), (2022, 2023)]
MARGIN_CALL_EQUITY_RATIO    = 0.70           # equity haircut after margin call

# ── sanity checks ──────────────────────────────────────────────
assert 0 <= PARTIAL_FILL_PROB <= 1
assert 0 <= REJECT_FILL_PROB  <= 1
assert 0 <= STOPRUN_PROB      <= 1
# ╰──────────────────────────────────────────────────────────────╯


# single RNG for the whole run ----------------------------------
rng = np.random.default_rng(seed=42)

# ─────────────────────────── HELPERS ───────────────────────────
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


# –– spread helpers –––––––––––––––––––––––––––––––––––––––––––––
def _ensure_spread_col(b: pd.DataFrame) -> None:
    if "spread" not in b.columns:
        b["spread"] = BASE_SPREAD_TICKS * TICK_SIZE


def _true_atr(b: pd.DataFrame) -> pd.Series:
    hl = b.high - b.low
    hc = (b.high - b.close.shift()).abs()
    lc = (b.low  - b.close.shift()).abs()
    tr = np.maximum.reduce([hl.values, hc.values, lc.values])
    tr = pd.Series(tr, index=b.index)
    return tr.rolling(ATR_WINDOW_MIN, min_periods=ATR_WINDOW_MIN).mean()


def apply_flat_uplift(b: pd.DataFrame, pct: float) -> None:
    _ensure_spread_col(b)
    b["spread"] *= (1 + pct)


def apply_vol_scaled_spread(b: pd.DataFrame, atr: pd.Series) -> None:
    _ensure_spread_col(b)
    vol_mult = (atr / atr.median()).clip(0, 1) * MAX_ATR_SPREAD_MULT
    b["spread"] *= (1 + vol_mult)


# –– misc helpers –––––––––––––––––––––––––––––––––––––––––––––––
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

        # BUY pays ask, SELL pays bid
        ask = bar.open + 0.5 * bar.spread
        bid = bar.open - 0.5 * bar.spread
        out.at[i, "datetime"]  = bar.time
        out.at[i, "entry_mid"] = ask if row.direction == "BUY" else bid
    return out


def partial_or_reject(s: pd.DataFrame) -> pd.DataFrame:
    rnd = rng.random(len(s))
    mult = np.ones(len(s))
    mult[rnd < PARTIAL_FILL_PROB] *= PARTIAL_FILL_FRACTION
    out  = s.loc[rnd >= REJECT_FILL_PROB].reset_index(drop=True)
    mult = mult[rnd >= REJECT_FILL_PROB]
    out["risk_mult"] = mult
    return out


def inject_weekend_gaps(b: pd.DataFrame) -> None:
    monday_open = (b.time.dt.weekday == 0) & (b.time.dt.hour == 0) & (b.time.dt.minute == 0)
    b.loc[monday_open, "open"] *= (1 + rng.uniform(-0.0025, 0.0025, monday_open.sum()))


def stop_run_wicks(tr: pd.DataFrame, b: pd.DataFrame, atr: pd.Series) -> None:
    if tr.empty or STOPRUN_PROB == 0:
        return
    picks = tr.sample(frac=STOPRUN_PROB, random_state=rng.integers(0, 2**32 - 1))
    for _, sig in picks.iterrows():
        idx = b[b.time >= sig.datetime].index[0]
        spike = STOPRUN_MULT_ATR * atr.iloc[idx]
        if sig.direction.upper() == "BUY":
            b.at[idx, "low"] = min(b.at[idx, "low"], sig.entry_mid - spike)
        else:
            b.at[idx, "high"] = max(b.at[idx, "high"], sig.entry_mid + spike)


# ───────────────────── SINGLE MONTE‑CARLO RUN ──────────────────
def run_single_trial(sig_base: pd.DataFrame,
                     bars_base: pd.DataFrame,
                     cfg_base : dict,
                     n: int) -> dict | None:

    sigs = (sig_base.sample(frac=1.0, random_state=rng.integers(0, 2**32-1), ignore_index=True)
            if SHUFFLE_SIGNALS else sig_base.copy())

    bars = bars_base.copy(deep=False)
    cfg  = cfg_base.copy()

    # --- market friction ---------------------------------------------------
    apply_flat_uplift(bars, rng.uniform(*SPREAD_UPLIFT_RANGE))
    cfg["slippage_usd_side_lot"] *= (1 + rng.uniform(*SLIPPAGE_UPLIFT_RANGE))

    atr_full = _true_atr(bars).bfill()
    apply_vol_scaled_spread(bars, atr_full)
    inject_weekend_gaps(bars)

    # --- execution quirks ---------------------------------------------------
    slide_entries(sigs, rng.uniform(*ENTRY_SLIDE_RANGE_PCT))
    sigs = skip_signals(sigs, SKIP_PROBABILITY)
    sigs = delay_and_fill(sigs, bars, atr_full)
    sigs = partial_or_reject(sigs)

    # --- walk‑forward -------------------------------------------------------
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

        # slice‑specific bars view (shallow)
        slice_bars = bars
        stop_run_wicks(slice_sigs, slice_bars, atr_full)

        bt = run_backtest(slice_sigs, slice_bars, cfg,
                          start_balance=equity,
                          trial_no=n)
        if bt.empty:
            start += pd.DateOffset(months=OOS_WINDOW_MONTHS)
            continue

        rows.append(bt)
        equity = bt.balance_after.iloc[-1]

        # margin‑call emulation
        if bt.balance_after.min() < equity * (1 - MARGIN_CALL_EQUITY_RATIO):
            equity *= MARGIN_CALL_EQUITY_RATIO
            break

        start += pd.DateOffset(months=OOS_WINDOW_MONTHS)

    if not rows:
        return None

    full = pd.concat(rows, ignore_index=True)
    
    # CALCULATE PERCENTAGE DRAWDOWN =========================================
    # 1. Compute equity peaks (cumulative maximum)
    full['equity_peak'] = full.balance_after.cummax()
    
    # 2. Calculate drawdown percentage
    full['drawdown_pct'] = (full['equity_peak'] - full.balance_after) / full['equity_peak'] * 100
    
    # 3. Find maximum percentage drawdown
    max_dd_pct = full['drawdown_pct'].max()
    # =======================================================================

    return {
        "trial"        : n,
        "equity_final" : full.balance_after.iloc[-1],
        "equity_min"   : full.balance_after.min(),
        "max_dd_pct"   : max_dd_pct,  # percentage
        "profit_factor": full.pnl[full.pnl > 0].sum() / 
                         max(1, abs(full.pnl[full.pnl < 0].sum())),
        "win_rate"     : (full.pnl > 0).mean()
    }

# ───────────────────────── MAIN DRIVER ─────────────────────────
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
        print("No successful trials – check data / parameters.")
        return

    out = pd.DataFrame(results)
    out.to_csv("stress_results_v5.csv", index=False)

    summary = (
        f"\n=== Monte‑Carlo v5.3 summary "
        f"({len(out)}/{TRIALS} runs) ===\n"
        f"equity_final   : ${out.equity_final.min():,.0f} – "
        f"{out.equity_final.mean():,.0f} – ${out.equity_final.max():,.0f}\n"
        f"max_dd%%  : ${out.max_dd_pct.min():.2f} % – "
        f"{out.max_dd_pct.mean():.2f}% – {out.max_dd_pct.max():.2f}%\n"
        f"win‑rate       : {out.win_rate.min():.2%} – "
        f"{out.win_rate.mean():.2%} – {out.win_rate.max():.2%}\n"
        f"profit factor  : {out.profit_factor.min():.2f} – "
        f"{out.profit_factor.mean():.2f} – {out.profit_factor.max():.2f}\n"
    )
    pathlib.Path("stress_summary_v5.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
