# ────────────────────────────────────────────────────────────────
# stress_test.py   ·   quick‑and‑dirty Monte‑Carlo harness  (v2)
# now rolls the balance forward between walk‑forward slices
# ----------------------------------------------------------------
from __future__ import annotations
import random, pathlib, yaml
import numpy as np
import pandas as pd

from backtest_engine import run_backtest, CFG as ENGINE_DEFAULTS

TRIALS                 = 500
SHUFFLE_SIGNALS        = True
SPREAD_UPLIFT_RANGE    = (0.0, 0.40)      # 0–40 %
SLIPPAGE_UPLIFT_RANGE  = (0.0, 0.40)      # 0–40 %
ENTRY_SLIDE_RANGE_PCT  = (0.0, 0.20)      # 0–20 % of zone width
OOS_WINDOW_MONTHS      = 3                # 9 in‑sample : 3 OOS

RAND = random.Random(42)                  # reproducibility
# ────────────────────────────────────────────────────────────────
# helpers
# ----------------------------------------------------------------
def load_signals() -> pd.DataFrame:
    return (pd.read_csv("signals_long.csv", parse_dates=["datetime"])
              .sort_values("datetime"))

def load_bars() -> pd.DataFrame:
    files = sorted(pathlib.Path("data").glob("XAUUSD_M1_*.csv"))
    dfs   = [pd.read_csv(f, parse_dates=["time_utc"])
               .rename(columns={"time_utc": "time"}) for f in files]
    bars  = pd.concat(dfs).sort_values("time")
    if bars.time.dt.tz is None:
        bars.time = bars.time.dt.tz_localize("UTC")
    bars.time = bars.time.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return bars

def uplift_spread(bars: pd.DataFrame, pct: float) -> pd.DataFrame:
    if "spread" not in bars.columns:
        return bars
    out = bars.copy()
    out["spread"] *= (1.0 + pct)
    return out

def slide_entries(sigs: pd.DataFrame, pct: float) -> pd.DataFrame:
    out = sigs.copy()
    delta = pct * out.zone_width
    out.entry_mid += np.where(out.direction == "BUY",  delta, -delta)
    return out
# ────────────────────────────────────────────────────────────────
# single Monte‑Carlo run
# ----------------------------------------------------------------
def run_single_trial(base_signals, base_bars, base_cfg, trial_no: int):
    sigs = base_signals.copy()
    bars = base_bars.copy()
    cfg  = base_cfg.copy()

    # 1) optional shuffle
    if SHUFFLE_SIGNALS:
        sigs = sigs.sample(frac=1.0,
                           random_state=RAND.randint(0, 1 << 32)
                           ).reset_index(drop=True)

    # 2) random market‑friction uplifts
    spr_up  = RAND.uniform(*SPREAD_UPLIFT_RANGE)
    slip_up = RAND.uniform(*SLIPPAGE_UPLIFT_RANGE)
    bars    = uplift_spread(bars, spr_up)
    cfg["slippage_usd_side_lot"] *= (1.0 + slip_up)

    # 3) entry‑price slide
    ent_slide = RAND.uniform(*ENTRY_SLIDE_RANGE_PCT)
    sigs      = slide_entries(sigs, ent_slide)

    # 4) rolling walk‑forward
    rows   : list[pd.DataFrame] = []
    balance = cfg["start_balance"]            # <‑‑ carry‑forward state
    train_months = 12 - OOS_WINDOW_MONTHS

    start = sigs.datetime.min().to_period("M").to_timestamp()
    end   = sigs.datetime.max()

    while start < end:
        train_end = start + pd.DateOffset(months=train_months)
        test_end  = train_end + pd.DateOffset(months=OOS_WINDOW_MONTHS)

        test_slice = sigs[(sigs.datetime >= train_end) &
                          (sigs.datetime <  test_end)]
        if test_slice.empty:
            start += pd.DateOffset(months=train_months)
            continue

        bt = run_backtest(test_slice, bars, cfg, start_balance=balance)
        if not bt.empty:
            balance = bt.balance_after.iloc[-1]   # <‑‑ roll equity forward
            rows.append(bt)

        start += pd.DateOffset(months=train_months)

    if not rows:
        return None

    full   = pd.concat(rows, ignore_index=True)
    equity = full.balance_after
    max_dd = (equity.cummax() - equity).max()

    return dict(
        trial          = trial_no,
        equity_final   = equity.iloc[-1],
        equity_min     = equity.min(),
        max_draw_down  = max_dd,
        profit_factor  = (full.pnl[full.pnl > 0].sum() /
                          abs(full.pnl[full.pnl < 0].sum() or 1)),
        win_rate       = (full.pnl > 0).mean()
    )
# ────────────────────────────────────────────────────────────────
# main driver
# ----------------------------------------------------------------
def main():
    base_signals = load_signals()
    base_bars    = load_bars()

    # merge user cfg → engine cfg
    usr_cfg = yaml.safe_load(open("config.yaml"))
    cfg     = ENGINE_DEFAULTS.copy()
    cfg.update({
        "risk_mult"               : usr_cfg["risk_mult"],
        "tp1_mult"                : usr_cfg["tp1_mult"],
        "tp2_mult"                : usr_cfg["tp2_mult"],
        "no_touch_timeout_hours"  : usr_cfg["no_touch_timeout_hours"],
        "force_close_hours"       : usr_cfg["force_close_hours"],
        "tick_size"               : usr_cfg["tick_size"],
        # align key spelling with back‑test engine
        "slippage_usd_side_lot"   : usr_cfg["slip_usd_side_lot"],
        "lot_table"               : [tuple(r) for r in usr_cfg["lot_table"]],
        "start_balance"           : usr_cfg["start_balance"],
    })

    results = []
    for i in range(1, TRIALS + 1):
        res = run_single_trial(base_signals, base_bars, cfg, i)
        if res:
            results.append(res)
        print(f"Trial {i:3d}/{TRIALS} "
              f"→ final ${res['equity_final']:,.0f}  "
              f"DD ${res['max_draw_down']:,.0f}")

    out = pd.DataFrame(results)
    out.to_csv("stress_results_2.csv", index=False)

    summary = (
        f"=== Monte‑Carlo summary ({len(out)} / {TRIALS} successful runs) ===\n"
        f"equity_final  :  min ${out.equity_final.min():,.0f}  "
        f"mean ${out.equity_final.mean():,.0f}  "
        f"max ${out.equity_final.max():,.0f}\n"
        f"max_draw_down :  min ${out.max_draw_down.min():,.0f}  "
        f"mean ${out.max_draw_down.mean():,.0f}  "
        f"max ${out.max_draw_down.max():,.0f}\n"
        f"win‑rate      :  min {out.win_rate.min():.2%}  "
        f"mean {out.win_rate.mean():.2%}  "
        f"max {out.win_rate.max():.2%}\n"
        f"profit factor :  min {out.profit_factor.min():.2f}  "
        f"mean {out.profit_factor.mean():.2f}  "
        f"max {out.profit_factor.max():.2f}\n"
    )
    pathlib.Path("stress_summary_2.txt").write_text(summary)
    print(summary)

if __name__ == "__main__":
    main()
