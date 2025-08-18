# ────────────────────────────────────────────────────────────────
# stress_test.py  ·  Monte‑Carlo torture‑test  (v4 “adverse‑slide”)
# ────────────────────────────────────────────────────────────────
from __future__ import annotations
import random, pathlib, yaml
import numpy as np
import pandas as pd
from backtest_engine import run_backtest, CFG as ENGINE_DEFAULTS

# ── global knobs ────────────────────────────────────────────────
TRIALS                 = 500
SHUFFLE_SIGNALS        = True

SPREAD_UPLIFT_RANGE    = (0.0, 0.80)      # 0‑80 %
SLIPPAGE_UPLIFT_RANGE  = (0.0, 0.80)      # 0‑80 %

ENTRY_SLIDE_RANGE_PCT  = (0.0, 0.40)      # 0‑40 % of zone width  (now adverse)
SKIP_PROBABILITY       = 0.15             # 15 % signals simply skipped
LATENCY_MINUTES        = (1, 3)           # 1‑3 min fill delay

OOS_WINDOW_MONTHS      = 3                # 9 in‑sample : 3 OOS
RAND = random.Random(42)                  # reproducible

# ── helpers ─────────────────────────────────────────────────────
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
    """
    Move the entry *against* the trader by pct × zone_width.
    BUY → entry lower     (worse fill)
    SELL → entry higher   (worse fill)
    """
    out   = sigs.copy()
    delta = pct * out.zone_width
    out.entry_mid += np.where(out.direction == "BUY", -delta, +delta)
    return out

def skip_signals(sigs: pd.DataFrame, p_skip: float) -> pd.DataFrame:
    mask = np.random.rand(len(sigs)) > p_skip
    return sigs.loc[mask].reset_index(drop=True)

def delay_fills(sigs: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """
    Push each signal forward 1‑3 minutes and force entry_mid to the *open*
    of the delayed bar – simulates latency & price drift.
    """
    delayed = sigs.copy()
    for i, row in delayed.iterrows():
        delay = RAND.randint(*LATENCY_MINUTES)
        ts    = row.datetime + pd.Timedelta(minutes=delay)
        bar   = bars[bars.time >= ts].iloc[0]          # first bar ≥ ts
        delayed.at[i, "datetime"]  = bar.time
        delayed.at[i, "entry_mid"] = bar.open
    return delayed
# ────────────────────────────────────────────────────────────────
# single Monte‑Carlo run
# ────────────────────────────────────────────────────────────────
def run_single_trial(base_signals, base_bars, base_cfg, trial_no: int):

    sigs = base_signals.copy()
    bars = base_bars.copy()
    cfg  = base_cfg.copy()

    # 1) optional shuffle
    if SHUFFLE_SIGNALS:
        sigs = sigs.sample(frac=1.0,
                           random_state=RAND.randint(0, 1 << 32)
                          ).reset_index(drop=True)

    # 2) market‑friction uplifts
    spr_up  = RAND.uniform(*SPREAD_UPLIFT_RANGE)
    slip_up = RAND.uniform(*SLIPPAGE_UPLIFT_RANGE)
    bars    = uplift_spread(bars, spr_up)
    cfg["slippage_usd_side_lot"] *= (1.0 + slip_up)

    # 3) entry & signal tweaks
    sigs = slide_entries(sigs, RAND.uniform(*ENTRY_SLIDE_RANGE_PCT))
    sigs = skip_signals(sigs, SKIP_PROBABILITY)
    sigs = delay_fills(sigs, bars)

    # 4) rolling walk‑forward – carry balance forward
    rows, balance = [], cfg["start_balance"]
    train_months  = 12 - OOS_WINDOW_MONTHS
    start         = sigs.datetime.min().to_period("M").to_timestamp()
    end           = sigs.datetime.max()

    while start < end:
        train_end = start + pd.DateOffset(months=train_months)
        test_end  = train_end + pd.DateOffset(months=OOS_WINDOW_MONTHS)

        slice_ = sigs[(sigs.datetime >= train_end) &
                      (sigs.datetime <  test_end)]
        if slice_.empty:
            start += pd.DateOffset(months=train_months)
            continue

        bt = run_backtest(slice_, bars, cfg, start_balance=balance)
        if not bt.empty:
            balance = bt.balance_after.iloc[-1]
            rows.append(bt)

        start += pd.DateOffset(months=train_months)

    if not rows:
        return None

    full   = pd.concat(rows, ignore_index=True)
    equity = full.balance_after
    dd     = (equity.cummax() - equity).max()

    return dict(
        trial         = trial_no,
        equity_final  = equity.iloc[-1],
        equity_min    = equity.min(),
        max_draw_down = dd,
        profit_factor = (full.pnl[full.pnl > 0].sum() /
                         abs(full.pnl[full.pnl < 0].sum() or 1)),
        win_rate      = (full.pnl > 0).mean()
    )
# ────────────────────────────────────────────────────────────────
# main driver
# ────────────────────────────────────────────────────────────────
def main():
    base_signals = load_signals()
    base_bars    = load_bars()

    usr = yaml.safe_load(open("config.yaml"))
    cfg = ENGINE_DEFAULTS.copy()
    cfg.update({
        "risk_mult"              : usr["risk_mult"],
        "tp1_mult"               : usr["tp1_mult"],
        "tp2_mult"               : usr["tp2_mult"],
        "no_touch_timeout_hours" : usr["no_touch_timeout_hours"],
        "force_close_hours"      : usr["force_close_hours"],
        "tick_size"              : usr["tick_size"],
        "slippage_usd_side_lot"  : usr["slip_usd_side_lot"],
        "lot_table"              : [tuple(r) for r in usr["lot_table"]],
        "start_balance"          : usr["start_balance"],
    })

    results = []
    for n in range(1, TRIALS + 1):
        res = run_single_trial(base_signals, base_bars, cfg, n)
        if res:
            results.append(res)
        print(f"Trial {n:3d}/{TRIALS}  →  "
              f"${res['equity_final']:,.0f}  DD ${res['max_draw_down']:,.0f}")

    out = pd.DataFrame(results)
    out.to_csv("stress_results.csv", index=False)

    summary = (
        f"=== Monte‑Carlo summary ({len(out)}/{TRIALS} runs) ===\n"
        f"equity_final   :  ${out.equity_final.min():,.0f} – "
        f"{out.equity_final.mean():,.0f} – ${out.equity_final.max():,.0f}\n"
        f"max_draw_down  :  ${out.max_draw_down.min():,.0f} – "
        f"{out.max_draw_down.mean():,.0f} – ${out.max_draw_down.max():,.0f}\n"
        f"win‑rate       :  {out.win_rate.min():.2%} – "
        f"{out.win_rate.mean():.2%} – {out.win_rate.max():.2%}\n"
        f"profit factor  :  {out.profit_factor.min():.2f} – "
        f"{out.profit_factor.mean():.2f} – {out.profit_factor.max():.2f}\n"
    )
    pathlib.Path("stress_summary_4.txt").write_text(summary)
    print(summary)

if __name__ == "__main__":
    main()
