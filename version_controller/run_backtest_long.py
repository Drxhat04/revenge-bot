"""
run_backtest_long.py · v3
────────────────────────────────────────────────────────────────────────────
Glue script: ❶ load config ❷ load signals_long.csv ❸ load M1 bars
❹ run back‑test engine ❺ write bt_results_long.csv and print headline stats
plus: equity‑curve health check (max‑DD, avg pnl, Sharpe).
"""

from __future__ import annotations

import pathlib
import pandas as pd
import yaml

from backtest_engine import CFG as ENGINE_DEFAULTS, run_backtest

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ---------------------------------------------------------------------------


def merge_cfg(config_path: str = "config.yaml") -> dict:
    """Merge user config with engine defaults, returning a fresh dict."""
    usr = yaml.safe_load(open(config_path))
    cfg = ENGINE_DEFAULTS.copy()
    cfg.update(
        {
            "risk_mult": usr["risk_mult"],
            "tp1_mult": usr["tp1_mult"],
            "tp2_mult": usr["tp2_mult"],
            "no_touch_timeout_hours": usr["no_touch_timeout_hours"],
            "force_close_hours": usr["force_close_hours"],
            "tick_size": usr["tick_size"],
            "slippage_usd_side_lot": usr["slip_usd_side_lot"],
            "lot_table": [tuple(r) for r in usr["lot_table"]],
            "start_balance": usr["start_balance"],
        }
    )
    return cfg


def load_bars(folder: str = "data") -> pd.DataFrame:
    """Load and concatenate all M1 CSVs, normalising the time column."""
    files = sorted(pathlib.Path(folder).glob("XAUUSD_M1_*.csv"))
    if not files:
        raise FileNotFoundError("No XAUUSD_M1_*.csv files found in /data")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["time_utc"])
        df.rename(columns={"time_utc": "time"}, inplace=True)
        dfs.append(df)

    bars = pd.concat(dfs).sort_values("time")

    # tz‑aware or tz‑naïve → normalise to tz‑naïve Berlin
    if bars["time"].dt.tz is None:
        bars["time"] = bars["time"].dt.tz_localize("UTC")
    bars["time"] = bars["time"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return bars


# ────────────────────────────────────────────────────────────────────────────
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = merge_cfg()

    # ---- Load signals -----------------------------------------------------
    signals = pd.read_csv("signals_long.csv", parse_dates=["datetime"])
    if signals.empty:
        print("⚠️  No signals in signals_long.csv → nothing to back‑test.")
        return

    required = {"datetime", "entry_low", "entry_high", "direction"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Missing columns in signals_long.csv: {missing}")

    signals = signals.sort_values("datetime")

    # ---- Load price bars --------------------------------------------------
    bars = load_bars()

    # ---- Run back‑test ----------------------------------------------------
    bt = run_backtest(signals, bars, cfg, start_balance=cfg["start_balance"])
    if bt.empty:
        print("⚠️  Back‑test produced zero trades. Check signal logic / filters.")
        return

    bt.to_csv("bt_results_long.csv", index=False)

    # ---- Headline stats ---------------------------------------------------
    final_bal = bt.balance_after.iloc[-1]
    wins      = (bt.pnl > 0).sum()
    losses    = (bt.pnl < 0).sum()
    breakeven = (bt.pnl.round(8) == 0).sum()
    loss_sum  = abs(bt.loc[bt.pnl < 0, "pnl"].sum())
    pf        = bt.loc[bt.pnl > 0, "pnl"].sum() / loss_sum if loss_sum else float("inf")

    # ── Equity‑curve audit ────────────────────────────────────────────────
    eq            = bt.balance_after
    peak          = eq.cummax()
    drawdown      = peak - eq
    max_dd        = drawdown.max()
    max_dd_pct    = (max_dd / peak.loc[drawdown.idxmax()]) * 100 if not peak.empty else 0
    avg_trade_pnl = bt.pnl.mean()
    sharpe_naive  = (bt.pnl.mean() / bt.pnl.std()) if bt.pnl.std() else float("inf")

    # ---- Print summary ----------------------------------------------------
    print("=" * 40)
    print("Back‑test complete")
    print(f"Trades               : {len(bt)}")
    print(f"Wins / Losses / BE   : {wins} / {losses} / {breakeven}")
    print(f"Profit Factor        : {pf:.2f}")
    print(f"Final Balance        : ${final_bal:,.2f}")
    print(f"Max Draw‑down        : ${max_dd:,.2f}  ({max_dd_pct:.2f} %)")
    print(f"Average trade        : ${avg_trade_pnl:.2f}")
    print(f"Naïve Sharpe (mean/σ): {sharpe_naive:.2f}")
    print("=" * 40)
    print("Detailed log → bt_results_long.csv")


if __name__ == "__main__":
    main()
