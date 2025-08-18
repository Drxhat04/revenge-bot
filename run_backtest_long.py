#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────
# run_backtest_long.py · v10.1 (UTC-native, engine v10.1)
#   • load config
#   • load signals_long_utc.csv (+15m Wilder ATR if needed)
#   • load M1 bars in UTC (dedup, skip weekends)
#   • call run_backtest() and write bt_results_long.csv
#   • print filled-trade stats and diagnostics
# ────────────────────────────────────────────────────────────────

from __future__ import annotations
import pathlib
import pandas as pd
from backtest_engine import load_cfg, run_backtest

# ╭─ Helpers ───────────────────────────────────────────────────╮
def load_bars(folder: str = "data") -> pd.DataFrame:
    """Load raw XAUUSD M1 CSVs in UTC, drop weekends, sort, and de-dup."""
    path = pathlib.Path(folder)
    files = sorted(path.glob("XAUUSD_M1_*.csv"))
    if not files:
        raise FileNotFoundError("No XAUUSD_M1_*.csv files found in /data")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["time_utc"])
        df = df.rename(columns={"time_utc": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        dfs.append(df)

    bars = pd.concat(dfs, ignore_index=True)
    bars = bars.sort_values("time")
    bars = bars.drop_duplicates(subset="time", keep="last")
    bars = bars[~bars.time.dt.weekday.isin([5, 6])]  # drop Sat/Sun
    return bars
# ╰──────────────────────────────────────────────────────────────╯


def attach_wilder_atr_15m(signals: pd.DataFrame, bars: pd.DataFrame, period: int = 14) -> None:
    """
    Compute 15-minute Wilder ATR and merge onto signals at the latest prior 15m close.
    """
    m15 = (bars.set_index("time")
                 .resample("15min")
                 .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                 .dropna()
                 .reset_index())

    prev_close = m15["close"].shift()
    tr = pd.concat([
        m15["high"] - m15["low"],
        (m15["high"] - prev_close).abs(),
        (m15["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder ATR via EMA with alpha = 1/period
    m15["atr"] = tr.ewm(alpha=1/period, adjust=False).mean()

    signals.sort_values("datetime", inplace=True)
    m15.sort_values("time", inplace=True)
    signals["atr"] = pd.merge_asof(
        signals[["datetime"]],
        m15[["time", "atr"]],
        left_on="datetime",
        right_on="time",
        direction="backward",
    )["atr"]


def ensure_signal_columns(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist. Derive entry_mid and zone_width if missing.
    """
    required = {"datetime", "entry_low", "entry_high", "direction"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Missing columns in signals_long_utc.csv: {missing}")

    if "entry_mid" not in signals.columns:
        signals["entry_mid"] = (signals.entry_low + signals.entry_high) / 2.0

    if "zone_width" not in signals.columns:
        signals["zone_width"] = (signals.entry_high - signals.entry_low).abs()

    signals["direction"] = signals["direction"].str.upper().str.strip()
    return signals


def main() -> None:
    cfg = load_cfg("config.yaml")

    # ── Load signals (UTC) ───────────────────────────────────────
    sig_file = "signals_long_utc.csv"
    signals = pd.read_csv(sig_file, parse_dates=["datetime"])
    if signals.empty:
        print(f"No signals in {sig_file} → nothing to back-test.")
        return

    signals["datetime"] = pd.to_datetime(signals["datetime"], utc=True)
    signals = ensure_signal_columns(signals)
    signals.sort_values("datetime", inplace=True)

    # ── Load bars (UTC) ───────────────────────────────────────────
    bars = load_bars()

    # ── Optional ATR merge (15m Wilder) ──────────────────────────
    if cfg.get("dynamic_stops", False):
        atr_period = int(cfg.get("atr_period", 14))
        attach_wilder_atr_15m(signals, bars, atr_period)

    # ── Run back-test ─────────────────────────────────────────────
    bt = run_backtest(signals, bars, cfg)
    if bt.empty:
        print("Back-test produced zero trades. Check signal logic or filters.")
        return

    bt.to_csv("bt_results_long.csv", index=False)

    # ── Stats (filled-trade centric) ─────────────────────────────
    total_signals = len(bt)
    filled = bt[bt["filled"] == True]
    fills = len(filled)

    wins = (filled.pnl > 0).sum()
    losses = (filled.pnl < 0).sum()
    breakeven = (filled.pnl == 0).sum()

    gross_win = filled.loc[filled.pnl > 0, "pnl"].sum()
    gross_loss = -filled.loc[filled.pnl < 0, "pnl"].sum()
    pf = gross_win / gross_loss if gross_loss else float("inf")

    final_bal = bt.balance_after.iloc[-1]

    eq = bt.balance_after
    drawdown = eq.cummax() - eq
    max_dd = drawdown.max()
    max_dd_base = eq.loc[drawdown.idxmax()] if not eq.empty else 1.0
    max_dd_pct = (max_dd / max_dd_base) * 100 if max_dd_base else 0.0

    sharpe = filled.pnl.mean() / filled.pnl.std() if filled.pnl.std() else float("inf")
    fill_rate = (fills / total_signals * 100) if total_signals else 0.0

    # Optional diagnostics if engine provided audit fields
    trig = str(cfg.get("entry_trigger", "band"))
    thr = float(cfg.get("band_entry_threshold", 0.0))
    fill_policy = str(cfg.get("band_fill_price", "threshold"))
    extra = ""
    if {"entry_price_used", "trigger_price"}.issubset(bt.columns):
        fill_to_trigger = (filled["entry_price_used"] - filled["trigger_price"]).abs()
        avg_fill_slip = fill_to_trigger.mean() if not fill_to_trigger.empty else 0.0
        extra = f"\nAvg |entry_price - trigger_price| : ${avg_fill_slip:.2f}"

    print("=" * 54)
    print("Back-test complete (engine v10.1)")
    print(f"Signals total        : {total_signals}")
    print(f"Filled trades        : {fills}  (Fill rate: {fill_rate:.1f}%)")
    print(f"W / L / BE           : {wins} / {losses} / {breakeven}")
    print(f"Profit Factor        : {pf:.2f}")
    print(f"Final Balance        : ${final_bal:,.2f}")
    print(f"Max Drawdown         : ${max_dd:,.2f}  ({max_dd_pct:.2f}%)")
    print(f"Avg filled P&L       : ${filled.pnl.mean():.2f}" if fills else "Avg filled P&L       : n/a")
    print(f"Naive Sharpe (filled): {sharpe:.2f}" if fills else "Naive Sharpe (filled): n/a")
    print(f"Entry trigger mode   : {trig}  |  threshold={thr:.2f}  |  price={fill_policy}{extra}")
    print("=" * 54)
    print("Detailed log → bt_results_long.csv")


if __name__ == "__main__":
    main()
