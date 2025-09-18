#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────
# run_backtest_long.py · v10.5 (UTC-native, engine v10.5 compatible)
#   • load config
#   • load signals_long_utc.csv (dedup exact duplicates)
#   • optional prefilter by sessions (mirror live gating)
#   • attach 15m Wilder ATR to signals (if dynamic_stops)
#   • load M1 bars in UTC (dedup, skip weekends)
#   • CROP bars to the smallest window needed (respects force_close+grace+max_ext)
#   • call run_backtest() and write bt_results_long.csv
#   • print expanded filled-trade stats + single-target + overtime diagnostics
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


# ── Session gating (mirror live) ───────────────────────────────
def _parse_sessions(cfg: dict) -> list[tuple[int, int]]:
    """
    Return list of (start_minute, end_minute) in UTC minutes since midnight.
    Supports windows that cross midnight.
    """
    out: list[tuple[int, int]] = []
    for win in cfg.get("sessions", []):
        s, e = [t.strip() for t in str(win).split("-")]
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        out.append((sh * 60 + sm, eh * 60 + em))
    return out

def _in_sessions(ts: pd.Timestamp, sess_windows: list[tuple[int, int]]) -> bool:
    if not sess_windows:
        return True
    m = ts.hour * 60 + ts.minute
    for s, e in sess_windows:
        if s <= e:
            if s <= m <= e:
                return True
        else:
            # crosses midnight (e.g., 18:00-06:00)
            if m >= s or m <= e:
                return True
    return False


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

    raw_count = len(signals)

    # Hard de-dup identical rows (fixes duplicated lines in your export)
    signals = signals.drop_duplicates(
        subset=["datetime", "direction", "entry_low", "entry_high"],
        keep="first",
    ).copy()
    after_exact_dedup = len(signals)

    # Optional: mirror live session gating before BT for realism
    sess_windows = _parse_sessions(cfg) if cfg.get("enforce_session_gating_in_backtest", True) else []
    if sess_windows:
        signals = signals[signals["datetime"].apply(lambda ts: _in_sessions(ts, sess_windows))].copy()

    after_session_filter = len(signals)

    # Sort final signals
    signals.sort_values("datetime", inplace=True)
    signals.reset_index(drop=True, inplace=True)

    # ── Load bars (UTC) ───────────────────────────────────────────
    bars = load_bars()

    # ── Crop bars to smallest needed window (speed-up + consistency) ──
    sig_min = signals["datetime"].min()
    sig_max = signals["datetime"].max()

    no_touch_hours = float(cfg.get("no_touch_timeout_hours", 0.0) or 0.0)
    force_close_hours = float(cfg.get("force_close_hours", 0.0) or 0.0)
    grace_hours = float(cfg.get("grace_hours", 0.0) or 0.0)
    policy = str(cfg.get("force_close_policy", "final_close")).lower()
    # if policy is tp1/tp2 we may extend; otherwise extension is 0
    default_ext = 0.0 if policy == "final_close" else 0.0
    max_extension_hours = cfg.get("max_extension_hours", default_ext)
    # interpret None as "to end of dataset"
    if max_extension_hours is None:
        ext_hours = (bars["time"].max() - sig_max).total_seconds() / 3600.0
        ext_hours = max(ext_hours, 0.0)
    else:
        ext_hours = float(max_extension_hours)

    buf_hours = 6.0  # small safety buffer

    bars_start = sig_min - pd.Timedelta(hours=no_touch_hours + buf_hours)
    bars_end_candidate = sig_max + pd.Timedelta(hours=force_close_hours + grace_hours + ext_hours + buf_hours)
    # never exceed available data
    bars_end = min(bars["time"].max(), bars_end_candidate)

    bars = bars[(bars["time"] >= bars_start) & (bars["time"] <= bars_end)].copy()
    print(f"Bars after crop: {len(bars):,}  {bars['time'].min()} → {bars['time'].max()}")

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

    # Overtime policy info from cfg
    force_close_policy = str(cfg.get("force_close_policy", "final_close")).lower()
    max_extension_hours = cfg.get("max_extension_hours", None)
    grace_hours = cfg.get("grace_hours", 0)
    overtime_sl_mode = str(cfg.get("overtime_sl_mode", "none")).lower()

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

    sharpe = filled.pnl.mean() / filled.pnl.std() if fills and filled.pnl.std() else float("inf")
    fill_rate = (fills / total_signals * 100) if total_signals else 0.0

    # TP hit counts & exit reasons
    tp1_hits = int(filled.get("tp1_hit", pd.Series(dtype=bool)).sum()) if "tp1_hit" in filled else 0
    tp2_hits = int(filled.get("tp2_hit", pd.Series(dtype=bool)).sum()) if "tp2_hit" in filled else 0
    tp3_hits = int(filled.get("tp3_hit", pd.Series(dtype=bool)).sum()) if "tp3_hit" in filled else 0
    tp4_hits = int(filled.get("tp4_hit", pd.Series(dtype=bool)).sum()) if "tp4_hit" in filled else 0

    exit_ct = filled.get("exit_reason", pd.Series(dtype=str)).value_counts(dropna=False)

    # Costs
    avg_commission = filled.get("commission", pd.Series(dtype=float)).mean() if "commission" in filled and fills else 0.0
    avg_slip_spread = filled.get("slippage_spread", pd.Series(dtype=float)).mean() if "slippage_spread" in filled and fills else 0.0
    avg_swap = filled.get("swap_cost", pd.Series(dtype=float)).mean() if "swap_cost" in filled and fills else 0.0

    # Optional diagnostics if engine provided audit fields
    trig = str(cfg.get("entry_trigger", "band"))
    thr = float(cfg.get("band_entry_threshold", 0.0))
    fill_policy = str(cfg.get("band_fill_price", "threshold"))
    extra = ""
    if {"entry_price_used", "trigger_price"}.issubset(bt.columns):
        fill_to_trigger = (filled["entry_price_used"] - filled["trigger_price"]).abs()
        avg_fill_slip = fill_to_trigger.mean() if fills else 0.0
        extra = f"\nAvg |entry_price - trigger_price| : ${avg_fill_slip:.2f}"

    # Management/units overview from cfg (for quick audit)
    be_after_tp1 = bool(cfg.get("be_after_tp1", True))
    use_tp3 = bool(cfg.get("use_tp3", False))
    use_tp4 = bool(cfg.get("use_tp4", False))
    umain = float(cfg.get("main_units", 1.0))
    urun = float(cfg.get("runner_units", 0.5))
    uextra = float(cfg.get("extra_runner_units", 0.0))
    tp1_mult = float(cfg.get("tp1_mult", 0.68))
    tp2_mult = float(cfg.get("tp2_mult", 1.16))
    tp3_mult = float(cfg.get("tp3_mult", 1.80))
    tp4_mult = float(cfg.get("tp4_mult", 2.40))

    # Single-target mode summary (purely informational)
    st_on = bool(cfg.get("force_single_entry", False))
    st_tp = cfg.get("single_entry_target", None)
    st_text = "off"
    if st_on and st_tp in (1, 2, 3, 4):
        st_text = f"ON → TP{st_tp} only"

    # Engine early-stop reason (e.g., margin stop)
    early_stop_reason = getattr(bt, "attrs", {}).get("early_stop_reason")

    # Overtime diagnostics (count how often overtime outcomes happened)
    overtime_reasons = ("force_tp1", "force_tp2", "final_close_timeout", "force_tp1_eod", "force_tp2_eod")
    overtime_count = int(filled["exit_reason"].isin(overtime_reasons).sum()) if "exit_reason" in filled else 0
    overtime_pct = (overtime_count / fills * 100.0) if fills else 0.0

    print("=" * 60)
    print("Back-test complete (runner v10.5; engine v10.5 compatible)")
    print(f"Signals (raw)        : {raw_count}")
    print(f" - after exact de-dup: {after_exact_dedup}")
    if sess_windows:
        print(f" - after sessions    : {after_session_filter}")
    print(f"BT rows (signals)    : {total_signals}")
    print(f"Filled trades        : {fills}  (Fill rate: {fill_rate:.1f}%)")
    print(f"W / L / BE           : {wins} / {losses} / {breakeven}")
    print(f"Profit Factor        : {pf:.2f}")
    print(f"Final Balance        : ${final_bal:,.2f}")
    print(f"Max Drawdown         : ${max_dd:,.2f}  ({max_dd_pct:.2f}%)")
    print(f"Naive Sharpe (filled): {sharpe:.2f}" if fills else "Naive Sharpe (filled): n/a")
    print(f"Entry trigger mode   : {trig}  |  threshold={thr:.2f}  |  price={fill_policy}{extra}")
    print(f"Single-target mode   : {st_text}")

    # Overtime policy printout (now includes grace + SL mode)
    me = "∞" if max_extension_hours in (None, "None") else str(max_extension_hours)
    print(f"Overtime policy      : {force_close_policy}  |  grace_hours={grace_hours}  |  sl_mode={overtime_sl_mode}  |  max_extension_hours={me}")
    if fills:
        print(f"Overtime usage       : {overtime_count} trades ({overtime_pct:.1f}%)")

    if early_stop_reason:
        print(f"Engine early stop    : {early_stop_reason}")

    print("-" * 60)
    print(f"TP hits              : TP1={tp1_hits}  |  TP2={tp2_hits}  |  TP3={tp3_hits}  |  TP4={tp4_hits}")
    print("Exit reasons         :")
    if not exit_ct.empty:
        for reason, cnt in exit_ct.items():
            print(f"  • {reason:>14}: {cnt}")
    else:
        print("  • (none)")
    print("-" * 60)
    print(f"Avg commission/trade : ${avg_commission:.2f}")
    print(f"Avg spread+slippage  : ${avg_slip_spread:.2f}")
    print(f"Avg swap/trade       : ${avg_swap:.2f}")
    print("-" * 60)
    print(f"Units (TP1/TP2/Extra): {umain} / {urun} / {uextra}  |  BE after TP1: {be_after_tp1}")
    print(f"TP multipliers       : TP1={tp1_mult:.2f}  TP2={tp2_mult:.2f}  "
          f"TP3={tp3_mult:.2f}({use_tp3})  TP4={tp4_mult:.2f}({use_tp4})")
    print("=" * 60)
    print("Detailed log → bt_results_long.csv")


if __name__ == "__main__":
    main()
