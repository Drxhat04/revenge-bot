# ────────────────────────────────────────────────────────────────
# backtest_engine.py · v4
# Pure‑Python simulator for the XAUUSD sweep / OB strategy
#
#  • main   = L lots   → exits at TP‑1
#  • runner = 0.5 L    → exits at TP‑2 / BE / force_close
#
# All tunables live in CFG (merge with cfg.update as before).
# ────────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import timedelta
import pandas as pd

# ╭─ DEFAULT PARAMS ─────────────────────────────────────────────╮
CFG: dict = {
    # geometry
    "risk_mult": 2.3, "tp1_mult": 0.68, "tp2_mult": 1.16,
    # timing
    "no_touch_timeout_hours": 3, "force_close_hours": 5,
    # costs
    "tick_size": 0.01, "slippage_usd_side_lot": 3.52,
    "dollar_per_unit_per_lot": 100.0,
    "tp_before_sl_same_bar": True,
    # lot ladder  (balance bands → base lot)
    "lot_table": [
        (0, 50, 0.02), (50, 100, 0.02), (100, 200, 0.05),
        (200, 500, 0.10), (500, 1_000, 0.25), (1_000, 5_000, 0.50),
        (5_000, 1e12, 1.00),
    ],
    "start_balance": 100.0,
}
# ╰──────────────────────────────────────────────────────────────╯


# ╭─ PRICE HELPERS ──────────────────────────────────────────────╮
def ask_price(bar) -> float:      # BUY pays this, SELL exits here
    return bar.close + (bar.spread / 2 if "spread" in bar else 0.0)


def bid_price(bar) -> float:      # SELL receives this, BUY exits here
    return bar.close - (bar.spread / 2 if "spread" in bar else 0.0)
# ╰──────────────────────────────────────────────────────────────╯


# ╭─ LOT SIZING ─────────────────────────────────────────────────╮
def lot_from_balance(balance: float,
                     table: list[tuple[float, float, float]]) -> float:
    """Step‑wise lot table."""
    for low, high, lot in table:
        if float(low) < balance <= float(high):
            return float(lot)
    return float(table[-1][2])
# ╰──────────────────────────────────────────────────────────────╯


# ╭─ DEAL COSTS (spread + slippage) ─────────────────────────────╮
def pts_to_price(points: float, tick: float) -> float:
    return points * tick


def deal_cost(bars: pd.DataFrame,
              entry_ts,
              exit_ts,
              lots: float,
              cfg: dict) -> float:
    """
    Round‑turn cost: spread (entry + exit) + per‑side slippage.
    `lots` is the total round‑turn volume (we pass 1.5 × L).
    """
    if "spread" not in bars.columns or cfg["tick_size"] == 0:
        spr_in = spr_out = 0.0
    else:
        spr_in  = pts_to_price(
            bars[bars.time >= entry_ts].iloc[0].spread, cfg["tick_size"]
        )
        out_bar = bars[bars.time == exit_ts]
        spr_out = pts_to_price(
            out_bar.spread.iloc[0], cfg["tick_size"]
        ) if not out_bar.empty else spr_in

    usd_spread = (spr_in + spr_out) * cfg["dollar_per_unit_per_lot"] * lots
    usd_slip   = cfg["slippage_usd_side_lot"] * 2 * lots
    return usd_spread + usd_slip
# ╰──────────────────────────────────────────────────────────────╯


# ╭─ TRADE SIMULATOR (per signal) ───────────────────────────────╮
def simulate_trade(sig: pd.Series,
                   bars: pd.DataFrame,
                   cfg: dict) -> dict:
    """
    Returns a dict with fill / P&L info *per **base lot L***.
    """
    if pd.isna(sig.entry_mid) or pd.isna(sig.zone_width):
        return {"filled": False, "reason": "invalid_signal"}

    # true cash entry at bid/ask of the first bar that touches the zone
    first_bar = bars[bars.time >= sig.datetime].iloc[0]
    entry = ask_price(first_bar) if sig.direction == "BUY" else bid_price(first_bar)

    risk     = cfg["risk_mult"] * sig.zone_width          # still sized on mid
    timeout  = sig.datetime + timedelta(hours=cfg["no_touch_timeout_hours"])
    hardcut  = sig.datetime + timedelta(days=5)

    # direction‑specific levels and test lambdas (always bid/ask aware!)
    if sig.direction == "BUY":
        sl, tp1, tp2 = (entry - risk,
                        entry + cfg["tp1_mult"]*risk,
                        entry + cfg["tp2_mult"]*risk)
        hit_sl  = lambda b: bid_price(b) <= sl
        hit_tp1 = lambda b: ask_price(b) >= tp1
        hit_tp2 = lambda b: ask_price(b) >= tp2
        sign = 1
    else:  # SELL
        sl, tp1, tp2 = (entry + risk,
                        entry - cfg["tp1_mult"]*risk,
                        entry - cfg["tp2_mult"]*risk)
        hit_sl  = lambda b: ask_price(b) >= sl
        hit_tp1 = lambda b: bid_price(b) <= tp1
        hit_tp2 = lambda b: bid_price(b) <= tp2
        sign = -1

    look = bars[(bars.time >= sig.datetime) & (bars.time <= hardcut)]
    if look.empty:
        return {"filled": False, "reason": "no_data"}

    # ── wait for touch / fill ───────────────────────────────────
    touch_idx = None
    for idx, bar in enumerate(look.itertuples()):
        if bar.low <= sig.entry_mid <= bar.high:
            touch_idx = idx
            break
        if bar.time >= timeout:
            return {"filled": False, "reason": "timeout"}

    if touch_idx is None:
        return {"filled": False, "reason": "no_fill"}

    after = look.iloc[touch_idx:]

    # ── two‑leg bookkeeping ────────────────────────────────────
    main_units, runner_units = 1.0, 0.5
    open_units = main_units + runner_units
    pnl_raw = 0.0
    tp1_hit = tp2_hit = False
    exit_time = exit_reason = None
    be_level = entry  # break‑even for runner

    for bar in after.itertuples():

        # life‑time stop
        if (bar.time - after.iloc[0].time) > timedelta(hours=cfg["force_close_hours"]):
            px = bid_price(bar) if sig.direction == "BUY" else ask_price(bar)
            pnl_raw += sign * (px - entry) * open_units * cfg["dollar_per_unit_per_lot"]
            exit_time, exit_reason = bar.time, "force_close"
            break

        if cfg["tp_before_sl_same_bar"]:
            # ---- TP‑1 ---------------------------------------------------
            if not tp1_hit and hit_tp1(bar):
                tp1_hit = True
                pnl_raw += sign * (tp1 - entry) * main_units * cfg["dollar_per_unit_per_lot"]
                open_units -= main_units          # only runner left
                sl = be_level                    # SL to BE

                # same‑bar TP‑2 or BE checks
                if hit_tp2(bar):
                    tp2_hit = True
                    pnl_raw += sign * (tp2 - entry) * runner_units * cfg["dollar_per_unit_per_lot"]
                    exit_time, exit_reason = bar.time, "tp2"
                    break
                if hit_sl(bar):
                    exit_time, exit_reason = bar.time, "be"
                    break
                continue

            # ---- SL before TP‑1 ---------------------------------------
            if not tp1_hit and hit_sl(bar):
                pnl_raw += sign * (sl - entry) * open_units * cfg["dollar_per_unit_per_lot"]
                exit_time, exit_reason = bar.time, "sl"
                break

            # ---- runner management ------------------------------------
            if tp1_hit:
                if hit_tp2(bar):
                    tp2_hit = True
                    pnl_raw += sign * (tp2 - entry) * runner_units * cfg["dollar_per_unit_per_lot"]
                    exit_time, exit_reason = bar.time, "tp2"
                    break
                if hit_sl(bar):
                    exit_time, exit_reason = bar.time, "be"
                    break
        # (An SL‑first logic block could be inserted in the `else`.)

    # still open at slice end → mark to mkt & close
    if exit_time is None:
        last = after.iloc[-1]
        px   = bid_price(last) if sig.direction == "BUY" else ask_price(last)
        pnl_raw += sign * (px - entry) * open_units * cfg["dollar_per_unit_per_lot"]
        exit_time, exit_reason = last.time, "final_close"

    return dict(
        filled=True, exit_time=exit_time, exit_reason=exit_reason,
        pnl_raw=pnl_raw, tp1_hit=tp1_hit, tp2_hit=tp2_hit
    )
# ╰──────────────────────────────────────────────────────────────╯


# ╭─ PORTFOLIO BACK‑TEST LOOP ───────────────────────────────────╮
def run_backtest(signals: pd.DataFrame,
                 bars: pd.DataFrame,
                 cfg: dict,
                 *,
                 start_balance: float | None = None,
                 trial_no: int | None = None) -> pd.DataFrame:
    """
    Returns a trade ledger (one row per signal).
    `signals` may contain an optional *risk_mult* (or risk_mult_sig) column
    produced by stress‑tests to down‑size / partial‑fill trades.
    """
    if start_balance is None:
        start_balance = cfg["start_balance"]

    balance = start_balance
    rows    = []

    # use itertuples → faster & gives attribute access
    for sig_idx, sig in enumerate(signals.itertuples()):

        res = simulate_trade(sig, bars, cfg)

        row = {
            "signal_time"   : sig.datetime,
            "direction"     : sig.direction,
            "entry_mid"     : sig.entry_mid,
            "balance_before": balance,
            **{k: res.get(k) for k in
               ("exit_time", "exit_reason", "tp1_hit", "tp2_hit", "filled")},
        }

        if res.get("filled"):
            # optional per‑signal sizing from stress‑tests
            lot_factor = getattr(sig, "risk_mult_sig",
                         getattr(sig, "risk_mult", 1.0))
            base_lot   = round(lot_from_balance(balance, cfg["lot_table"])
                               * lot_factor, 3)
            total_lots = base_lot * 1.5      # L  +  0.5 L

            cost = deal_cost(bars, sig.datetime, res["exit_time"],
                             total_lots, cfg)
            pnl  = res["pnl_raw"] * base_lot - cost
            balance += pnl

            # ── DEBUG DUMP (first trade, first trial only) ─────────────
            if trial_no == 1 and sig_idx == 0:
                print(
                    f"[debug] raw={res['pnl_raw'] * base_lot:8.2f}  "
                    f"cost={cost:8.2f}  "
                    f"pnl={pnl:8.2f}  "
                    f"spread_pts="
                    f"{bars[bars.time >= sig.datetime].iloc[0].spread}"
                )

            row.update(
                lot_main   = base_lot,
                lot_runner = round(base_lot * 0.5, 3),
                pnl        = pnl,
            )
        else:
            row.update(lot_main=0.0, lot_runner=0.0, pnl=0.0)

        row["balance_after"] = balance
        rows.append(row)

    return pd.DataFrame(rows)
# ╰──────────────────────────────────────────────────────────────╯
