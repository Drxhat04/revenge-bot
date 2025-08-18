# config.py
import yaml
from pathlib import Path

DEFAULTS = {
    "symbol": "XAUUSD",

    # Session (UTC)
    "sessions": ["06:00-09:00", "10:00-14:00"],

    # Asia session used by the scanner (UTC)
    "asia_start": "18:00",
    "asia_end":   "09:00",
    "tokyo_only": False,

    # Sweep / zone geometry
    "zone_width_usd": 5.0,
    "sweep_buffer_usd": 0.1,
    "require_pierce": False,
    "require_fvg": True,
    "gap_filter": 0.005,

    # Risk & targets
    "risk_mult": 2.3,
    "tp1_mult": 0.68,
    "tp2_mult": 1.16,
    "risk_per_trade": 0.10,      # note: very aggressive; override as needed
    "max_risk_ratio": 0.03,

    # Volatility controls
    "volatility_adjust": True,
    "atr_period": 14,
    "atr_multiplier": 3.0,
    "min_stop_usd": 1.0,

    # Timeouts
    "no_touch_timeout_hours": 3,
    "force_close_hours": 24,

    # Entry logic (engine v10+)
    # "band" → trade when price reaches a threshold inside the chosen half-zone
    # "mid"  → require midpoint touch
    "entry_trigger": "band",
    "band_side_buy": "upper",        # "upper"=[mid,high], "lower"=[low,mid]
    "band_side_sell": "lower",       # "lower"=[low,mid],  "upper"=[mid,high]
    "band_entry_threshold": 0.00,    # 0.00=at mid edge, 0.50=halfway, 1.00=far edge
    "band_fill_price": "threshold",  # "threshold" or "touch"
    "entry_min_delay_minutes": 1,    # prevent same-minute fills

    # Lot sizing / leverage
    "min_lot_size": 0.01,
    "max_lots": 2.3,
    "lot_size_increment": 0.01,
    "leverage": 1000.0,
    "dollar_per_unit_per_lot": 100.0,  # 1 USD move × 100 oz = $100 per 1 lot

    # Dealing costs & execution realism
    "tick_size": 0.0,                    # 0 if using zero-spread CSVs without points
    "use_csv_spread": False,             # set True if your CSV has a 'spread' column
    "commission_per_lot_per_side": 5.5,  # $ per lot per side
    "slip_usd_side_lot": 5.0,            # per side per lot
    "slippage_scale_threshold_lots": 1.5,
    "slippage_scale_factor": 1.0,

    # Swap management
    "swap_cutoff_hour": 19,      # UTC
    "swap_buffer_hours": 3,
    "swap_avoidance": False,
    "swap_fee_long": -5.10,
    "swap_fee_short": 0.0,
    "wednesday_multiplier": 3,

    # Risk controls
    "day_loss_limit_pct": 0.05,

    # Capital
    "start_balance": 10000.0,

    # MT5 metadata (if applicable)
    "deviation": 20,
    "magic": 234000,
}

def load_config(path: str = "config.yaml") -> dict:
    cfg = DEFAULTS.copy()
    if Path(path).exists():
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
            cfg.update(user_cfg)
    # Alias for lot cap used by engine
    cfg["lot_cap"] = cfg.get("max_lots", DEFAULTS["max_lots"])
    return cfg

# Global config cache
CONFIG = load_config()
