from __future__ import annotations
import os, json, threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any
from config import CONFIG

# Capture import error so we can report it from main.py
_BASE_IMPORT_ERROR: Exception | None = None
try:
    from telegram_notifier import TelegramNotifier as _BaseTN
except Exception as e:
    _BASE_IMPORT_ERROR = e
    _BaseTN = None  # importing failed

_STORE_PATH = CONFIG.get("telegram_store_file", "telegram_message_map.json")

import math

# ---------- local helpers (no dependency on main.py) ----------
def _coerce_ts_utc(ts) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        if not isinstance(t, datetime):
            return None
        if t.tzinfo is None:
            # Treat naive as UTC already
            return t.replace(tzinfo=timezone.utc).astimezone(timezone.utc).replace(tzinfo=None)
        return t.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None

def _is_fresh(ts_utc, *, freshness_minutes: int | None = None) -> bool:
    fm = int(CONFIG.get("freshness_minutes", 45)) if freshness_minutes is None else int(freshness_minutes)
    if fm <= 0:
        return True
    sig = _coerce_ts_utc(ts_utc)
    if sig is None:
        return False
    now = datetime.utcnow()
    return (now - sig) <= timedelta(minutes=fm)

def _is_near_trigger(direction: str,
                     bid: float,
                     ask: float,
                     trigger: float,
                     band_lo: float,
                     band_hi: float,
                     *,
                     trigger_tolerance_usd: float | None = None,
                     allow_outside_band_tolerance: float | None = None) -> bool:
    tol = float(CONFIG.get("trigger_tolerance_usd", 0.02)) if trigger_tolerance_usd is None else float(trigger_tolerance_usd)
    band_tol = float(CONFIG.get("allow_outside_band_tolerance", 0.00)) if allow_outside_band_tolerance is None else float(allow_outside_band_tolerance)
    px = float(ask if str(direction).upper() == "BUY" else bid)
    lo = float(min(band_lo, band_hi))
    hi = float(max(band_lo, band_hi))
    if not (lo - band_tol <= px <= hi + band_tol):
        return False
    return abs(px - float(trigger)) <= tol

_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "tickets": {},   # str(ticket) -> {"mid": int, "base": str}
    "pending": {},   # key -> {"mid": int, "base": str}
}

def _load() -> None:
    global _STATE
    if os.path.exists(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                _STATE = json.load(f)
        except Exception:
            pass

def _save() -> None:
    try:
        with open(_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(_STATE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_load()

def _sig_key(symbol: str, side: str, lo: float, hi: float, when_utc: datetime) -> str:
    t = when_utc.replace(tzinfo=timezone.utc).isoformat()
    return f"{symbol}:{side}:{round(lo,2)}:{round(hi,2)}:{t}"

class Notifier:
    """
    Safe, config-driven wrapper around TelegramNotifier.
    If disabled or misconfigured, methods become no-ops, but 'reason' explains why.
    """
    def __init__(self):
        tg_cfg = CONFIG.get("telegram", {}) or {}
        self.enabled: bool = False
        self.post_signals = self.post_orders = self.post_exits = False
        self.client = None
        self.reason: str | None = None

        # 1) feature flag
        if not bool(tg_cfg.get("enabled", True)):
            self.reason = "telegram.enabled is false in config"
            return

        # 2) import failure?
        if _BaseTN is None:
            self.reason = f"failed to import telegram_notifier: {_BASE_IMPORT_ERROR!r}"
            return

        # 3) init the client
        self.post_signals = bool(tg_cfg.get("post_signals", True))
        self.post_orders  = bool(tg_cfg.get("post_orders", True))
        self.post_exits   = bool(tg_cfg.get("post_exits", True))
        try:
            self.client = _BaseTN(
                bot_token=tg_cfg.get("bot_token"),
                chat_id=tg_cfg.get("chat_id"),
            )
            self.enabled = True
            self.reason = None
        except Exception as e:
            self.enabled = False
            self.reason = f"Telegram client init failed: {e!r}"

    # ---------- signal lifecycle ----------
    def post_signal(
        self,
        *,
        symbol: str,
        side: str,
        entry_zone,
        sl: float,
        tps=None,
        when_utc: datetime,
        # Optional live context for gating + display
        bid: float | None = None,
        ask: float | None = None,
        trigger: float | None = None,
        band_lo: float | None = None,
        band_hi: float | None = None,
        freshness_minutes: int | None = None,
        trigger_tolerance_usd: float | None = None,
        allow_outside_band_tolerance: float | None = None,
        force_post: bool = False,   # NEW: bypass near-trigger gating for announcements
    ):
        """
        Post a signal card. If live context is provided, we will:
          - suppress sending if not _is_fresh(when_utc) unless force_post is True
          - suppress sending if not _is_near_trigger(...) unless force_post is True
        Also prints Sent: UTC (Hamburg local) and Price now / Trigger / Δ.
        """
        if not (self.enabled and self.post_signals):
            return None

        # Freshness gate (unless forced)
        if not force_post:
            if not _is_fresh(when_utc, freshness_minutes=freshness_minutes):
                return None

        # Compute display price_now/trigger/delta if possible
        price_now = None
        delta = None
        near_ok = True
        if bid is not None and ask is not None and trigger is not None and band_lo is not None and band_hi is not None:
            price_now = float(ask if str(side).upper() == "BUY" else bid)
            delta = price_now - float(trigger)
            near_ok = _is_near_trigger(
                side, bid, ask, float(trigger), float(band_lo), float(band_hi),
                trigger_tolerance_usd=trigger_tolerance_usd,
                allow_outside_band_tolerance=allow_outside_band_tolerance
            )
            # Only enforce near_ok when not forced
            if not near_ok and not force_post:
                return None

        mid, base = self.client.post_signal(
            symbol=symbol,
            side=side,
            entry_zone=entry_zone,
            sl=sl,
            tps=tps,
            when_utc=when_utc,
            price_now=price_now,
            trigger=trigger,
            delta=delta,
        )
        key = _sig_key(
            symbol, side,
            entry_zone[0] if isinstance(entry_zone, tuple) else entry_zone,
            entry_zone[1] if isinstance(entry_zone, tuple) else entry_zone,
            when_utc
        )
        with _LOCK:
            _STATE["pending"][key] = {"mid": mid, "base": base}
            _save()
        return mid, base, key

    def promote_pending_to_ticket(self, key: str, ticket: int):
        if not self.enabled:
            return
        with _LOCK:
            rec = _STATE["pending"].pop(key, None)
            if rec:
                _STATE["tickets"][str(ticket)] = rec
                _save()

    def mark_from_pending(self, key: str, cb, *args, **kwargs):
        if not self.enabled:
            return
        with _LOCK:
            rec = _STATE["pending"].get(key)
        if rec:
            cb(rec["mid"], rec["base"], *args, **kwargs)

    # ---------- ticket-bound status updates ----------
    def _with_ticket(self, ticket: int, fn, *args, **kwargs):
        if not self.enabled:
            return
        with _LOCK:
            rec = _STATE["tickets"].get(str(ticket))
        if rec:
            fn(rec["mid"], rec["base"], *args, **kwargs)

    def mark_placed(self, ticket: int, *, sent_price: float, fill: str):
        if not (self.enabled and self.post_orders): return
        self._with_ticket(ticket, self.client.mark_placed, sent_price=sent_price, fill=fill)

    def mark_retry(self, ticket: int, *, attempt: int, retcode, last_price: float | None = None):
        if not (self.enabled and self.post_orders): return
        self._with_ticket(ticket, self.client.mark_retry, attempt=attempt, retcode=retcode, last_price=last_price)

    def mark_failed(self, ticket: int | None, *, retcode):
        if not (self.enabled and self.post_orders): return
        # ticket may be None — nothing to edit unless pending was promoted.

    def mark_tp_hit(self, ticket: int, *, tp_idx: int, hit_price: float):
        if not (self.enabled and self.post_exits): return
        self._with_ticket(ticket, self.client.mark_tp_hit, tp_idx=tp_idx, hit_price=hit_price)

    def mark_sl_hit(self, ticket: int, *, hit_price: float):
        if not (self.enabled and self.post_exits): return
        self._with_ticket(ticket, self.client.mark_sl_hit, hit_price=hit_price)

    def mark_partial(self, ticket: int, *, lots_closed: float, exit_price: float):
        if not (self.enabled and self.post_exits): return
        self._with_ticket(ticket, self.client.mark_partial_close, lots_closed=lots_closed, exit_price=exit_price)

    def mark_closed(self, ticket: int, *, reason: str, exit_price: float):
        if not (self.enabled and self.post_exits): return
        self._with_ticket(ticket, self.client.mark_closed, reason=reason, exit_price=exit_price)
