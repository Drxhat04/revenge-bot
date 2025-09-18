# telegram_notifier.py
from __future__ import annotations
import os
import time
import requests
from typing import Optional, Sequence, Tuple, Dict, Any
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ========== Low-level Bot API client with simple retries ==========
class _TGClient:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, *, timeout: int = 10):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        if not self.token or not self.chat_id:
            raise RuntimeError("Telegram: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        self.base = f"https://api.telegram.org/bot{self.token}"
        self.timeout = timeout

    def _post(self, method: str, data: Dict[str, Any], files=None) -> Dict[str, Any]:
        url = f"{self.base}/{method}"
        for attempt in range(3):
            try:
                resp = requests.post(url, data=data, files=files, timeout=self.timeout)
                if resp.status_code == 429:
                    retry_after = 1
                    try:
                        retry_after = int(resp.json().get("parameters", {}).get("retry_after", 1))
                    except Exception:
                        pass
                    time.sleep(retry_after + 0.5)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                if not payload.get("ok", False):
                    raise RuntimeError(f"Telegram API error: {payload}")
                return payload
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 + attempt * 0.5)
        raise RuntimeError("Telegram API call failed after retries")

    def send_message(self, text: str, *, parse_mode: str = "HTML", disable_preview: bool = True) -> Dict[str, Any]:
        return self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": "true" if disable_preview else "false",
            },
        )

    def edit_message_text(self, message_id: int, text: str, *, parse_mode: str = "HTML", disable_preview: bool = True) -> Dict[str, Any]:
        return self._post(
            "editMessageText",
            {
                "chat_id": self.chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": "true" if disable_preview else "false",
            },
        )

    def send_photo(self, image_path: str, caption: Optional[str] = None, *, parse_mode: str = "HTML") -> Dict[str, Any]:
        with open(image_path, "rb") as f:
            return self._post(
                "sendPhoto",
                {"chat_id": self.chat_id, "caption": caption or "", "parse_mode": parse_mode},
                files={"photo": f},
            )

# ========== High-level notifier that formats messages ==========
class TelegramNotifier:
    def __init__(self, *, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.client = _TGClient(bot_token, chat_id)

    # ----- formatting helpers -----
    @staticmethod
    def _fmt_header(symbol: str, side: str) -> str:
        return f"{symbol.upper()} {side.upper()} SIGNAL"

    @staticmethod
    def _fmt_entry(entry_zone: Tuple[float, float] | float) -> str:
        if isinstance(entry_zone, tuple):
            lo, hi = entry_zone
            return f"Entry Zone: {lo:g} – {hi:g}"
        return f"Entry: {float(entry_zone):g}"

    @staticmethod
    def _fmt_sl(sl: float) -> str:
        return f"Stop Loss (SL): {sl:g}"

    @staticmethod
    def _fmt_tps(tps: Sequence[float]) -> str:
        lines = ["\nTake Profit Targets:"]
        for i, v in enumerate(tps, start=1):
            lines.append(f"TP{i}: {float(v):g}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_footer() -> str:
        return "\n\nEnsure strict risk management and always protect your capital."

    @staticmethod
    def _fmt_status_block(status_lines: Sequence[str]) -> str:
        if not status_lines:
            return ""
        return "\n\nStatus:\n" + "\n".join(f"• {line}" for line in status_lines)

    @staticmethod
    def _fmt_times(when_utc: datetime, *, local_tz: str = "Europe/Berlin") -> str:
        if when_utc.tzinfo is None:
            u = when_utc.replace(tzinfo=timezone.utc)
        else:
            u = when_utc.astimezone(timezone.utc)
        l = u.astimezone(ZoneInfo(local_tz))
        return f"Sent: {u.strftime('%Y-%m-%d %H:%M:%S UTC')} ({l.strftime('%Y-%m-%d %H:%M:%S %Z')})"

    @staticmethod
    def _fmt_price_block(price_now: float | None, trigger: float | None, delta: float | None) -> str:
        if price_now is None or trigger is None or delta is None:
            return ""
        # Keep compact single line
        return f"\nPrice now: {price_now:g}   Trigger: {trigger:g}   Δ: {delta:+.2f}"

    @staticmethod
    def build_signal_text(
        *,
        symbol: str,
        side: str,
        entry_zone,
        sl: float,
        tps: Sequence[float] | None,
        when_utc: datetime,
        price_now: float | None = None,
        trigger: float | None = None,
        delta: float | None = None,
        extra_status: Sequence[str] | None = None
    ) -> str:
        parts = [
            TelegramNotifier._fmt_header(symbol, side),
            "",
            TelegramNotifier._fmt_times(when_utc),
            TelegramNotifier._fmt_entry(entry_zone),
            TelegramNotifier._fmt_sl(sl),
        ]
        if tps and len(tps) > 0:
            parts.append(TelegramNotifier._fmt_tps(tps))
        pb = TelegramNotifier._fmt_price_block(price_now, trigger, delta)
        if pb:
            parts.append(pb)
        parts.append(TelegramNotifier._fmt_footer())
        base = "\n".join(parts)
        if extra_status:
            base += TelegramNotifier._fmt_status_block(extra_status)
        return base

    def post_signal(
        self,
        *,
        symbol: str,
        side: str,
        entry_zone,
        sl: float,
        tps: Sequence[float] | None = None,
        when_utc: datetime,
        price_now: float | None = None,
        trigger: float | None = None,
        delta: float | None = None,
        initial_status: Sequence[str] | None = ("Sending…",)
    ) -> tuple[int, str]:
        base_text = self.build_signal_text(
            symbol=symbol,
            side=side,
            entry_zone=entry_zone,
            sl=sl,
            tps=tps,
            when_utc=when_utc,
            price_now=price_now,
            trigger=trigger,
            delta=delta,
            extra_status=initial_status
        )
        res = self.client.send_message(base_text)
        mid = int(res["result"]["message_id"])
        return mid, base_text

    def _edit_with_status(self, message_id: int, base_text: str, status_lines: Sequence[str]) -> None:
        new_text = base_text + self._fmt_status_block(status_lines)
        self.client.edit_message_text(message_id, new_text)

    # ----- status updaters -----
    def mark_placed(self, message_id: int, base_text: str, *, sent_price: float, fill: str) -> None:
        self._edit_with_status(message_id, base_text, [f"Placed @ {sent_price:g} ({fill})"])

    def mark_retry(self, message_id: int, base_text: str, *, attempt: int, retcode: Any, last_price: Optional[float] = None) -> None:
        line = f"Retry {attempt} (retcode={retcode})"
        if last_price is not None:
            line += f" @ {last_price:g}"
        self._edit_with_status(message_id, base_text, [line])

    def mark_failed(self, message_id: int, base_text: str, *, retcode: Any) -> None:
        self._edit_with_status(message_id, base_text, [f"Failed (retcode={retcode})"])

    def mark_tp_hit(self, message_id: int, base_text: str, *, tp_idx: int, hit_price: float) -> None:
        self._edit_with_status(message_id, base_text, [f"TP{tp_idx} hit @ {hit_price:g}"])

    def mark_sl_hit(self, message_id: int, base_text: str, *, hit_price: float) -> None:
        self._edit_with_status(message_id, base_text, [f"SL hit @ {hit_price:g}"])

    def mark_partial_close(self, message_id: int, base_text: str, *, lots_closed: float, exit_price: float) -> None:
        self._edit_with_status(message_id, base_text, [f"Partial close {lots_closed:g} lots @ {exit_price:g}"])

    def mark_closed(self, message_id: int, base_text: str, *, reason: str, exit_price: float) -> None:
        self._edit_with_status(message_id, base_text, [f"Closed ({reason}) @ {exit_price:g}"])

    def send_chart(self, image_path: str, caption: Optional[str] = None) -> None:
        self.client.send_photo(image_path, caption=caption or "Entry context")
