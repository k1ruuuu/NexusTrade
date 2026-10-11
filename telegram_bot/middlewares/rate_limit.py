from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable, Optional

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.telegram.rate_limit")


class RateLimitMiddleware(BaseMiddleware):
    """
    Sliding-window anti-spam rate limiter for Aiogram v3 handlers.
    Limits each user to `max_calls` within `period_seconds`.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        max_calls: Optional[int] = None,
        period_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.max_calls = max_calls or self.settings.telegram_rate_limit_calls
        self.period_seconds = period_seconds or self.settings.telegram_rate_limit_period
        self._clock = clock
        self._user_timestamps: dict[int, deque[float]] = defaultdict(deque)

    def check_and_record(self, user_id: int) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = self._clock()
        history = self._user_timestamps[user_id]
        cutoff = now - self.period_seconds

        while history and history[0] <= cutoff:
            history.popleft()

        if len(history) >= self.max_calls:
            retry_after = max(0.1, self.period_seconds - (now - history[0]))
            return False, round(retry_after, 1)

        history.append(now)
        return True, 0.0

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = getattr(event, "from_user", None)
        user_id = getattr(user, "id", 0) or 0

        allowed, retry_after = self.check_and_record(int(user_id))
        if not allowed:
            logger.warning(
                "Throttled Telegram user request",
                user_id=user_id,
                retry_after=retry_after,
            )
            if isinstance(event, Message):
                await event.answer(
                    f"⏳ <b>Terlalu Cepat:</b> Mohon tunggu <code>{retry_after:.1f} detik</code> sebelum mengirim perintah berikutnya.",
                    parse_mode="HTML",
                )
            elif isinstance(event, CallbackQuery):
                await event.answer(
                    f"Mohon tunggu {retry_after:.1f} detik sebelum menekan tombol lagi.",
                    show_alert=False,
                )
            return None

        return await handler(event, data)
