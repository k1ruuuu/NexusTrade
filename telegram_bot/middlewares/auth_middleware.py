from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.telegram.auth")


class WhitelistAuthMiddleware(BaseMiddleware):
    """
    Aiogram v3 authorization guard verifying incoming user IDs against
    TELEGRAM_ALLOWED_USER_IDS. Politely drops unauthorized commands.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        allowed_user_ids: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        raw_ids = (
            allowed_user_ids
            if allowed_user_ids is not None
            else self.settings.telegram_allowed_user_ids
        )
        self.allowed_ids: set[int] = {int(uid) for uid in raw_ids}

    def is_authorized(self, user_id: Optional[int]) -> bool:
        if user_id is None:
            return False
        if self.settings.telegram_open_access:
            return True
        return int(user_id) in self.allowed_ids or self.settings.is_admin_user(user_id)
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = getattr(event, "from_user", None)
        if user is None and "event_from_user" in data:
            user = data["event_from_user"]

        user_id = getattr(user, "id", None)
        if not self.is_authorized(user_id):
            logger.warning(
                "Blocked unauthorized Telegram user command",
                user_id=user_id,
                username=getattr(user, "username", None),
            )
            if isinstance(event, Message):
                await event.answer(
                    "🔒 <b>Akses Dibatasi:</b> Akun Telegram Anda tidak memiliki izin untuk mengakses terminal ini.",
                    parse_mode="HTML",
                )
            elif isinstance(event, CallbackQuery):
                await event.answer(
                    "Akses Dibatasi: ID Telegram Anda belum terdaftar.",
                    show_alert=True,
                )
            return None

        data["authorized"] = True
        data["user_role"] = "admin" if self.settings.is_admin_user(user_id) else "user"
        return await handler(event, data)
