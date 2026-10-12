from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from ai_engine.auto_trader.admin_store import AdminStore
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from telegram_bot.handlers.handler_utils import _PENDING_SETUP_MODE

logger = get_logger("nexustrade.telegram.auth")


class WhitelistAuthMiddleware(BaseMiddleware):
    """
    Aiogram v3 authorization, maintenance guard, subscription expiry verifier,
    and automatic user activity audit logger.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        allowed_user_ids: Optional[list[int]] = None,
        admin_store: Optional[AdminStore] = None,
    ) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        raw_ids = (
            allowed_user_ids
            if allowed_user_ids is not None
            else self.settings.telegram_allowed_user_ids
        )
        self.allowed_ids: set[int] = {int(uid) for uid in raw_ids}
        self.admin_store = admin_store or AdminStore(self.settings)

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
        username = (
            getattr(user, "username", None)
            or getattr(user, "full_name", None)
            or f"user_{user_id}"
        )
        is_admin = self.settings.is_admin_user(user_id)
        role = "admin" if is_admin else "user"

        is_msg = isinstance(event, Message) or (
            hasattr(event, "text") and not hasattr(event, "data")
        )
        is_cb = isinstance(event, CallbackQuery) or hasattr(event, "data")

        if is_msg:
            event_type = "COMMAND" if (getattr(event, "text", "") or "").startswith("/") else "MESSAGE"
            action_str = str(getattr(event, "text", None) or "[non-text message]")
            stripped_act = action_str.strip()
            if (
                user_id is not None
                and int(user_id) in _PENDING_SETUP_MODE
                and not stripped_act.startswith("/")
            ):
                action_str = "[REDACTED_INTERACTIVE_SETUP_CREDENTIALS]"
            elif stripped_act.lower().startswith("/setup") and len(stripped_act.split()) > 1:
                action_str = "/setup [REDACTED_CREDENTIALS]"
        elif is_cb:
            event_type = "CALLBACK"
            action_str = f"cb:{getattr(event, 'data', '')}"
        else:
            event_type = "EVENT"
            action_str = type(event).__name__

        # 1. Maintenance Mode Guard (Non-admin users are blocked when Maintenance is ON)
        if self.admin_store.is_maintenance_mode() and not is_admin:
            self.admin_store.log_activity(
                user_id=user_id,
                username=username,
                role=role,
                event_type=event_type,
                action=action_str,
                status="DENIED_MAINTENANCE",
            )
            if is_msg and callable(getattr(event, "answer", None)):
                await event.answer(
                    "🛠️ <b>Server Sedang Maintenance:</b>\n"
                    "Mohon maaf, saat ini server sedang dalam pemeliharaan. "
                    "Silakan coba kembali beberapa saat lagi.",
                    parse_mode="HTML",
                )
            elif is_cb and callable(getattr(event, "answer", None)):
                await event.answer(
                    "🛠️ Server sedang dalam pemeliharaan (Maintenance).",
                    show_alert=True,
                )
            return None

        # 2. User Authorization & Subscription Expiration Check
        allowed, reason, managed_rec = self.admin_store.check_user_access(
            user_id, self.allowed_ids
        )
        if not allowed:
            status_code = "DENIED_EXPIRED" if reason == "expired" else "DENIED_UNAUTHORIZED"
            self.admin_store.log_activity(
                user_id=user_id,
                username=username,
                role=role,
                event_type=event_type,
                action=action_str,
                status=status_code,
            )
            logger.warning(
                "Blocked Telegram user command",
                user_id=user_id,
                username=username,
                reason=reason,
            )
            if reason == "expired" and managed_rec is not None:
                if is_msg and callable(getattr(event, "answer", None)):
                    await event.answer(
                        f"⏳ <b>Masa Aktif Berakhir:</b>\n"
                        f"Masa aktif akses Anda telah habis pada <code>{managed_rec.expires_at_iso}</code>. "
                        f"Silakan hubungi Admin untuk perpanjangan.",
                        parse_mode="HTML",
                    )
                elif is_cb and callable(getattr(event, "answer", None)):
                    await event.answer(
                        "⏳ Masa aktif akses Anda telah berakhir. Hubungi Admin.",
                        show_alert=True,
                    )
                return None

            if is_msg and callable(getattr(event, "answer", None)):
                await event.answer(
                    "🔒 <b>Akses Dibatasi:</b> Akun Telegram Anda tidak memiliki izin untuk mengakses terminal ini.",
                    parse_mode="HTML",
                )
            elif is_cb and callable(getattr(event, "answer", None)):
                await event.answer(
                    "Akses Dibatasi: ID Telegram Anda belum terdaftar.",
                    show_alert=True,
                )
            return None

        # 3. Record Allowed Activity in Audit Trail
        self.admin_store.log_activity(
            user_id=user_id,
            username=username,
            role=role,
            event_type=event_type,
            action=action_str,
            status="ALLOWED",
        )
        data["authorized"] = True
        data["user_role"] = role
        data["admin_store"] = self.admin_store
        return await handler(event, data)
