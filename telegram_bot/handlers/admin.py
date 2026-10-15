from __future__ import annotations

from typing import Any, Optional

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import Message

from ai_engine.auto_trader.admin_store import AdminStore
from config.settings import Settings, get_settings
from telegram_bot.handlers.admin_callbacks import (
    admin_callbacks_router,
    build_admin_panel_keyboard,
    execute_admin_broadcast,
    handle_admin_callbacks,
    handle_pending_admin_input,
    perform_admin_user_delete,
    perform_admin_user_upsert,
)
from telegram_bot.handlers.handler_utils import (
    _PENDING_ADMIN_ACTION,
    clear_pending_wizards,
    extract_user_identity,
    parse_command_args,
)
from telegram_bot.views.admin_formatters import (
    format_admin_audit_card,
    format_admin_broadcast_result_card,
    format_admin_dashboard_card,
    format_admin_maintenance_card,
    format_admin_users_card,
)

admin_router = Router(name="admin")
admin_router.include_router(admin_callbacks_router)

__all__ = [
    "_PENDING_ADMIN_ACTION",
    "admin_router",
    "build_admin_panel_keyboard",
    "handle_admin_callbacks",
    "handle_admin_command",
    "handle_pending_admin_input",
]


@admin_router.message(Command("admin"))
async def handle_admin_command(message: Message, **data: Any) -> None:
    """
    Handle /admin command and subcommands:
    - `/admin` -> Open Admin Control Panel
    - `/admin adduser <user_id> <1hari|1minggu|1bulan> [label]`
    - `/admin edituser <user_id> <1hari|1minggu|1bulan>`
    - `/admin deluser <user_id>`
    - `/admin users`
    - `/admin broadcast <pesan>`
    - `/admin maintenance <on|off>`
    - `/admin audit [user_id]`
    """
    settings: Settings = data.get("settings") or get_settings()
    admin_store: AdminStore = data.get("admin_store") or AdminStore(settings)
    bot: Optional[Bot] = data.get("bot") or getattr(message, "bot", None)

    user_id, admin_name = extract_user_identity(message, default_name="Admin")

    if not settings.is_admin_user(user_id):
        await message.answer(
            "🚫 <b>Akses Ditolak:</b> Menu <code>/admin</code> hanya dapat diakses oleh <b>Role Admin</b>.",
            parse_mode="HTML",
        )
        return

    args = parse_command_args(message.text)

    if not args:
        clear_pending_wizards(user_id)
        maint_on = admin_store.is_maintenance_mode()
        users = admin_store.list_managed_users()
        recipients = admin_store.get_broadcast_recipients()
        logs = admin_store.get_audit_logs(limit=50)
        card = format_admin_dashboard_card(
            admin_name=admin_name,
            maintenance_on=maint_on,
            managed_users=users,
            total_recipients=len(recipients),
            recent_logs_count=len(logs),
        )
        await message.answer(
            card,
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(maintenance_on=maint_on),
        )
        return

    sub = args[0].lower()

    if sub == "users":
        users = admin_store.list_managed_users()
        await message.answer(
            format_admin_users_card(users),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        return

    if sub in ("adduser", "add", "edituser", "edit"):
        if len(args) < 3 or not args[1].lstrip("-").isdigit():
            await message.answer(
                "⚠️ <b>Format kurang lengkap.</b>\n"
                "Gunakan:\n"
                "<code>/admin adduser &lt;id_tele&gt; &lt;1hari|1minggu|1bulan&gt; [nama]</code>\n"
                "<code>/admin edituser &lt;id_tele&gt; &lt;1hari|1minggu|1bulan&gt;</code>",
                parse_mode="HTML",
            )
            return
        target_uid = int(args[1])
        duration_code = args[2]
        label = " ".join(args[3:]) if len(args) >= 4 else f"User {target_uid}"
        await perform_admin_user_upsert(
            message=message,
            admin_store=admin_store,
            admin_id=user_id,
            admin_name=admin_name,
            target_uid=target_uid,
            duration_code=duration_code,
            label=label,
            is_add=sub in ("adduser", "add"),
        )
        return

    if sub in ("deluser", "delete", "del", "rm"):
        if len(args) < 2 or not args[1].lstrip("-").isdigit():
            await message.answer(
                "⚠️ Gunakan format: <code>/admin deluser &lt;id_tele&gt;</code>",
                parse_mode="HTML",
            )
            return
        await perform_admin_user_delete(
            message=message,
            settings=settings,
            admin_store=admin_store,
            admin_id=user_id,
            admin_name=admin_name,
            target_uid=int(args[1]),
        )
        return

    if sub == "broadcast":
        raw_text = (message.text or "").strip()
        parts_b = raw_text.split(maxsplit=2)
        if len(parts_b) < 3 or not parts_b[2].strip():
            await message.answer(
                "⚠️ Gunakan format: <code>/admin broadcast &lt;pesan yang ingin dikirim ke seluruh user&gt;</code>",
                parse_mode="HTML",
            )
            return
        bcast_msg = parts_b[2].strip()
        sent_n, fail_n, total_n = await execute_admin_broadcast(bot, admin_store, bcast_msg)
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"broadcast ({sent_n}/{total_n}): {bcast_msg[:50]}",
            status="SUCCESS",
        )
        await message.answer(
            format_admin_broadcast_result_card(bcast_msg, sent_n, fail_n, total_n),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        return

    if sub in ("maintenance", "maint"):
        if len(args) < 2 or args[1].lower() not in ("on", "off"):
            current = admin_store.is_maintenance_mode()
            await message.answer(
                format_admin_maintenance_card(current),
                parse_mode="HTML",
                reply_markup=build_admin_panel_keyboard(current),
            )
            return
        enable = args[1].lower() == "on"
        admin_store.set_maintenance_mode(enable, admin_id=user_id)
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"maintenance:{'ON' if enable else 'OFF'}",
            status="SUCCESS",
        )
        await message.answer(
            format_admin_maintenance_card(enable),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(enable),
        )
        return

    if sub == "audit":
        filter_uid: Optional[int] = None
        if len(args) >= 2 and args[1].lstrip("-").isdigit():
            filter_uid = int(args[1])
        logs = admin_store.get_audit_logs(limit=20, user_id=filter_uid)
        await message.answer(
            format_admin_audit_card(logs, filter_user_id=filter_uid),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        return

    await message.answer(
        "ℹ️ Sub-perintah admin tidak dikenali. Ketik <code>/admin</code> untuk membuka Panel Admin.",
        parse_mode="HTML",
        reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
    )
