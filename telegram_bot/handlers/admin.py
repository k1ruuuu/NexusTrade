from __future__ import annotations

import html
from typing import Any, Optional

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from ai_engine.auto_trader.admin_store import AdminStore
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from telegram_bot.views.admin_formatters import (
    format_admin_audit_card,
    format_admin_broadcast_result_card,
    format_admin_dashboard_card,
    format_admin_maintenance_card,
    format_admin_user_updated_card,
    format_admin_users_card,
)

logger = get_logger("nexustrade.telegram.admin")

admin_router = Router(name="admin")

# Pending interactive states for Admin wizards: admin_user_id -> dict[str, Any]
_PENDING_ADMIN_ACTION: dict[int, dict[str, Any]] = {}


def build_admin_panel_keyboard(maintenance_on: bool = False) -> InlineKeyboardMarkup:
    """Build main inline keyboard for /admin dashboard."""
    maint_text = (
        "🟢 Matikan Maintenance (OFF)"
        if maintenance_on
        else "🔴 Nyalakan Maintenance (ON)"
    )
    maint_cb = "admin:maint:off" if maintenance_on else "admin:maint:on"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👥 Management User", callback_data="admin:users"),
                InlineKeyboardButton(text="📢 Broadcast Pesan", callback_data="admin:broadcast_prompt"),
            ],
            [
                InlineKeyboardButton(text=maint_text, callback_data=maint_cb),
                InlineKeyboardButton(text="📋 Audit Aktivitas User", callback_data="admin:audit"),
            ],
            [
                InlineKeyboardButton(text="➕ Add User (1 Hari)", callback_data="admin:add_prompt:1hari"),
                InlineKeyboardButton(text="➕ Add User (1 Minggu)", callback_data="admin:add_prompt:1minggu"),
                InlineKeyboardButton(text="➕ Add User (1 Bulan)", callback_data="admin:add_prompt:1bulan"),
            ],
            [
                InlineKeyboardButton(text="✏️ Edit Expired User", callback_data="admin:edit_prompt"),
                InlineKeyboardButton(text="🗑️ Hapus User", callback_data="admin:del_prompt"),
            ],
            [
                InlineKeyboardButton(text="🏠 Kembali ke Menu Utama", callback_data="cmd:dashboard"),
            ],
        ]
    )


def _require_admin(user_id: Optional[int], settings: Settings) -> bool:
    return settings.is_admin_user(user_id)


async def _execute_broadcast(
    bot: Optional[Bot],
    admin_store: AdminStore,
    broadcast_text: str,
) -> tuple[int, int, int]:
    """Send custom broadcast message to all registered user IDs."""
    recipients = admin_store.get_broadcast_recipients()
    sent_count = 0
    failed_count = 0
    formatted_msg = (
        f"📢 <b>PENGUMUMAN RESMI ADMIN — NEXUSTRADE AI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{html.escape(broadcast_text)}"
    )

    for uid in recipients:
        if bot is None:
            sent_count += 1
            continue
        try:
            await bot.send_message(chat_id=uid, text=formatted_msg, parse_mode="HTML")
            sent_count += 1
        except Exception as exc:
            logger.warning("Broadcast delivery failed for user", user_id=uid, error=str(exc))
            failed_count += 1

    return sent_count, failed_count, len(recipients)


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

    user = message.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    admin_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Admin"
    )

    if not _require_admin(user_id, settings):
        await message.answer(
            "🚫 <b>Akses Ditolak:</b> Menu <code>/admin</code> hanya dapat diakses oleh <b>Role Admin</b>.",
            parse_mode="HTML",
        )
        return

    parts = (message.text or "").strip().split()
    args = parts[1:] if parts and parts[0].startswith("/") else parts

    if not args:
        _PENDING_ADMIN_ACTION.pop(user_id, None)
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

    # 1. Management User: adduser / edituser / deluser / users
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
        try:
            rec = admin_store.add_or_update_user(
                user_id=target_uid,
                duration_code=duration_code,
                label=label,
                admin_id=user_id,
            )
        except ValueError as exc:
            await message.answer(f"⚠️ {html.escape(str(exc))}", parse_mode="HTML")
            return

        action_title = "TAMBAH USER BARU" if sub in ("adduser", "add") else "EDIT EXPIRED USER"
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"{sub}:{target_uid}:{rec.plan_label}",
            status="SUCCESS",
        )
        await message.answer(
            format_admin_user_updated_card(action_title, rec),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        return

    if sub in ("deluser", "delete", "del", "rm"):
        if len(args) < 2 or not args[1].lstrip("-").isdigit():
            await message.answer(
                "⚠️ Gunakan format: <code>/admin deluser &lt;id_tele&gt;</code>",
                parse_mode="HTML",
            )
            return
        target_uid = int(args[1])
        admin_store.delete_user(target_uid)
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"deluser:{target_uid}",
            status="SUCCESS",
        )
        await message.answer(
            f"🗑️ <b>USER DIHAPUS</b>\n"
            f"User dengan ID Telegram <code>{target_uid}</code> telah dihapus dan aksesnya dicabut.",
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        return

    # 2. Broadcast custom message to all users
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
        sent_n, fail_n, total_n = await _execute_broadcast(bot, admin_store, bcast_msg)
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

    # 3. Maintenance Mode (on / off)
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

    # 4. Audit User Activities
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


@admin_router.message(
    lambda msg: bool(
        msg.from_user
        and int(msg.from_user.id) in _PENDING_ADMIN_ACTION
        and msg.text
        and not msg.text.strip().startswith("/")
    )
)
async def handle_pending_admin_input(message: Message, **data: Any) -> None:
    """Handle interactive follow-up replies for Admin Add/Edit/Delete User and Broadcast wizards."""
    settings: Settings = data.get("settings") or get_settings()
    admin_store: AdminStore = data.get("admin_store") or AdminStore(settings)
    bot: Optional[Bot] = data.get("bot") or getattr(message, "bot", None)

    user = message.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    admin_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Admin"
    )
    if not _require_admin(user_id, settings):
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        return

    pending = _PENDING_ADMIN_ACTION.pop(user_id, None)
    if not pending:
        return

    mode = pending.get("mode")
    text = (message.text or "").strip()

    if mode == "add_user":
        duration_code = str(pending.get("duration", "1bulan"))
        tokens = text.split()
        if not tokens or not tokens[0].lstrip("-").isdigit():
            await message.answer(
                "⚠️ ID Telegram harus berupa angka. Contoh: <code>555666777</code>",
                parse_mode="HTML",
            )
            return
        target_uid = int(tokens[0])
        label = " ".join(tokens[1:]) if len(tokens) > 1 else f"User {target_uid}"
        rec = admin_store.add_or_update_user(
            user_id=target_uid,
            duration_code=duration_code,
            label=label,
            admin_id=user_id,
        )
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"adduser:{target_uid}:{rec.plan_label}",
            status="SUCCESS",
        )
        await message.answer(
            format_admin_user_updated_card("TAMBAH USER BARU", rec),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        return

    if mode == "edit_user":
        tokens = text.split()
        if len(tokens) < 2 or not tokens[0].lstrip("-").isdigit():
            await message.answer(
                "⚠️ Format edit: <code>&lt;id_tele&gt; &lt;1hari|1minggu|1bulan&gt;</code>\nContoh: <code>555666777 1minggu</code>",
                parse_mode="HTML",
            )
            return
        target_uid = int(tokens[0])
        try:
            rec = admin_store.edit_user_expiry(
                user_id=target_uid,
                duration_code=tokens[1],
                admin_id=user_id,
            )
        except ValueError as exc:
            await message.answer(f"⚠️ {html.escape(str(exc))}", parse_mode="HTML")
            return
        if rec is not None:
            admin_store.log_activity(
                user_id=user_id,
                username=admin_name,
                role="admin",
                event_type="ADMIN",
                action=f"edituser:{target_uid}:{rec.plan_label}",
                status="SUCCESS",
            )
            await message.answer(
                format_admin_user_updated_card("EDIT EXPIRED USER", rec),
                parse_mode="HTML",
                reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
            )
        return

    if mode == "del_user":
        tokens = text.split()
        if not tokens or not tokens[0].lstrip("-").isdigit():
            await message.answer("⚠️ Kirimkan ID Telegram berupa angka.", parse_mode="HTML")
            return
        target_uid = int(tokens[0])
        admin_store.delete_user(target_uid)
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"deluser:{target_uid}",
            status="SUCCESS",
        )
        await message.answer(
            f"🗑️ User <code>{target_uid}</code> telah dihapus dan aksesnya dicabut.",
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        return

    if mode == "broadcast":
        sent_n, fail_n, total_n = await _execute_broadcast(bot, admin_store, text)
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"broadcast ({sent_n}/{total_n}): {text[:50]}",
            status="SUCCESS",
        )
        await message.answer(
            format_admin_broadcast_result_card(text, sent_n, fail_n, total_n),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )


@admin_router.callback_query(F.data.startswith("admin:"))
async def handle_admin_callbacks(callback: CallbackQuery, **data: Any) -> None:
    """Handle inline button callbacks for the Admin Control Panel."""
    settings: Settings = data.get("settings") or get_settings()
    admin_store: AdminStore = data.get("admin_store") or AdminStore(settings)

    msg = callback.message
    if msg is None:
        await callback.answer()
        return

    user = callback.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    admin_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Admin"
    )

    if not _require_admin(user_id, settings):
        await callback.answer("Akses khusus Admin", show_alert=True)
        return

    cb_data = callback.data or ""

    if cb_data == "admin:panel":
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        maint_on = admin_store.is_maintenance_mode()
        users = admin_store.list_managed_users()
        recipients = admin_store.get_broadcast_recipients()
        logs = admin_store.get_audit_logs(limit=50)
        await msg.answer(
            format_admin_dashboard_card(
                admin_name=admin_name,
                maintenance_on=maint_on,
                managed_users=users,
                total_recipients=len(recipients),
                recent_logs_count=len(logs),
            ),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(maint_on),
        )
        await callback.answer()
        return

    if cb_data == "admin:users":
        users = admin_store.list_managed_users()
        await msg.answer(
            format_admin_users_card(users),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        await callback.answer()
        return

    if cb_data.startswith("admin:add_prompt:"):
        dur = cb_data.split(":")[-1]
        _PENDING_ADMIN_ACTION[user_id] = {"mode": "add_user", "duration": dur}
        await msg.answer(
            f"➕ <b>TAMBAH USER (Paket: {html.escape(dur.upper())})</b>\n"
            f"Silakan balas pesan ini dengan mengirim <b>ID Telegram User</b> (dan nama catatan opsional):\n\n"
            f"Contoh: <code>555666777 Budi Trader</code>",
            parse_mode="HTML",
        )
        await callback.answer(f"Kirim ID Telegram untuk paket {dur}")
        return

    if cb_data == "admin:edit_prompt":
        _PENDING_ADMIN_ACTION[user_id] = {"mode": "edit_user"}
        await msg.answer(
            "✏️ <b>EDIT EXPIRED USER</b>\n"
            "Silakan balas pesan ini dengan format:\n"
            "<code>&lt;id_tele&gt; &lt;1hari|1minggu|1bulan&gt;</code>\n\n"
            "Contoh: <code>555666777 1bulan</code>",
            parse_mode="HTML",
        )
        await callback.answer("Kirim ID & durasi baru")
        return

    if cb_data == "admin:del_prompt":
        _PENDING_ADMIN_ACTION[user_id] = {"mode": "del_user"}
        await msg.answer(
            "🗑️ <b>HAPUS USER</b>\n"
            "Silakan balas pesan ini dengan mengirim <b>ID Telegram User</b> yang ingin dihapus:\n\n"
            "Contoh: <code>555666777</code>",
            parse_mode="HTML",
        )
        await callback.answer("Kirim ID Telegram yang akan dihapus")
        return

    if cb_data == "admin:broadcast_prompt":
        _PENDING_ADMIN_ACTION[user_id] = {"mode": "broadcast"}
        recipients = admin_store.get_broadcast_recipients()
        await msg.answer(
            f"📢 <b>BROADCAST PESAN KE SELURUH USER ({len(recipients)} User)</b>\n"
            f"Silakan balas pesan ini dengan mengetik <b>pesan custom</b> yang ingin Anda kirim ke semua user:",
            parse_mode="HTML",
        )
        await callback.answer("Ketik pesan broadcast Anda")
        return

    if cb_data in ("admin:maint:on", "admin:maint:off"):
        enable = cb_data.endswith(":on")
        admin_store.set_maintenance_mode(enable, admin_id=user_id)
        admin_store.log_activity(
            user_id=user_id,
            username=admin_name,
            role="admin",
            event_type="ADMIN",
            action=f"maintenance:{'ON' if enable else 'OFF'}",
            status="SUCCESS",
        )
        await msg.answer(
            format_admin_maintenance_card(enable),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(enable),
        )
        await callback.answer("Maintenance ON" if enable else "Maintenance OFF")
        return

    if cb_data == "admin:audit":
        logs = admin_store.get_audit_logs(limit=20)
        await msg.answer(
            format_admin_audit_card(logs),
            parse_mode="HTML",
            reply_markup=build_admin_panel_keyboard(admin_store.is_maintenance_mode()),
        )
        await callback.answer()
