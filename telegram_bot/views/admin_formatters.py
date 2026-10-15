from __future__ import annotations

import html
from typing import Optional

from ai_engine.auto_trader.admin_store import AuditLogRecord, ManagedUserRecord


def format_admin_dashboard_card(
    admin_name: str,
    maintenance_on: bool,
    managed_users: list[ManagedUserRecord],
    total_recipients: int,
    recent_logs_count: int,
) -> str:
    """Format the main /admin control panel card."""
    safe_name = html.escape(admin_name or "Admin")
    maint_badge = (
        "🔴 <b>ON (SERVER SEDANG MAINTENANCE)</b>"
        if maintenance_on
        else "🟢 <b>OFF (SERVER BERJALAN NORMAL)</b>"
    )
    active_count = sum(1 for u in managed_users if not u.is_expired)
    expired_count = sum(1 for u in managed_users if u.is_expired)

    return (
        f"👑 <b>PANEL KONTROL ADMIN — NEXUSTRADE AI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Selamat datang, Admin <b>{safe_name}</b>!\n\n"
        f"<b>📊 Ringkasan Status Server &amp; User:</b>\n"
        f"• <b>Status Maintenance:</b> {maint_badge}\n"
        f"• <b>User Terdaftar (Aktif):</b> <code>{active_count} user</code> (Expired: <code>{expired_count}</code>)\n"
        f"• <b>Total Target Broadcast:</b> <code>{total_recipients} user</code>\n"
        f"• <b>Log Audit Tercatat:</b> <code>{recent_logs_count}+ aktivitas</code>\n\n"
        f"<b>⚡ Pintasan Perintah Admin:</b>\n"
        f"1️⃣ <b>Management User:</b>\n"
        f"  • Tambah: <code>/admin adduser &lt;id_tele&gt; &lt;1hari|1minggu|1bulan&gt;</code>\n"
        f"  • Edit Expired: <code>/admin edituser &lt;id_tele&gt; &lt;1hari|1minggu|1bulan&gt;</code>\n"
        f"  • Hapus User: <code>/admin deluser &lt;id_tele&gt;</code>\n"
        f"2️⃣ <b>Broadcast Pesan:</b>\n"
        f"  • <code>/admin broadcast &lt;pesan pengumuman Anda&gt;</code>\n"
        f"3️⃣ <b>Maintenance Server:</b>\n"
        f"  • <code>/admin maintenance on</code> atau <code>/admin maintenance off</code>\n"
        f"4️⃣ <b>Audit Aktivitas:</b>\n"
        f"  • <code>/admin audit [id_tele]</code>\n\n"
        f"<i>👇 Atau gunakan tombol menu interaktif di bawah ini:</i>"
    )


def format_admin_users_card(users: list[ManagedUserRecord]) -> str:
    """Format the Admin User Management list and instructions card."""
    if not users:
        user_lines_str = "  <i>Belum ada user berjangka waktu yang ditambahkan.</i>"
    else:
        lines = []
        for idx, u in enumerate(users[:15], start=1):
            status_icon = "🔴 EXPIRED" if u.is_expired else "🟢 AKTIF"
            lines.append(
                f"{idx}. <code>{u.user_id}</code> ({html.escape(u.label)})\n"
                f"   • Paket: <b>{html.escape(u.plan_label)}</b> | Status: <b>{status_icon}</b>\n"
                f"   • Sisa: <code>{html.escape(u.remaining_human)}</code> (s/d <code>{html.escape(u.expires_at_iso)}</code>)"
            )
        user_lines_str = "\n".join(lines)

    return (
        f"👥 <b>MANAGEMENT USER — ADMIN PANEL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Daftar User &amp; Masa Aktif ({len(users)}):</b>\n"
        f"{user_lines_str}\n\n"
        f"<b>📌 Cara Tambah / Edit / Hapus User:</b>\n"
        f"• Tekan tombol <b>➕ Tambah (1 Hari / 1 Minggu / 1 Bulan)</b> di bawah lalu kirim ID Telegram user.\n"
        f"• Atau ketik langsung:\n"
        f"  <code>/admin adduser 555666777 1hari</code>\n"
        f"  <code>/admin adduser 555666777 1minggu</code>\n"
        f"  <code>/admin adduser 555666777 1bulan</code>\n"
        f"  <code>/admin edituser 555666777 1bulan</code>\n"
        f"  <code>/admin deluser 555666777</code>"
    )


def format_admin_user_updated_card(
    action_title: str,
    record: ManagedUserRecord,
) -> str:
    return (
        f"✅ <b>{html.escape(action_title)} BERHASIL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>ID Telegram User:</b> <code>{record.user_id}</code> ({html.escape(record.label)})\n"
        f"• <b>Paket Masa Aktif:</b> <b>{html.escape(record.plan_label)}</b>\n"
        f"• <b>Berlaku Sampai:</b> <code>{html.escape(record.expires_at_iso)}</code>\n"
        f"• <b>Sisa Waktu:</b> <code>{html.escape(record.remaining_human)}</code>"
    )


def format_admin_broadcast_result_card(
    message_text: str,
    sent_count: int,
    failed_count: int,
    total_targets: int,
) -> str:
    return (
        f"📢 <b>LAPORAN BROADCAST PESAN ADMIN</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Total Target User:</b> <code>{total_targets} user</code>\n"
        f"• <b>Berhasil Terkirim:</b> 🟢 <code>{sent_count} user</code>\n"
        f"• <b>Gagal Terkirim:</b> 🔴 <code>{failed_count} user</code>\n\n"
        f"<b>📝 Isi Pesan Broadcast:</b>\n"
        f"<i>{html.escape(message_text)}</i>"
    )


def format_admin_maintenance_card(enabled: bool) -> str:
    if enabled:
        return (
            "🛠️ <b>MODE MAINTENANCE DIAKTIFKAN (ON)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "• <b>Status Server:</b> 🔴 <b>SEDANG MAINTENANCE</b>\n"
            "• <b>Efek ke User:</b> Seluruh akses user biasa ke fitur apa pun akan ditahan sementara dan menampilkan pesan bahwa server sedang dalam pemeliharaan.\n"
            "• <b>Akses Admin:</b> Admin tetap dapat mengakses seluruh fitur seperti biasa.\n\n"
            "<i>Tekan tombol 🟢 Matikan Maintenance (OFF) di bawah jika pemeliharaan sudah selesai.</i>"
        )
    return (
        "✅ <b>MODE MAINTENANCE DINONAKTIFKAN (OFF)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "• <b>Status Server:</b> 🟢 <b>BERJALAN NORMAL (ONLINE)</b>\n"
        "• <b>Efek ke User:</b> Seluruh user aktif kini dapat kembali mengakses seluruh fitur bot secara normal."
    )


def format_admin_audit_card(
    logs: list[AuditLogRecord],
    filter_user_id: Optional[int] = None,
) -> str:
    """Format the Admin User Activity Audit Log card."""
    title_suffix = f" (User ID: {filter_user_id})" if filter_user_id else " (Semua User)"
    if not logs:
        body = "  <i>Belum ada riwayat aktivitas user yang tercatat.</i>"
    else:
        rows = []
        for item in logs[:18]:
            if item.status == "ALLOWED" or item.status == "SUCCESS":
                st_badge = "🟢"
            elif item.status == "DENIED_MAINTENANCE":
                st_badge = "🛠️"
            elif item.status == "DENIED_EXPIRED":
                st_badge = "⏳"
            else:
                st_badge = "🔴"
            rows.append(
                f"{st_badge} <code>[{html.escape(item.created_at[5:19])}]</code> "
                f"<b>{html.escape(item.username)}</b> (<code>{item.user_id}</code> • {html.escape(item.role)})\n"
                f"   ↳ <code>{html.escape(item.event_type)}</code>: <code>{html.escape(item.action[:60])}</code> (<i>{html.escape(item.status)}</i>)"
            )
        body = "\n".join(rows)

    return (
        f"📋 <b>AUDIT AKTIVITAS USER{html.escape(title_suffix)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{body}\n\n"
        f"<i>💡 Filter per user: <code>/admin audit &lt;id_tele&gt;</code></i>"
    )
