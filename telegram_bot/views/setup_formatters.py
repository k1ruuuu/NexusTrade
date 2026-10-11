from __future__ import annotations

import html
from typing import Optional

from ai_engine.auto_trader.binance_client import BinanceAccountVerification
from ai_engine.auto_trader.engine import AutoTradeExecutionReport
from ai_engine.auto_trader.user_vault import (
    UserBinanceCredentials,
    UserPositionRecord,
    UserRole,
)
from telegram_bot.views.formatter_utils import format_price


def format_role_badge(role: UserRole, source: Optional[str] = None) -> str:
    if role == "admin":
        if source == "admin_env":
            return "👑 <b>ADMIN (Kredensial .env Server)</b>"
        return "👑 <b>ADMIN (Setup Pribadi)</b>"
    return "👤 <b>USER PRIBADI (Brankas Terisolasi)</b>"


def format_setup_wizard_card(
    user_name: str,
    user_id: int,
    role: UserRole,
    current_creds: Optional[UserBinanceCredentials] = None,
) -> str:
    """Format interactive /setup onboarding & instruction card."""
    safe_name = html.escape(user_name or "Trader")
    role_badge = format_role_badge(role, current_creds.source if current_creds else None)

    if current_creds is not None:
        prov_lbl = (
            f"📱 EXNESS GO / MT5 ({html.escape(current_creds.server_name)})"
            if current_creds.provider == "exness_mt5"
            else ("🧪 BINANCE TESTNET" if current_creds.testnet else "🏦 BINANCE LIVE")
        )
        src_lbl = (
            "Brankas /setup Pribadi"
            if current_creds.source == "user_setup"
            else "Fallback .env Khusus Admin"
        )
        status_block = (
            f"<b>✅ Status Akun Saat Ini: TERHUBUNG</b>\n"
            f"• <b>Provider &amp; Jaringan:</b> <code>{prov_lbl}</code>\n"
            f"• <b>Sumber Kredensial:</b> <code>{src_lbl}</code>\n"
            f"• <b>Login / API Key:</b> <code>{html.escape(current_creds.masked_api_key)}</code>\n"
            f"• <b>Saldo Terverifikasi:</b> <code>${current_creds.verified_usdt_balance:,.2f} USD/USDT</code>\n\n"
        )
    else:
        status_block = (
            "<b>⚠️ Status Akun Saat Ini: BELUM TERHUBUNG</b>\n"
            "<i>Anda wajib menghubungkan akun Exness Go / MT5 atau Binance pribadi terlebih dahulu sebelum menggunakan fitur Auto-Trade.</i>\n\n"
        )

    return (
        f"🔐 <b>SETUP AKUN TRADING PRIBADI — NEXUSTRADE AI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Halo <b>{safe_name}</b> (ID: <code>{user_id}</code>)\n"
        f"• <b>Role Anda:</b> {role_badge}\n\n"
        f"{status_block}"
        f"<b>🌐 Pilihan Provider yang Didukung (BUY &amp; SELL 2 Arah):</b>\n"
        f"• <b>📱 Exness Go / MetaTrader 5:</b> Mendukung <b>SEMUA ASET</b> (Crypto, Emas <code>XAUUSDm</code>, Forex <code>EURUSDm</code>, Saham AS <code>NVDA</code>, Indeks <code>US500</code>) untuk <b>BUY &amp; SELL</b> + SL/TP otomatis.\n"
        f"• <b>🟡 Binance (Spot / Futures):</b> Khusus aset Crypto (<code>BTCUSDT</code>, <code>ETHUSDT</code>, dll.).\n\n"
        f"<b>🚀 Cara Menghubungkan Akun:</b>\n"
        f"• <b>Cara 1 (Interaktif):</b> Tekan tombol <b>📱 Pilih Exness Go / MT5</b> atau <b>🧪 Pilih Binance</b> di bawah.\n"
        f"• <b>Cara 2 (Perintah Langsung):</b>\n"
        f"  <code>/setup exness LOGIN_MT5 PASSWORD_ATAU_TOKEN [Exness-MT5Trial|Exness-MT5Real]</code>\n"
        f"  <code>/setup testnet API_KEY_BINANCE API_SECRET_BINANCE</code>\n"
        f"  <code>/setup live API_KEY_BINANCE API_SECRET_BINANCE</code>"
    )


def format_setup_prompt_key_card(testnet: bool, provider: str = "binance_spot") -> str:
    if provider == "exness_mt5":
        srv = "Exness-MT5Trial (Demo)" if testnet else "Exness-MT5Real (Live)"
        return (
            f"📱 <b>PROVIDER DIPILIH: EXNESS GO / METATRADER 5 ({srv})</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Silakan balas pesan ini dengan mengirim <b>Login ID MT5 / Account ID</b> dan <b>Password Trading / MetaApi Token</b> (serta nama Server opsional):\n\n"
            f"<code>LOGIN_ID_MT5 PASSWORD_ANDA Exness-MT5Trial</code>\n\n"
            f"<i>🔒 Pesan Anda akan langsung dihapus otomatis oleh bot begitu diterima. Ketik /cancel untuk membatalkan.</i>"
        )
    mode_str = "🧪 BINANCE SPOT TESTNET" if testnet else "🏦 BINANCE SPOT LIVE"
    return (
        f"🔑 <b>MODE DIPILIH: {mode_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Silakan balas pesan ini dengan mengirim <b>API Key</b> dan <b>API Secret</b> Anda dipisahkan dengan spasi:\n\n"
        f"<code>API_KEY_ANDA API_SECRET_ANDA</code>\n\n"
        f"<i>🔒 Pesan Anda akan langsung dihapus otomatis oleh bot begitu diterima demi menjaga kerahasiaan kunci Anda. "
        f"Ketik /cancel untuk membatalkan.</i>"
    )


def format_setup_success_card(
    creds: UserBinanceCredentials,
    verification: BinanceAccountVerification,
    secret_deleted: bool = True,
) -> str:
    if creds.provider == "exness_mt5":
        mode_str = f"📱 EXNESS GO / MT5 ({html.escape(creds.server_name)})"
        header_title = "✅ <b>SETUP AKUN EXNESS GO / MT5 BERHASIL!</b>"
    else:
        mode_str = "🧪 BINANCE TESTNET" if creds.testnet else "🏦 BINANCE LIVE"
        header_title = "✅ <b>SETUP AKUN BINANCE BERHASIL!</b>"
    role_badge = format_role_badge(creds.role, creds.source)
    del_note = (
        "🗑️ <i>Pesan chat berisi kredensial rahasia Anda telah dihapus otomatis demi keamanan.</i>\n\n"
        if secret_deleted
        else ""
    )
    return (
        f"{header_title}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{del_note}"
        f"• <b>ID Telegram:</b> <code>{creds.user_id}</code> ({html.escape(creds.username)})\n"
        f"• <b>Role Akun:</b> {role_badge}\n"
        f"• <b>Jaringan:</b> <code>{mode_str}</code> ({html.escape(verification.account_type)})\n"
        f"• <b>API Key Terenkripsi:</b> <code>{html.escape(creds.masked_api_key)}</code>\n"
        f"• <b>Izin Spot Trading:</b> 🟢 <b>AKTIF (Terverifikasi)</b>\n"
        f"• <b>Saldo USDT Tersedia:</b> <code>${verification.usdt_free:,.2f} USDT</code>\n"
        f"• <b>Total Saldo USDT:</b> <code>${verification.total_usdt:,.2f} USDT</code>\n"
        f"• <b>Batas Risiko / Trade:</b> <code>{creds.risk_per_trade_pct * 100:.2f}%</code>\n\n"
        f"<b>📌 Langkah Selanjutnya:</b>\n"
        f"• Ketik <code>/autotrade on</code> untuk mengaktifkan izin Auto-Trade.\n"
        f"• Ketik <code>/autotrade scan BTCUSDT</code> agar AI menganalisis 6 Gerbang (Berita + Teknikal 5Thn + S/R) &amp; membuka posisi jika akurat.\n"
        f"• Ketik <code>/myaccount</code> untuk memantau akun &amp; posisi Anda."
    )


def format_myaccount_card(
    user_id: int,
    user_name: str,
    role: UserRole,
    creds: Optional[UserBinanceCredentials],
    positions: list[UserPositionRecord],
) -> str:
    safe_name = html.escape(user_name or "Trader")
    role_badge = format_role_badge(role, creds.source if creds else None)

    if creds is None:
        return (
            f"👤 <b>PROFIL AKUN &amp; AUTO-TRADE — {safe_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>ID Telegram:</b> <code>{user_id}</code>\n"
            f"• <b>Role Anda:</b> {role_badge}\n"
            f"• <b>Status Binance:</b> 🔴 <b>BELUM DI-SETUP</b>\n\n"
            f"⚠️ <i>Anda belum menghubungkan akun Binance pribadi. Data di <code>.env</code> hanya berlaku untuk Admin. "
            f"Silakan ketik <code>/setup</code> terlebih dahulu.</i>"
        )

    if creds.provider == "exness_mt5":
        mode_str = f"📱 EXNESS GO / MT5 ({html.escape(creds.server_name)})"
    else:
        mode_str = "🧪 BINANCE TESTNET" if creds.testnet else "🏦 BINANCE LIVE"
    src_str = "Setup Pribadi (/setup)" if creds.source == "user_setup" else "Konfigurasi Admin (.env)"
    auto_badge = "🟢 <b>AKTIF (ON)</b>" if creds.autotrade_enabled else "⚪ <b>NONAKTIF (OFF)</b>"

    if positions:
        pos_lines = []
        for p in positions[:5]:
            pos_lines.append(
                f"  • <b>{html.escape(p.symbol)} ({p.side})</b>: <code>{p.quantity}</code> @ <code>{format_price(p.entry_price)}</code> "
                f"| SL: <code>{format_price(p.stop_loss)}</code> | TP1: <code>{format_price(p.take_profit_1)}</code>"
            )
        pos_block = "\n".join(pos_lines)
    else:
        pos_block = "  <i>Belum ada posisi terbuka saat ini.</i>"

    return (
        f"👤 <b>PROFIL AKUN TRADING — {safe_name}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>ID Telegram:</b> <code>{user_id}</code>\n"
        f"• <b>Role:</b> {role_badge}\n"
        f"• <b>Sumber Akun:</b> <code>{src_str}</code>\n"
        f"• <b>Provider &amp; Server:</b> <code>{mode_str}</code>\n"
        f"• <b>API Key:</b> <code>{html.escape(creds.masked_api_key)}</code>\n"
        f"• <b>Saldo USDT:</b> <code>${creds.verified_usdt_balance:,.2f} USDT</code>\n"
        f"• <b>Status Auto-Trade:</b> {auto_badge}\n"
        f"• <b>Risiko / Transaksi:</b> <code>{creds.risk_per_trade_pct * 100:.2f}%</code>\n\n"
        f"<b>📂 Posisi Auto-Trade Aktif ({len(positions)}):</b>\n"
        f"{pos_block}\n\n"
        f"<i>Perintah: <code>/autotrade on</code> | <code>/autotrade off</code> | <code>/autotrade scan BTCUSDT</code> | <code>/unlink</code></i>"
    )


def format_autotrade_report_card(report: AutoTradeExecutionReport) -> str:
    """Format the 6-Gate Pre-Trade Checklist & execution result card."""
    sym = html.escape(report.symbol)
    tf = html.escape(report.timeframe)

    if report.setup_required:
        return (
            f"🔒 <b>AUTO-TRADE DITOLAK — WAJIB /SETUP DULU</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Akun Telegram Anda (ID: <code>{report.user_id}</code>) belum terhubung ke akun Binance pribadi.\n\n"
            f"• <b>Aturan Sistem:</b> Kredensial pada <code>.env</code> hanya berlaku untuk <b>Role Admin</b>.\n"
            f"• Setiap user wajib menghubungkan akun Binance masing-masing terlebih dahulu.\n\n"
            f"👉 Silakan ketik <code>/setup</code> untuk menghubungkan akun Binance Anda."
        )

    gate_lines = []
    for g in report.gates:
        icon = "✅" if g.passed else "❌"
        gate_lines.append(
            f"{icon} <b>Gerbang {g.gate_number} ({html.escape(g.name)}):</b>\n"
            f"   <i>{html.escape(g.detail)}</i>"
        )
    gates_block = "\n".join(gate_lines)

    if report.executed and report.order_result and report.position_record:
        ord_r = report.order_result
        pos_r = report.position_record
        header_badge = "🚀 <b>POSISI DIBUKA OTOMATIS (LOLOS 6 GERBANG!)</b>"
        footer_block = (
            f"\n<b>🧾 Detail Order Binance Anda:</b>\n"
            f"• <b>Order ID:</b> <code>{html.escape(ord_r.order_id)}</code> ({html.escape(ord_r.status)})\n"
            f"• <b>Aksi &amp; Simbol:</b> <b>{ord_r.side} {html.escape(ord_r.symbol)}</b>\n"
            f"• <b>Jumlah &amp; Harga:</b> <code>{ord_r.executed_qty}</code> @ <code>{format_price(ord_r.avg_price)}</code> (<code>${ord_r.notional_usdt:,.2f}</code>)\n"
            f"• <b>Stop Loss (2xATR):</b> <code>{format_price(pos_r.stop_loss)}</code>\n"
            f"• <b>Take Profit 1 &amp; 2:</b> <code>{format_price(pos_r.take_profit_1)}</code> | <code>{format_price(pos_r.take_profit_2)}</code>"
        )
    else:
        header_badge = "🛡️ <b>POSISI DITAHAN — BELUM LOLOS 6 GERBANG AKURASI</b>"
        reason = html.escape(report.rejection_summary or "Syarat pembukaan posisi belum terpenuhi.")
        footer_block = (
            f"\n<b>💡 Keputusan AI:</b>\n"
            f"<i>AI tidak membuka posisi sembarangan. {reason}</i>"
        )

    return (
        f"🤖 <b>LAPORAN AUTO-TRADE 6 GERBANG — {sym} ({tf})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Status Eksekusi:</b> {header_badge}\n\n"
        f"<b>📋 Hasil Pemeriksaan 6 Gerbang AI:</b>\n"
        f"{gates_block}\n"
        f"{footer_block}"
    )
