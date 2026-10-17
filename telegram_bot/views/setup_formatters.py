from __future__ import annotations

import html
from typing import Optional

from ai_engine.auto_trader.mt5_exness_client import BinanceAccountVerification
from ai_engine.auto_trader.engine import AutoTradeExecutionReport
from ai_engine.auto_trader.session_manager import (
    AutoTradeSessionState,
    PositionMonitorResult,
)
from ai_engine.auto_trader.user_vault import (
    UserBinanceCredentials,
    UserPositionRecord,
    UserRole,
)
from telegram_bot.views.formatter_utils import format_price


EVENT_ICONS: dict[str, str] = {
    "FOMC": "🏛️",
    "IMF": "🌐",
    "NFP": "💼",
    "CPI": "📊",
    "PPI": "🏭",
    "PCE": "📈",
    "GDP": "🇺🇸",
    "RETAIL": "🛒",
    "PMI": "🏗️",
    "JOBLESS": "📉",
}


def get_macro_event_icon(event_mode: Optional[str]) -> str:
    if not event_mode:
        return "🎯"
    return EVENT_ICONS.get(event_mode.upper(), "🎯")


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
        prov_lbl = f"📱 EXNESS GO / MT5 ({html.escape(current_creds.server_name)})"
        src_lbl = (
            "Brankas /setup Pribadi"
            if current_creds.source == "user_setup"
            else "Fallback .env Khusus Admin"
        )
        status_block = (
            f"<b>✅ Status Akun Saat Ini: TERHUBUNG</b>\n"
            f"• <b>Provider &amp; Server:</b> <code>{prov_lbl}</code>\n"
            f"• <b>Sumber Kredensial:</b> <code>{src_lbl}</code>\n"
            f"• <b>Login ID MT5:</b> <code>{html.escape(current_creds.masked_api_key)}</code>\n"
            f"• <b>Saldo Terverifikasi:</b> <code>${current_creds.verified_usdt_balance:,.2f} USD</code>\n\n"
        )
    else:
        status_block = (
            "<b>⚠️ Status Akun Saat Ini: BELUM TERHUBUNG</b>\n"
            "<i>Anda wajib menghubungkan akun Exness Go / MT5 pribadi terlebih dahulu sebelum menggunakan fitur Auto-Trade.</i>\n\n"
        )

    return (
        f"🔐 <b>SETUP AKUN EXNESS GO / MT5 — NEXUSTRADE AI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Halo <b>{safe_name}</b> (ID: <code>{user_id}</code>)\n"
        f"• <b>Role Anda:</b> {role_badge}\n\n"
        f"{status_block}"
        f"<b>🌐 Cakupan Broker Exness Go / MetaTrader 5 (BUY &amp; SELL 2 Arah):</b>\n"
        f"• Mendukung <b>SEMUA ASET</b> (Crypto <code>BTCUSDm</code>, Emas <code>XAUUSDm</code>, Forex <code>EURUSDm</code>, Saham AS <code>NVDAm</code>, Indeks <code>US500m</code>) untuk <b>BUY &amp; SELL</b> + SL/TP otomatis.\n\n"
        f"<b>🚀 Cara Menghubungkan Akun Exness Go / MT5:</b>\n"
        f"• <b>Cara 1 (Interaktif):</b> Tekan tombol <b>📱 Pilih Exness Go / MT5</b> di bawah.\n"
        f"• <b>Cara 2 (Perintah Langsung):</b>\n"
        f"  <code>/setup exness LOGIN_MT5 PASSWORD_TRADING [Exness-MT5Trial7|Exness-MT5Real]</code>"
    )


def format_setup_prompt_key_card(testnet: bool, provider: str = "exness_mt5") -> str:
    srv = "Exness-MT5Trial7 (Demo)" if testnet else "Exness-MT5Real (Live)"
    return (
        f"📱 <b>PROVIDER DIPILIH: EXNESS GO / METATRADER 5 ({srv})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Silakan balas pesan ini dengan mengirim <b>Login ID MT5</b> dan <b>Password Trading</b> (serta nama Server opsional):\n\n"
        f"<code>LOGIN_ID_MT5 PASSWORD_ANDA Exness-MT5Trial7</code>\n\n"
        f"<i>🔒 Pesan Anda akan langsung dihapus otomatis oleh bot begitu diterima. Ketik /cancel untuk membatalkan.</i>"
    )


def format_setup_success_card(
    creds: UserBinanceCredentials,
    verification: BinanceAccountVerification,
    secret_deleted: bool = True,
) -> str:
    mode_str = f"📱 EXNESS GO / MT5 ({html.escape(creds.server_name)})"
    header_title = "✅ <b>SETUP AKUN EXNESS GO / MT5 BERHASIL!</b>"
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
        f"• <b>Login MT5 Terenkripsi:</b> <code>{html.escape(creds.masked_api_key)}</code>\n"
        f"• <b>Izin Trading MT5:</b> 🟢 <b>AKTIF (Terverifikasi)</b>\n"
        f"• <b>Free Margin Tersedia:</b> <code>${verification.usdt_free:,.2f} USD</code>\n"
        f"• <b>Total Equity Akun:</b> <code>${verification.total_usdt:,.2f} USD</code>\n"
        f"• <b>Batas Risiko / Trade:</b> <code>{creds.risk_per_trade_pct * 100:.2f}%</code>\n\n"
        f"<b>📌 Langkah Selanjutnya:</b>\n"
        f"• Ketik <code>/autotrade XAUUSD 0.05</code> atau <code>/autotrade BTCUSD 0.01</code> untuk menyalakan Auto-Trade 24/7 Non-Stop.\n"
        f"• Ketik <code>/myaccount</code> untuk memantau saldo &amp; posisi Exness Go Anda."
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
            f"• <b>Status Exness Go / MT5:</b> 🔴 <b>BELUM DI-SETUP</b>\n\n"
            f"⚠️ <i>Anda belum menghubungkan akun Exness Go / MT5 pribadi. Data di <code>.env</code> hanya berlaku untuk Admin. "
            f"Silakan ketik <code>/setup</code> terlebih dahulu.</i>"
        )

    mode_str = f"📱 EXNESS GO / MT5 ({html.escape(creds.server_name)})"
    src_str = "Setup Pribadi (/setup)" if creds.source == "user_setup" else "Konfigurasi Admin (.env)"
    auto_badge = "🟢 <b>AKTIF (ON)</b>" if creds.autotrade_enabled else "⚪ <b>NONAKTIF (OFF)</b>"
    if positions:
        pos_lines = []
        for p in positions[:5]:
            tp_str = f"TP1: <code>{format_price(p.take_profit_1)}</code>" if p.take_profit_1 > 0 else "TP: <i>Dihapus (Runner)</i>"
            pos_lines.append(
                f"  • <b>{html.escape(p.symbol)} ({p.side})</b>: <code>{p.quantity}</code> @ <code>{format_price(p.entry_price)}</code> "
                f"| SL: <code>{format_price(p.stop_loss)}</code> | {tp_str}"
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
        f"<i>Perintah: <code>/autotrade XAUUSD 0.05</code> | <code>/autotrade BTCUSD</code> | <code>/autotrade off</code> | <code>/unlink</code></i>"
    )


def format_timed_session_started_card(
    session: AutoTradeSessionState,
    first_cycle_result: object,
) -> str:
    """
    Format the Real-Time 24/7 Auto-Trade Live Dashboard Card showing:
    - Multi-Timeframe (1D, 1H, 15M, 5M) Trend Analysis + 1m Execution Action
    - Real-Time Floating Profit, Realized Profit, Total Net Profit, & Live Exness Price (updated in-place without delay)
    """
    sym = html.escape(session.symbol)
    dur = html.escape(session.duration_label)

    if isinstance(first_cycle_result, AutoTradeExecutionReport) and first_cycle_result.setup_required:
        return format_autotrade_report_card(first_cycle_result)

    pos = session.live_open_position
    if pos is None and isinstance(first_cycle_result, AutoTradeExecutionReport):
        pos = first_cycle_result.position_record
    elif pos is None and isinstance(first_cycle_result, PositionMonitorResult):
        pos = first_cycle_result.position

    floating_pnl = session.live_floating_pnl_usdt
    floating_pct = session.live_floating_pnl_pct
    if isinstance(first_cycle_result, PositionMonitorResult) and first_cycle_result.action_taken in ("HOLDING", "TRAILING_UPDATED"):
        floating_pnl = first_cycle_result.unrealized_pnl_usdt
        floating_pct = first_cycle_result.unrealized_pnl_pct

    realized_pnl = session.realized_pnl_usdt
    total_net_pnl = round(realized_pnl + floating_pnl, 2)
    float_icon = "🟢" if floating_pnl > 0 else ("🔴" if floating_pnl < 0 else "⚪")
    net_icon = "🟢" if total_net_pnl > 0 else ("🔴" if total_net_pnl < 0 else "⚪")

    live_px = session.live_current_price
    if live_px <= 0 and pos is not None:
        live_px = pos.entry_price
    elif live_px <= 0 and isinstance(first_cycle_result, AutoTradeExecutionReport) and first_cycle_result.decision is not None:
        live_px = first_cycle_result.decision.entry_price

    if pos is not None and pos.status == "OPEN":
        sl_dist = abs(pos.entry_price - pos.stop_loss)
        tp1_dist = abs(pos.take_profit_1 - pos.entry_price)
        tp2_dist = abs(pos.take_profit_2 - pos.entry_price)
        tp_line = (
            "• <b>Take Profit (TP):</b> <i>Dihapus / Tanpa Batas (Runner Mode Volatilitas)</i>"
            if pos.take_profit_1 <= 0
            else f"• <b>Take Profit (1:2 | 1:3):</b> <code>{format_price(pos.take_profit_1)}</code> (<code>{format_price(tp1_dist)}</code>) | <code>{format_price(pos.take_profit_2)}</code> (<code>{format_price(tp2_dist)}</code>)"
        )
        cycle_block = (
            f"🚀 <b>STATUS POSISI AKTIF (DIKAWAL REAL-TIME 5M):</b>\n"
            f"• <b>Posisi &amp; Ticket:</b> <b>{pos.side} {html.escape(pos.symbol)}</b> (<code>{pos.quantity} Lot</code> | Ticket <code>#{html.escape(pos.order_id)}</code>)\n"
            f"• <b>Harga Entry vs Live (5m):</b> <code>{format_price(pos.entry_price)}</code> ➔ <b><code>{format_price(live_px)}</code></b>\n"
            f"• <b>Stop Loss (1R):</b> <code>{format_price(pos.stop_loss)}</code> (Jarak <code>{format_price(sl_dist)}</code>)\n"
            f"{tp_line}"
        )
    elif isinstance(first_cycle_result, AutoTradeExecutionReport):
        reason = html.escape(
            first_cycle_result.rejection_summary or session.last_check_summary
        )
        cycle_block = (
            f"🔍 <b>STATUS PEMANTAUAN 24/7 (MENUNGGU ENTRY 5M TERBAIK):</b>\n"
            f"• <b>Filter 6 Gerbang:</b> <i>{reason}</i>\n"
            f"• <b>Tindakan AI:</b> Memindai tren <b>1D, 1H, 15M, 5M</b> non-stop &amp; langsung membuka posisi di <b>5M</b> begitu ke-6 Gerbang selaras."
        )
    else:
        last_reason = html.escape(session.last_check_summary)
        cycle_block = (
            f"🔍 <b>STATUS PEMANTAUAN 24/7 (MENUNGGU ENTRY 5M TERBAIK):</b>\n"
            f"• <b>Filter 6 Gerbang:</b> <i>{last_reason}</i>\n"
            f"• <b>Tindakan AI:</b> Memindai tren <b>1D, 1H, 15M, 5M</b> non-stop &amp; langsung membuka posisi di <b>5M</b> begitu ke-6 Gerbang selaras."
        )

    lot_lbl = (
        f"<code>{session.custom_lot:.2f} Lot (Custom)</code>"
        if session.custom_lot is not None and session.custom_lot > 0
        else "<code>Otomatis (Sesuai Risiko Modal)</code>"
    )
    upd_str = html.escape(session.last_updated_iso or session.started_at_iso)
    ev_icon = get_macro_event_icon(session.macro_event_mode)
    event_line = (
        f"• <b>Mode Event Makro:</b> {ev_icon} <b>{session.macro_event_mode} Event Sniper</b> (Blackout Bypass Aktif)\n"
        if session.macro_event_mode
        else ""
    )
    if session.ea_mode == "EA_EXPERT_ADVISOR":
        hdr_title = "EA EXPERT ADVISOR MT5 (RSI 14) —"
        sub_title = "SESI EA MT5 (RSI 14 | SL 50 PIPS | TP 150 PIPS)"
        cmd_prefix = "/ea-EA"
        strat_line = "🤖 <b>RSI 14 (RSI ≥ 70 SELL, RSI ≤ 30 BUY | SL 50p, TP 150p)</b>"
        mode_line = f"🟢 <b>EA Expert Advisor</b> (<code>{html.escape(session.remaining_human)}</code>)"
    elif session.ea_mode == "EA_SCALPING":
        hdr_title = "EA SCALPING M1/M5 (FAST MOMENTUM) —"
        sub_title = "SESI EA SCALPING (M1 MOMENTUM | SL 20 PIPS | TP 40 PIPS)"
        cmd_prefix = "/ea-scp"
        strat_line = "⚡ <b>EMA Ribbon + Momentum Pullback (SL 20p, TP 40p)</b>"
        mode_line = f"🟢 <b>EA Scalping</b> (<code>{html.escape(session.remaining_human)}</code>)"
    elif session.is_trade_mode:
        hdr_title = "TRADE (4 INDIKATOR MURNI) —"
        sub_title = "SESI TRADE 4 INDIKATOR MURNI"
        cmd_prefix = "/trade"
        strat_line = "💎 <b>4 Indikator Murni (UT Bot + Supertrend + Candle + OC Signals)</b>"
        mode_line = f"🟢 <b>4 Indikator Murni</b> (<code>{html.escape(session.remaining_human)}</code>)"
    else:
        hdr_title = "AUTO-TRADE 24/7 NON-STOP AKTIF —"
        sub_title = f"SESI AUTO-TRADE OTOMATIS ({dur.upper()})"
        cmd_prefix = "/autotrade"
        strat_line = "🤖 <b>UT Bot + Supertrend + OC Signals + Candles</b>"
        mode_line = f"🟢 <b>24/7 Non-Stop</b> (<code>{html.escape(session.remaining_human)}</code>)"
    return (
        f"🤖 <b>{hdr_title} {sym} (ACTION 5M)</b>\n"
        f"⏱️ <i>{sub_title} AKTIF</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>ID Sesi:</b> <code>#AT-{session.short_id}</code> <i>(Ketik <code>{cmd_prefix} stop {session.short_id}</code> untuk menghentikan)</i>\n"
        f"• <b>Mode Sistem:</b> {mode_line}\n"
        f"{event_line}"
        f"• <b>Batas Layer Bersamaan:</b> <b>{session.max_positions} Posisi</b>\n"
        f"• <b>Indikator Strategi:</b> {strat_line}\n"
        f"• <b>Proteksi Anti-Banned:</b> 🛡️ <i>Throttle 500–1500ms • Step Filter 10pt • Anti-Error 130</i>\n"
        f"• <b>Analisa Tren:</b> <code>1D • 1H • 15M • 5M</code> ➔ <b>Action Open/Close:</b> <code>5M</code>\n"
        f"• <b>Ukuran Lot Order:</b> {lot_lbl}\n"
        f"• <b>Peta Tren MTF:</b> <i>{html.escape(session.mtf_status_line)}</i>\n\n"
        f"<b>💰 LIVE PROFIT MONITOR (REAL-TIME TANPA DELAY):</b>\n"
        f"• <b>Floating Profit (Berjalan):</b> {float_icon} <b><code>${floating_pnl:+.2f} USD ({floating_pct:+.2f}%)</code></b>\n"
        f"• <b>Realized Profit (Terealisasi):</b> <code>${realized_pnl:+.2f} USD</code> (🟢 <code>{session.winning_trades}W</code> / 🔴 <code>{session.losing_trades}L</code>)\n"
        f"• <b>Total Akumulasi Profit:</b> {net_icon} <b><code>${total_net_pnl:+.2f} USD</code></b>\n"
        f"• <b>Harga Live (5m) &amp; Siklus:</b> <code>{format_price(live_px)}</code> | Siklus <code>#{session.cycles_completed}</code> (<code>{upd_str}</code>)\n\n"
        f"{cycle_block}\n\n"
        f"<i>💡 Data profit di atas diperbarui otomatis secara real-time. Ketik <code>{cmd_prefix} off</code> untuk menonaktifkan sesi ini.</i>"
    )

def format_autotrade_report_card(report: AutoTradeExecutionReport) -> str:
    """Format the 6-Gate Pre-Trade Checklist & execution result card."""
    sym = html.escape(report.symbol)
    tf = html.escape(report.timeframe)

    if report.setup_required:
        is_trade = getattr(report, "trade_mode", False)
        prefix_title = "TRADE" if is_trade else "AUTO-TRADE"
        return (
            f"🔒 <b>{prefix_title} DITOLAK — WAJIB /SETUP DULU</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Akun Telegram Anda (ID: <code>{report.user_id}</code>) belum terhubung ke akun Exness Go / MT5 pribadi.\n\n"
            f"• <b>Aturan Sistem:</b> Kredensial pada <code>.env</code> hanya berlaku untuk <b>Role Admin</b>.\n"
            f"• Setiap user wajib menghubungkan akun Exness Go / MT5 masing-masing terlebih dahulu.\n\n"
            f"👉 Silakan ketik <code>/setup</code> untuk menghubungkan akun Exness Go / MT5 Anda."
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
        sl_note = " (🛡️ <i>Terkalibrasi Saldo Akun • Anti-MC</i>)" if getattr(report.decision, "event_mode", None) else ""
        tp_str = (
            "<i>Dihapus / Tanpa Batas (Runner Mode)</i>"
            if pos_r.take_profit_1 <= 0
            else f"<code>{format_price(pos_r.take_profit_1)}</code> | <code>{format_price(pos_r.take_profit_2)}</code>"
        )
        footer_block = (
            f"\n<b>🧾 Detail Order Exness Go / MT5 Anda:</b>\n"
            f"• <b>Order ID:</b> <code>{html.escape(ord_r.order_id)}</code> ({html.escape(ord_r.status)})\n"
            f"• <b>Aksi &amp; Simbol:</b> <b>{ord_r.side} {html.escape(ord_r.symbol)}</b>\n"
            f"• <b>Lot &amp; Harga Masuk:</b> <code>{ord_r.executed_qty} Lot</code> @ <code>{format_price(ord_r.avg_price)}</code> (<code>${ord_r.notional_usdt:,.2f}</code>)\n"
            f"• <b>Stop Loss (SL):</b> <code>{format_price(pos_r.stop_loss)}</code>{sl_note}\n"
            f"• <b>Take Profit (TP):</b> {tp_str}"
        )
    else:
        header_badge = "🛡️ <b>POSISI DITAHAN — BELUM LOLOS 6 GERBANG AKURASI</b>"
        reason = html.escape(report.rejection_summary or "Syarat pembukaan posisi belum terpenuhi.")
        footer_block = (
            f"\n<b>💡 Keputusan AI:</b>\n"
            f"<i>AI tidak membuka posisi sembarangan. {reason}</i>"
        )

    event_header = ""
    if report.decision and getattr(report.decision, "event_mode", None):
        ev_m = report.decision.event_mode
        ev_icon = get_macro_event_icon(ev_m)
        st_m = getattr(report.decision, "macro_stance", "NEUTRAL")
        icon_st = "🦅" if st_m == "HAWKISH" else ("🕊️" if st_m == "DOVISH" else "⚖️")
        bias_g = getattr(report.decision, "gold_action_bias", "HOLD")
        event_header = (
            f"{ev_icon} <b>US MACRO EVENT SNIPER: {ev_m}</b>\n"
            f"• <b>Sikap Kebijakan AS:</b> {icon_st} <b>{st_m}</b> (Transmisi Emas ➔ <b>{bias_g}</b>)\n"
            f"• <b>Status Blackout:</b> 🟢 <b>Dilewati (Bypassed)</b> — AI Aktif Berburu Setup\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    if getattr(report, "ea_mode", None) == "EA_EXPERT_ADVISOR":
        header_name = "EA EXPERT ADVISOR MT5 (RSI 14)"
        sub_rule = "🤖 <i>Open BUY jika RSI 14 ≤ 30, Open SELL jika RSI 14 ≥ 70 • SL 50 Pips / TP 150 Pips</i>\n"
    elif getattr(report, "ea_mode", None) == "EA_SCALPING":
        header_name = "EA SCALPING M1/M5 (MOMENTUM)"
        sub_rule = "⚡ <i>Fast Scalping M1/M5 Micro-Momentum • SL 20 Pips / TP 40 Pips</i>\n"
    elif getattr(report, "trade_mode", False):
        header_name = "NEXUSTRADE ALGO-TRADE (4 INDIKATOR MURNI)"
        sub_rule = "💎 <i>Eksekusi HANYA jika ke-4 Indikator Murni (UT Bot, Supertrend, Candle, OC Signals) selaras</i>\n"
    else:
        header_name = "LAPORAN AUTO-TRADE 6 GERBANG"
        sub_rule = ""
    return (
        f"🤖 <b>{header_name} — {sym} ({tf})</b>\n"
        f"{sub_rule}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        + event_header
        + f"<b>Status Eksekusi:</b> {header_badge}\n\n"
        f"<b>📋 Hasil Pemeriksaan 6 Gerbang AI:</b>\n"
        f"{gates_block}\n"
        f"{footer_block}"
    )
