from __future__ import annotations

from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from ai_engine.auto_trader.mt5_exness_client import MT5ExnessClient
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.session_manager import is_session_duration_token
from ai_engine.auto_trader.user_vault import UserVault
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe
from telegram_bot.handlers.handler_utils import (
    _PENDING_ADMIN_ACTION,
    _PENDING_SETUP_MODE,
    _PENDING_SETUP_PROVIDER,
    clear_pending_wizards,
    extract_user_identity,
    parse_command_args,
)
from telegram_bot.views.setup_formatters import (
    format_autotrade_report_card,
    format_myaccount_card,
    format_setup_prompt_key_card,
    format_setup_success_card,
    format_timed_session_started_card,
)

autotrade_router = Router(name="autotrade")


def build_setup_keyboard(has_creds: bool = False, autotrade_on: bool = False) -> InlineKeyboardMarkup:
    """Build interactive inline keyboard for /setup, /myaccount, and /autotrade."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="📱 Pilih Exness Go / MT5 (Trial)",
                callback_data="setup:mode:exness_trial",
            ),
            InlineKeyboardButton(
                text="📱 Pilih Exness Go / MT5 (Real)",
                callback_data="setup:mode:exness_live",
            ),
        ],
        [
            InlineKeyboardButton(
                text="⚡ Demo Exness Go Instan",
                callback_data="setup:demo:exness",
            ),
        ],
    ]
    if has_creds:
        toggle_txt = "🔴 Matikan Auto-Trade" if autotrade_on else "🟢 Aktifkan Auto-Trade"
        toggle_cb = "autotrade:toggle:off" if autotrade_on else "autotrade:toggle:on"
        rows.append(
            [
                InlineKeyboardButton(text=toggle_txt, callback_data=toggle_cb),
                InlineKeyboardButton(
                    text="🤖 Scan & Auto-Trade BTC",
                    callback_data="autotrade:scan:BTCUSDT",
                ),
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(text="👤 Akun Saya", callback_data="setup:myaccount"),
                InlineKeyboardButton(text="🗑️ Putuskan Akun (/unlink)", callback_data="setup:unlink"),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(text="🏠 Menu Utama", callback_data="cmd:dashboard"),
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


@autotrade_router.message(Command("autotrade"))
async def handle_autotrade(message: Message, **data: Any) -> None:
    """
    Handle /autotrade command:
    - `/autotrade` or `/autotrade status` -> Show current auto-trade status (requires /setup)
    - `/autotrade on` -> Enable auto-trade for this user (requires /setup)
    - `/autotrade off` -> Disable auto-trade for this user
    - `/autotrade scan [symbol] [timeframe]` or `/autotrade [symbol]` -> Run 6-Gate check & execute if passed
    """
    settings: Settings = data.get("settings") or get_settings()
    vault: UserVault = data.get("user_vault") or UserVault(settings)
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)
    auto_trader: SmartAutoTraderEngine = data.get("auto_trader") or SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
    )

    user_id, user_name = extract_user_identity(message, default_name="Trader")
    creds = vault.resolve_credentials(user_id)

    if creds is None:
        report = await auto_trader.evaluate_and_execute(user_id=user_id, symbol="BTCUSDT")
        await message.answer(
            format_autotrade_report_card(report),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        return

    args = parse_command_args(message.text)
    sub = args[0].lower() if args else "status"

    if sub == "on":
        vault.set_autotrade_enabled(user_id, True)
        await auto_trader.sync_live_mt5_positions(user_id)
        updated = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        await message.answer(
            "🟢 <b>Auto-Trade Diaktifkan!</b>\n"
            "AI sekarang diizinkan mengeksekusi posisi pada akun Exness Go / MT5 Anda jika ke-6 Gerbang Akurasi terpenuhi.\n\n"
            + format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), updated, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=True),
        )
        return

    if sub in ("off", "stop"):
        stopped_sessions = auto_trader.stop_user_sessions(user_id)
        vault.set_autotrade_enabled(user_id, False)
        await auto_trader.sync_live_mt5_positions(user_id)
        updated = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        stop_note = (
            f" ({len(stopped_sessions)} sesi berjangka waktu dihentikan)"
            if stopped_sessions
            else ""
        )
        await message.answer(
            f"⚪ <b>Auto-Trade Dinonaktifkan{stop_note}.</b>\n"
            f"Bot tidak akan membuka posisi otomatis baru sampai Anda mengaktifkannya kembali.\n\n"
            + format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), updated, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=False),
        )
        return

    if sub == "status":
        await auto_trader.sync_live_mt5_positions(user_id)
        positions = vault.get_open_positions(user_id)
        active_sessions = auto_trader.get_user_sessions(user_id)
        sess_lines = ""
        if active_sessions:
            s_items = [
                f"• <b>{s.symbol} ({s.duration_label})</b>: Sisa <code>{s.remaining_human}</code> | "
                f"Siklus: <code>{s.cycles_completed}x</code> | <i>{s.last_check_summary}</i>"
                for s in active_sessions
            ]
            sess_lines = "\n\n<b>⏱️ Sesi Auto-Trade Berjangka Aktif:</b>\n" + "\n".join(s_items)
        await message.answer(
            format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), creds, positions)
            + sess_lines,
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=creds.autotrade_enabled),
        )
        return

    # Parse `/autotrade <SYMBOL> [DURATION] [LOT] [TIMEFRAME]` in any order
    from ai_engine.auto_trader.session_manager import parse_custom_lot_token

    raw_tokens = args[1:] if sub == "scan" else args
    symbol = normalize_symbol(raw_tokens[0]) if len(raw_tokens) >= 1 else "BTCUSDT"
    duration_token: str | None = None
    custom_lot: float | None = None
    timeframe = "15m"

    for tok in raw_tokens[1:]:
        lot_val = parse_custom_lot_token(tok)
        if lot_val is not None and custom_lot is None:
            custom_lot = lot_val
        elif is_session_duration_token(tok) and duration_token is None:
            duration_token = tok
        else:
            timeframe = normalize_timeframe(tok)

    if duration_token is not None:
        async def _notify_user(uid: int, text_html: str) -> None:
            try:
                await message.answer(text_html, parse_mode="HTML")
            except Exception:
                pass

        session_state, first_res = await auto_trader.start_timed_session(
            user_id=user_id,
            symbol=symbol,
            duration_code=duration_token,
            timeframe=timeframe,
            notify_callback=_notify_user,
            custom_lot=custom_lot,
        )
        await message.answer(
            format_timed_session_started_card(session_state, first_res),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=True),
        )
        return

    # If user already has an OPEN position on `symbol`, sync with Exness MT5 first then monitor & hold it
    await auto_trader.sync_live_mt5_positions(user_id)
    existing_open = vault.get_open_positions(user_id, symbol=symbol)
    if existing_open:
        mon_res = await auto_trader.monitor_and_manage_open_position(
            user_id=user_id,
            symbol=symbol,
            timeframe=timeframe,
        )
        p = mon_res.position or existing_open[0]
        if mon_res.action_taken in ("CLOSED_TP", "CLOSED_SL"):
            await message.answer(
                f"🎯 <b>POSISI {p.symbol} ({p.side}) DITUTUP OTOMATIS</b>\n"
                f"• <b>Alasan:</b> {mon_res.exit_reason}\n"
                f"• <b>Harga Keluar:</b> <code>${mon_res.current_price:,.2f}</code>\n"
                f"• <b>Realized PnL:</b> <code>${mon_res.realized_pnl_usdt:+.2f} USDT</code>",
                parse_mode="HTML",
                reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=creds.autotrade_enabled),
            )
            return
        await message.answer(
            f"🛡️ <b>MENAHAN POSISI {p.symbol} ({p.side}) MENUJU TARGET PROFIT BESAR</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Harga Masuk:</b> <code>${p.entry_price:,.2f}</code> | <b>Harga Saat Ini:</b> <code>${mon_res.current_price:,.2f}</code>\n"
            f"• <b>Floating PnL:</b> <code>${mon_res.unrealized_pnl_usdt:+.2f} USDT ({mon_res.unrealized_pnl_pct:+.2f}%)</code>\n"
            f"• <b>Stop Loss / Trailing Stop:</b> <code>${p.stop_loss:,.2f}</code>\n"
            f"• <b>Target Profit 1 &amp; 2:</b> <code>${p.take_profit_1:,.2f}</code> | <code>${p.take_profit_2:,.2f}</code>\n\n"
            f"<i>💡 AI tidak menutup posisi terlalu cepat. Posisi terus dikawal dengan Trailing Stop hingga menyentuh TP1/TP2.</i>",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=creds.autotrade_enabled),
        )
        return

    report = await auto_trader.evaluate_and_execute(
        user_id=user_id,
        symbol=symbol,
        timeframe=timeframe,
        custom_lot=custom_lot,
    )
    await message.answer(
        format_autotrade_report_card(report),
        parse_mode="HTML",
        reply_markup=build_setup_keyboard(
            has_creds=True,
            autotrade_on=creds.autotrade_enabled,
        ),
    )


@autotrade_router.callback_query(F.data.startswith("setup:") | F.data.startswith("autotrade:"))
async def handle_setup_callbacks(callback: CallbackQuery, **data: Any) -> None:
    """Handle inline keyboard actions for /setup, /myaccount, /unlink, and /autotrade."""
    settings: Settings = data.get("settings") or get_settings()
    vault: UserVault = data.get("user_vault") or UserVault(settings)
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)
    auto_trader: SmartAutoTraderEngine = data.get("auto_trader") or SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
    )

    msg = callback.message
    if msg is None:
        await callback.answer()
        return

    user_id, user_name = extract_user_identity(callback, default_name="Trader")
    cb_data = callback.data or ""

    if cb_data in ("setup:mode:exness_trial", "setup:mode:testnet"):
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        _PENDING_SETUP_MODE[user_id] = True
        _PENDING_SETUP_PROVIDER[user_id] = ("exness_mt5", "Exness-MT5Trial7")
        await msg.answer(format_setup_prompt_key_card(testnet=True, provider="exness_mt5"), parse_mode="HTML")
        await callback.answer("Mode Exness Go / MT5 Trial dipilih")
        return

    if cb_data in ("setup:mode:exness_live", "setup:mode:live"):
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        _PENDING_SETUP_MODE[user_id] = False
        _PENDING_SETUP_PROVIDER[user_id] = ("exness_mt5", "Exness-MT5Real")
        await msg.answer(format_setup_prompt_key_card(testnet=False, provider="exness_mt5"), parse_mode="HTML")
        await callback.answer("Mode Exness Go / MT5 Real dipilih")
        return

    if cb_data in ("setup:demo:exness", "setup:demo"):
        mt5_client: MT5ExnessClient = data.get("mt5_client") or MT5ExnessClient()
        demo_login = f"EXNESS_DEMO_{user_id}"
        demo_pass = f"ExnessGo_{user_id}_Trial"
        verification = await mt5_client.verify_credentials(
            demo_login, demo_pass, server_name="Exness-MT5Trial7", testnet=True
        )
        creds = vault.save_user_credentials(
            user_id=user_id,
            api_key=demo_login,
            api_secret=demo_pass,
            username=user_name,
            testnet=True,
            verified_usdt_balance=verification.total_usdt,
            autotrade_enabled=True,
            provider="exness_mt5",
            server_name="Exness-MT5Trial7",
        )
        clear_pending_wizards(user_id)
        await msg.answer(
            format_setup_success_card(creds, verification, secret_deleted=False),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=True),
        )
        await callback.answer("Akun Demo Exness Go / MT5 aktif!")
        return

    if cb_data == "setup:myaccount":
        await auto_trader.sync_live_mt5_positions(user_id)
        creds = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        await msg.answer(
            format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), creds, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(
                has_creds=creds is not None,
                autotrade_on=bool(creds and creds.autotrade_enabled),
            ),
        )
        await callback.answer()
        return

    if cb_data == "setup:unlink":
        clear_pending_wizards(user_id)
        vault.delete_user_credentials(user_id)
        await msg.answer(
            "🗑️ Kredensial Exness Go / MT5 pribadi Anda telah dihapus dari brankas dan Auto-Trade dimatikan.",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        await callback.answer("Akun diputuskan")
        return

    if cb_data in ("autotrade:toggle:on", "autotrade:toggle:off"):
        enable = cb_data.endswith(":on")
        ok = vault.set_autotrade_enabled(user_id, enable)
        if not ok:
            report = await auto_trader.evaluate_and_execute(user_id=user_id, symbol="BTCUSDT")
            await msg.answer(
                format_autotrade_report_card(report),
                parse_mode="HTML",
                reply_markup=build_setup_keyboard(has_creds=False),
            )
            await callback.answer("Wajib /setup dulu", show_alert=True)
            return
        creds = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        await msg.answer(
            format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), creds, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=enable),
        )
        await callback.answer("Auto-Trade ON" if enable else "Auto-Trade OFF")
        return

    if cb_data.startswith("autotrade:scan:"):
        sym = normalize_symbol(cb_data.split(":")[-1])
        report = await auto_trader.evaluate_and_execute(user_id=user_id, symbol=sym, timeframe="15m")
        creds = vault.resolve_credentials(user_id)
        await msg.answer(
            format_autotrade_report_card(report),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(
                has_creds=creds is not None,
                autotrade_on=bool(creds and creds.autotrade_enabled),
            ),
        )
        await callback.answer()
