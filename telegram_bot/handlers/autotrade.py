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

from ai_engine.auto_trader.binance_client import BinanceSpotClient
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.user_vault import UserVault
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe
from telegram_bot.handlers.handler_utils import (
    _PENDING_ADMIN_ACTION,
    _PENDING_SETUP_MODE,
    clear_pending_wizards,
    extract_user_identity,
    parse_command_args,
)
from telegram_bot.views.setup_formatters import (
    format_autotrade_report_card,
    format_myaccount_card,
    format_setup_prompt_key_card,
    format_setup_success_card,
)

autotrade_router = Router(name="autotrade")


def build_setup_keyboard(has_creds: bool = False, autotrade_on: bool = False) -> InlineKeyboardMarkup:
    """Build interactive inline keyboard for /setup, /myaccount, and /autotrade."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="🧪 Pilih Binance Testnet",
                callback_data="setup:mode:testnet",
            ),
            InlineKeyboardButton(
                text="🏦 Pilih Binance Live",
                callback_data="setup:mode:live",
            ),
        ],
        [
            InlineKeyboardButton(
                text="⚡ Coba Akun Demo Testnet Instan",
                callback_data="setup:demo",
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
    binance_client: BinanceSpotClient = data.get("binance_client") or BinanceSpotClient()
    auto_trader: SmartAutoTraderEngine = data.get("auto_trader") or SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
        binance_client=binance_client,
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
        updated = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        await message.answer(
            "🟢 <b>Auto-Trade Diaktifkan!</b>\n"
            "AI sekarang diizinkan mengeksekusi posisi pada akun Binance Anda jika ke-6 Gerbang Akurasi terpenuhi.\n\n"
            + format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), updated, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=True),
        )
        return

    if sub == "off":
        vault.set_autotrade_enabled(user_id, False)
        updated = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        await message.answer(
            "⚪ <b>Auto-Trade Dinonaktifkan.</b>\n"
            "Bot tidak akan membuka posisi otomatis baru sampai Anda mengaktifkannya kembali.\n\n"
            + format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), updated, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=False),
        )
        return

    if sub == "status":
        positions = vault.get_open_positions(user_id)
        await message.answer(
            format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), creds, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=creds.autotrade_enabled),
        )
        return

    if sub == "scan":
        symbol = normalize_symbol(args[1]) if len(args) >= 2 else "BTCUSDT"
        timeframe = normalize_timeframe(args[2]) if len(args) >= 3 else "15m"
    else:
        symbol = normalize_symbol(args[0])
        timeframe = normalize_timeframe(args[1]) if len(args) >= 2 else "15m"

    report = await auto_trader.evaluate_and_execute(
        user_id=user_id,
        symbol=symbol,
        timeframe=timeframe,
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
    binance_client: BinanceSpotClient = data.get("binance_client") or BinanceSpotClient()
    auto_trader: SmartAutoTraderEngine = data.get("auto_trader") or SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
        binance_client=binance_client,
    )

    msg = callback.message
    if msg is None:
        await callback.answer()
        return

    user_id, user_name = extract_user_identity(callback, default_name="Trader")
    cb_data = callback.data or ""

    if cb_data == "setup:mode:testnet":
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        _PENDING_SETUP_MODE[user_id] = True
        await msg.answer(format_setup_prompt_key_card(testnet=True), parse_mode="HTML")
        await callback.answer("Mode Binance Testnet dipilih")
        return

    if cb_data == "setup:mode:live":
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        _PENDING_SETUP_MODE[user_id] = False
        await msg.answer(format_setup_prompt_key_card(testnet=False), parse_mode="HTML")
        await callback.answer("Mode Binance Live dipilih")
        return

    if cb_data == "setup:demo":
        demo_key = f"TESTNET_DEMO_{user_id}"
        demo_secret = f"DEMO_SECRET_KEY_{user_id}_NEXUS"
        verification = await binance_client.verify_credentials(demo_key, demo_secret, testnet=True)
        creds = vault.save_user_credentials(
            user_id=user_id,
            api_key=demo_key,
            api_secret=demo_secret,
            username=user_name,
            testnet=True,
            verified_usdt_balance=verification.total_usdt,
            autotrade_enabled=True,
        )
        clear_pending_wizards(user_id)
        await msg.answer(
            format_setup_success_card(creds, verification, secret_deleted=False),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=True),
        )
        await callback.answer("Akun Demo Testnet aktif!")
        return

    if cb_data == "setup:myaccount":
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
            "🗑️ Kredensial Binance pribadi Anda telah dihapus dari brankas dan Auto-Trade dimatikan.",
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
