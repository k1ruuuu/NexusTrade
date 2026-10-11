from __future__ import annotations

import html
from typing import Any, Optional

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
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe
from telegram_bot.views.setup_formatters import (
    format_autotrade_report_card,
    format_myaccount_card,
    format_setup_prompt_key_card,
    format_setup_success_card,
    format_setup_wizard_card,
)

logger = get_logger("nexustrade.telegram.setup")

setup_router = Router(name="setup")

# Per-user pending interactive setup mode: user_id -> testnet (bool)
_PENDING_SETUP_MODE: dict[int, bool] = {}


def build_setup_keyboard(has_creds: bool = False, autotrade_on: bool = False) -> InlineKeyboardMarkup:
    """Build interactive inline keyboard for /setup and /myaccount."""
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


async def _try_delete_secret_message(message: Message) -> bool:
    """Immediately delete a user's Telegram message containing API credentials."""
    try:
        delete_fn = getattr(message, "delete", None)
        if callable(delete_fn):
            await delete_fn()
            return True
    except Exception as exc:
        logger.debug("Could not auto-delete secret message", error=str(exc))
    return False


async def _process_and_save_credentials(
    message: Message,
    user_id: int,
    user_name: str,
    api_key: str,
    api_secret: str,
    testnet: bool,
    risk_pct: Optional[float],
    vault: UserVault,
    binance_client: BinanceSpotClient,
) -> None:
    """Delete secret message, verify against Binance API, save encrypted in UserVault, and reply."""
    deleted = await _try_delete_secret_message(message)

    verification = await binance_client.verify_credentials(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    if not verification.valid:
        err = html.escape(verification.error_message or "API Key atau Secret tidak valid.")
        await message.answer(
            f"❌ <b>Verifikasi Akun Binance Gagal:</b>\n"
            f"• <b>Alasan:</b> {err}\n\n"
            f"<i>Pastikan API Key &amp; Secret benar serta izin Spot Trading aktif. Ketik <code>/setup</code> untuk mencoba lagi.</i>",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        return

    creds = vault.save_user_credentials(
        user_id=user_id,
        api_key=api_key,
        api_secret=api_secret,
        username=user_name,
        testnet=testnet,
        risk_per_trade_pct=risk_pct,
        verified_usdt_balance=verification.total_usdt,
        autotrade_enabled=True,
    )
    _PENDING_SETUP_MODE.pop(user_id, None)

    card = format_setup_success_card(creds, verification, secret_deleted=deleted)
    await message.answer(
        card,
        parse_mode="HTML",
        reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=creds.autotrade_enabled),
    )


@setup_router.message(Command("setup"))
async def handle_setup(message: Message, **data: Any) -> None:
    """
    Handle /setup command:
    - `/setup` -> Open interactive multi-user Binance setup wizard
    - `/setup [testnet|live] <API_KEY> <API_SECRET> [risk_pct]` -> Direct verification & encrypted save
    """
    settings: Settings = data.get("settings") or get_settings()
    vault: UserVault = data.get("user_vault") or UserVault(settings)
    binance_client: BinanceSpotClient = data.get("binance_client") or BinanceSpotClient()

    user = message.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    user_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Trader"
    )
    role = vault.get_user_role(user_id)

    parts = (message.text or "").strip().split()
    args = parts[1:] if parts and parts[0].startswith("/") else parts

    if not args:
        current_creds = vault.resolve_credentials(user_id)
        card = format_setup_wizard_card(user_name, user_id, role, current_creds)
        await message.answer(
            card,
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(
                has_creds=current_creds is not None,
                autotrade_on=bool(current_creds and current_creds.autotrade_enabled),
            ),
        )
        return

    # Parse direct command: /setup [testnet|live] <API_KEY> <API_SECRET> [risk_pct]
    first_lower = args[0].lower()
    if first_lower in ("testnet", "live", "spot"):
        testnet = first_lower == "testnet"
        key_args = args[1:]
    else:
        testnet = True
        key_args = args

    if len(key_args) < 2:
        await message.answer(
            "⚠️ <b>Format /setup kurang lengkap.</b>\n"
            "Gunakan format:\n"
            "<code>/setup testnet API_KEY_ANDA API_SECRET_ANDA</code>\n"
            "atau tekan tombol pilihan di bawah:",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        return

    api_key = key_args[0]
    api_secret = key_args[1]
    risk_pct: Optional[float] = None
    if len(key_args) >= 3:
        try:
            raw_r = float(key_args[2])
            risk_pct = raw_r / 100.0 if raw_r > 0.1 else raw_r
        except ValueError:
            risk_pct = None

    await _process_and_save_credentials(
        message=message,
        user_id=user_id,
        user_name=user_name,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        risk_pct=risk_pct,
        vault=vault,
        binance_client=binance_client,
    )


@setup_router.message(Command("cancel"))
async def handle_cancel_setup(message: Message) -> None:
    """Cancel pending interactive /setup or /admin wizard input."""
    from telegram_bot.handlers.admin import _PENDING_ADMIN_ACTION

    user_id = int(getattr(message.from_user, "id", 0) or 0)
    _PENDING_SETUP_MODE.pop(user_id, None)
    _PENDING_ADMIN_ACTION.pop(user_id, None)
    await message.answer("⚪ Sesi input interaktif telah dibatalkan.", parse_mode="HTML")

@setup_router.message(Command("myaccount"))
async def handle_myaccount(message: Message, **data: Any) -> None:
    """Handle /myaccount command displaying personal Binance setup & open positions."""
    settings: Settings = data.get("settings") or get_settings()
    vault: UserVault = data.get("user_vault") or UserVault(settings)

    user = message.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    user_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Trader"
    )
    role = vault.get_user_role(user_id)
    creds = vault.resolve_credentials(user_id)
    positions = vault.get_open_positions(user_id)

    card = format_myaccount_card(user_id, user_name, role, creds, positions)
    await message.answer(
        card,
        parse_mode="HTML",
        reply_markup=build_setup_keyboard(
            has_creds=creds is not None,
            autotrade_on=bool(creds and creds.autotrade_enabled),
        ),
    )


@setup_router.message(Command("unlink"))
async def handle_unlink(message: Message, **data: Any) -> None:
    """Handle /unlink command removing personal Binance credentials from the vault."""
    settings: Settings = data.get("settings") or get_settings()
    vault: UserVault = data.get("user_vault") or UserVault(settings)

    user_id = int(getattr(message.from_user, "id", 0) or 0)
    _PENDING_SETUP_MODE.pop(user_id, None)
    deleted = vault.delete_user_credentials(user_id)
    if deleted:
        await message.answer(
            "🗑️ <b>Akun Binance Pribadi Dihapus:</b>\n"
            "Kredensial API Key &amp; Secret Anda telah dihapus permanen dari brankas terenkripsi dan Auto-Trade dinonaktifkan.",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
    else:
        await message.answer(
            "ℹ️ Belum ada kredensial Binance pribadi yang tersimpan untuk akun Anda. Ketik <code>/setup</code> untuk mendaftar.",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )


@setup_router.message(Command("autotrade"))
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

    user = message.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    user_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Trader"
    )
    creds = vault.resolve_credentials(user_id)

    # Mandatory /setup check before any autotrade action
    if creds is None:
        report = await auto_trader.evaluate_and_execute(user_id=user_id, symbol="BTCUSDT")
        await message.answer(
            format_autotrade_report_card(report),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        return

    parts = (message.text or "").strip().split()
    args = parts[1:] if parts and parts[0].startswith("/") else parts
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

    # /autotrade scan [symbol] [tf] OR /autotrade <SYMBOL> [tf]
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


@setup_router.message(
    lambda msg: bool(
        msg.from_user
        and int(msg.from_user.id) in _PENDING_SETUP_MODE
        and msg.text
        and not msg.text.strip().startswith("/")
    )
)
async def handle_pending_setup_credentials(message: Message, **data: Any) -> None:
    """Capture interactive `<API_KEY> <API_SECRET>` reply after user selected Testnet or Live button."""
    settings: Settings = data.get("settings") or get_settings()
    vault: UserVault = data.get("user_vault") or UserVault(settings)
    binance_client: BinanceSpotClient = data.get("binance_client") or BinanceSpotClient()

    user = message.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    user_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Trader"
    )
    testnet = _PENDING_SETUP_MODE.get(user_id, True)
    tokens = (message.text or "").strip().split()

    if len(tokens) < 2:
        await _try_delete_secret_message(message)
        await message.answer(
            "⚠️ Format belum sesuai. Kirimkan <b>API Key</b> dan <b>API Secret</b> dipisahkan spasi:\n"
            "<code>API_KEY_ANDA API_SECRET_ANDA</code>",
            parse_mode="HTML",
        )
        return

    await _process_and_save_credentials(
        message=message,
        user_id=user_id,
        user_name=user_name,
        api_key=tokens[0],
        api_secret=tokens[1],
        testnet=testnet,
        risk_pct=None,
        vault=vault,
        binance_client=binance_client,
    )


@setup_router.callback_query(F.data.startswith("setup:") | F.data.startswith("autotrade:"))
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

    user = callback.from_user
    user_id = int(getattr(user, "id", 0) or 0)
    user_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Trader"
    )
    cb_data = callback.data or ""

    if cb_data == "setup:mode:testnet":
        from telegram_bot.handlers.admin import _PENDING_ADMIN_ACTION

        _PENDING_ADMIN_ACTION.pop(user_id, None)
        _PENDING_SETUP_MODE[user_id] = True
        await msg.answer(format_setup_prompt_key_card(testnet=True), parse_mode="HTML")
        await callback.answer("Mode Binance Testnet dipilih")
        return

    if cb_data == "setup:mode:live":
        from telegram_bot.handlers.admin import _PENDING_ADMIN_ACTION

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
        _PENDING_SETUP_MODE.pop(user_id, None)
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
        _PENDING_SETUP_MODE.pop(user_id, None)
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
