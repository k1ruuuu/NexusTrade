from __future__ import annotations

import html
from typing import Any, Optional

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ai_engine.auto_trader.binance_client import BinanceSpotClient
from ai_engine.auto_trader.mt5_exness_client import MT5ExnessClient
from ai_engine.auto_trader.user_vault import BrokerProvider
from ai_engine.auto_trader.user_vault import UserVault
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from telegram_bot.handlers.autotrade import (
    autotrade_router,
    build_setup_keyboard,
    handle_autotrade,
    handle_setup_callbacks,
)
from telegram_bot.handlers.handler_utils import (
    _PENDING_SETUP_MODE,
    _PENDING_SETUP_PROVIDER,
    clear_pending_wizards,
    extract_user_identity,
    parse_command_args,
)
from telegram_bot.views.setup_formatters import (
    format_myaccount_card,
    format_setup_success_card,
    format_setup_wizard_card,
)

logger = get_logger("nexustrade.telegram.setup")

setup_router = Router(name="setup")
setup_router.include_router(autotrade_router)

__all__ = [
    "_PENDING_SETUP_MODE",
    "build_setup_keyboard",
    "handle_autotrade",
    "handle_cancel_setup",
    "handle_myaccount",
    "handle_pending_setup_credentials",
    "handle_setup",
    "handle_setup_callbacks",
    "handle_unlink",
    "setup_router",
]


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
    provider: BrokerProvider = "binance_spot",
    server_name: Optional[str] = None,
    mt5_client: Optional[MT5ExnessClient] = None,
    metaapi_token_override: Optional[str] = None,
) -> None:
    """Delete secret message, verify against Broker/Exchange API, save encrypted in UserVault, and reply."""
    deleted = await _try_delete_secret_message(message)

    if provider == "exness_mt5":
        eff_mt5 = mt5_client or MT5ExnessClient()
        eff_srv = server_name or ("Exness-MT5Trial" if testnet else "Exness-MT5Real")
        verification = await eff_mt5.verify_credentials(
            account_id_or_login=api_key,
            token_or_password=api_secret,
            server_name=eff_srv,
            testnet=testnet,
            metaapi_token_override=metaapi_token_override,
        )
    else:
        eff_srv = server_name or ("Binance-Futures" if provider == "binance_futures" else "Binance-Spot")
        verification = await binance_client.verify_credentials(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )

    if not verification.valid:
        err = html.escape(verification.error_message or "Kredensial tidak valid.")
        await message.answer(
            f"❌ <b>Verifikasi Akun Broker Gagal:</b>\n"
            f"• <b>Alasan:</b> {err}\n\n"
            f"<i>Pastikan Login/API Key &amp; Password/Secret benar. Ketik <code>/setup</code> untuk mencoba lagi.</i>",
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
        provider=provider,
        server_name=eff_srv,
    )
    clear_pending_wizards(user_id)

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

    user_id, user_name = extract_user_identity(message, default_name="Trader")
    role = vault.get_user_role(user_id)
    args = parse_command_args(message.text)

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

    first_lower = args[0].lower()
    provider: BrokerProvider = "binance_spot"
    server_name: Optional[str] = None
    if first_lower in ("exness", "mt5", "exness_mt5", "exnessgo"):
        provider = "exness_mt5"
        key_args = args[1:]
        if len(key_args) >= 3 and not key_args[2].replace(".", "").isdigit():
            server_name = key_args[2]
        testnet = "real" not in (server_name or "").lower()
    elif first_lower in ("futures", "binance_futures"):
        provider = "binance_futures"
        testnet = True
        key_args = args[1:]
    elif first_lower in ("testnet", "live", "spot"):
        testnet = first_lower == "testnet"
        key_args = args[1:]
    else:
        testnet = True
        key_args = args

    if len(key_args) < 2:
        await message.answer(
            "⚠️ <b>Format /setup kurang lengkap.</b>\n"
            "Gunakan format:\n"
            "<code>/setup exness LOGIN_MT5 PASSWORD_ANDA Exness-MT5Trial</code>\n"
            "<code>/setup testnet API_KEY_ANDA API_SECRET_ANDA</code>\n"
            "atau tekan tombol pilihan di bawah:",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        return

    api_key = key_args[0]
    api_secret = key_args[1]
    risk_pct: Optional[float] = None
    metaapi_tok: Optional[str] = None
    if provider == "exness_mt5" and len(key_args) >= 4:
        metaapi_tok = key_args[3]
    elif len(key_args) >= 3 and server_name is None:
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
        provider=provider,
        server_name=server_name,
        mt5_client=data.get("mt5_client"),
        metaapi_token_override=metaapi_tok,
    )

@setup_router.message(Command("cancel"))
async def handle_cancel_setup(message: Message) -> None:
    """Cancel pending interactive /setup or /admin wizard input."""
    user_id, _ = extract_user_identity(message)
    clear_pending_wizards(user_id)
    await message.answer("⚪ Sesi input interaktif telah dibatalkan.", parse_mode="HTML")


@setup_router.message(Command("myaccount"))
async def handle_myaccount(message: Message, **data: Any) -> None:
    """Handle /myaccount command displaying personal Binance setup & open positions."""
    settings: Settings = data.get("settings") or get_settings()
    vault: UserVault = data.get("user_vault") or UserVault(settings)

    user_id, user_name = extract_user_identity(message, default_name="Trader")
    role = vault.get_user_role(user_id)
    creds = vault.resolve_credentials(user_id)
    if creds is not None and creds.source == "user_setup":
        if creds.provider == "exness_mt5" and not MT5ExnessClient._is_exness_trial_demo(creds.api_key, creds.testnet):
            eff_mt5: MT5ExnessClient = data.get("mt5_client") or MT5ExnessClient()
            ver = await eff_mt5.verify_credentials(
                creds.api_key, creds.api_secret, server_name=creds.server_name, testnet=creds.testnet
            )
            if ver.valid and ver.total_usdt >= 0:
                vault.update_verified_balance(user_id, ver.total_usdt)
                creds.verified_usdt_balance = ver.total_usdt
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

    user_id, _ = extract_user_identity(message)
    clear_pending_wizards(user_id)
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

    user_id, user_name = extract_user_identity(message, default_name="Trader")
    testnet = _PENDING_SETUP_MODE.get(user_id, True)
    prov_tuple = _PENDING_SETUP_PROVIDER.get(user_id, ("binance_spot", "Binance-Spot"))
    provider: BrokerProvider = prov_tuple[0]  # type: ignore[assignment]
    default_srv = prov_tuple[1]
    tokens = (message.text or "").strip().split()

    if len(tokens) < 2:
        await _try_delete_secret_message(message)
        await message.answer(
            "⚠️ Format belum sesuai. Kirimkan <b>Login/API Key</b> dan <b>Password/Secret</b> dipisahkan spasi:\n"
            "<code>LOGIN_ATAU_KEY PASSWORD_ATAU_SECRET</code>",
            parse_mode="HTML",
        )
        return

    srv_name = tokens[2] if (provider == "exness_mt5" and len(tokens) >= 3) else default_srv
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
        provider=provider,
        server_name=srv_name,
        mt5_client=data.get("mt5_client"),
    )
