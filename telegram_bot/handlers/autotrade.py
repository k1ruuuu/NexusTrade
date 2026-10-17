from __future__ import annotations

from typing import Any, Optional

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
from ai_engine.auto_trader.session_manager import (
    format_position_closed_notification,
)
from ai_engine.auto_trader.session_models import (
    AutoTradeSessionState,
    parse_custom_lot_token,
    parse_macro_event_mode_token,
    parse_max_positions_token,
)
from data_pipeline.market_symbols import normalize_symbol
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

async def _safe_answer_callback(
    callback: CallbackQuery,
    text: Optional[str] = None,
    show_alert: bool = False,
) -> None:
    """Defensively answer callback query; suppresses timeout exceptions if query expired."""
    try:
        await callback.answer(text=text, show_alert=show_alert)
    except Exception:
        pass


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
                InlineKeyboardButton(
                    text="🛑 Tutup Semua Posisi & Hapus Pending",
                    callback_data="autotrade:panic",
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


@autotrade_router.message(
    Command(
        "autotrade",
        "trade",
        "ea-scp",
        "ea_scp",
        "eascp",
        "ea_scalping",
        "eascalping",
        "scalping",
        "ea-EA",
        "ea_EA",
        "ea-ea",
        "ea_ea",
        "eaea",
        "ea_grid",
        "ea-grid",
        "eagrid",
        "grid",
        "ea_break",
        "ea-break",
        "eabreak",
        "nxbreak",
        "ea_parsial",
        "ea-parsial",
        "eaparsial",
        "nxparsial",
        "ea_reverse",
        "ea-reverse",
        "eareverse",
        "nxreverse",
        "ea_poc",
        "ea-poc",
        "eapoc",
        "nxpoc",
        ignore_case=True,
    )
)
async def handle_autotrade(message: Message, **data: Any) -> None:
    """
    Handle /autotrade, /trade, /ea-scp, and /ea-EA commands:
    - `/ea-EA {symbol} [lot]` -> Starts EA Expert Advisor MT5 (RSI 14 | SL 50p | TP 150p | 3 Anti-Ban Protections)
    - `/ea-scp {symbol} [lot]` -> Starts EA Scalping M1/M5 (Fast Momentum | SL 20p | TP 40p)
    - `/trade [scan] [symbol]` -> Strict 4-Indicator Confluence (UT Bot, Supertrend, Candle, OC Signals)
    - `/autotrade [scan] [symbol]` -> 6-Gate Auto-Trader
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
    cmd_token = (message.text or "").strip().split()[0].lower() if message.text else "/autotrade"
    is_trade_mode = cmd_token.startswith("/trade")
    is_ea_scp = any(cmd_token.startswith(p) for p in ("/ea-scp", "/ea_scp", "/eascp", "/ea_scalping", "/eascalping", "/scalping"))
    is_ea_ea = any(cmd_token.startswith(p) for p in ("/ea-ea", "/ea_ea", "/eaea"))
    is_ea_grid = any(cmd_token.startswith(p) for p in ("/ea_grid", "/ea-grid", "/eagrid", "/grid"))
    is_nx_break = any(cmd_token.startswith(p) for p in ("/ea_break", "/ea-break", "/eabreak", "/nxbreak"))
    is_nx_parsial = any(cmd_token.startswith(p) for p in ("/ea_parsial", "/ea-parsial", "/eaparsial", "/nxparsial"))
    is_nx_reverse = any(cmd_token.startswith(p) for p in ("/ea_reverse", "/ea-reverse", "/eareverse", "/nxreverse"))
    is_nx_poc = any(cmd_token.startswith(p) for p in ("/ea_poc", "/ea-poc", "/eapoc", "/nxpoc"))
    ea_mode: Optional[str] = (
        "EA_NX_BREAK"
        if is_nx_break
        else (
            "EA_NX_PARSIAL"
            if is_nx_parsial
            else (
                "EA_NX_REVERSE"
                if is_nx_reverse
                else (
                    "EA_NX_POC"
                    if is_nx_poc
                    else (
                        "EA_GRID_SCALPER"
                        if is_ea_grid
                        else ("EA_SCALPING" if is_ea_scp else ("EA_EXPERT_ADVISOR" if is_ea_ea else None))
                    )
                )
            )
        )
    )
    cmd_name = (
        "/ea_break"
        if is_nx_break
        else (
            "/ea_parsial"
            if is_nx_parsial
            else (
                "/ea_reverse"
                if is_nx_reverse
                else (
                    "/ea_poc"
                    if is_nx_poc
                    else (
                        "/ea_grid"
                        if is_ea_grid
                        else ("/ea-scp" if is_ea_scp else ("/ea-EA" if is_ea_ea else ("/trade" if is_trade_mode else "/autotrade")))
                    )
                )
            )
        )
    )
    args = parse_command_args(message.text)
    sub = args[0].lower() if args else "status"
    raw_tokens = args[1:] if sub == "scan" else args
    req_symbol = (
        normalize_symbol(raw_tokens[0])
        if (len(raw_tokens) >= 1 and raw_tokens[0].lower() not in ("on", "off", "stop", "status"))
        else "BTCUSDT"
    )
    creds = vault.resolve_credentials(user_id)

    if creds is None:
        report = await auto_trader.evaluate_and_execute(
            user_id=user_id, symbol=req_symbol, trade_mode=is_trade_mode, ea_mode=ea_mode
        )
        await message.answer(
            format_autotrade_report_card(report),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        return

    if sub == "on":
        vault.set_autotrade_enabled(user_id, True)
        await auto_trader.sync_live_mt5_positions(user_id)
        updated = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        if is_ea_grid:
            on_title = (
                "🟢 <b>Mode EA Grid/Trap Scalper M1 Diaktifkan!</b>\n"
                "Eksekusi trap stop ganda (BUY_STOP &amp; SELL_STOP) pada M1 dengan auto-compounding lot ($10 ➔ 0.01 lot), "
                "target profit $5,000, dan pembersihan posisi massal.\n\n"
            )
        elif is_nx_break:
            on_title = (
                "🟢 <b>Mode EA NXBreak (M1/M5 Breakout) Diaktifkan!</b>\n"
                "Breakout candle berkualitas (Body ≥ 55%, Range 0.4–2.5 ATR, Buffer 0.05 ATR) dengan SL 300pt & TP 1:1 RR.\n\n"
            )
        elif is_nx_parsial:
            on_title = (
                "🟢 <b>Mode EA NXParsial (3-Leg Basket) Diaktifkan!</b>\n"
                "Breakout candle 3-leg berurutan dengan shared risk budget dan target berjenjang (1R, 3R, 6R).\n\n"
            )
        elif is_nx_reverse:
            on_title = (
                "🟢 <b>Mode EA NXReverse (Trend & 1:5 RR Recovery) Diaktifkan!</b>\n"
                "EMA 20>50>200 + Breakout Rolling 20-bar SNR, dynamic swing stop, TP 1:5 RR, dan recovery switching 10-bar.\n\n"
            )
        elif is_nx_poc:
            on_title = (
                "🟢 <b>Mode EA NX-POC (Volume Profile Rejection) Diaktifkan!</b>\n"
                "100-bar 40-bin Point of Control rejection pinbar dengan filter ADX ≤ 25 dan target 1:1 RR menuju POC.\n\n"
            )
        elif is_ea_ea:
            on_title = (
                "🟢 <b>Mode EA Expert Advisor MT5 Diaktifkan!</b>\n"
                "Eksekusi otomatis berbasis <b>RSI 14</b> (RSI >= 70 SELL, RSI <= 30 BUY), "
                "SL 50 Pips, TP 150 Pips (1:3 RRR), dan 3 Proteksi Anti-Broker-Ban.\n\n"
            )
        elif is_ea_scp:
            on_title = (
                "🟢 <b>Mode EA Scalping Diaktifkan!</b>\n"
                "• <b>Membaca Arah Trend:</b> EMA 9/21/50 M1/M5 momentum.\n"
                "• <b>Membaca Candlestick:</b> Rejection Pinbar, Engulfing, & Momentum Body.\n"
                "• <b>Membaca Pergerakan Market:</b> Prediksi ekspansi arah selanjutnya.\n"
                "• <b>SL & TP Berdekatan:</b> SL 12 Pips / TP 14 Pips untuk eksekusi kilat.\n"
                "• <b>Auto Re-Entry Seketika:</b> Langsung entry kembali saat terkena TP maupun SL.\n\n"
            )
        elif is_trade_mode:
            on_title = (
                "🟢 <b>Mode /trade (4 Indikator Murni) Diaktifkan!</b>\n"
                "AI siap memantau pasar dan <b>HANYA</b> akan membuka posisi BUY/SELL jika ke-4 Indikator "
                "(UT Bot, Supertrend, Candlestick Patterns, OC Signals) selaras dan didukung pembacaan candle, tren MTF, serta berita.\n\n"
            )
        else:
            on_title = (
                "🟢 <b>Auto-Trade 24/7 Non-Stop Diaktifkan!</b>\n"
                "AI siap menganalisis tren <b>1D, 1H, 15M, 5M</b> dan mengeksekusi open/close posisi di <b>1M</b> secara 24/7 pada akun Exness Go / MT5 Anda.\n\n"
            )
        await message.answer(
            on_title + format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), updated, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=True),
        )
        return
    if sub in ("off", "stop"):
        target_token = args[1].strip() if len(args) > 1 else None
        if target_token and target_token.lower() not in ("all", "semua"):
            stopped_one = auto_trader.stop_session_by_id(user_id, target_token)
            if stopped_one is not None:
                remaining_sess = auto_trader.get_user_sessions(user_id)
                rem_note = f"\n\nSisa sesi aktif: {len(remaining_sess)} sesi." if remaining_sess else "\n\nSemua sesi Auto-Trade telah selesai."
                if not remaining_sess:
                    vault.set_autotrade_enabled(user_id, False)
                await message.answer(
                    f"🛑 <b>Sesi Auto-Trade [ID #{stopped_one.short_id}] ({stopped_one.symbol}) Telah Dihentikan.</b>\n"
                    f"• <b>Status:</b> Nonaktif (Dibatalkan user)\n"
                    f"• <b>Total Siklus:</b> <code>{stopped_one.cycles_completed}x</code>\n"
                    f"• <b>Realized PnL:</b> <code>${stopped_one.realized_pnl_usdt:+.2f} USD</code>"
                    + rem_note,
                    parse_mode="HTML",
                    reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=bool(remaining_sess)),
                )
                return
            else:
                stopped_sym = auto_trader.stop_user_sessions(user_id, symbol=target_token)
                if stopped_sym:
                    remaining_sess = auto_trader.get_user_sessions(user_id)
                    if not remaining_sess:
                        vault.set_autotrade_enabled(user_id, False)
                    await message.answer(
                        f"🛑 <b>{len(stopped_sym)} Sesi Auto-Trade pada {target_token.upper()} Telah Dihentikan.</b>",
                        parse_mode="HTML",
                        reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=bool(remaining_sess)),
                    )
                    return
                else:
                    await message.answer(
                        f"⚠️ <b>Sesi Auto-Trade dengan ID '<code>{html.escape(target_token)}</code>' tidak ditemukan atau sudah nonaktif.</b>\n"
                        f"Ketik <code>/autotrade status</code> untuk melihat daftar ID sesi yang sedang aktif.",
                        parse_mode="HTML",
                    )
                    return

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
            f"⚪ <b>Seluruh Sesi Auto-Trade Dinonaktifkan{stop_note}.</b>\n"
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
                f"• <b>[ID #{s.short_id}] {s.symbol}</b> ({s.duration_label} | {s.custom_lot or 'Auto'} Lot | Max {s.max_positions} Posisi):\n"
                f"   Sisa: <code>{s.remaining_human}</code> | Siklus: <code>{s.cycles_completed}x</code> | <i>{s.last_check_summary}</i>\n"
                f"   👉 <i>Ketik <code>/autotrade stop {s.short_id}</code> untuk menghentikan sesi ini saja.</i>"
                for s in active_sessions
            ]
            sess_lines = "\n\n<b>⏱️ Sesi Auto-Trade Aktif Berdasarkan ID:</b>\n\n" + "\n\n".join(s_items)
        await message.answer(
            format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), creds, positions)
            + sess_lines,
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=creds.autotrade_enabled),
        )
        return

    # Parse `/autotrade <SYMBOL> [LOT] [EVENT_MODE] [MAX_POSITIONS]` -> Starts 24/7 Auto-Trade
    raw_tokens = args[1:] if sub == "scan" else args
    symbol = normalize_symbol(raw_tokens[0]) if len(raw_tokens) >= 1 else "BTCUSDT"
    custom_lot: float | None = None
    macro_event_mode: str | None = None
    max_positions: int = 1

    for tok in raw_tokens[1:]:
        pos_val = parse_max_positions_token(tok)
        if pos_val is not None and max_positions == 1:
            max_positions = pos_val
            continue
        lot_val = parse_custom_lot_token(tok)
        if lot_val is not None and custom_lot is None:
            custom_lot = lot_val
            continue
        ev_mode = parse_macro_event_mode_token(tok)
        if ev_mode is not None and macro_event_mode is None:
            macro_event_mode = ev_mode
            continue
    if sub != "scan":
        live_msg_ref: list[Any] = []
        kb_markup = build_setup_keyboard(has_creds=True, autotrade_on=True)

        async def _notify_user(uid: int, text_html: str) -> None:
            try:
                await message.answer(text_html, parse_mode="HTML")
            except Exception:
                pass

        async def _update_live_card(uid: int, card_html: str) -> None:
            if not live_msg_ref:
                return
            target = live_msg_ref[0]
            edit_fn = getattr(target, "edit_text", None)
            if callable(edit_fn):
                try:
                    await edit_fn(card_html, parse_mode="HTML", reply_markup=kb_markup)
                except Exception:
                    pass

        session_state, first_res = await auto_trader.start_timed_session(
            user_id=user_id,
            symbol=symbol,
            duration_code="24/7",
            timeframe="1m",
            notify_callback=_notify_user,
            custom_lot=custom_lot,
            live_card_callback=_update_live_card,
            macro_event_mode=macro_event_mode,
            max_positions=max_positions,
            is_trade_mode=is_trade_mode,
            ea_mode=ea_mode,
        )
        sent_card = await message.answer(
            format_timed_session_started_card(session_state, first_res),
            parse_mode="HTML",
            reply_markup=kb_markup,
        )
        if sent_card is not None:
            live_msg_ref.append(sent_card)
        return
    # If user already has an OPEN position on `symbol`, sync with Exness MT5 first then monitor & hold it
    await auto_trader.sync_live_mt5_positions(user_id)
    existing_open = vault.get_open_positions(user_id, symbol=symbol)
    if existing_open:
        mon_res = await auto_trader.monitor_and_manage_open_position(
            user_id=user_id,
            symbol=symbol,
            timeframe="1m",
        )
        p = mon_res.position or existing_open[0]
        if mon_res.action_taken in ("CLOSED_TP", "CLOSED_SL"):
            await message.answer(
                format_position_closed_notification(
                    mon_res,
                    header_title=f"POSISI {p.symbol} ({p.side}) DITUTUP OTOMATIS",
                    pnl_label="Realized PnL",
                ),
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
        timeframe="1m",
        custom_lot=custom_lot,
        macro_event_mode=macro_event_mode,
        max_positions=max_positions,
        trade_mode=is_trade_mode,
        ea_mode=ea_mode,
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
    # Immediate answer to dismiss client spinning state (<20ms) and prevent timeout
    await _safe_answer_callback(callback)
    if msg is None:
        return

    user_id, user_name = extract_user_identity(callback, default_name="Trader")
    cb_data = callback.data or ""

    if cb_data.startswith("autotrade:close_all_grid:") or cb_data == "autotrade:panic":
        parts = cb_data.split(":")
        target_sym = parts[2] if len(parts) > 2 else "XAUUSD"
        creds = vault.resolve_credentials(user_id)
        if creds:
            res = await auto_trader.mt5_client.close_all_and_delete_pending(
                account_id_or_login=creds.api_key,
                token_or_password=creds.api_secret,
                symbol=target_sym,
                server_name=creds.server_name,
            )
            await msg.answer(
                f"🛑 <b>MASS CLEANUP BERHASIL — {target_sym}</b>\n"
                f"• Posisi Pasar Ditutup: <b>{res.get('positions_closed', 0)} posisi</b>\n"
                f"• Pending Trap Dihapus: <b>{res.get('pending_deleted', 0)} order</b>\n\n"
                f"<i>Seluruh order dan pending trap pada {target_sym} telah dibersihkan.</i>",
                parse_mode="HTML",
                reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=False),
            )
            await _safe_answer_callback(callback, "Semua posisi & pending order dibersihkan!")
        else:
            await _safe_answer_callback(callback, "Akun belum terhubung (/setup)")
        return

    if cb_data in ("setup:mode:exness_trial", "setup:mode:testnet"):
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        _PENDING_SETUP_MODE[user_id] = True
        _PENDING_SETUP_PROVIDER[user_id] = ("exness_mt5", "Exness-MT5Trial7")
        await msg.answer(format_setup_prompt_key_card(testnet=True, provider="exness_mt5"), parse_mode="HTML")
        await _safe_answer_callback(callback, "Mode Exness Go / MT5 Trial dipilih")
        return

    if cb_data in ("setup:mode:exness_live", "setup:mode:live"):
        _PENDING_ADMIN_ACTION.pop(user_id, None)
        _PENDING_SETUP_MODE[user_id] = False
        _PENDING_SETUP_PROVIDER[user_id] = ("exness_mt5", "Exness-MT5Real")
        await msg.answer(format_setup_prompt_key_card(testnet=False, provider="exness_mt5"), parse_mode="HTML")
        await _safe_answer_callback(callback, "Mode Exness Go / MT5 Real dipilih")
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
        await _safe_answer_callback(callback, "Akun Demo Exness Go / MT5 aktif!")
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
        await _safe_answer_callback(callback)
        return

    if cb_data == "setup:unlink":
        clear_pending_wizards(user_id)
        auto_trader.stop_user_sessions(user_id)
        vault.delete_user_credentials(user_id)
        await msg.answer(
            "🗑️ Kredensial Exness Go / MT5 pribadi Anda telah dihapus dari brankas dan Auto-Trade dimatikan.",
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=False),
        )
        await _safe_answer_callback(callback, "Akun diputuskan")
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
            await _safe_answer_callback(callback, "Wajib /setup dulu", show_alert=True)
            return
        creds = vault.resolve_credentials(user_id)
        positions = vault.get_open_positions(user_id)
        await msg.answer(
            format_myaccount_card(user_id, user_name, vault.get_user_role(user_id), creds, positions),
            parse_mode="HTML",
            reply_markup=build_setup_keyboard(has_creds=True, autotrade_on=enable),
        )
        await _safe_answer_callback(callback, "Auto-Trade ON" if enable else "Auto-Trade OFF")
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
        await _safe_answer_callback(callback)
