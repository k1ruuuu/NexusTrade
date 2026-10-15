from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile

from ai_engine.backtester.engine import EventDrivenBacktester
from ai_engine.auto_trader.admin_store import AdminStore
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.user_vault import UserVault
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from telegram_bot.handlers.callbacks import callbacks_router
from telegram_bot.handlers.admin import admin_router
from telegram_bot.handlers.commands import (
    build_symbol_action_keyboard,
    commands_router,
)
from telegram_bot.handlers.signals import signals_router
from telegram_bot.handlers.setup import setup_router
from telegram_bot.middlewares.auth_middleware import WhitelistAuthMiddleware
from telegram_bot.middlewares.rate_limit import RateLimitMiddleware
from telegram_bot.views.formatters import format_signal_card
from telegram_bot.views.setup_formatters import format_autotrade_report_card
from tradingview.alert_parser import TradingViewAlertPayload
from tradingview.chart_generator import ChartSnapshotGenerator

logger = get_logger("nexustrade.telegram.bot")


class NexusTradeBotService:
    """
    Aiogram v3 Dispatcher & Bot lifecycle manager with Whitelist Auth,
    Rate Limiting, and Webhook-to-Telegram broadcast integration.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        ensemble_agent: Optional[EnsembleDecisionAgent] = None,
        chart_generator: Optional[ChartSnapshotGenerator] = None,
        backtester: Optional[EventDrivenBacktester] = None,
        user_vault: Optional[UserVault] = None,
        admin_store: Optional[AdminStore] = None,
        auto_trader: Optional[SmartAutoTraderEngine] = None,
        bot: Optional[Bot] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ensemble_agent = ensemble_agent or EnsembleDecisionAgent(self.settings)
        self.chart_generator = chart_generator or ChartSnapshotGenerator(self.settings)
        self.backtester = backtester or EventDrivenBacktester(self.settings)
        self.user_vault = user_vault or UserVault(self.settings)
        self.admin_store = admin_store or AdminStore(self.settings, db_path=self.user_vault.db_path)
        self.bot = bot or Bot(
            token=self.settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self.auto_trader = auto_trader or SmartAutoTraderEngine(
            settings=self.settings,
            ensemble_agent=self.ensemble_agent,
            user_vault=self.user_vault,
            notify_user_callback=self._send_user_notification,
        )
        if getattr(self.auto_trader, "notify_user_callback", None) is None:
            self.auto_trader.notify_user_callback = self._send_user_notification
        self.dp = self._build_dispatcher()
        self.broadcast_history: list[dict[str, Any]] = []
    def _build_dispatcher(self) -> Dispatcher:
        dp = Dispatcher(
            settings=self.settings,
            ensemble_agent=self.ensemble_agent,
            chart_generator=self.chart_generator,
            backtester=self.backtester,
            user_vault=self.user_vault,
            admin_store=self.admin_store,
            auto_trader=self.auto_trader,
        )

        rate_mw = RateLimitMiddleware(self.settings)
        auth_mw = WhitelistAuthMiddleware(self.settings, admin_store=self.admin_store)

        dp.message.middleware(rate_mw)
        dp.message.middleware(auth_mw)
        dp.callback_query.middleware(rate_mw)
        dp.callback_query.middleware(auth_mw)

        for router in (
            commands_router,
            signals_router,
            setup_router,
            admin_router,
            callbacks_router,
        ):
            if getattr(router, "_parent_router", None) is not None:
                router._parent_router = None
            dp.include_router(router)
        return dp

    async def dispatch_webhook_alert(
        self,
        alert: TradingViewAlertPayload,
        decision: SignalDecision,
        chart_path: Optional[Path] = None,
    ) -> bool:
        """
        Broadcast confirmed TradingView webhook trading signal to the configured
        Telegram admin channel/chat ID with formatted trade card and chart image.
        """
        target_chat_id = self.settings.telegram_admin_chat_id
        if not target_chat_id and self.settings.telegram_allowed_user_ids:
            target_chat_id = int(self.settings.telegram_allowed_user_ids[0])

        if not target_chat_id:
            logger.warning("No TELEGRAM_ADMIN_CHAT_ID configured for webhook broadcast")
            return False

        caption = format_signal_card(decision)
        reply_markup = build_symbol_action_keyboard(decision.symbol)

        record: dict[str, Any] = {
            "chat_id": target_chat_id,
            "symbol": decision.symbol,
            "action": decision.action,
            "confidence": decision.confidence,
            "caption": caption,
            "chart_path": str(chart_path) if chart_path else None,
            "autotrade_reports": [],
        }
        self.broadcast_history.append(record)

        sent_ok = False
        try:
            if chart_path is not None and chart_path.exists():
                try:
                    await self.bot.send_photo(
                        chat_id=target_chat_id,
                        photo=FSInputFile(str(chart_path)),
                        caption=caption,
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup,
                    )
                except Exception as photo_exc:
                    if "caption is too long" in str(photo_exc).lower():
                        await self.bot.send_photo(
                            chat_id=target_chat_id,
                            photo=FSInputFile(str(chart_path)),
                            caption=f"📡 <b>GRAFIK SINYAL AI — {decision.symbol} ({decision.timeframe})</b>",
                            parse_mode=ParseMode.HTML,
                        )
                        await self.bot.send_message(
                            chat_id=target_chat_id,
                            text=caption,
                            parse_mode=ParseMode.HTML,
                            reply_markup=reply_markup,
                        )
                    else:
                        raise
            else:
                await self.bot.send_message(
                    chat_id=target_chat_id,
                    text=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
            logger.info(
                "Dispatched webhook signal to Telegram",
                chat_id=target_chat_id,
                symbol=decision.symbol,
                action=decision.action,
                confidence=decision.confidence,
            )
            sent_ok = True
        except Exception as exc:
            logger.warning(
                "Telegram API send failed during webhook dispatch (recorded in broadcast_history)",
                chat_id=target_chat_id,
                symbol=decision.symbol,
                error=str(exc),
            )

        # LF-11: If any authorized user/admin has autotrade_enabled=True, also trigger 6-Gate Auto-Trade
        static_allowed = {int(x) for x in self.settings.telegram_allowed_user_ids}
        for uid in self.user_vault.list_autotrade_enabled_user_ids():
            allowed, _, _ = self.admin_store.check_user_access(uid, static_allowed)
            if not allowed:
                continue
            if self.admin_store.is_maintenance_mode() and not self.settings.is_admin_user(uid):
                continue
            try:
                report = await self.auto_trader.evaluate_and_execute(
                    user_id=uid,
                    symbol=decision.symbol,
                    timeframe=decision.timeframe,
                    decision_override=decision,
                    require_autotrade_toggle=True,
                )
                record["autotrade_reports"].append(report)
                await self.bot.send_message(
                    chat_id=uid,
                    text=format_autotrade_report_card(report),
                    parse_mode=ParseMode.HTML,
                )
            except Exception as exc:
                logger.warning(
                    "Webhook auto-trade execution/notify failed for user",
                    user_id=uid,
                    symbol=decision.symbol,
                    error=str(exc),
                )

        return sent_ok

    async def _send_user_notification(self, user_id: int, html_text: str) -> None:
        """Send a real-time HTML notification (e.g., TP/SL hit on Exness MT5) to `user_id`."""
        try:
            await self.bot.send_message(
                chat_id=int(user_id),
                text=html_text,
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            logger.debug("Failed to send user notification", user_id=user_id, error=str(exc))

    async def _mt5_position_watcher_loop(self) -> None:
        """
        Background watcher that polls open positions every 15 seconds and dispatches
        real-time Telegram notifications whenever TP or SL is hit on the Exness MT5 server.
        """
        while True:
            try:
                await asyncio.sleep(15.0)
                for uid in self.user_vault.list_autotrade_enabled_user_ids():
                    if self.user_vault.get_open_positions(uid):
                        await self.auto_trader.sync_live_mt5_positions(uid, notify_closures=True)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("MT5 position watcher iteration error", error=str(exc))

    async def start_polling(self, handle_signals: bool = True) -> None:
        """Start Aiogram v3 long-polling loop + background Exness MT5 TP/SL watcher."""
        logger.info("Starting NexusTrade AI Telegram Bot polling")
        watcher_task = asyncio.create_task(self._mt5_position_watcher_loop())
        try:
            await self.dp.start_polling(self.bot, handle_signals=handle_signals)
        finally:
            watcher_task.cancel()
