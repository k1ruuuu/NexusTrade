from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile

from ai_engine.backtester.engine import EventDrivenBacktester
from ai_engine.auto_trader.binance_client import BinanceSpotClient
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.user_vault import UserVault
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from telegram_bot.handlers.callbacks import callbacks_router
from telegram_bot.handlers.commands import (
    build_symbol_action_keyboard,
    commands_router,
)
from telegram_bot.handlers.signals import signals_router
from telegram_bot.handlers.setup import setup_router
from telegram_bot.middlewares.auth_middleware import WhitelistAuthMiddleware
from telegram_bot.middlewares.rate_limit import RateLimitMiddleware
from telegram_bot.views.formatters import format_signal_card
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
        binance_client: Optional[BinanceSpotClient] = None,
        auto_trader: Optional[SmartAutoTraderEngine] = None,
        bot: Optional[Bot] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ensemble_agent = ensemble_agent or EnsembleDecisionAgent(self.settings)
        self.chart_generator = chart_generator or ChartSnapshotGenerator(self.settings)
        self.backtester = backtester or EventDrivenBacktester(self.settings)
        self.user_vault = user_vault or UserVault(self.settings)
        self.binance_client = binance_client or BinanceSpotClient()
        self.auto_trader = auto_trader or SmartAutoTraderEngine(
            settings=self.settings,
            ensemble_agent=self.ensemble_agent,
            user_vault=self.user_vault,
            binance_client=self.binance_client,
        )
        self.bot = bot or Bot(
            token=self.settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self.dp = self._build_dispatcher()
        self.broadcast_history: list[dict[str, Any]] = []

    def _build_dispatcher(self) -> Dispatcher:
        dp = Dispatcher(
            settings=self.settings,
            ensemble_agent=self.ensemble_agent,
            chart_generator=self.chart_generator,
            backtester=self.backtester,
            user_vault=self.user_vault,
            binance_client=self.binance_client,
            auto_trader=self.auto_trader,
        )

        auth_mw = WhitelistAuthMiddleware(self.settings)
        rate_mw = RateLimitMiddleware(self.settings)

        dp.message.middleware(auth_mw)
        dp.message.middleware(rate_mw)
        dp.callback_query.middleware(auth_mw)
        dp.callback_query.middleware(rate_mw)

        dp.include_router(commands_router)
        dp.include_router(signals_router)
        dp.include_router(setup_router)
        dp.include_router(callbacks_router)
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

        record = {
            "chat_id": target_chat_id,
            "symbol": decision.symbol,
            "action": decision.action,
            "confidence": decision.confidence,
            "caption": caption,
            "chart_path": str(chart_path) if chart_path else None,
        }
        self.broadcast_history.append(record)

        try:
            if chart_path is not None and chart_path.exists():
                await self.bot.send_photo(
                    chat_id=target_chat_id,
                    photo=FSInputFile(str(chart_path)),
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
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
            return True
        except Exception as exc:
            logger.warning(
                "Telegram API send failed during webhook dispatch (recorded in broadcast_history)",
                chat_id=target_chat_id,
                symbol=decision.symbol,
                error=str(exc),
            )
            return False

    async def start_polling(self, handle_signals: bool = True) -> None:
        """Start Aiogram v3 long-polling loop."""
        logger.info("Starting NexusTrade AI Telegram Bot polling")
        await self.dp.start_polling(self.bot, handle_signals=handle_signals)
