from __future__ import annotations

from typing import Any

from aiogram import F, Router
from aiogram.types import CallbackQuery

from ai_engine.backtester.engine import EventDrivenBacktester
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.settings import Settings, get_settings
from telegram_bot.handlers.commands import (
    build_live_portfolio_state,
    build_main_dashboard_keyboard,
)
from telegram_bot.handlers.signals import (
    send_backtest_response,
    send_learn_response,
    send_sentiment_response,
    send_signal_response,
    send_volatility_response,
)
from telegram_bot.views.formatters import (
    format_portfolio_card,
    format_status_card,
    format_welcome_card,
)
from tradingview.chart_generator import ChartSnapshotGenerator

callbacks_router = Router(name="callbacks")


@callbacks_router.callback_query(F.data.startswith("cmd:"))
async def handle_dashboard_callback(callback: CallbackQuery, **data: Any) -> None:
    """Process interactive inline keyboard menu actions."""
    settings: Settings = data.get("settings") or get_settings()
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)
    charts: ChartSnapshotGenerator = data.get("chart_generator") or ChartSnapshotGenerator(settings)
    backtester: EventDrivenBacktester = data.get("backtester") or EventDrivenBacktester(settings)

    payload = (callback.data or "").split(":")
    action = payload[1] if len(payload) > 1 else "dashboard"
    msg = callback.message

    try:
        await callback.answer()
    except Exception:
        pass

    if msg is None:
        return

    symbol = payload[2].upper() if len(payload) > 2 else "BTCUSDT"

    if action == "dashboard":
        user_name = (
            callback.from_user.full_name
            if callback.from_user and callback.from_user.full_name
            else "Quant Trader"
        )
        card = format_welcome_card(user_name=user_name, settings=settings)
        await msg.answer(card, parse_mode="HTML", reply_markup=build_main_dashboard_keyboard())
    elif action == "status":
        card = format_status_card(settings=settings, position_sizer=agent.position_sizer)
        await msg.answer(card, parse_mode="HTML", reply_markup=build_main_dashboard_keyboard())
    elif action == "portfolio":
        uid = int(getattr(callback.from_user, "id", 0) or 0) if callback.from_user else None
        portfolio_state = build_live_portfolio_state(
            agent=agent,
            settings=settings,
            user_id=uid,
            user_vault=data.get("user_vault"),
        )
        card = format_portfolio_card(portfolio_state)
        await msg.answer(card, parse_mode="HTML", reply_markup=build_main_dashboard_keyboard())
    elif action == "signal":
        await send_signal_response(msg, symbol, "15m", agent, charts)
    elif action == "sentiment":
        await send_sentiment_response(msg, symbol, agent)
    elif action == "learn":
        await send_learn_response(msg, symbol, "15m", agent)
    elif action == "volatility":
        await send_volatility_response(msg, symbol, agent)
    elif action == "backtest":
        days = int(payload[3]) if len(payload) > 3 and payload[3].isdigit() else 90
        await send_backtest_response(msg, symbol, days, agent, backtester)

