from __future__ import annotations

from typing import Any

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message

from ai_engine.backtester.engine import EventDrivenBacktester
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe
from telegram_bot.handlers.commands import (
    build_main_dashboard_keyboard,
    build_symbol_action_keyboard,
)
from telegram_bot.handlers.handler_utils import parse_command_args as _parse_command_args
from telegram_bot.views.formatters import (
    format_backtest_card,
    format_online_learning_card,
    format_sentiment_card,
    format_signal_card,
    format_volatility_card,
)
from tradingview.chart_generator import ChartSnapshotGenerator

signals_router = Router(name="signals")


async def send_signal_response(
    target_msg: Any,
    symbol: str,
    timeframe: str,
    agent: EnsembleDecisionAgent,
    charts: ChartSnapshotGenerator,
) -> None:
    """Shared signal + 1200x800 chart sender for /signal and inline callbacks."""
    df = agent.load_market_series(symbol=symbol, timeframe=timeframe)
    if timeframe in ("15m", "5m"):
        decision = agent.analyze_multi_timeframe_autotrade(symbol=symbol)
    else:
        decision = agent.analyze_symbol(symbol=symbol, timeframe=timeframe, df=df)
    chart_path = await charts.generate_chart_async(
        df=df,
        symbol=decision.symbol,
        timeframe=timeframe,
        decision=decision,
    )
    caption = format_signal_card(decision)
    kb = build_symbol_action_keyboard(decision.symbol)
    try:
        await target_msg.answer_photo(
            photo=FSInputFile(str(chart_path)),
            caption=caption,
            parse_mode="HTML",
            reply_markup=kb,
        )
    except Exception as exc:
        if "caption is too long" in str(exc).lower():
            await target_msg.answer_photo(
                photo=FSInputFile(str(chart_path)),
                caption=f"📡 <b>GRAFIK SINYAL AI — {decision.symbol} ({timeframe})</b>",
                parse_mode="HTML",
            )
            await target_msg.answer(
                caption,
                parse_mode="HTML",
                reply_markup=kb,
            )
        else:
            raise


async def send_sentiment_response(
    target_msg: Any,
    symbol: str,
    agent: EnsembleDecisionAgent,
) -> None:
    """Shared sentiment analysis sender for /sentiment and inline callbacks."""
    market_df = agent.load_market_series(symbol=symbol, timeframe="15m", limit=120)
    result = agent.sentiment_analyzer.analyze_headlines(symbol=symbol, market_df=market_df)
    await target_msg.answer(
        format_sentiment_card(result),
        parse_mode="HTML",
        reply_markup=build_symbol_action_keyboard(result.symbol),
    )


async def send_volatility_response(
    target_msg: Any,
    symbol: str,
    agent: EnsembleDecisionAgent,
) -> None:
    """Shared volatility forecast sender for /volatility and inline callbacks."""
    df = agent.load_market_series(symbol=symbol, timeframe="15m")
    forecast = agent.volatility_estimator.estimate(df=df, symbol=symbol)
    await target_msg.answer(
        format_volatility_card(forecast),
        parse_mode="HTML",
        reply_markup=build_symbol_action_keyboard(forecast.symbol),
    )


async def send_backtest_response(
    target_msg: Any,
    symbol: str,
    days: int,
    agent: EnsembleDecisionAgent,
    backtester: EventDrivenBacktester,
) -> None:
    """Shared backtest simulation + equity curve sender for /backtest and inline callbacks."""
    df = agent.load_market_series(symbol=symbol, timeframe="1d", limit=max(days + 60, 200))
    report = backtester.run(df=df, symbol=symbol, days=days, generate_plot=True)
    caption = format_backtest_card(report)

    if report.chart_path and report.chart_path.exists():
        await target_msg.answer_photo(
            photo=FSInputFile(str(report.chart_path)),
            caption=caption,
            parse_mode="HTML",
            reply_markup=build_main_dashboard_keyboard(),
        )
    else:
        await target_msg.answer(
            caption,
            parse_mode="HTML",
            reply_markup=build_main_dashboard_keyboard(),
        )


async def send_learn_response(
    target_msg: Any,
    symbol: str,
    timeframe: str,
    agent: EnsembleDecisionAgent,
) -> None:
    """Shared online learning cycle sender for /learn and inline callbacks."""
    report = agent.learn_online_cycle(symbol=symbol, timeframe=timeframe)
    await target_msg.answer(
        format_online_learning_card(report),
        parse_mode="HTML",
        reply_markup=build_symbol_action_keyboard(str(report["symbol"])),
    )


@signals_router.message(Command("signal"))
async def handle_signal(message: Message, **data: Any) -> None:
    """Handle /signal <symbol> [timeframe]."""
    settings: Settings = data.get("settings") or get_settings()
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)
    charts: ChartSnapshotGenerator = data.get("chart_generator") or ChartSnapshotGenerator(settings)

    args = _parse_command_args(message.text)
    symbol = normalize_symbol(args[0]) if len(args) >= 1 else "BTCUSDT"
    timeframe = normalize_timeframe(args[1]) if len(args) >= 2 else "15m"
    await send_signal_response(message, symbol, timeframe, agent, charts)


@signals_router.message(Command("sentiment"))
async def handle_sentiment(message: Message, **data: Any) -> None:
    """Handle /sentiment <symbol>."""
    settings: Settings = data.get("settings") or get_settings()
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)

    args = _parse_command_args(message.text)
    symbol = normalize_symbol(args[0]) if args else "BTCUSDT"
    await send_sentiment_response(message, symbol, agent)


@signals_router.message(Command("volatility"))
async def handle_volatility(message: Message, **data: Any) -> None:
    """Handle /volatility <symbol>."""
    settings: Settings = data.get("settings") or get_settings()
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)

    args = _parse_command_args(message.text)
    symbol = normalize_symbol(args[0]) if args else "BTCUSDT"
    await send_volatility_response(message, symbol, agent)


@signals_router.message(Command("backtest"))
async def handle_backtest(message: Message, **data: Any) -> None:
    """Handle /backtest <symbol> [days]."""
    settings: Settings = data.get("settings") or get_settings()
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)
    backtester: EventDrivenBacktester = data.get("backtester") or EventDrivenBacktester(settings)

    args = _parse_command_args(message.text)
    symbol = normalize_symbol(args[0]) if len(args) >= 1 else "BTCUSDT"
    days = 90
    if len(args) >= 2:
        try:
            days = max(15, min(int(args[1]), 1000))
        except ValueError:
            days = 90
    await send_backtest_response(message, symbol, days, agent, backtester)


@signals_router.message(Command("learn"))
async def handle_learn(message: Message, **data: Any) -> None:
    """Handle /learn [symbol] [timeframe]."""
    settings: Settings = data.get("settings") or get_settings()
    agent: EnsembleDecisionAgent = data.get("ensemble_agent") or EnsembleDecisionAgent(settings)

    args = _parse_command_args(message.text)
    symbol = normalize_symbol(args[0]) if len(args) >= 1 else "BTCUSDT"
    timeframe = normalize_timeframe(args[1]) if len(args) >= 2 else "15m"
    await send_learn_response(message, symbol, timeframe, agent)
