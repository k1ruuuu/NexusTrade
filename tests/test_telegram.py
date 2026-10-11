from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_engine.backtester.engine import EventDrivenBacktester
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.settings import Settings
from telegram_bot.bot import NexusTradeBotService
from telegram_bot.handlers.commands import (
    handle_analyze,
    handle_portfolio,
    handle_start,
    handle_status,
)
from telegram_bot.handlers.signals import (
    handle_backtest,
    handle_sentiment,
    handle_signal,
    handle_volatility,
)
from telegram_bot.middlewares.auth_middleware import WhitelistAuthMiddleware
from telegram_bot.middlewares.rate_limit import RateLimitMiddleware
from telegram_bot.views.formatters import (
    format_sentiment_card,
    format_signal_card,
    format_status_card,
    format_volatility_card,
    format_welcome_card,
)
from tradingview.alert_parser import AlertParser
from tradingview.chart_generator import ChartSnapshotGenerator


def _make_fake_message(user_id: int = 111, text: str = "/start") -> Any:
    msg = SimpleNamespace()
    msg.from_user = SimpleNamespace(id=user_id, full_name="Alice Quant", username="alice_q")
    msg.text = text
    msg.answer = AsyncMock()
    msg.answer_photo = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_whitelist_auth_middleware() -> None:
    settings = Settings(telegram_allowed_user_ids=[111, 222])
    mw = WhitelistAuthMiddleware(settings=settings)
    handler = AsyncMock(return_value="ok")

    # Authorized user passes through
    auth_msg = _make_fake_message(user_id=111, text="/status")
    data: dict[str, Any] = {}
    res = await mw(handler, auth_msg, data)
    assert res == "ok"
    assert data.get("authorized") is True
    handler.assert_awaited_once()

    # Unauthorized user is dropped
    handler.reset_mock()
    unauth_msg = _make_fake_message(user_id=999, text="/status")
    res_unauth = await mw(handler, unauth_msg, {})
    assert res_unauth is None
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_rate_limit_middleware() -> None:
    now = [100.0]
    mw = RateLimitMiddleware(max_calls=2, period_seconds=10.0, clock=lambda: now[0])
    handler = AsyncMock(return_value="ok")
    msg = _make_fake_message(user_id=111)

    assert await mw(handler, msg, {}) == "ok"
    assert await mw(handler, msg, {}) == "ok"
    # 3rd call within 10s window is throttled
    assert await mw(handler, msg, {}) is None

    # Advance clock past window
    now[0] += 11.0
    assert await mw(handler, msg, {}) == "ok"


def test_message_formatters_html_cards(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    agent = EnsembleDecisionAgent(settings=settings)

    welcome_html = format_welcome_card("Alice <Admin>", settings)
    assert "Alice &lt;Admin&gt;" in welcome_html
    assert "/signal" in welcome_html

    status_html = format_status_card(settings, agent.position_sizer, processed_webhooks=4)
    assert "ARMED" in status_html
    assert "2.0x ATR(14)" in status_html

    decision = agent.synthesize_scores(
        s_tech=0.88,
        s_vol=0.25,
        s_sent=0.71,
        symbol="BTCUSDT",
        entry_price=67420.50,
        atr_14=450.0,
    )
    signal_html = format_signal_card(decision)
    assert "STRONG BUY" in signal_html
    assert "Entry Price:" in signal_html
    assert "Stop Loss (2xATR):" in signal_html
    assert "Take Profit 1 (1:2):" in signal_html
    assert "Take Profit 2 (1:3):" in signal_html
    assert "█" in signal_html
    assert "Resistance Atas (R1 | R2):" in signal_html
    assert "Support Bawah (S1 | S2):" in signal_html
    assert decision.support_2 < decision.support_1 < decision.entry_price < decision.resistance_1 < decision.resistance_2

    sent_res = agent.sentiment_analyzer.analyze_headlines(symbol="BTCUSDT")
    sent_html = format_sentiment_card(sent_res)
    assert "FINBERT SENTIMENT ENGINE" in sent_html

    df = agent.load_market_series("BTCUSDT", "15m", limit=100)
    vol_res = agent.volatility_estimator.estimate(df, symbol="BTCUSDT")
    vol_html = format_volatility_card(vol_res)
    assert "OPTIVER REALIZED VOLATILITY" in vol_html


@pytest.mark.asyncio
async def test_command_and_signal_handlers_and_webhook_dispatch(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        telegram_allowed_user_ids=[111],
        telegram_admin_chat_id=111,
    )
    agent = EnsembleDecisionAgent(settings=settings)
    charts = ChartSnapshotGenerator(settings=settings)
    backtester = EventDrivenBacktester(settings=settings)
    ctx = {
        "settings": settings,
        "ensemble_agent": agent,
        "chart_generator": charts,
        "backtester": backtester,
    }

    start_msg = _make_fake_message(111, "/start")
    await handle_start(start_msg, **ctx)
    start_msg.answer.assert_awaited_once()

    status_msg = _make_fake_message(111, "/status")
    await handle_status(status_msg, **ctx)
    status_msg.answer.assert_awaited_once()

    analyze_msg = _make_fake_message(111, "/analyze XAUUSD")
    await handle_analyze(analyze_msg, **ctx)
    analyze_msg.answer.assert_awaited_once()

    portfolio_msg = _make_fake_message(111, "/portfolio")
    await handle_portfolio(portfolio_msg, **ctx)
    portfolio_msg.answer.assert_awaited_once()

    signal_msg = _make_fake_message(111, "/signal XAUUSD")
    await handle_signal(signal_msg, **ctx)
    signal_msg.answer_photo.assert_awaited_once()

    sent_msg = _make_fake_message(111, "/sentiment BTCUSDT")
    await handle_sentiment(sent_msg, **ctx)
    sent_msg.answer.assert_awaited_once()

    vol_msg = _make_fake_message(111, "/volatility BTCUSDT")
    await handle_volatility(vol_msg, **ctx)
    vol_msg.answer.assert_awaited_once()

    bt_msg = _make_fake_message(111, "/backtest BTCUSDT 60")
    await handle_backtest(bt_msg, **ctx)
    bt_msg.answer_photo.assert_awaited_once()

    # Test Webhook-to-Telegram dispatcher with mocked Bot API
    fake_bot = SimpleNamespace(
        send_photo=AsyncMock(),
        send_message=AsyncMock(),
    )
    bot_service = NexusTradeBotService(
        settings=settings,
        ensemble_agent=agent,
        chart_generator=charts,
        backtester=backtester,
        bot=fake_bot,  # type: ignore[arg-type]
    )
    alert = AlertParser(settings).parse(
        {
            "secret": settings.tradingview_webhook_secret,
            "ticker": "BINANCE:BTCUSDT",
            "interval": "15m",
            "action": "BUY",
            "price": 67420.50,
            "strategy": "TrendBreakout_v2",
            "indicators": {"rsi": 62.4, "macd": 142.1, "atr": 450.0},
            "timestamp": "2026-10-11T04:50:00Z",
        }
    )
    decision = agent.analyze_symbol("BTCUSDT", "15m", price_override=67420.50, action_hint="BUY")
    df = agent.load_market_series("BTCUSDT", "15m", limit=80)
    chart_path = charts.generate_chart(df, symbol="BTCUSDT", timeframe="15m", decision=decision)

    sent_ok = await bot_service.dispatch_webhook_alert(alert, decision, chart_path)
    assert sent_ok is True
    fake_bot.send_photo.assert_awaited_once()
    assert len(bot_service.broadcast_history) == 1
