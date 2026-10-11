from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_engine.backtester.engine import EventDrivenBacktester
from datetime import datetime, timedelta, timezone
from ai_engine.auto_trader.admin_store import AdminStore
from ai_engine.auto_trader.binance_client import BinanceSpotClient
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.user_vault import UserVault
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.settings import Settings
from telegram_bot.bot import NexusTradeBotService
from telegram_bot.handlers.admin import handle_admin_command
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
from telegram_bot.handlers.setup import (
    handle_autotrade,
    handle_myaccount,
    handle_setup,
    handle_unlink,
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
    msg.delete = AsyncMock()
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


@pytest.mark.asyncio
async def test_multi_user_setup_role_isolation_and_6gate_autotrade(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        telegram_allowed_user_ids=[111, 222],
        telegram_admin_user_ids=[111],
        telegram_admin_chat_id=111,
        binance_api_key="TESTNET_ADMIN_ENV_KEY_9999",
        binance_api_secret="ADMIN_ENV_SECRET_9999_SECURE",
        binance_testnet=True,
        use_live_market_data=False,
        use_live_news_feed=False,
    )
    vault = UserVault(settings=settings)
    binance_client = BinanceSpotClient()
    agent = EnsembleDecisionAgent(settings=settings)
    auto_trader = SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
        binance_client=binance_client,
    )
    ctx = {
        "settings": settings,
        "ensemble_agent": agent,
        "user_vault": vault,
        "binance_client": binance_client,
        "auto_trader": auto_trader,
    }

    # 1. Admin (user 111) can access .env fallback; Regular user (user 222) CANNOT
    admin_creds = vault.resolve_credentials(111)
    assert admin_creds is not None
    assert admin_creds.role == "admin"
    assert admin_creds.source == "admin_env"

    user222_creds = vault.resolve_credentials(222)
    assert user222_creds is None

    # 2. Regular user 222 trying /autotrade BEFORE /setup is blocked
    blocked_msg = _make_fake_message(222, "/autotrade scan BTCUSDT")
    await handle_autotrade(blocked_msg, **ctx)
    blocked_msg.answer.assert_awaited_once()
    blocked_reply = blocked_msg.answer.call_args[0][0]
    assert "WAJIB /SETUP DULU" in blocked_reply

    # 3. Regular user 222 runs /setup -> secret message is auto-deleted & stored encrypted
    setup_msg = _make_fake_message(222, "/setup testnet TESTNET_USER222_KEY SECRET_USER222_PASS 1.0")
    await handle_setup(setup_msg, **ctx)
    setup_msg.delete.assert_awaited_once()
    setup_msg.answer.assert_awaited_once()
    setup_reply = setup_msg.answer.call_args[0][0]
    assert "SETUP AKUN BINANCE BERHASIL" in setup_reply

    # Verify cross-user cryptographic isolation (User 111 cannot decrypt User 222's token)
    enc_token = vault.encrypt_for_user(222, "TOP_SECRET_222")
    with pytest.raises(ValueError, match="integrity check failed"):
        vault.decrypt_for_user(111, enc_token)

    # 4. 6-Gate Pre-Trade Check: Conflicting news sentiment blocks BUY execution (Gate 2)
    bad_news_decision = agent.synthesize_scores(
        s_tech=0.92,
        s_vol=0.20,
        s_sent=-0.45,
        symbol="BTCUSDT",
        entry_price=85000.0,
        atr_14=500.0,
        headlines=["SEC announces emergency probe"],
    )
    bad_report = await auto_trader.evaluate_and_execute(
        user_id=222,
        symbol="BTCUSDT",
        decision_override=bad_news_decision,
    )
    assert bad_report.executed is False
    assert any(g.gate_number == 2 and not g.passed for g in bad_report.gates)

    # 5. All 6 Gates pass -> Order executed on User 222's personal account & position recorded
    good_decision = agent.synthesize_scores(
        s_tech=0.92,
        s_vol=0.20,
        s_sent=0.80,
        symbol="BTCUSDT",
        entry_price=85000.0,
        atr_14=500.0,
        headlines=["Institutional ETF inflows surge to record high"],
    )
    good_report = await auto_trader.evaluate_and_execute(
        user_id=222,
        symbol="BTCUSDT",
        decision_override=good_decision,
    )
    assert good_report.executed is True
    assert good_report.position_record is not None
    assert len(vault.get_open_positions(222, "BTCUSDT")) == 1
    assert len(vault.get_open_positions(111, "BTCUSDT")) == 0

    # 6. Duplicate position on same symbol is blocked by Gate 5
    dup_report = await auto_trader.evaluate_and_execute(
        user_id=222,
        symbol="BTCUSDT",
        decision_override=good_decision,
    )
    assert dup_report.executed is False
    assert any(g.gate_number == 5 and not g.passed for g in dup_report.gates)

    # 7. /myaccount and /unlink work for User 222
    acc_msg = _make_fake_message(222, "/myaccount")
    await handle_myaccount(acc_msg, **ctx)
    assert "TEST****_KEY" in acc_msg.answer.call_args[0][0]

    unlink_msg = _make_fake_message(222, "/unlink")
    await handle_unlink(unlink_msg, **ctx)
    assert vault.resolve_credentials(222) is None


@pytest.mark.asyncio
async def test_admin_panel_user_management_broadcast_maintenance_and_audit(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        telegram_allowed_user_ids=[111],
        telegram_admin_user_ids=[111],
        telegram_admin_chat_id=111,
    )
    admin_store = AdminStore(settings=settings)
    mw = WhitelistAuthMiddleware(settings=settings, admin_store=admin_store)
    handler = AsyncMock(return_value="ok")
    fake_bot = SimpleNamespace(send_message=AsyncMock())
    ctx = {
        "settings": settings,
        "admin_store": admin_store,
        "bot": fake_bot,
    }

    # 0. Non-admin cannot access /admin
    non_admin_msg = _make_fake_message(777, "/admin")
    await handle_admin_command(non_admin_msg, **ctx)
    assert "Akses Ditolak" in non_admin_msg.answer.call_args[0][0]

    # 1. Management User: Admin adds user 555 with 1 hari expiry, then edits to 1 bulan
    add_msg = _make_fake_message(111, "/admin adduser 555 1hari Budi Trader")
    await handle_admin_command(add_msg, **ctx)
    assert "TAMBAH USER BARU BERHASIL" in add_msg.answer.call_args[0][0]
    assert "1 Hari" in add_msg.answer.call_args[0][0]

    # User 555 is now authorized by WhitelistAuthMiddleware
    u555_msg = _make_fake_message(555, "/status")
    res_555 = await mw(handler, u555_msg, {})
    assert res_555 == "ok"

    # Admin edits User 555 expiry to 1 minggu, then 1 bulan
    edit_msg = _make_fake_message(111, "/admin edituser 555 1bulan")
    await handle_admin_command(edit_msg, **ctx)
    assert "EDIT EXPIRED USER BERHASIL" in edit_msg.answer.call_args[0][0]
    assert "1 Bulan" in edit_msg.answer.call_args[0][0]

    # Simulate expired user 666 -> blocked with Masa Aktif Berakhir message
    past_dt = datetime.now(timezone.utc) - timedelta(days=5)
    admin_store.add_or_update_user(666, duration_code="1hari", label="ExpiredUser", admin_id=111, now_override=past_dt)
    exp_msg = _make_fake_message(666, "/signal BTCUSDT")
    res_exp = await mw(handler, exp_msg, {})
    assert res_exp is None
    assert "Masa Aktif Berakhir" in exp_msg.answer.call_args[0][0]

    # 2. Maintenance Mode: Admin turns Maintenance ON -> User 555 blocked with maintenance message, Admin still allowed
    maint_on_msg = _make_fake_message(111, "/admin maintenance on")
    await handle_admin_command(maint_on_msg, **ctx)
    assert admin_store.is_maintenance_mode() is True

    u555_maint_msg = _make_fake_message(555, "/signal BTCUSDT")
    res_maint = await mw(handler, u555_maint_msg, {})
    assert res_maint is None
    assert "Server Sedang Maintenance" in u555_maint_msg.answer.call_args[0][0]

    # Admin is still allowed during Maintenance ON
    admin_maint_msg = _make_fake_message(111, "/admin")
    res_admin_maint = await mw(handler, admin_maint_msg, {})
    assert res_admin_maint == "ok"

    # Admin turns Maintenance OFF -> User 555 allowed again
    maint_off_msg = _make_fake_message(111, "/admin maintenance off")
    await handle_admin_command(maint_off_msg, **ctx)
    assert admin_store.is_maintenance_mode() is False
    assert await mw(handler, _make_fake_message(555, "/status"), {}) == "ok"

    # 3. Broadcast: Admin sends custom broadcast message to all active users (111 and 555)
    bcast_msg = _make_fake_message(111, "/admin broadcast Jadwal rilis fitur baru malam ini pukul 20:00 WIB")
    await handle_admin_command(bcast_msg, **ctx)
    assert "LAPORAN BROADCAST PESAN ADMIN" in bcast_msg.answer.call_args[0][0]
    assert fake_bot.send_message.await_count >= 2

    # 4. Delete User: Admin deletes User 555 -> User 555 is revoked and blocked
    del_msg = _make_fake_message(111, "/admin deluser 555")
    await handle_admin_command(del_msg, **ctx)
    assert "USER DIHAPUS" in del_msg.answer.call_args[0][0]
    assert await mw(handler, _make_fake_message(555, "/status"), {}) is None

    # 5. Audit: Admin views full user activity audit trail
    audit_msg = _make_fake_message(111, "/admin audit")
    await handle_admin_command(audit_msg, **ctx)
    audit_reply = audit_msg.answer.call_args[0][0]
    assert "AUDIT AKTIVITAS USER" in audit_reply
    assert "DENIED_MAINTENANCE" in audit_reply
    assert "DENIED_EXPIRED" in audit_reply
