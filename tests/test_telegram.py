from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_engine.backtester.engine import EventDrivenBacktester
from datetime import datetime, timedelta, timezone
from ai_engine.auto_trader.admin_store import AdminStore
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.mt5_exness_client import (
    MT5ExnessClient,
    anchor_targets_to_fill_price,
    estimate_mt5_pnl_usdt,
)
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
    agent = EnsembleDecisionAgent(settings=settings)
    auto_trader = SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
    )
    ctx = {
        "settings": settings,
        "ensemble_agent": agent,
        "user_vault": vault,
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
    assert "SETUP AKUN EXNESS GO / MT5 BERHASIL" in setup_reply
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


@pytest.mark.asyncio
async def test_logic_flow_audit_regressions_lf01_to_lf11(tmp_path: Path) -> None:
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
    admin_store = AdminStore(settings=settings, db_path=vault.db_path)
    agent = EnsembleDecisionAgent(settings=settings)
    charts = ChartSnapshotGenerator(settings=settings)
    backtester = EventDrivenBacktester(settings=settings)
    auto_trader = SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
    )

    # Setup User 222 on Exness Go / MT5
    vault.save_user_credentials(
        user_id=222,
        api_key="EXNESS_DEMO_222_KEY",
        api_secret="SECRET_USER222_PASS",
        username="User222",
        testnet=True,
        verified_usdt_balance=1000.0,
        autotrade_enabled=True,
    )
    # LF-02: When user turns /autotrade off, evaluate_and_execute rejects execution at Gate 1
    vault.set_autotrade_enabled(222, False)
    off_rep = await auto_trader.evaluate_and_execute(user_id=222, symbol="BTCUSDT")
    assert off_rep.executed is False
    assert "OFF" in (off_rep.rejection_summary or "")

    # LF-03: Insufficient balance (< $10 USDT) is rejected at Gate 1 without fabricating $50 equity
    vault.set_autotrade_enabled(222, True)
    vault.update_verified_balance(222, 4.50)
    low_bal_rep = await auto_trader.evaluate_and_execute(user_id=222, symbol="BTCUSDT")
    assert low_bal_rep.executed is False
    assert "$10.00" in (low_bal_rep.rejection_summary or "")
    vault.update_verified_balance(222, 2500.0)

    # LF-04: Broadcast with bot=None reports failed_count > 0 and sent_count == 0 (no false positive)
    bcast_no_bot = _make_fake_message(111, "/admin broadcast Pesan Uji Tanpa Bot")
    bcast_no_bot.bot = None
    await handle_admin_command(
        bcast_no_bot,
        settings=settings,
        admin_store=admin_store,
        bot=None,
    )
    bcast_reply = bcast_no_bot.answer.call_args[0][0]
    assert "Berhasil Terkirim:</b> 🟢 <code>0 user</code>" in bcast_reply

    # LF-05: Admin cannot delete their own Admin account, and deleting User 222 disables their autotrade
    assert admin_store.delete_user(111) is False
    assert admin_store.delete_user(222) is True
    creds_222_after_del = vault.resolve_credentials(222)
    assert creds_222_after_del is not None and creds_222_after_del.autotrade_enabled is False

    # LF-11: Webhook dispatches notification ONLY when autotrade is OFF, and ALSO triggers Auto-Trade when ON
    fake_bot = SimpleNamespace(send_photo=AsyncMock(), send_message=AsyncMock())
    bot_service = NexusTradeBotService(
        settings=settings,
        ensemble_agent=agent,
        chart_generator=charts,
        backtester=backtester,
        user_vault=vault,
        admin_store=admin_store,
        auto_trader=auto_trader,
        bot=fake_bot,  # type: ignore[arg-type]
    )
    alert = AlertParser(settings).parse(
        {
            "secret": settings.tradingview_webhook_secret,
            "ticker": "BINANCE:BTCUSDT",
            "interval": "15m",
            "action": "BUY",
            "price": 85000.0,
            "strategy": "TrendBreakout_v2",
        }
    )
    strong_buy_dec = agent.synthesize_scores(
        s_tech=0.92,
        s_vol=0.20,
        s_sent=0.80,
        symbol="BTCUSDT",
        entry_price=85000.0,
        atr_14=500.0,
        headlines=["Record institutional inflows"],
    )

    # Case A: Admin autotrade is OFF -> Webhook sends signal notification ONLY (0 autotrade reports)
    vault.set_autotrade_enabled(111, False)
    await bot_service.dispatch_webhook_alert(alert, strong_buy_dec, None)
    assert len(bot_service.broadcast_history[-1]["autotrade_reports"]) == 0

    # Case B: Admin autotrade is ON -> Webhook sends signal notification AND executes Auto-Trade
    vault.set_autotrade_enabled(111, True)
    await bot_service.dispatch_webhook_alert(alert, strong_buy_dec, None)
    assert len(bot_service.broadcast_history[-1]["autotrade_reports"]) == 1
    assert bot_service.broadcast_history[-1]["autotrade_reports"][0].executed is True


@pytest.mark.asyncio
async def test_exness_mt5_provider_all_asset_buy_and_sell_execution(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        telegram_allowed_user_ids=[111, 333],
        telegram_admin_user_ids=[111],
        telegram_admin_chat_id=111,
        use_live_market_data=False,
        use_live_news_feed=False,
    )
    vault = UserVault(settings=settings)
    agent = EnsembleDecisionAgent(settings=settings)
    auto_trader = SmartAutoTraderEngine(
        settings=settings,
        ensemble_agent=agent,
        user_vault=vault,
    )
    ctx = {
        "settings": settings,
        "ensemble_agent": agent,
        "user_vault": vault,
        "auto_trader": auto_trader,
    }

    # 1a. Invalid numeric Exness login is rejected by Exness MT5 server, NOT faked as $10,000 sandbox
    raw_num_msg = _make_fake_message(333, "/setup exness 88333444 ExnessGoPass2026 Exness-MT5Trial7")
    await handle_setup(raw_num_msg, **ctx)
    assert "Verifikasi Akun Broker Gagal" in raw_num_msg.answer.call_args[0][0]
    # 1b. User 333 sets up an Exness Go / MT5 account via /setup exness
    ex_msg = _make_fake_message(333, "/setup exness EXNESS_DEMO_333 ExnessGoPass2026 Exness-MT5Trial")
    await handle_setup(ex_msg, **ctx)
    ex_msg.delete.assert_awaited_once()
    ex_reply = ex_msg.answer.call_args[0][0]
    assert "SETUP AKUN EXNESS GO / MT5 BERHASIL" in ex_reply
    creds = vault.resolve_credentials(333)
    assert creds is not None
    assert creds.provider == "exness_mt5"
    assert creds.server_name == "Exness-MT5Trial"

    # 2. Symmetric SELL execution on Gold (XAUUSD -> XAUUSDm) with neutral news (s_sent=0.0)
    sell_gold_dec = agent.synthesize_scores(
        s_tech=-0.92,
        s_vol=0.20,
        s_sent=0.0,
        symbol="XAUUSD",
        entry_price=4175.0,
        atr_14=12.0,
        headlines=["Gold market awaits macro catalyst"],
    )
    assert sell_gold_dec.action == "SELL"
    assert sell_gold_dec.confidence >= 0.72

    sell_report = await auto_trader.evaluate_and_execute(
        user_id=333,
        symbol="XAUUSD",
        decision_override=sell_gold_dec,
    )
    assert sell_report.executed is True
    assert sell_report.order_result is not None
    assert sell_report.order_result.side == "SELL"
    assert sell_report.order_result.symbol == "XAUUSDm"
    assert sell_report.position_record is not None
    assert sell_report.position_record.stop_loss > sell_report.position_record.entry_price
    assert sell_report.position_record.take_profit_1 < sell_report.position_record.entry_price

    # 3. Multi-Asset BUY execution on US Equities (NVDA -> NVDAm) on Exness Go / MT5
    buy_nvda_dec = agent.synthesize_scores(
        s_tech=0.90,
        s_vol=0.22,
        s_sent=0.65,
        symbol="NVDA",
        entry_price=138.50,
        atr_14=2.40,
        headlines=["AI data center GPU demand surges"],
    )
    buy_report = await auto_trader.evaluate_and_execute(
        user_id=333,
        symbol="NVDA",
        decision_override=buy_nvda_dec,
        custom_lot=0.05,
    )
    assert buy_report.executed is True
    assert buy_report.order_result is not None
    assert buy_report.order_result.side == "BUY"
    assert buy_report.order_result.symbol == "NVDAm"
    assert buy_report.order_result.executed_qty == 0.05

    # 4. Timed session `/autotrade XAUUSD 5H 0.05` with Custom Lot + Position Holding / Trailing Stop / Auto-Close
    sess_msg = _make_fake_message(333, "/autotrade XAUUSD 5H 0.05")
    await handle_autotrade(sess_msg, **ctx)
    sess_reply = sess_msg.answer.call_args[0][0]
    assert "SESI AUTO-TRADE OTOMATIS (5 JAM) AKTIF" in sess_reply
    assert "0.05 Lot (Custom)" in sess_reply

    active_sessions = auto_trader.get_user_sessions(333)
    assert len(active_sessions) == 1
    assert active_sessions[0].symbol == "XAUUSD"
    assert active_sessions[0].duration_seconds == 5 * 3600
    assert active_sessions[0].custom_lot == 0.05

    # Open position on XAUUSD is HELD on small move (4175 -> 4170), NOT closed prematurely
    hold_res = await auto_trader.monitor_and_manage_open_position(
        user_id=333,
        symbol="XAUUSD",
        price_override=4170.0,
        high_override=4172.0,
        low_override=4169.0,
    )
    assert hold_res.action_taken == "HOLDING"
    assert len(vault.get_open_positions(333, "XAUUSD")) == 1

    # Move >= 1.0*ATR in favor (4175 -> 4158) ratchets Trailing Stop into profit while STILL holding
    trail_res = await auto_trader.monitor_and_manage_open_position(
        user_id=333,
        symbol="XAUUSD",
        price_override=4158.0,
        high_override=4161.0,
        low_override=4157.0,
    )
    assert trail_res.action_taken == "TRAILING_UPDATED"
    assert trail_res.new_stop_loss is not None and trail_res.new_stop_loss < 4175.0
    assert len(vault.get_open_positions(333, "XAUUSD")) == 1

    # When session timer expires BEFORE reaching TP, close_session_open_positions automatically closes it
    expired_closed = await auto_trader.session_manager.close_session_open_positions(
        active_sessions[0],
        reason="Batas waktu sesi 5 Jam habis sebelum menyentuh TP",
    )
    assert len(expired_closed) == 1
    assert expired_closed[0].action_taken in ("CLOSED_TP", "CLOSED_SL")
    assert len(vault.get_open_positions(333, "XAUUSD")) == 0

    # /autotrade off stops the running 5H session
    stop_msg = _make_fake_message(333, "/autotrade off")
    await handle_autotrade(stop_msg, **ctx)
    assert len(auto_trader.get_user_sessions(333)) == 0

    # 5. Regression tests for F-01, F-02, F-03, F-04, F-06:
    # F-01: Anchoring SL/TP1/TP2 to actual broker fill price (Spot 4210 vs Futures 4235)
    anc_sl, anc_tp1, anc_tp2 = anchor_targets_to_fill_price(
        side="BUY",
        fill_price=4210.0,
        ref_entry=4235.0,
        ref_sl=4229.0,
        ref_tp1=4244.0,
        ref_tp2=4250.0,
    )
    assert anc_sl == 4204.0 and anc_sl < 4210.0
    assert anc_tp1 == 4219.0 and anc_tp1 > 4210.0
    assert anc_tp2 == 4225.0 and anc_tp2 > anc_tp1

    # F-03: Contract-size-aware MT5 PnL calculation (XAUUSD 1 lot = 100 oz -> 0.05 lot * $10 move = +$50.00 USD)
    assert estimate_mt5_pnl_usdt("XAUUSDm", "BUY", 4210.0, 4220.0, 0.05) == 50.0
    assert estimate_mt5_pnl_usdt("XAUUSDm", "SELL", 4220.0, 4210.0, 0.05) == 50.0

    # F-04: Non-digit fallback ticket on real account returns False (never false-positive True)
    mt5_cli = MT5ExnessClient()
    assert await mt5_cli.close_mt5_ticket("434355782", "Pass123", "MT5-FALLBACK", 0.01) is False
    assert await mt5_cli.modify_mt5_ticket("434355782", "Pass123", "MT5-FALLBACK", 4200.0, 4230.0) is False

    # F-06: Direct `/setup exness LOGIN SECRET` command is redacted in Admin Audit Logs
    admin_store = AdminStore(settings=settings, db_path=vault.db_path)
    mw = WhitelistAuthMiddleware(settings=settings, admin_store=admin_store)
    secret_cmd_msg = _make_fake_message(333, "/setup exness 434355782 SuperSecretPass999 Exness-MT5Trial7")
    await mw(AsyncMock(return_value="ok"), secret_cmd_msg, {})
    logs = admin_store.get_audit_logs(limit=5)
    assert any("[REDACTED_CREDENTIALS]" in l.action for l in logs)
    assert all("SuperSecretPass999" not in l.action for l in logs)

    # 6. Sideways ADX(14) Regime Guard: blocks entry when market is sideways (`is_sideways=True`)
    vault.set_autotrade_enabled(333, True)
    sideways_dec = agent.synthesize_scores(
        s_tech=0.88,
        s_vol=0.20,
        s_sent=0.60,
        symbol="XAUUSD",
        entry_price=4200.0,
        atr_14=10.0,
        headlines=["Gold steady in narrow range"],
    )
    sideways_dec.adx_14 = 14.2
    sideways_dec.is_sideways = True
    sideways_dec.market_structure_status = "Sideways / Ranging (ADX 14.2 < 20 • Rawan Fakeout)"
    sideways_rep = await auto_trader.evaluate_and_execute(
        user_id=333,
        symbol="XAUUSD",
        decision_override=sideways_dec,
    )
    assert sideways_rep.executed is False
    assert any(
        g.gate_number == 3 and not g.passed and "PASAR SIDEWAYS" in g.detail
        for g in sideways_rep.gates
    )

    # 7. In-Position Emergency Breaking News Protection:
    # Open a BUY position on XAUUSD, then simulate sudden severe negative breaking news -> Emergency Early Exit
    vault.set_autotrade_enabled(333, True)
    open_buy_dec = agent.synthesize_scores(
        s_tech=0.92,
        s_vol=0.20,
        s_sent=0.65,
        symbol="XAUUSD",
        entry_price=4200.0,
        atr_14=10.0,
        headlines=["Strong central bank gold buying"],
    )
    open_buy_rep = await auto_trader.evaluate_and_execute(
        user_id=333,
        symbol="XAUUSD",
        decision_override=open_buy_dec,
        custom_lot=0.05,
    )
    assert open_buy_rep.executed is True
    assert len(vault.get_open_positions(333, "XAUUSD")) == 1

    emerg_exit_res = await auto_trader.monitor_and_manage_open_position(
        user_id=333,
        symbol="XAUUSD",
        price_override=4198.5,
        high_override=4199.0,
        low_override=4198.0,
        headlines_override=[
            "Emergency crash panic selloff as central bank liquidates gold reserves and hikes rates sharply",
            "Gold plunges on severe liquidity crisis and regulatory ban",
        ],
    )
    assert emerg_exit_res.action_taken in ("CLOSED_SL", "CLOSED_TP")
    assert "Proteksi Darurat Berita Buruk" in (emerg_exit_res.exit_reason or "")
    assert len(vault.get_open_positions(333, "XAUUSD")) == 0
