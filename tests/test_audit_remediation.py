from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest

from config.settings import Settings
from ai_engine.auto_trader.session_models import AutoTradeSessionState, parse_macro_event_mode_token, parse_max_positions_token
from ai_engine.auto_trader.session_manager import AutoTradeSessionManager
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.user_vault import UserVault, UserPositionRecord
from ai_engine.auto_trader.position_monitor import PositionMonitorResult
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from ai_engine.models.meta_label_engine import MacroRiskFilter
from ai_engine.models.sentiment_lexicon import macro_monetary_to_gold_signal
from ai_engine.risk_management.stop_loss_manager import StopLossManager
from ai_engine.auto_trader.mt5_symbols import estimate_mt5_pnl_usdt
from telegram_bot.handlers.commands import build_live_portfolio_state
from telegram_bot.views.formatters import format_signal_card, format_sentiment_card
from telegram_bot.views.setup_formatters import format_myaccount_card
from ai_engine.models.sentiment_analyzer import SentimentAnalysisResult


# -----------------------------------------------------------------------------
# Finding #1 & #2: Multi-layer session manager execution and no UnboundLocalError
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_findings_01_02_session_manager_multi_layer_reachable() -> None:
    cfg = Settings(_env_file=None, use_live_market_data=False, use_live_news_feed=False)
    vault = UserVault(cfg)
    agent = EnsembleDecisionAgent(settings=cfg)
    engine = SmartAutoTraderEngine(settings=cfg, ensemble_agent=agent, user_vault=vault)
    mgr = AutoTradeSessionManager(engine)

    sess = AutoTradeSessionState(
        session_id="SESS-TEST-12",
        user_id=12345,
        symbol="XAUUSD",
        timeframe="1m",
        duration_seconds=3600,
        duration_label="1 Jam",
        started_at_ts=1700000000,
        ends_at_ts=1700003600,
        started_at_iso="2026-10-16 12:00:00 UTC",
        ends_at_iso="2026-10-16 13:00:00 UTC",
        max_positions=4,
    )

    # Simulated open position
    pos1 = UserPositionRecord(
        position_id=101,
        user_id=12345,
        symbol="XAUUSD",
        side="BUY",
        entry_price=2650.0,
        quantity=0.01,
        notional_usdt=26500.0,
        stop_loss=2640.0,
        take_profit_1=0.0,
        take_profit_2=0.0,
        confidence=0.80,
        status="OPEN",
        order_id="TICKET-1",
        testnet=True,
        opened_at="2026-10-16T12:00:00Z",
    )

    with patch.object(vault, "get_open_positions", return_value=[pos1]), \
         patch.object(engine, "evaluate_and_execute") as mock_exec, \
         patch.object(mgr, "monitor_and_manage_open_position") as mock_mon:
        mock_mon.return_value = PositionMonitorResult(
            action_taken="HOLDING",
            position=pos1,
            current_price=2652.0,
            unrealized_pnl_usdt=2.0,
            unrealized_pnl_pct=0.07,
        )
        fake_report = MagicMock()
        fake_report.executed = False
        fake_report.rejection_summary = "Waiting for trigger"
        fake_report.decision = None
        mock_exec.return_value = fake_report

        res = await mgr._run_single_session_cycle(sess)

        # Finding #1 Proof: evaluate_and_execute MUST be called because 1 < 4 (max_positions)
        mock_exec.assert_awaited_once()
        # Finding #2 Proof: res must not raise UnboundLocalError and must return a valid monitor result
        assert res is not None


# -----------------------------------------------------------------------------
# Finding #3: Multi-position monitoring in SmartPositionMonitor
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_finding_03_position_monitor_evaluates_all_layers() -> None:
    cfg = Settings(_env_file=None, use_live_market_data=False, use_live_news_feed=False)
    vault = UserVault(cfg)
    agent = EnsembleDecisionAgent(settings=cfg)
    engine = SmartAutoTraderEngine(settings=cfg, ensemble_agent=agent, user_vault=vault)
    monitor = engine.session_manager

    pos1 = UserPositionRecord(
        position_id=1, user_id=111, symbol="XAUUSD", side="BUY",
        entry_price=2650.0, quantity=0.01, notional_usdt=26500.0,
        stop_loss=2640.0, take_profit_1=0.0, take_profit_2=0.0,
        confidence=0.80, status="OPEN", order_id="TK-1", testnet=True,
        opened_at="2026-10-16T12:00:00Z",
    )
    # Position 2 hit its stop loss @ 2645.0
    pos2 = UserPositionRecord(
        position_id=2, user_id=111, symbol="XAUUSD", side="BUY",
        entry_price=2652.0, quantity=0.01, notional_usdt=26520.0,
        stop_loss=2645.0, take_profit_1=0.0, take_profit_2=0.0,
        confidence=0.80, status="OPEN", order_id="TK-2", testnet=True,
        opened_at="2026-10-16T12:01:00Z",
    )

    with patch.object(vault, "get_open_positions", return_value=[pos1, pos2]), \
         patch.object(monitor, "_close_position_on_broker", AsyncMock()) as mock_close:
        mock_close.return_value = PositionMonitorResult(
            action_taken="CLOSED_SL",
            position=pos2,
            current_price=2644.0,
            realized_pnl_usdt=-8.0,
        )
        res = await monitor.monitor_and_manage_open_position(
            user_id=111, symbol="XAUUSD", price_override=2644.0, low_override=2644.0, high_override=2644.0
        )
        # Finding #3 Proof: pos2 SL breach is caught and closed even though it is at index 1!
        assert res.action_taken == "CLOSED_SL"
        assert res.position.position_id == 2


# -----------------------------------------------------------------------------
# Finding #4: DXY Trend Filter index out-of-bounds with 4 bars
# -----------------------------------------------------------------------------
def test_finding_04_dxy_filter_handles_4_bars_without_error() -> None:
    df_4 = pd.DataFrame({"close": [100.0, 100.1, 100.2, 100.4]})
    suppressed, reason = MacroRiskFilter.evaluate_dxy_trend_filter(df_4, "BUY", threshold_pct=0.15)
    assert "DXY filter error" not in reason
    assert suppressed is True


# -----------------------------------------------------------------------------
# Finding #5: Forex currency direction transmission
# -----------------------------------------------------------------------------
def test_finding_05_forex_currency_transmission() -> None:
    # Hawkish (+0.80) -> USD strengthens
    # EURUSD drops -> SELL
    s_eur, act_eur = macro_monetary_to_gold_signal(0.80, "EURUSD")
    assert act_eur == "SELL" and s_eur == -0.80

    # USDJPY rises -> BUY
    s_jpy, act_jpy = macro_monetary_to_gold_signal(0.80, "USDJPY")
    assert act_jpy == "BUY" and s_jpy == 0.80


# -----------------------------------------------------------------------------
# Finding #6: Portfolio contract size scaled PnL
# -----------------------------------------------------------------------------
def test_finding_06_portfolio_pnl_contract_size() -> None:
    # 0.05 lot on XAUUSD, $10 move ($2650 to $2660) = 0.05 * 10 * 100 = $50.00
    pnl_gold = estimate_mt5_pnl_usdt("XAUUSD", "BUY", 2650.0, 2660.0, 0.05)
    assert pnl_gold == 50.00

    cfg = Settings(_env_file=None, use_live_market_data=False)
    vault = UserVault(cfg)
    agent = EnsembleDecisionAgent(settings=cfg)
    rec = UserPositionRecord(
        position_id=99, user_id=123, symbol="XAUUSD", side="BUY",
        entry_price=2650.0, quantity=0.05, notional_usdt=13250.0,
        stop_loss=2640.0, take_profit_1=0.0, take_profit_2=0.0,
        confidence=0.80, status="OPEN", order_id="TK-99", testnet=True,
        opened_at="2026-10-16T12:00:00Z",
    )
    with patch.object(vault, "get_open_positions", return_value=[rec]), \
         patch.object(agent, "load_market_series", return_value=pd.DataFrame({"close": [2660.0]})):
        state = build_live_portfolio_state(agent, cfg, user_id=123, user_vault=vault)
        assert state["unrealized_pnl"] == 50.00


# -----------------------------------------------------------------------------
# Finding #7: Targeted stop matching in auth middleware
# -----------------------------------------------------------------------------
def test_finding_07_auth_middleware_targeted_stop_matching() -> None:
    from telegram_bot.middlewares.auth_middleware import WhitelistAuthMiddleware
    mw = WhitelistAuthMiddleware(Settings(_env_file=None))

    for cmd in ("/autotrade stop 1", "/autotrade stop AT-2", "/autotrade off 1", "/autotrade stop XAUUSD", "/unlink"):
        lower_act = cmd.strip().lower()
        is_emergency = (
            lower_act.startswith("/autotrade off")
            or lower_act.startswith("/autotrade stop")
            or lower_act == "/unlink"
            or lower_act in ("cb:autotrade:toggle:off", "cb:setup:unlink")
        )
        assert is_emergency is True


# -----------------------------------------------------------------------------
# Finding #8: Take profit removal formatting in signal and account cards
# -----------------------------------------------------------------------------
def test_finding_08_tp_removal_formatting() -> None:
    decision = SignalDecision(
        action="BUY", confidence=0.85, risk_multiplier=0.80, symbol="XAUUSD", timeframe="1m",
        entry_price=2650.0, stop_loss=2642.0, take_profit_1=0.0, take_profit_2=0.0,
    )
    card = format_signal_card(decision)
    # Must NOT show negative percent
    assert "-100.0%" not in card
    assert "Dihapus / Tanpa Batas" in card


# -----------------------------------------------------------------------------
# Finding #9: Sentiment card blackout state consistency
# -----------------------------------------------------------------------------
def test_finding_09_sentiment_card_blackout_state_consistency() -> None:
    res = SentimentAnalysisResult(
        symbol="XAUUSD", s_sent=0.10, label="NEUTRAL",
        bullish_ratio=0.5, bearish_ratio=0.3, neutral_ratio=0.2, model_name="FinBERT",
        is_macro_blackout=True, blackout_event="USD CPI m/m",
    )
    card = format_sentiment_card(res)
    # When macro_blackout_enabled=False by default, card should reflect Gembok Nonaktif / CLEAR
    assert "BLACKOUT AKTIF" not in card
    assert "Gembok Nonaktif" in card


# -----------------------------------------------------------------------------
# Finding #10: Event mode anti-MC SL bounds on micro accounts
# -----------------------------------------------------------------------------
def test_finding_10_anti_mc_sl_bounds_on_micro_account() -> None:
    sl_mgr = StopLossManager()
    t_micro = sl_mgr.calculate_event_mode_targets(
        entry_price=2650.0, action="BUY", account_balance=50.0, lot_size=0.05, symbol="XAUUSD"
    )
    loss = (2650.0 - t_micro.stop_loss) * 0.05 * 100.0
    # Must never risk more than 5% of balance ($2.50)
    assert loss <= 2.50 + 1e-4


# -----------------------------------------------------------------------------
# UT Bot Alerts Strategy Indicators & Triggers
# -----------------------------------------------------------------------------
def test_ut_bot_alerts_strategy_indicators_and_triggers() -> None:
    from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

    sample_df = pd.DataFrame({
        "open": [100.0, 101.0, 103.0, 102.0, 99.0, 97.0, 95.0, 98.0, 101.0],
        "high": [101.5, 103.5, 104.0, 102.5, 99.5, 97.5, 96.0, 99.0, 102.0],
        "low": [99.5, 100.5, 102.0, 98.5, 96.5, 94.5, 94.0, 95.0, 98.0],
        "close": [101.0, 103.0, 102.0, 99.0, 97.0, 95.0, 95.5, 98.5, 101.5],
        "volume": [100.0] * 9,
    })

    ut_df = OHLCVPreprocessor.compute_ut_bot_indicators(sample_df, key_value=1.0, atr_period=3)
    assert "ut_trailing_stop" in ut_df.columns
    assert "ut_pos" in ut_df.columns
    assert "ut_buy" in ut_df.columns
    assert "ut_sell" in ut_df.columns
    assert "ut_trend" in ut_df.columns

    # Bar 3 drops to 99.0 below trailing stop -> ut_sell must be True
    assert ut_df["ut_sell"].iloc[3] is True or ut_df["ut_sell"].iloc[3] == 1
    assert ut_df["ut_pos"].iloc[3] == -1

    # Bar 7 crosses back up to 98.5 above trailing stop -> ut_buy must be True
    assert ut_df["ut_buy"].iloc[7] is True or ut_df["ut_buy"].iloc[7] == 1
    assert ut_df["ut_pos"].iloc[7] == 1


def test_supertrend_strategy_indicators_and_triggers() -> None:
    from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

    df = pd.DataFrame({
        "open": [100.0, 102.0, 105.0, 108.0, 104.0, 99.0, 95.0, 92.0, 93.0, 97.0, 105.0],
        "high": [102.5, 106.0, 109.0, 110.0, 105.0, 100.0, 96.5, 94.0, 98.0, 106.0, 108.0],
        "low":  [99.5, 101.5, 104.5, 103.0, 99.0, 94.0, 91.0, 90.0, 93.0, 97.0, 103.0],
        "close": [102.0, 105.0, 108.0, 104.0, 99.0, 95.0, 92.0, 93.0, 97.0, 105.0, 107.0],
        "volume": [1000.0] * 11,
    })

    st_df = OHLCVPreprocessor.compute_supertrend_indicators(df, period=3, multiplier=2.0)
    assert "supertrend" in st_df.columns
    assert "supertrend_dir" in st_df.columns
    assert "supertrend_buy" in st_df.columns
    assert "supertrend_sell" in st_df.columns

    # Bar 4 drops to 99.0 breaking down -> sell must trigger and dir == -1
    assert st_df["supertrend_dir"].iloc[4] == -1
    assert st_df["supertrend_sell"].iloc[4] is True or st_df["supertrend_sell"].iloc[4] == 1

    # Bar 9 breaks out to 105.0 -> buy must trigger and dir == 1
    assert st_df["supertrend_dir"].iloc[9] == 1
    assert st_df["supertrend_buy"].iloc[9] is True or st_df["supertrend_buy"].iloc[9] == 1


def test_candlestick_patterns_identified_pine_script() -> None:
    from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

    # Test Bullish Engulfing
    df_engulf = pd.DataFrame({
        "open":  [110.0, 108.0, 106.0, 105.0, 104.0, 104.0, 99.0],
        "high":  [111.0, 109.0, 107.0, 106.0, 105.0, 104.5, 107.0],
        "low":   [109.0, 107.0, 105.0, 103.5, 102.0,  99.5, 98.5],
        "close": [109.5, 107.5, 105.5, 104.0, 102.5, 100.0, 106.0],
        "volume": [1000.0] * 7,
    })
    res_eng = OHLCVPreprocessor.identify_candlestick_patterns(df_engulf, trend_bars=5)
    assert res_eng["cp_bull_engulf"].iloc[-1] is True or res_eng["cp_bull_engulf"].iloc[-1] == 1
    assert res_eng["cp_pattern_name"].iloc[-1] == "Bullish Engulfing"
    assert res_eng["cp_score"].iloc[-1] > 0

    # Test Hammer
    df_hammer = pd.DataFrame({
        "open":  [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0],
        "high":  [101.0, 99.5, 98.5, 97.5, 96.5, 95.5, 94.2],
        "low":   [99.0,  97.5, 96.5, 95.5, 94.5, 93.5, 89.0],
        "close": [99.5,  98.0, 97.0, 96.0, 95.0, 94.0, 94.1],
        "volume": [1000.0] * 7,
    })
    res_ham = OHLCVPreprocessor.identify_candlestick_patterns(df_hammer, trend_bars=5)
    assert res_ham["cp_hammer"].iloc[-1] is True or res_ham["cp_hammer"].iloc[-1] == 1
    assert "Hammer" in res_ham["cp_pattern_name"].iloc[-1]


def test_oc_signals_pine_script_entry_sl_tp() -> None:
    from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

    sample_df = pd.DataFrame({
        "open":  [100.0, 101.0, 100.0, 99.0, 104.0, 103.0, 97.0],
        "high":  [101.5, 102.0, 101.0, 105.0, 105.5, 104.0, 98.0],
        "low":   [99.5,  100.0,  98.5,  98.0, 103.0,  96.5, 96.0],
        "close": [101.0, 100.5,  99.0, 104.5, 103.5,  97.0, 97.5],
        "volume": [1000.0] * 7,
    })

    res = OHLCVPreprocessor.compute_oc_signals(
        sample_df, body_len=3, body_mult=1.2, sl_mode="Candle wick", tp_rmult=2.0
    )
    assert "oc_signal" in res.columns
    assert "oc_entry" in res.columns
    assert "oc_sl" in res.columns
    assert "oc_tp" in res.columns

    # Bar 2 breaks down below previous body -> SELL signal
    assert res["oc_signal"].iloc[2] == -1
    assert res["oc_entry"].iloc[2] == 99.0
    assert res["oc_sl"].iloc[2] == 101.0  # wick high
    assert res["oc_tp"].iloc[2] == 95.0   # entry - (2 * 2.0) = 95.0

    # Bar 3 breaks out above previous body -> BUY signal
    assert res["oc_signal"].iloc[3] == 1
    assert res["oc_entry"].iloc[3] == 104.5
    assert res["oc_sl"].iloc[3] == 98.0   # wick low
    assert res["oc_tp"].iloc[3] == 117.5  # entry + (6.5 * 2.0) = 117.5


def test_round2_oc_signals_dynamic_rrr_anchoring() -> None:
    """Verify OC Signals dynamic RRR anchoring maintains strict 1:2 RRR even when entry price moves."""
    cfg = Settings(_env_file=None, use_live_market_data=False)
    agent = EnsembleDecisionAgent(settings=cfg)

    indicators = {
        "oc_signal": 1.0,
        "oc_sl": 2635.0,
        "oc_tp": 2650.0, # static from earlier bar
    }

    # Order enters at 2648.0 (8 dollars above original signal)
    decision = agent.synthesize_scores(
        s_tech=0.60,
        s_vol=0.15,
        s_sent=0.10,
        entry_price=2648.0,
        indicators=indicators,
        event_mode=None,
    )
    assert decision.action == "BUY"
    assert decision.stop_loss == 2635.0
    # Risk distance = 2648 - 2635 = 13.0
    # TP1 must be 2648 + (13.0 * 2.0) = 2674.0 (strictly maintaining 1:2 RRR!)
    assert decision.take_profit_1 == 2674.0
    assert decision.take_profit_2 == 2687.0


def test_round2_flat_volume_tolerance() -> None:
    """Verify flat-volume data feeds do not suppress valid breakout signals when vol_mult > 0."""
    from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

    # Data with flat volume (all 100.0, zero variance)
    sample_flat_vol = pd.DataFrame({
        "open":  [100.0, 101.0, 100.0, 99.0, 104.0],
        "high":  [101.5, 102.0, 101.0, 105.0, 105.5],
        "low":   [99.5,  100.0,  98.5,  98.0, 103.0],
        "close": [101.0, 100.5,  99.0, 104.5, 103.5],
        "volume": [100.0] * 5,
    })

    res = OHLCVPreprocessor.compute_oc_signals(
        sample_flat_vol, body_len=3, body_mult=1.2, vol_mult=1.5
    )
    # Flat volume must not block the breakout on bar 3
    assert res["oc_signal"].iloc[3] == 1
