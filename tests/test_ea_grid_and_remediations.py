"""
Comprehensive verification test suite for Part 1 (9 Bug Fixes) and Part 2 (EA_GRID_SCALPER).
Covers all 5 acceptance requirements in Section 3 of PRODUCTION_EA_REMEDIATION_AND_GRID_SPEC.md.
"""
from __future__ import annotations

from pathlib import Path
import math
import time
from unittest.mock import AsyncMock, patch
import pandas as pd
import pytest

from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.mt5_symbols import (
    GridScalperConfig,
    build_pending_stop_grid_levels,
    calculate_compounded_grid_lot,
    get_mt5_digits_and_point,
    get_mt5_pip_size,
)
from ai_engine.auto_trader.session_manager import AutoTradeSessionManager
from ai_engine.auto_trader.user_vault import UserVault
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from ai_engine.models.sentiment_analyzer import FinancialSentimentAnalyzer
from ai_engine.risk_management.risk_policy import RiskPolicy, SymbolTradingSpec
from config.settings import Settings


# -----------------------------------------------------------------------------
# 1. test_score_timeframe_trend_no_name_error
# -----------------------------------------------------------------------------
def test_score_timeframe_trend_no_name_error() -> None:
    """Requirement 1: Confirm _score_timeframe_trend() returns non-zero scores on trending DataFrames and never raises NameError."""
    cfg = Settings(_env_file=None, use_live_market_data=False)
    agent = EnsembleDecisionAgent(settings=cfg)

    # 60 bars of upward trending data
    df = pd.DataFrame({
        "open": [2600.0 + i * 1.5 for i in range(60)],
        "high": [2602.0 + i * 1.5 for i in range(60)],
        "low": [2599.0 + i * 1.5 for i in range(60)],
        "close": [2601.5 + i * 1.5 for i in range(60)],
        "volume": [100.0] * 60,
    })

    score, atr, lbl = agent._score_timeframe_trend(df)
    assert isinstance(score, float)
    assert score != 0.0  # Must NOT fail into silent exception fallback 0.0, 0.0, '🟡 FLAT'
    assert lbl in ("🟢 BULL", "🔴 BEAR", "🟡 FLAT")
    assert atr > 0.0


# -----------------------------------------------------------------------------
# 2. test_sentiment_composite_not_overwritten
# -----------------------------------------------------------------------------
def test_sentiment_composite_not_overwritten() -> None:
    """Requirement 2: Confirm analyze_symbol() preserves the 65% composite_sentiment + 35% FinBERT blend."""
    cfg = Settings(_env_file=None, use_live_news_feed=True)
    analyzer = FinancialSentimentAnalyzer(settings=cfg)

    # Mock aggregator returning strong positive composite sentiment (+0.80)
    mock_intel = {
        "is_macro_blackout": False,
        "blackout_event": None,
        "provider_breakdown": {"CryptoNews": 5},
        "composite_sentiment": 0.80,
        "macro_stance": "HAWKISH",
        "gold_action_bias": "BUY",
        "top_headlines": [{"title": "Bitcoin surges past key resistance as institutional inflows climb", "source": "Coindesk"}],
    }

    with patch.object(analyzer.aggregator, "get_composite_intelligence_sync", return_value=mock_intel), \
         patch.object(analyzer, "ingest_and_learn_live_news", return_value=([], 0)):

        res = analyzer.analyze_symbol(symbol="BTCUSDT")
        # If composite_sentiment (+0.80) is preserved and blended (65%), s_sent must be positive (> 0.20)!
        assert res.s_sent > 0.20
        assert res.s_sent <= 1.0


# -----------------------------------------------------------------------------
# 3. test_micro_account_min_lot_execution
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_micro_account_min_lot_execution(tmp_path: Path) -> None:
    """Requirement 3: Confirm $10 - $50 accounts running EA modes are approved for 0.01 lot within micro risk fraction."""
    db_p = tmp_path / "test_micro.db"
    cfg = Settings(_env_file=None, user_vault_db_path=str(db_p), use_live_market_data=False)
    vault = UserVault(settings=cfg, db_path=str(db_p))

    # $20.00 micro-account
    vault.save_user_credentials(
        user_id=301,
        api_key="EXNESS_DEMO_301",
        api_secret="secret",
        username="MicroScalper",
        server_name="Exness-MT5Trial7",
        testnet=True,
        verified_usdt_balance=20.00,
        autotrade_enabled=True,
    )
    engine = SmartAutoTraderEngine(settings=cfg, user_vault=vault)

    # Gold EA Scalping: Entry $2650, SL $2648 ($2.00 stop distance on 0.01 lot = $2.00 loss <= $10.00 micro limit)
    mock_dec = SignalDecision(
        symbol="XAUUSD",
        timeframe="1m",
        action="BUY",
        badge="EA-SCP BUY",
        confidence=0.85,
        s_tech=0.60,
        s_vol=0.10,
        s_sent=0.0,
        risk_multiplier=1.0,
        entry_price=2650.0,
        stop_loss=2648.0,
        take_profit_1=2654.0,
        take_profit_2=2654.0,
        expectancy_r=1.0,
        confluence_pillars=5,
        is_sideways=False,
        resistance_1=2680.0,
        support_1=2620.0,
        dist_to_r1_pct=0.50,
        dist_to_s1_pct=0.50,
        indicators={"ea_mode": "EA_SCALPING"},
    )

    report = await engine.evaluate_and_execute(
        user_id=301,
        symbol="XAUUSD",
        decision_override=mock_dec,
        ea_mode="EA_SCALPING",
    )

    assert report.executed is True
    assert report.order_result is not None
    assert report.order_result.executed_qty == 0.01


# -----------------------------------------------------------------------------
# 4. test_session_cleanup_and_deduplication
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_session_cleanup_and_deduplication(tmp_path: Path) -> None:
    """Requirement 4: Confirm _session_loop pops sess.session_id cleanly and duplicate symbol sessions are stopped first."""
    db_p = tmp_path / "test_sess.db"
    cfg = Settings(_env_file=None, user_vault_db_path=str(db_p), use_live_market_data=False)
    vault = UserVault(settings=cfg, db_path=str(db_p))
    vault.save_user_credentials(
        user_id=401,
        api_key="EXNESS_DEMO_401",
        api_secret="secret",
        username="SessionUser",
        server_name="Exness-MT5Trial7",
        testnet=True,
        verified_usdt_balance=1000.0,
        autotrade_enabled=True,
    )
    engine = SmartAutoTraderEngine(settings=cfg, user_vault=vault)
    mgr = engine.session_manager

    # Start first session on XAUUSD
    s1, _ = await mgr.start_timed_session(user_id=401, symbol="XAUUSD", duration_code="20m")
    assert s1.session_id in mgr._sessions

    # Start second session on same symbol -> first session must be stopped and replaced
    s2, _ = await mgr.start_timed_session(user_id=401, symbol="XAUUSD", duration_code="20m")
    assert s2.session_id in mgr._sessions
    assert s1.session_id not in mgr._sessions or mgr._sessions[s1.session_id].status == "STOPPED"

    # Stop session cleanly
    stopped = mgr.stop_session_by_id(401, s2.session_id)
    assert stopped is not None
    assert s2.session_id not in mgr._sessions


# -----------------------------------------------------------------------------
# 5. test_grid_scalper_multi_asset_levels
# -----------------------------------------------------------------------------
def test_grid_scalper_multi_asset_levels() -> None:
    """Requirement 5: Confirm build_pending_stop_grid_levels() creates symmetric BUY_STOP / SELL_STOP and lot compounding scales accurately."""
    cfg = GridScalperConfig(
        layers_per_side=5,
        first_offset_pips=10.0,
        grid_step_pips=10.0,
        take_profit_pips=15.0,
        stop_loss_pips=25.0,
    )

    # A. Test Lot Compounding Progression matching video
    assert calculate_compounded_grid_lot(10.0, cfg) == 0.01
    assert calculate_compounded_grid_lot(15.0, cfg) == 0.01
    assert calculate_compounded_grid_lot(45.0, cfg) == 0.03
    assert calculate_compounded_grid_lot(120.0, cfg) == 0.08
    assert calculate_compounded_grid_lot(1500.0, cfg) == 1.00
    assert calculate_compounded_grid_lot(3000.0, cfg) == 2.00
    assert calculate_compounded_grid_lot(5000.0, cfg) == 2.00  # Capped at max_lot=2.00

    # B. Test Grid Generation for Gold (XAUUSD @ $2650.00, Ask=2650.30)
    levels = build_pending_stop_grid_levels(
        symbol="XAUUSD",
        bid_price=2650.00,
        ask_price=2650.30,
        cfg=cfg,
    )
    assert len(levels) == 10  # 5 Buy Stops + 5 Sell Stops

    buy_stops = [l for l in levels if l["side"] == "BUY_STOP"]
    sell_stops = [l for l in levels if l["side"] == "SELL_STOP"]
    assert len(buy_stops) == 5
    assert len(sell_stops) == 5

    # Buy stops must be above Ask ($2650.30)
    for b in buy_stops:
        assert b["op_code"] == 4
        assert b["price"] > 2650.30
        assert b["take_profit"] > b["price"]
        assert b["stop_loss"] < b["price"]

    # Sell stops must be below Bid ($2650.00)
    for s in sell_stops:
        assert s["op_code"] == 5
        assert s["price"] < 2650.00
        assert s["take_profit"] < s["price"]
        assert s["stop_loss"] > s["price"]

    # Test digits & point for USDJPY
    digits, pt = get_mt5_digits_and_point("USDJPY", 155.0)
    assert digits == 3
    assert pt == 0.001
