"""
Test Suite for Liquidity & Smart Money Concepts (SMC):
1. Deteksi Equal Highs (EQH) dan Equal Lows (EQL) beserta skor kerapian.
2. Identifikasi Buy Side Liquidity (BSL) di atas Highs dan Sell Side Liquidity (SSL) di bawah Lows.
3. Deteksi Sapuan Likuiditas (Liquidity Sweep / Stop Hunt / Purge & Rejection):
   - Bullish SSL Sweep (Harga memancing SL buyer di bawah EQL lalu berbalik naik).
   - Bearish BSL Sweep (Harga memancing SL seller di atas EQH lalu berbalik turun).
   - Membedakan Sweep (Jebakan) vs Displacement (Tembus Bodi Penuh).
4. Penempatan SL Cerdas (Anti-Stop-Hunt Buffer): Jangan taruh SL di tempat yang terlalu jelas!
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from data_pipeline.liquidity_analyzer import (
    LiquidityAnalyzer,
    LiquidityPool,
    LiquiditySweepEvent,
    LiquidityAnalysisResult,
)
from ai_engine.risk_management.stop_loss_manager import StopLossManager


def create_candle_df(records: list[dict[str, float]]) -> pd.DataFrame:
    """Helper to convert records to DataFrame."""
    return pd.DataFrame(records)


def test_rule_1_detect_equal_highs_and_lows():
    """Rule 1: Tandai Equal Highs (EQH) dan Equal Lows (EQL)."""
    # Buat rangkaian lilin dengan 2 puncak sejajar di 120.0 (Equal Highs) dan 2 lembah sejajar di 100.0 (Equal Lows)
    records = []
    # Bar 0-2: naik ke 120
    records.extend([
        {"open": 105.0, "high": 112.0, "low": 104.0, "close": 110.0},
        {"open": 110.0, "high": 120.0, "low": 108.0, "close": 118.0},  # Peak 1 (120.0)
        {"open": 118.0, "high": 119.0, "low": 108.0, "close": 110.0},
    ])
    # Bar 3-5: turun ke lembah 1 (100.0)
    records.extend([
        {"open": 110.0, "high": 112.0, "low": 100.0, "close": 102.0},  # Trough 1 (100.0)
        {"open": 102.0, "high": 110.0, "low": 101.0, "close": 108.0},
        {"open": 108.0, "high": 115.0, "low": 107.0, "close": 114.0},
    ])
    # Bar 6-8: naik ke puncak 2 sejajar (120.1 - Equal Highs)
    records.extend([
        {"open": 114.0, "high": 120.1, "low": 112.0, "close": 118.0},  # Peak 2 (120.1)
        {"open": 118.0, "high": 119.0, "low": 108.0, "close": 110.0},
        {"open": 110.0, "high": 112.0, "low": 105.0, "close": 107.0},
    ])
    # Bar 9-11: turun ke lembah 2 sejajar (100.05 - Equal Lows)
    records.extend([
        {"open": 107.0, "high": 108.0, "low": 100.05, "close": 102.0},  # Trough 2 (100.05)
        {"open": 102.0, "high": 108.0, "low": 101.0, "close": 106.0},
        {"open": 106.0, "high": 110.0, "low": 105.0, "close": 108.0},
    ])
    df = create_candle_df(records)

    eq_highs, eq_lows = LiquidityAnalyzer.detect_equal_highs_and_lows(df, tolerance_pct=0.002, window=1)

    assert len(eq_highs) >= 1
    assert abs(eq_highs[0].price_level - 120.05) <= 0.15
    assert eq_highs[0].pool_type == "EQUAL_HIGHS"
    assert eq_highs[0].kind == "BSL"
    assert eq_highs[0].neatness_score >= 0.50

    assert len(eq_lows) >= 1
    assert abs(eq_lows[0].price_level - 100.02) <= 0.15
    assert eq_lows[0].pool_type == "EQUAL_LOWS"
    assert eq_lows[0].kind == "SSL"


def test_rule_2_bsl_above_highs_and_ssl_below_lows():
    """Rule 2: Atas High = BSL, Bawah Low = SSL (Magnet Likuiditas Harga)."""
    # Price currently at 110.0, High at 125.0 (BSL), Low at 95.0 (SSL)
    records = [
        {"open": 100.0, "high": 125.0, "low": 98.0, "close": 120.0},
        {"open": 120.0, "high": 122.0, "low": 95.0, "close": 105.0},
        {"open": 105.0, "high": 115.0, "low": 102.0, "close": 110.0},
    ]
    df = create_candle_df(records)

    res = LiquidityAnalyzer.analyze_liquidity(df, symbol="XAUUSD", timeframe="5m")
    assert res.nearest_bsl is not None and res.nearest_bsl >= 110.0
    assert res.nearest_ssl is not None and res.nearest_ssl <= 110.0
    assert "Buy Side Liquidity (BSL)" in res.rule_2_bsl_ssl_summary
    assert "Sell Side Liquidity (SSL)" in res.rule_2_bsl_ssl_summary


def test_bullish_liquidity_sweep_ssl_stop_hunt():
    """Verify Bullish Liquidity Sweep: price dips below SSL to hunt SL, then rejects with strong wick."""
    # SSL level established at 100.0
    ssl_level = 100.0
    # Final candle sweeps down to 96.0 (below 100.0), but closes back at 102.0 with a long lower wick
    records = [
        {"open": 108.0, "high": 110.0, "low": 104.0, "close": 105.0},
        {"open": 105.0, "high": 106.0, "low": 96.0, "close": 102.0},  # Sweep wick: low 96, open 105, close 102 -> lower wick = 102-96=6
    ]
    df = create_candle_df(records)

    sweep = LiquidityAnalyzer.detect_liquidity_sweep(
        df=df,
        bsl_levels=[120.0],
        ssl_levels=[ssl_level],
        lookback_bars=2,
    )
    assert sweep is not None
    assert sweep.sweep_type == "BULLISH_SSL_SWEEP"
    assert sweep.sweep_extreme_price == 96.0
    assert sweep.is_fakeout_reversal is True
    assert "BULLISH LIQUIDITY SWEEP" in sweep.details


def test_bearish_liquidity_sweep_bsl_stop_hunt():
    """Verify Bearish Liquidity Sweep: price pokes above BSL to hunt SL, then rejects with strong upper wick."""
    bsl_level = 120.0
    # Final candle spikes up to 125.0 (above 120.0), but closes back down at 118.0 with a long upper wick
    records = [
        {"open": 112.0, "high": 116.0, "low": 111.0, "close": 115.0},
        {"open": 115.0, "high": 125.0, "low": 114.0, "close": 117.0},  # Sweep wick: high 125, open 115, close 117 -> upper wick = 125-117=8
    ]
    df = create_candle_df(records)

    sweep = LiquidityAnalyzer.detect_liquidity_sweep(
        df=df,
        bsl_levels=[bsl_level],
        ssl_levels=[100.0],
        lookback_bars=2,
    )
    assert sweep is not None
    assert sweep.sweep_type == "BEARISH_BSL_SWEEP"
    assert sweep.sweep_extreme_price == 125.0
    assert sweep.is_fakeout_reversal is True
    assert "BEARISH LIQUIDITY SWEEP" in sweep.details


def test_rule_3_anti_stop_hunt_sl_placement():
    """Rule 3: Jangan taruh SL di tempat yang terlalu jelas (Anti-Stop-Hunt Protection)."""
    sl_mgr = StopLossManager()
    entry = 105.0
    # Obvious retail support / Equal Lows sitting right at 100.0
    obvious_ssl = 100.0
    latest_hl = 101.0

    # Without obvious liquidity awareness: SL would sit right at or near 100.0 (easy prey for stop hunt)
    targets_safe = sl_mgr.calculate_structural_targets(
        entry_price=entry,
        action="BUY",
        latest_hl=latest_hl,
        atr_14=5.0,
        min_rrr=2.0,
        obvious_ssl_level=obvious_ssl,
    )

    # With Anti-Stop-Hunt buffer: SL MUST be placed safely below the obvious retail SSL pool (below 100.0)
    assert targets_safe.stop_loss < obvious_ssl
    # Verify distance gives room beyond the retail liquidity pool
    assert (obvious_ssl - targets_safe.stop_loss) >= 1.0
    # Verify 1:2 RRR is preserved based on the new safe SL distance
    risk_1r = entry - targets_safe.stop_loss
    assert abs((targets_safe.take_profit_1 - entry) - (2.0 * risk_1r)) <= 0.05


def test_sweep_vs_breakout_resistance_sweep():
    """Verify Scenario 1: Resistance Sweep (Tembus wick, close di bawah, balik cepat)."""
    r_level = 120.0
    records = [
        {"open": 110.0, "high": 115.0, "low": 109.0, "close": 114.0},
        {"open": 114.0, "high": 123.0, "low": 112.0, "close": 118.0},  # Wick breaches 120, but closes at 118 (below)
    ]
    df = create_candle_df(records)

    assessment = LiquidityAnalyzer.evaluate_sweep_vs_breakout(
        df=df,
        resistance_level=r_level,
        lookback_bars=2,
    )
    assert assessment is not None
    assert assessment.verdict == "SWEEP_JEBAKAN"
    assert assessment.pierce_type == "WICK"
    assert assessment.close_position == "DI_DALAM_LEVEL"
    assert assessment.retest_status == "GAGAL_BALIK_ARAH"
    assert assessment.retest_held is False
    assert "1. 🔍 TEMBUS PAKAI: WICK" in assessment.three_checks_display
    assert "2. 🚪 CANDLE CLOSE: DI_DALAM_LEVEL" in assessment.three_checks_display


def test_sweep_vs_breakout_resistance_true_breakout():
    """Verify Scenario 2: True Breakout (Body besar close di atas, retest bertahan)."""
    r_level = 120.0
    records = [
        {"open": 112.0, "high": 116.0, "low": 111.0, "close": 115.0},
        # Bar 1: Strong body breakout closing at 124.0 (clearly above 120.0)
        {"open": 115.0, "high": 125.0, "low": 114.5, "close": 124.0},
        # Bar 2: Retest pulling back to test 120.1 and closing back up at 122.5 (holds as new support)
        {"open": 124.0, "high": 124.5, "low": 120.1, "close": 122.5},
    ]
    df = create_candle_df(records)

    assessment = LiquidityAnalyzer.evaluate_sweep_vs_breakout(
        df=df,
        resistance_level=r_level,
        lookback_bars=3,
    )
    assert assessment is not None
    assert assessment.verdict == "BREAKOUT_ASLI"
    assert assessment.pierce_type == "BODY"
    assert assessment.close_position == "DI_LUAR_LEVEL"
    assert assessment.retest_status == "BERTAHAN_FLIP_SR"
    assert assessment.retest_held is True
    assert "Resistance berubah fungsi menjadi Support" in assessment.details


def test_sweep_vs_breakout_support_sweep():
    """Verify Support Sweep: Equal Lows swept by wick, candle closes above support."""
    s_level = 100.0
    records = [
        {"open": 105.0, "high": 108.0, "low": 104.0, "close": 106.0},
        # Wick dips down to 96.0, but closes at 101.5 (above support)
        {"open": 106.0, "high": 107.0, "low": 96.0, "close": 101.5},
    ]
    df = create_candle_df(records)

    assessment = LiquidityAnalyzer.evaluate_sweep_vs_breakout(
        df=df,
        support_level=s_level,
        lookback_bars=2,
    )
    assert assessment is not None
    assert assessment.verdict == "SWEEP_JEBAKAN"
    assert assessment.pierce_type == "WICK"
    assert assessment.close_position == "DI_DALAM_LEVEL"
    assert assessment.retest_held is False
    assert "Waspada Berbalik Naik Cepat" in assessment.details
