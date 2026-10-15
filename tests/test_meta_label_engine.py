from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from ai_engine.models.meta_label_engine import (
    AdvancedFeatureExtractor,
    MacroRiskFilter,
    TripleBarrierConfig,
    TripleBarrierLabeler,
    TwoStageMetaLabelEngine,
)
from ai_engine.models.ohlcv_predictor import OHLCVPredictor
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from config.settings import Settings


# -----------------------------------------------------------------------------
# Section 2 & 6: Triple-Barrier Labeling Tests
# -----------------------------------------------------------------------------
def test_triple_barrier_numerical_specification() -> None:
    """
    SPEC Section 2.3 Concrete Numerical Example:
    Close = 65,000, ATR = 150, k_tp = 2.0, k_sl = 1.5, horizon = 30 bars.
    Upper = 65,300, Lower = 64,775.
    """
    config = TripleBarrierConfig(k_tp=2.0, k_sl=1.5, horizon_bars=30)

    # 1. Path hits Upper barrier first (bar t+12 hits 65,300) -> Label +1
    dates = pd.date_range("2026-10-14", periods=35, freq="15min", tz="UTC")
    closes = np.full(35, 65000.0)
    highs = np.full(35, 65100.0)
    lows = np.full(35, 64900.0)

    # At bar 12, price breaches 65,300
    highs[12] = 65350.0

    df_buy = pd.DataFrame(
        {"close": closes, "high": highs, "low": lows, "atr_14": 150.0},
        index=dates,
    )
    labels_buy = TripleBarrierLabeler.apply_triple_barrier(df_buy, config)
    assert labels_buy.iloc[0] == 1, "Should assign label +1 when upper barrier touched first"

    # 2. Path hits Lower barrier first (bar t+5 dips to 64,700) -> Label -1
    highs_sell = np.full(35, 65100.0)
    lows_sell = np.full(35, 64900.0)
    lows_sell[5] = 64700.0
    df_sell = pd.DataFrame(
        {"close": closes, "high": highs_sell, "low": lows_sell, "atr_14": 150.0},
        index=dates,
    )
    labels_sell = TripleBarrierLabeler.apply_triple_barrier(df_sell, config)
    assert labels_sell.iloc[0] == -1, "Should assign label -1 when lower barrier touched first"

    # 3. Path touches neither barrier within horizon -> Label 0
    highs_flat = np.full(35, 65050.0)
    lows_flat = np.full(35, 64950.0)
    df_flat = pd.DataFrame(
        {"close": closes, "high": highs_flat, "low": lows_flat, "atr_14": 150.0},
        index=dates,
    )
    labels_flat = TripleBarrierLabeler.apply_triple_barrier(df_flat, config)
    assert labels_flat.iloc[0] == 0, "Should assign label 0 when time horizon expires without barrier touch"


def test_ohlcv_predictor_triple_barrier_mapping() -> None:
    """Test OHLCVPredictor.create_triple_barrier_labels maps to 0=SELL, 1=HOLD, 2=BUY."""
    dates = pd.date_range("2026-10-14", periods=40, freq="15min", tz="UTC")
    closes = np.full(40, 100.0)
    highs = np.full(40, 101.0)
    lows = np.full(40, 99.0)
    # Breach upper barrier on first bar
    highs[2] = 110.0

    df = pd.DataFrame(
        {"open": closes, "close": closes, "high": highs, "low": lows, "atr_14": 2.0},
        index=dates,
    )
    labels = OHLCVPredictor.create_triple_barrier_labels(df, k_tp=2.0, k_sl=1.5, horizon_bars=10)
    assert labels.iloc[0] == 2  # 2 corresponds to BUY class


# -----------------------------------------------------------------------------
# Section 4: Multi-Dimensional Feature Vector Tests
# -----------------------------------------------------------------------------
def test_advanced_feature_extractor() -> None:
    """Test mathematical definitions of EMA Slope 50, Wick Asymmetry, S/R distance, and VolZ."""
    dates = pd.date_range("2026-10-14 08:00:00", periods=60, freq="15min", tz="UTC")
    close = 100.0 + np.linspace(0, 10, 60)
    high = close + 1.5
    low = close - 0.5
    volume = np.full(60, 1000.0)
    volume[-1] = 3000.0  # Volume surge

    df = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr_14": np.full(60, 2.0),
        },
        index=dates,
    )

    features = AdvancedFeatureExtractor.compute_features(df, sr_level=105.0)

    # Check that required columns exist
    for col in ("ema_slope_50", "wick_asymmetry", "dist_sr", "volume_zscore", "is_london_session"):
        assert col in features.columns, f"{col} must be computed in feature vector"

    # Volume Z-score on the surge candle should be positive
    assert features["volume_zscore"].iloc[-1] > 1.5
    # EMA slope on an upward trend should be positive
    assert features["ema_slope_50"].iloc[-1] > 0.0
    # London session active between 08:00 and 16:00 UTC
    assert features["is_london_session"].iloc[0] == 1.0


# -----------------------------------------------------------------------------
# Section 3: Two-Stage Meta-Labeling Model Tests
# -----------------------------------------------------------------------------
def test_meta_label_engine_training_and_execution_gate() -> None:
    """Test training secondary meta-model and evaluating execution gate threshold."""
    meta_engine = TwoStageMetaLabelEngine(gate_threshold=0.72)

    # Synthesize sample dataset
    n_samples = 100
    dates = pd.date_range("2026-01-01", periods=n_samples, freq="15min", tz="UTC")
    features_df = pd.DataFrame(
        {
            "volume_zscore": np.random.normal(0, 1, n_samples),
            "wick_asymmetry": np.random.uniform(-1, 1, n_samples),
            "ema_slope_50": np.random.uniform(-0.5, 0.5, n_samples),
            "dist_sr": np.random.uniform(-2, 2, n_samples),
            "is_london_session": np.ones(n_samples),
            "is_ny_session": np.zeros(n_samples),
            "is_overlap_session": np.zeros(n_samples),
        },
        index=dates,
    )

    primary_preds = pd.Series(np.random.choice([1, -1], n_samples), index=dates)
    actual_labels = pd.Series(np.random.choice([1, -1, 0], n_samples), index=dates)

    # Fit meta model
    acc = meta_engine.fit_meta_model(features_df, primary_preds, actual_labels)
    assert meta_engine.is_fitted is True
    assert 0.0 <= acc <= 1.0

    # Test probability prediction
    row = features_df.iloc[0]
    prob = meta_engine.predict_meta_probability(row, primary_confidence=0.80)
    assert 0.0 <= prob <= 1.0

    # Test execution gate (P >= 0.72)
    auth_pass, reason_pass = meta_engine.evaluate_execution_gate("BUY", 0.75, 0.72)
    assert auth_pass is True
    assert "Disetujui" in reason_pass

    auth_fail, reason_fail = meta_engine.evaluate_execution_gate("BUY", 0.65, 0.72)
    assert auth_fail is False
    assert "Ditolak" in reason_fail


# -----------------------------------------------------------------------------
# Section 5: Macro Risk Filters (DXY Trend Filter)
# -----------------------------------------------------------------------------
def test_dxy_trend_filter_suppresses_crypto_gold_buy() -> None:
    """SPEC Section 5.1: If DXY 1-hour change > +0.15%, suppress all crypto/gold BUY signals."""
    dates = pd.date_range("2026-10-14 12:00:00", periods=5, freq="15min", tz="UTC")
    # DXY rises from 100.0 to 100.25 (+0.25% > +0.15% threshold)
    closes_spike = [100.00, 100.05, 100.10, 100.18, 100.25]
    dxy_spike = pd.DataFrame({"close": closes_spike}, index=dates)

    suppressed, reason = MacroRiskFilter.evaluate_dxy_trend_filter(
        dxy_df=dxy_spike, target_action="BUY", threshold_pct=0.15
    )
    assert suppressed is True
    assert "DXY Spike Filter Aktif" in reason

    # SELL action should NOT be suppressed by DXY spike
    suppressed_sell, _ = MacroRiskFilter.evaluate_dxy_trend_filter(
        dxy_df=dxy_spike, target_action="SELL", threshold_pct=0.15
    )
    assert suppressed_sell is False

    # Modest DXY rise (+0.05% < +0.15%) should not suppress BUY
    closes_calm = [100.00, 100.01, 100.02, 100.03, 100.05]
    dxy_calm = pd.DataFrame({"close": closes_calm}, index=dates)
    suppressed_calm, _ = MacroRiskFilter.evaluate_dxy_trend_filter(
        dxy_df=dxy_calm, target_action="BUY", threshold_pct=0.15
    )
    assert suppressed_calm is False


# -----------------------------------------------------------------------------
# Ensemble Integration & Meta Probability Population
# -----------------------------------------------------------------------------
def test_ensemble_agent_populates_meta_probability() -> None:
    """Verify that EnsembleDecisionAgent populates decision.meta_probability."""
    cfg = Settings(_env_file=None, use_live_market_data=False, use_live_news_feed=False)
    agent = EnsembleDecisionAgent(settings=cfg)

    dates = pd.date_range("2026-10-14", periods=60, freq="15min", tz="UTC")
    close = np.linspace(80000, 81000, 60)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 50,
            "low": close - 50,
            "close": close,
            "volume": np.full(60, 1000.0),
        },
        index=dates,
    )
    decision = agent.analyze_symbol(symbol="BTCUSDT", timeframe="15m", df=df)

    assert hasattr(decision, "meta_probability")
    assert 0.0 <= decision.meta_probability <= 1.0
    assert hasattr(decision, "dxy_filter_triggered")
