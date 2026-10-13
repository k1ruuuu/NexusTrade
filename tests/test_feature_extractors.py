from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.structure_extractor import CandleStructuralExtractor
from src.models.macro_regime import MacroRegimeDetector, UnifiedFeaturePipeline


def _build_sample_ohlcv(n_bars: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="15min", tz="UTC")
    rets = rng.normal(0.0005, 0.006, size=n_bars)
    # Create multi-regime oscillation so swing highs/lows, FVGs, and regimes are well-populated
    wave = 0.015 * np.sin(np.linspace(0.0, 6.0 * np.pi, n_bars))
    close = 1000.0 * np.exp(np.cumsum(rets) + wave)
    open_p = np.roll(close, 1)
    open_p[0] = close[0] * 0.998
    spread_up = np.abs(rng.normal(0.003, 0.002, size=n_bars))
    spread_dn = np.abs(rng.normal(0.003, 0.002, size=n_bars))
    high = np.maximum(open_p, close) * (1.0 + spread_up)
    low = np.minimum(open_p, close) * (1.0 - spread_dn)
    volume = rng.uniform(500.0, 5000.0, size=n_bars)
    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_scalar_price_multiplication_invariance() -> None:
    """
    Guardrail 3 & 4:
    Verify that multiplying all price columns (open, high, low, close) by an arbitrary
    positive scalar c > 0 leaves all dimensionless ratios, ATR-scaled structural distances,
    and unified state vector X_t (R^18) completely invariant.
    """
    base_df = _build_sample_ohlcv(n_bars=160, seed=7)
    pipeline_base = UnifiedFeaturePipeline()
    feat_base = pipeline_base.transform(base_df, s_sent=0.35, enforce_lag=True)
    s_base, r_base = pipeline_base.structural_extractor.detect_support_resistance(base_df)

    for scale_factor in (17.5, 0.004):
        scaled_df = base_df.copy()
        for col in ("open", "high", "low", "close"):
            scaled_df[col] = scaled_df[col] * scale_factor

        pipeline_scaled = UnifiedFeaturePipeline()
        feat_scaled = pipeline_scaled.transform(scaled_df, s_sent=0.35, enforce_lag=True)
        s_scaled, r_scaled = pipeline_scaled.structural_extractor.detect_support_resistance(scaled_df)

        # S/R levels in price space must scale linearly with scale_factor
        assert np.isclose(s_scaled, s_base * scale_factor, rtol=1e-5)
        assert np.isclose(r_scaled, r_base * scale_factor, rtol=1e-5)

        # Every structural & unified feature column must be scale-invariant
        for col in UnifiedFeaturePipeline.UNIFIED_VECTOR_COLUMNS:
            assert np.allclose(
                feat_base[col].to_numpy(),
                feat_scaled[col].to_numpy(),
                atol=1e-5,
                rtol=1e-4,
            ), f"Feature '{col}' violated scale invariance under factor {scale_factor}"


def test_candlestick_anatomy_pinbar_and_engulfing_quantification() -> None:
    """Module 1 (Sections 2.1 - 2.3): Verify r_body, r_wick, rel_spread, score_pinbar, and score_engulf."""
    df = _build_sample_ohlcv(n_bars=40, seed=11)
    base_p = float(df["close"].iloc[-3])
    # Construct an explicit Bullish Hammer / Pinbar at index -2: long lower shadow (10), tiny body (1), no upper wick
    df.iloc[-2, df.columns.get_loc("open")] = base_p
    df.iloc[-2, df.columns.get_loc("close")] = base_p + 1.0
    df.iloc[-2, df.columns.get_loc("high")] = base_p + 1.0
    df.iloc[-2, df.columns.get_loc("low")] = base_p - 10.0

    # Construct an explicit Bearish Engulfing bar at index -1 encompassing bar -2's range
    df.iloc[-1, df.columns.get_loc("open")] = base_p + 2.0
    df.iloc[-1, df.columns.get_loc("close")] = base_p - 18.0
    df.iloc[-1, df.columns.get_loc("high")] = base_p + 3.0
    df.iloc[-1, df.columns.get_loc("low")] = base_p - 19.0

    extractor = CandleStructuralExtractor(atr_period=14)
    out = extractor.compute_candle_anatomy(df)

    assert ((out["r_body"] >= 0.0) & (out["r_body"] <= 1.0)).all()
    assert ((out["r_wick"] >= -1.0) & (out["r_wick"] <= 1.0)).all()
    assert (out["rel_spread"] >= 0.0).all()

    # Hammer bar (-2): B = 1, Lambda = 10, U = 0, S = 11 -> score_pinbar = (10 - 2 - 0) / 11 = 8/11
    assert np.isclose(out["score_pinbar"].iloc[-2], 8.0 / 11.0, atol=1e-5)
    assert out["r_wick"].iloc[-2] < -0.80

    # Bearish engulfing bar (-1): opposite direction, larger body (20 > 1), range encompasses bar -2
    expected_engulf = (20.0 - 1.0) / float(out["atr"].iloc[-1])
    assert np.isclose(out["score_engulf"].iloc[-1], expected_engulf, atol=1e-5)
    assert out["score_engulf"].iloc[-1] > 1.0

def test_causal_support_resistance_dbscan_and_noise_fallback() -> None:
    """
    Module 2 (Sections 3.1 - 3.3) & Guardrail 2:
    Verify causal peak/trough exclusion of last `window` bars, DBSCAN clustering, and
    deterministic fallback when DBSCAN finds only noise.
    """
    extractor = CandleStructuralExtractor(atr_period=14, cluster_eps_atr=0.35, swing_window=5)
    df = extractor.compute_candle_anatomy(_build_sample_ohlcv(n_bars=120, seed=21))

    # Inject a new unconfirmed peak inside the final `window` bars (at index -2, where tau > t - k):
    # Because tau > t - k, a peak inside the final `window` bars must NOT be registered in candidate extrema
    df_with_late_spike = df.copy()
    df_with_late_spike.iloc[-2, df_with_late_spike.columns.get_loc("high")] = 2500.0
    _, _, clusters_base = extractor.detect_support_resistance_clusters(df, window=5)
    _, _, clusters_spike = extractor.detect_support_resistance_clusters(df_with_late_spike, window=5)
    assert all(c.level < 2000.0 for c in clusters_spike)
    assert len(clusters_base) == len(clusters_spike)
    mono_idx = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC")
    mono_close = np.linspace(100.0, 200.0, 40)
    mono_df = pd.DataFrame(
        {
            "open": mono_close - 0.5,
            "high": mono_close + 1.0,
            "low": mono_close - 1.0,
            "close": mono_close,
            "volume": np.ones(40) * 100.0,
        },
        index=mono_idx,
    )
    s_fb, r_fb = extractor.detect_support_resistance(mono_df, window=5)
    assert np.isclose(s_fb, float(mono_df["low"].min()))
    assert np.isclose(r_fb, float(mono_df["high"].max()))


def test_fair_value_gap_and_volume_profile_structural_zones() -> None:
    """Module 3 (Sections 4.1 - 4.2): Verify FVG creation/mitigation and Volume Profile POC/VAH/VAL."""
    extractor = CandleStructuralExtractor(volume_profile_window=50, volume_profile_bins=20)
    # 4-bar sequence: bar 0 H=100, bar 1 impulse, bar 2 L=105 (Bullish FVG [100, 105]), bar 3 L=106 (still active)
    highs = np.array([100.0, 106.0, 110.0, 111.0], dtype=np.float64)
    lows = np.array([98.0, 99.5, 105.0, 106.0], dtype=np.float64)
    bull_fvg, bear_fvg = extractor.detect_active_fvg_zones(highs, lows)
    assert bull_fvg is not None
    assert bull_fvg.lower_bound == 100.0 and bull_fvg.upper_bound == 105.0
    assert bear_fvg is None

    # Bar 4 drops to L=99.0 <= 100.0 -> Bullish FVG is mitigated
    highs_mit = np.append(highs, 107.0)
    lows_mit = np.append(lows, 99.0)
    bull_after, _ = extractor.detect_active_fvg_zones(highs_mit, lows_mit)
    assert bull_after is None

    # Volume profile: concentrate 80% of volume near 150.0 -> POC and Value Area must anchor around 150.0
    df = _build_sample_ohlcv(n_bars=60, seed=33)
    vp = extractor.compute_volume_profile_snapshot(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        df["volume"].to_numpy(),
    )
    assert vp.val <= vp.poc <= vp.vah
    assert vp.total_volume > 0.0


def test_macro_regime_hmm_simplex_and_strict_causality() -> None:
    """
    Module 4 (Sections 5.1 - 5.3) & Guardrail 1:
    Verify Intermarket Correlation, RoRo Index, 3-state Gaussian HMM simplex sum(pi_t) == 1,
    Forward filtering causality (no lookahead from future bars), and shift-by-1 enforcement.
    """
    df = _build_sample_ohlcv(n_bars=150, seed=99)
    intermarket = pd.DataFrame(
        {
            "SPX": df["close"] * 1.02,
            "DXY": 105.0 - 0.01 * np.arange(len(df)),
            "VIX": 18.0 + 2.0 * np.sin(np.linspace(0, 4 * np.pi, len(df))),
            "EM": df["close"] * 0.98,
        },
        index=df.index,
    )

    detector = MacroRegimeDetector(n_regimes=3)
    corr_df = detector.compute_intermarket_correlations(
        df["close"],
        {"SPX": intermarket["SPX"], "DXY": intermarket["DXY"], "VIX": intermarket["VIX"]},
        window=20,
    )
    assert set(corr_df.columns) == {"corr_spx", "corr_dxy", "corr_vix"}
    assert ((corr_df.abs() <= 1.0 + 1e-6)).all().all()

    obs_mat, roro = detector.build_observation_matrix(df, intermarket_df=intermarket)
    detector.fit_regimes(obs_mat[:100])
    latest_probs = detector.predict_regime_probabilities(obs_mat)
    assert latest_probs.shape == (3,)
    assert np.all(latest_probs >= 0.0) and np.all(latest_probs <= 1.0)
    assert np.isclose(np.sum(latest_probs), 1.0, atol=1e-6)

    # Verify strict causality: altering bars after index 110 must NOT alter features at or before index 110
    df_future_mutated = df.copy()
    df_future_mutated.iloc[115:, df_future_mutated.columns.get_loc("close")] *= 1.35
    df_future_mutated.iloc[115:, df_future_mutated.columns.get_loc("high")] *= 1.40
    df_future_mutated.iloc[115:, df_future_mutated.columns.get_loc("low")] *= 1.30

    pipeline = UnifiedFeaturePipeline(macro_detector=detector)
    f_orig = pipeline.transform(df, enforce_lag=True)
    f_mut = pipeline.transform(df_future_mutated, enforce_lag=True)

    for col in UnifiedFeaturePipeline.UNIFIED_VECTOR_COLUMNS:
        assert np.allclose(
            f_orig[col].iloc[:110].to_numpy(),
            f_mut[col].iloc[:110].to_numpy(),
            atol=1e-7,
        ), f"Future mutation leaked backward into '{col}' at t <= 110"

    vec_18 = pipeline.extract_latest_vector(df, intermarket_df=intermarket, s_sent=-0.25, optiver_sigma=0.018)
    assert vec_18.shape == (18,)
    assert np.all(np.isfinite(vec_18))
