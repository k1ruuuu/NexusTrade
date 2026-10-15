from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from config.logging_config import get_logger

logger = get_logger("nexustrade.ai_engine.meta_label")


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for Triple-Barrier path-dependent labeling."""

    k_tp: float = 2.0
    k_sl: float = 1.5
    horizon_bars: int = 30


class TripleBarrierLabeler:
    """
    Implements Marcos Lopez de Prado's Triple-Barrier Labeling Method.
    Evaluates price path dependency over subsequent bars:
      - Label +1: Touched Upper Barrier (Take Profit) first
      - Label -1: Touched Lower Barrier (Stop Loss) first
      - Label  0: Neither barrier touched before time expiry (Vertical Barrier)
    """

    @staticmethod
    def apply_triple_barrier(
        df: pd.DataFrame,
        config: TripleBarrierConfig = TripleBarrierConfig(),
    ) -> pd.Series:
        """
        Assign path-dependent labels {-1, 0, +1} to each bar based on subsequent price action.
        Uses ATR(14) or dynamic volatility to scale barriers.
        """
        labels = pd.Series(0, index=df.index, dtype=int)
        closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
        highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=np.float64)
        lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=np.float64)

        if "atr_14" in df.columns:
            atrs = pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=np.float64)
        elif "atr" in df.columns:
            atrs = pd.to_numeric(df["atr"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            # Fallback ATR estimate: rolling 14-period true range
            prev_closes = np.roll(closes, 1)
            prev_closes[0] = closes[0]
            tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
            atrs = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()

        n = len(df)
        horizon = config.horizon_bars

        for i in range(max(0, n - horizon)):
            entry_price = closes[i]
            atr = atrs[i]
            if np.isnan(atr) or atr <= 0 or entry_price <= 0:
                labels.iloc[i] = 0
                continue

            upper_barrier = entry_price + (config.k_tp * atr)
            lower_barrier = entry_price - (config.k_sl * atr)

            end_idx = min(i + horizon + 1, n)
            path_highs = highs[i + 1 : end_idx]
            path_lows = lows[i + 1 : end_idx]

            hit_upper = np.where(path_highs >= upper_barrier)[0]
            hit_lower = np.where(path_lows <= lower_barrier)[0]

            first_upper = hit_upper[0] if len(hit_upper) > 0 else 999999
            first_lower = hit_lower[0] if len(hit_lower) > 0 else 999999

            if first_upper < first_lower:
                labels.iloc[i] = 1
            elif first_lower < first_upper:
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0

        return labels


class AdvancedFeatureExtractor:
    """
    Computes multi-dimensional features strictly preventing lookahead bias:
      1. Trend & Regime: EMA Slope 50 Ratio
      2. Candle Anatomy: Wick Asymmetry Index (WAI)
      3. Structural Location: Normalized S/R Distance
      4. Volatility & Volume: Volume Z-Score
    """

    @staticmethod
    def compute_features(
        df: pd.DataFrame,
        sr_level: Optional[float] = None,
        epsilon: float = 1e-6,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        closes = pd.to_numeric(df["close"], errors="coerce")
        opens = pd.to_numeric(df["open"], errors="coerce")
        highs = pd.to_numeric(df["high"], errors="coerce")
        lows = pd.to_numeric(df["low"], errors="coerce")
        volumes = (
            pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
            if "volume" in df.columns
            else pd.Series(1.0, index=df.index)
        )

        # ATR calculation
        if "atr_14" in df.columns:
            atr = pd.to_numeric(df["atr_14"], errors="coerce")
        else:
            prev_close = closes.shift(1).fillna(closes)
            tr = pd.concat([
                (highs - lows).abs(),
                (highs - prev_close).abs(),
                (lows - prev_close).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=1).mean()
        atr = atr.replace(0.0, np.nan).fillna(closes * 0.012)

        # 1. Trend & Regime: EMA Slope 50 Ratio
        # Slope_50(t) = (EMA_50(t) - EMA_50(t-5)) / (5 * ATR(t))
        ema_50 = closes.ewm(span=50, adjust=False).mean()
        ema_50_lag5 = ema_50.shift(5)
        out["ema_slope_50"] = ((ema_50 - ema_50_lag5) / (5.0 * atr)).fillna(0.0)

        # 2. Candle Anatomy: Wick Asymmetry Index (WAI)
        # UpperWick = High - max(Open, Close)
        # LowerWick = min(Open, Close) - Low
        # WAI(t) = (UpperWick - LowerWick) / (High - Low + epsilon)
        body_top = np.maximum(opens, closes)
        body_bottom = np.minimum(opens, closes)
        upper_wick = highs - body_top
        lower_wick = body_bottom - lows
        hl_span = (highs - lows) + epsilon
        out["wick_asymmetry"] = ((upper_wick - lower_wick) / hl_span).fillna(0.0)

        # 3. Structural Location: Normalized S/R Distance
        # Dist_SR(t) = (Close(t) - mu_SR) / ATR(t)
        if sr_level is not None and sr_level > 0:
            out["dist_sr"] = ((closes - sr_level) / atr).fillna(0.0)
        else:
            rolling_sr = closes.rolling(50, min_periods=1).mean()
            out["dist_sr"] = ((closes - rolling_sr) / atr).fillna(0.0)

        # 4. Volatility & Volume: Volume Z-Score
        # VolZ(t) = (Volume(t) - mu_Vol(20)) / sigma_Vol(20)
        vol_mean_20 = volumes.rolling(20, min_periods=1).mean()
        vol_std_20 = volumes.rolling(20, min_periods=2).std(ddof=0).replace(0.0, np.nan)
        out["volume_zscore"] = ((volumes - vol_mean_20) / vol_std_20).fillna(0.0).clip(-3.0, 5.0)

        # 5. Session indicators (London / NY overlap)
        if isinstance(df.index, pd.DatetimeIndex):
            hours = df.index.hour
            out["is_london_session"] = ((hours >= 8) & (hours < 16)).astype(float)
            out["is_ny_session"] = ((hours >= 13) & (hours < 21)).astype(float)
            out["is_overlap_session"] = ((hours >= 13) & (hours < 16)).astype(float)
        else:
            out["is_london_session"] = 0.0
            out["is_ny_session"] = 0.0
            out["is_overlap_session"] = 0.0

        return out


class TwoStageMetaLabelEngine:
    """
    Two-Stage Meta-Labeling Architecture:
      - Stage 1: Primary Directional Model predicting trade direction (-1 for SELL, +1 for BUY).
      - Stage 2: Secondary Meta-Model predicting probability of trade profitability P(m_t = 1 | X_2),
                 filtering out false positives.
      - Execution Gate: Requires primary_action != 0 AND P(m_t = 1 | X_2) >= 0.72.
    """

    META_FEATURES = (
        "volume_zscore",
        "wick_asymmetry",
        "ema_slope_50",
        "dist_sr",
        "is_london_session",
        "is_ny_session",
        "is_overlap_session",
    )

    def __init__(self, gate_threshold: float = 0.72) -> None:
        self.gate_threshold = gate_threshold
        self._meta_model: Optional[Any] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit_meta_model(
        self,
        features_df: pd.DataFrame,
        primary_predictions: pd.Series,
        actual_labels: pd.Series,
    ) -> float:
        """
        Train secondary meta-model on historical outcomes.
        Meta-target m_t = 1 if primary_predictions == actual_labels and actual_labels != 0 else 0.
        """
        meta_target = (
            (primary_predictions == actual_labels) & (actual_labels != 0)
        ).astype(int)

        valid_cols = [c for c in self.META_FEATURES if c in features_df.columns]
        if not valid_cols or len(features_df) < 30:
            return 0.50

        X = features_df[valid_cols].fillna(0.0)
        y = meta_target.to_numpy()

        if len(np.unique(y)) < 2:
            self._is_fitted = False
            return 0.50

        try:
            model = lgb.LGBMClassifier(
                n_estimators=60,
                learning_rate=0.05,
                max_depth=3,
                num_leaves=7,
                subsample=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(X, y)
            self._meta_model = model
            self._is_fitted = True
            train_preds = model.predict_proba(X)[:, 1]
            acc = float(np.mean((train_preds >= 0.5) == y))
            return acc
        except Exception as exc:
            logger.warning("Failed to fit LightGBM meta-model; fallback to LogisticRegression", error=str(exc))
            try:
                fallback = LogisticRegression(random_state=42, max_iter=200)
                fallback.fit(X, y)
                self._meta_model = fallback
                self._is_fitted = True
                return float(fallback.score(X, y))
            except Exception:
                self._is_fitted = False
                return 0.50

    def predict_meta_probability(
        self,
        features_row: Union[pd.Series, dict[str, float]],
        primary_confidence: float = 0.50,
    ) -> float:
        """
        Predict probability P(m_t = 1 | X_2) that the current trade setup will be profitable.
        """
        if not self._is_fitted or self._meta_model is None:
            # Cold-start fallback: calibrate using primary confidence and feature heuristics
            vol_z = float(features_row.get("volume_zscore", 0.0))
            wick_a = float(features_row.get("wick_asymmetry", 0.0))
            heuristic_boost = 0.05 * np.clip(vol_z, -1.0, 2.0) - 0.05 * abs(wick_a)
            return float(np.clip(primary_confidence + heuristic_boost, 0.0, 1.0))

        valid_cols = [c for c in self.META_FEATURES]
        if isinstance(features_row, pd.Series):
            vals = [float(features_row.get(c, 0.0)) for c in valid_cols]
        else:
            vals = [float(features_row.get(c, 0.0)) for c in valid_cols]

        X_input = pd.DataFrame([vals], columns=valid_cols)
        try:
            prob = float(self._meta_model.predict_proba(X_input)[0, 1])
            # Blend model probability (75%) with primary model confidence (25%)
            blended = 0.75 * prob + 0.25 * primary_confidence
            return float(np.clip(blended, 0.0, 1.0))
        except Exception:
            return float(primary_confidence)

    def evaluate_execution_gate(
        self,
        primary_action: str,
        meta_probability: float,
        custom_threshold: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Execution Gate (PRD Section 3.3):
        Authorizes execution only if primary_action in ('BUY', 'SELL') and meta_probability >= threshold (0.72).
        """
        threshold = custom_threshold if custom_threshold is not None else self.gate_threshold
        if primary_action not in ("BUY", "SELL"):
            return False, "Sinyal Arah Netral (HOLD)"

        if meta_probability >= threshold:
            return True, f"Eksekusi Disetujui (Probabilitas Sukses Meta-Model {meta_probability * 100:.1f}% >= {threshold * 100:.0f}%)"
        else:
            return False, f"Eksekusi Ditolak (Probabilitas Sukses Meta-Model {meta_probability * 100:.1f}% < {threshold * 100:.0f}%)"


class MacroRiskFilter:
    """
    Macro Intermarket & Event Risk Filters (Section 5):
      1. DXY Trend Filter: If 1-hour percentage change of Dollar Index (DXY) > +0.15%,
         suppress all crypto and gold BUY signals.
      2. High-Impact Macro Lockout: ForexFactory calendar integration.
    """

    @staticmethod
    def evaluate_dxy_trend_filter(
        dxy_df: Optional[pd.DataFrame],
        target_action: str,
        threshold_pct: float = 0.15,
    ) -> Tuple[bool, str]:
        """
        Evaluates DXY 1-hour change. If DXY spikes > +0.15%, suppresses BUY orders on Gold and Crypto.
        Returns: (suppress_trade: bool, reason: str)
        """
        if dxy_df is None or len(dxy_df) < 2:
            return False, "DXY data unavailable; filter inactive"

        try:
            closes = pd.to_numeric(dxy_df["close"], errors="coerce").dropna()
            if len(closes) < 2:
                return False, "Insufficient DXY bars"

            # Compute 1-hour return (or latest 4x 15m bars if 15m data)
            lookback = 4 if len(closes) >= 4 else 1
            dxy_chg = (float(closes.iloc[-1]) / float(closes.iloc[-1 - lookback])) - 1.0
            dxy_chg_pct = dxy_chg * 100.0

            if target_action == "BUY" and dxy_chg_pct > threshold_pct:
                return True, f"DXY Spike Filter Aktif: Dolar AS naik +{dxy_chg_pct:.2f}% (Batas +{threshold_pct:.2f}%) menolak entri BUY"
            return False, f"DXY Stabil ({dxy_chg_pct:+.2f}%)"
        except Exception as exc:
            return False, f"DXY filter error: {exc}"
