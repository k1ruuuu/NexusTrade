from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import threading

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.preprocessor_microstructure import MicrostructureVolatilityPreprocessor

logger = get_logger("nexustrade.ai_engine.volatility")

VolatilityRegime = Literal["LOW", "NORMAL", "ELEVATED", "EXTREME"]


@dataclass
class VolatilityForecast:
    """Output of the Optiver-inspired HAR-RV Realized Volatility Estimator."""

    symbol: str
    s_vol: float
    current_rv: float
    predicted_rv: float
    annualized_volatility: float
    percentile_rank: float
    regime: VolatilityRegime
    parkinson_vol: float
    spread_volatility: float


class RealizedVolatilityEstimator:
    """
    Optiver-style Heterogeneous Autoregressive Realized Volatility (HAR-RV) regressor.
    Forecasts next-period realized volatility and outputs normalized risk scaler S_vol in [0.0, 1.0].
    """

    FEATURE_COLS: tuple[str, ...] = (
        "rv_window_10",
        "rv_window_20",
        "rv_window_60",
        "parkinson_vol_20",
        "spread_volatility_20",
        "book_depth_imbalance",
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        preprocessor: Optional[MicrostructureVolatilityPreprocessor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = preprocessor or MicrostructureVolatilityPreprocessor(self.settings)
        self.regressor: Ridge = Ridge(alpha=1.0, random_state=42)
        self._is_fitted: bool = False
        self._models_by_key: dict[str, Ridge] = {}
        self._quantiles_by_key: dict[str, np.ndarray] = {}
        self._historical_quantiles: np.ndarray = np.linspace(0.005, 0.08, 101)
        self._fit_lock = threading.RLock()

    @staticmethod
    def infer_bars_per_year(df: pd.DataFrame, default_bars: int = 365) -> int:
        """Infer annualization bars-per-year from DataFrame index interval (NT-14)."""
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 2:
            med_sec = float(df.index.to_series().diff().dt.total_seconds().median())
            if med_sec > 0:
                return max(1, int(round((365.0 * 86400.0) / med_sec)))
        return default_bars

    def fit(self, df: pd.DataFrame, symbol: str = "GLOBAL") -> RealizedVolatilityEstimator:
        """Fit HAR-RV Ridge regressor isolated per symbol key (NT-14)."""
        key = (symbol or str(df["symbol"].iloc[-1]) if "symbol" in df.columns and len(df) > 0 else "GLOBAL").upper()
        processed = self.preprocessor.process_optiver_order_book(
            df, windows=(10, 20, 60), enforce_lag=True
        )
        target = processed["rv_window_10"].shift(-5).dropna()
        aligned = processed.loc[target.index]

        if len(aligned) >= 30:
            x_mat = aligned[list(self.FEATURE_COLS)].fillna(0.0).to_numpy(dtype=np.float64)
            y_vec = target.to_numpy(dtype=np.float64)
            with self._fit_lock:
                reg = Ridge(alpha=1.0, random_state=42)
                reg.fit(x_mat, y_vec)
                self.regressor = reg
                self._models_by_key[key] = reg
                self._is_fitted = True
            rv_values = processed["rv_window_20"].dropna().to_numpy(dtype=np.float64)
            rv_values = rv_values[rv_values > 0]
            if len(rv_values) >= 10:
                q_arr = np.quantile(rv_values, np.linspace(0.0, 1.0, 101))
                self._historical_quantiles = q_arr
                self._quantiles_by_key[key] = q_arr

        return self

    def estimate(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        bars_per_year: Optional[int] = None,
    ) -> VolatilityForecast:
        """
        Predict next-period realized volatility isolated per symbol and map into S_vol in [0.0, 1.0].
        """
        key = symbol.upper()
        if key not in self._models_by_key and len(df) >= 40:
            self.fit(df, symbol=key)
        processed = self.preprocessor.process_optiver_order_book(
            df, windows=(10, 20, 60), enforce_lag=False
        )
        latest = processed.iloc[[-1]]
        x_latest = latest[list(self.FEATURE_COLS)].fillna(0.0).to_numpy(dtype=np.float64)

        current_rv = float(latest["rv_window_20"].iloc[0])
        parkinson_val = float(latest["parkinson_vol_20"].iloc[0])
        spread_vol = float(latest["spread_volatility_20"].iloc[0])

        active_reg = self._models_by_key.get(key) or (self.regressor if self._is_fitted else None)
        if active_reg is not None:
            pred_rv = float(max(1e-6, active_reg.predict(x_latest)[0]))
        else:
            rv_10 = float(latest["rv_window_10"].iloc[0])
            rv_60 = float(latest["rv_window_60"].iloc[0])
            pred_rv = max(1e-6, 0.50 * rv_10 + 0.35 * current_rv + 0.15 * rv_60)
        rv_history = processed["rv_window_20"].dropna().to_numpy(dtype=np.float64)
        rv_history = rv_history[rv_history > 0]
        if len(rv_history) >= 10:
            percentile_rank = float(np.mean(rv_history <= pred_rv))
            q10 = float(np.percentile(rv_history, 10))
            q95 = float(np.percentile(rv_history, 95))
            span = max(q95 - q10, 1e-6)
            # Calibrate S_vol into [0.05, 0.95] under normal-to-elevated distributions
            raw_s_vol = 0.10 + 0.75 * ((pred_rv - q10) / span)
        else:
            percentile_rank = 0.50
            raw_s_vol = 0.35

        s_vol = float(np.clip(raw_s_vol, 0.0, 1.0))

        # Per-bar return std estimate scaled to annualized volatility using actual bar frequency (NT-14)
        eff_bars_per_year = bars_per_year if bars_per_year is not None else self.infer_bars_per_year(df)
        per_bar_vol = current_rv / np.sqrt(20.0)
        annualized_vol = float(per_bar_vol * np.sqrt(eff_bars_per_year))
        regime = self.classify_regime(s_vol, self.settings.max_volatility_threshold)

        return VolatilityForecast(
            symbol=symbol.upper(),
            s_vol=round(s_vol, 4),
            current_rv=round(current_rv, 6),
            predicted_rv=round(pred_rv, 6),
            annualized_volatility=round(annualized_vol, 4),
            percentile_rank=round(percentile_rank, 4),
            regime=regime,
            parkinson_vol=round(parkinson_val, 6),
            spread_volatility=round(spread_vol, 6),
        )

    @staticmethod
    def classify_regime(
        s_vol: float,
        max_threshold: float = 0.85,
    ) -> VolatilityRegime:
        """Classify scalar realized volatility score S_vol in [0.0, 1.0] into discrete regime."""
        if s_vol < 0.25:
            return "LOW"
        if s_vol < 0.60:
            return "NORMAL"
        if s_vol < max_threshold:
            return "ELEVATED"
        return "EXTREME"
