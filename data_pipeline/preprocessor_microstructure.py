from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.feature_store import FeatureStore

logger = get_logger("nexustrade.data_pipeline.microstructure")


class MicrostructureVolatilityPreprocessor:
    """
    Optiver Realized Volatility & Jane Street Microstructure feature engineering.
    Computes WAP, log-return realized volatility, book depth imbalance, bid-ask spread
    volatility, Parkinson range volatility, and correlation-filtered Jane Street features.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        feature_store: Optional[FeatureStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.feature_store = feature_store or FeatureStore(self.settings)

    @staticmethod
    def compute_wap(
        bid_price1: pd.Series,
        ask_price1: pd.Series,
        bid_size1: pd.Series,
        ask_size1: pd.Series,
    ) -> pd.Series:
        """Compute Optiver Weighted Average Price (WAP) from Level-1 order book."""
        denom = (bid_size1 + ask_size1).replace(0.0, np.nan)
        wap = (bid_price1 * ask_size1 + ask_price1 * bid_size1) / denom
        mid = 0.5 * (bid_price1 + ask_price1)
        return wap.fillna(mid)

    @staticmethod
    def compute_log_returns(price_series: pd.Series) -> pd.Series:
        """Compute continuously compounded log returns ln(P_t / P_{t-1})."""
        safe_prices = pd.to_numeric(price_series, errors="coerce").replace(0.0, np.nan)
        log_ret = np.log(safe_prices / safe_prices.shift(1))
        return pd.Series(log_ret, index=price_series.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def compute_realized_volatility(
        log_returns: pd.Series,
        window: int = 20,
        enforce_lag: bool = True,
    ) -> pd.Series:
        """
        Compute Optiver-style realized volatility: sqrt(sum_{i=1..window} r_i^2).
        When enforce_lag=True, shifts by 1 bar to prevent lookahead bias.
        """
        squared_returns = np.square(log_returns.astype(np.float64))
        rv = np.sqrt(squared_returns.rolling(window=window, min_periods=2).sum())
        if enforce_lag:
            rv = rv.shift(1)
        return rv.fillna(0.0)

    @staticmethod
    def compute_parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
        enforce_lag: bool = True,
    ) -> pd.Series:
        """Compute Parkinson high-low range volatility estimator."""
        safe_low = low.replace(0.0, np.nan)
        hl_log_sq = np.square(np.log(high / safe_low))
        scale = 1.0 / (4.0 * window * np.log(2.0))
        parkinson = np.sqrt(scale * hl_log_sq.rolling(window=window, min_periods=2).sum())
        if enforce_lag:
            parkinson = parkinson.shift(1)
        return parkinson.fillna(0.0)

    def process_optiver_order_book(
        self,
        book_df: pd.DataFrame,
        windows: tuple[int, ...] = (10, 20, 60),
        enforce_lag: bool = True,
    ) -> pd.DataFrame:
        """
        Process Optiver order book snapshots or OHLCV proxy data into realized
        volatility rolling window benchmarks and microstructure metrics.
        """
        out = book_df.copy()
        cols = {c.lower(): c for c in out.columns}

        has_l1_book = all(
            k in cols for k in ("bid_price1", "ask_price1", "bid_size1", "ask_size1")
        )
        if has_l1_book:
            bp1 = pd.to_numeric(out[cols["bid_price1"]], errors="coerce")
            ap1 = pd.to_numeric(out[cols["ask_price1"]], errors="coerce")
            bs1 = pd.to_numeric(out[cols["bid_size1"]], errors="coerce").clip(lower=0.0)
            as1 = pd.to_numeric(out[cols["ask_size1"]], errors="coerce").clip(lower=0.0)

            out["wap"] = self.compute_wap(bp1, ap1, bs1, as1)
            total_depth = (bs1 + as1).replace(0.0, np.nan)
            imbalance = ((bs1 - as1) / total_depth).fillna(0.0)
            spread_pct = ((ap1 - bp1) / out["wap"].replace(0.0, np.nan)).fillna(0.0)
            price_ref = out["wap"]
        elif "close" in cols:
            close = pd.to_numeric(out[cols["close"]], errors="coerce")
            high = pd.to_numeric(out.get(cols.get("high", "close"), close), errors="coerce")
            low = pd.to_numeric(out.get(cols.get("low", "close"), close), errors="coerce")
            open_p = pd.to_numeric(out.get(cols.get("open", "close"), close), errors="coerce")

            out["wap"] = (high + low + 2.0 * close) / 4.0
            hl_span = (high - low).replace(0.0, np.nan)
            imbalance = ((close - open_p) / hl_span).clip(-1.0, 1.0).fillna(0.0)
            spread_pct = ((high - low) / out["wap"].replace(0.0, np.nan)).fillna(0.0)
            price_ref = out["wap"]
        else:
            raise ValueError(
                "Input DataFrame must contain Level-1 order book columns or OHLCV 'close' column"
            )

        log_ret = self.compute_log_returns(price_ref)
        out["log_return"] = log_ret.shift(1).fillna(0.0) if enforce_lag else log_ret
        out["book_depth_imbalance"] = imbalance.shift(1).fillna(0.0) if enforce_lag else imbalance
        out["bid_ask_spread_pct"] = spread_pct.shift(1).fillna(0.0) if enforce_lag else spread_pct

        spread_vol = spread_pct.rolling(window=20, min_periods=2).std(ddof=0).fillna(0.0)
        out["spread_volatility_20"] = spread_vol.shift(1).fillna(0.0) if enforce_lag else spread_vol

        for w in windows:
            out[f"rv_window_{w}"] = self.compute_realized_volatility(
                log_ret, window=w, enforce_lag=enforce_lag
            )

        if "high" in cols and "low" in cols:
            out["parkinson_vol_20"] = self.compute_parkinson_volatility(
                pd.to_numeric(out[cols["high"]], errors="coerce"),
                pd.to_numeric(out[cols["low"]], errors="coerce"),
                window=20,
                enforce_lag=enforce_lag,
            )
        else:
            out["parkinson_vol_20"] = out[f"rv_window_{windows[min(1, len(windows) - 1)]}"]

        primary_rv = out[f"rv_window_{windows[min(1, len(windows) - 1)]}"]
        rolling_min = primary_rv.rolling(window=100, min_periods=5).min()
        rolling_max = primary_rv.rolling(window=100, min_periods=5).max()
        denom = (rolling_max - rolling_min).replace(0.0, np.nan)
        out["rv_regime_score"] = (
            ((primary_rv - rolling_min) / denom).clip(0.0, 1.0).fillna(0.5)
        )

        return self.feature_store.downcast_dataframe(out)

    @staticmethod
    def filter_jane_street_correlations(
        df: pd.DataFrame,
        correlation_threshold: float = 0.92,
        feature_prefix: str = "feature_",
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Extract feature correlation filters for Jane Street anonymized microstructure features.
        Removes redundant features whose absolute Pearson correlation exceeds threshold.
        """
        feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
        if len(feature_cols) <= 1:
            return df.copy(), []

        numeric_features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        corr_matrix = numeric_features.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        dropped_features = [
            col
            for col in upper_triangle.columns
            if any(upper_triangle[col] > correlation_threshold)
        ]
        retained_df = df.drop(columns=dropped_features).copy()

        retained_feature_cols = [c for c in feature_cols if c not in dropped_features]
        if len(retained_feature_cols) >= 2:
            f0 = pd.to_numeric(retained_df[retained_feature_cols[0]], errors="coerce").fillna(0.0)
            f1 = pd.to_numeric(retained_df[retained_feature_cols[1]], errors="coerce").fillna(0.0)
            retained_df["js_cross_interaction"] = (f0 * f1).astype(np.float32)

        if "weight" in retained_df.columns and "resp" in retained_df.columns:
            weight = pd.to_numeric(retained_df["weight"], errors="coerce").fillna(0.0)
            resp = pd.to_numeric(retained_df["resp"], errors="coerce").fillna(0.0)
            retained_df["utility_score"] = (weight * resp).astype(np.float32)
            retained_df["action_label"] = ((weight * resp) > 0.0).astype(np.int32)

        return retained_df, dropped_features
