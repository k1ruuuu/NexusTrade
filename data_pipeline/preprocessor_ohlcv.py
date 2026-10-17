from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.feature_store import FeatureStore

logger = get_logger("nexustrade.data_pipeline.ohlcv")

SupportedTimeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d"]

PANDAS_RESAMPLE_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


class OHLCVPreprocessor:
    """Multi-timeframe OHLCV cleaning, alignment, and zero-lookahead feature engineering."""

    REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    FEATURE_COLUMNS: tuple[str, ...] = (
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_pct_b",
        "atr_14",
        "ema_20",
        "ema_50",
        "ema_200",
        "sma_50",
        "sma_200",
        "vwap",
        "donchian_upper_20",
        "donchian_lower_20",
        "donchian_mid_20",
        "rolling_return_1",
        "rolling_return_5",
        "rolling_return_20",
        "return_zscore_20",
        "volume_surge_20",
        "hl_range_pct",
        "r_body",
        "r_wick",
        "rel_spread",
        "score_pinbar",
        "score_engulf",
        "ut_trailing_stop",
        "ut_trend",
        "ut_buy",
        "ut_sell",
    )
    def __init__(
        self,
        settings: Optional[Settings] = None,
        feature_store: Optional[FeatureStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.feature_store = feature_store or FeatureStore(self.settings)

    @staticmethod
    def parse_utc_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamp columns or index to UTC DatetimeIndex and ISO-8601 string."""
        cleaned = df.copy()
        cleaned.columns = [str(col).strip().lower() for col in cleaned.columns]

        if isinstance(cleaned.index, pd.DatetimeIndex):
            ts = (
                cleaned.index.tz_localize("UTC")
                if cleaned.index.tz is None
                else cleaned.index.tz_convert("UTC")
            )
        elif "timestamp" in cleaned.columns:
            ts = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
        elif "date" in cleaned.columns and "time" in cleaned.columns:
            combined = cleaned["date"].astype(str).str.strip() + "T" + cleaned["time"].astype(str).str.strip()
            ts = pd.to_datetime(combined, utc=True, errors="coerce")
        elif "time" in cleaned.columns:
            ts = pd.to_datetime(cleaned["time"], utc=True, errors="coerce")
        elif "date" in cleaned.columns:
            ts = pd.to_datetime(cleaned["date"], utc=True, errors="coerce")
        else:
            raise ValueError("DataFrame must contain a DatetimeIndex or 'timestamp'/'time'/'date' column")

        cleaned.index = pd.DatetimeIndex(ts, name="timestamp")
        cleaned = cleaned[~cleaned.index.isna()]
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
        cleaned["timestamp_iso"] = cleaned.index.strftime("%Y-%m-%dT%H:%M:%SZ")
        return cleaned

    def clean_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
    ) -> pd.DataFrame:
        """Sanitize OHLCV columns, enforce high/low bounds, and standardize UTC timestamps."""
        standardized = self.parse_utc_timestamps(df)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in standardized.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        for col in self.REQUIRED_COLUMNS:
            standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

        standardized = standardized.dropna(subset=["open", "high", "low", "close"])
        standardized["volume"] = standardized["volume"].fillna(0.0).clip(lower=0.0)
        standardized = standardized[
            (standardized["open"] > 0)
            & (standardized["high"] > 0)
            & (standardized["low"] > 0)
            & (standardized["close"] > 0)
        ]

        # Enforce structural OHLC consistency
        standardized["high"] = standardized[["open", "high", "low", "close"]].max(axis=1)
        standardized["low"] = standardized[["open", "high", "low", "close"]].min(axis=1)
        standardized["symbol"] = symbol.upper()
        standardized["timeframe"] = timeframe
        return standardized

    def resample_ohlcv(
        self,
        df: pd.DataFrame,
        target_timeframe: SupportedTimeframe,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Resample OHLCV bars to a higher timeframe using right-closed label alignment."""
        if target_timeframe not in PANDAS_RESAMPLE_MAP:
            raise ValueError(
                f"Unsupported timeframe '{target_timeframe}'. Valid: {list(PANDAS_RESAMPLE_MAP)}"
            )

        inferred_symbol = symbol or (
            str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else "BTCUSDT"
        )
        cleaned = self.clean_ohlcv(df, symbol=inferred_symbol, timeframe=target_timeframe)
        rule = PANDAS_RESAMPLE_MAP[target_timeframe]

        agg_spec: dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        if "trade_count" in cleaned.columns:
            cleaned["trade_count"] = pd.to_numeric(cleaned["trade_count"], errors="coerce").fillna(0.0)
            agg_spec["trade_count"] = "sum"

        resampled = (
            cleaned[list(agg_spec.keys())]
            .resample(rule, closed="left", label="left")
            .agg(agg_spec)
            .dropna(subset=["open", "high", "low", "close"])
        )
        resampled["timestamp_iso"] = resampled.index.strftime("%Y-%m-%dT%H:%M:%SZ")
        resampled["symbol"] = inferred_symbol.upper()
        resampled["timeframe"] = target_timeframe
        return resampled

    @staticmethod
    def compute_ut_bot_indicators(
        df: pd.DataFrame,
        key_value: float = 1.0,
        atr_period: int = 10,
        use_heikin_ashi: bool = False,
    ) -> pd.DataFrame:
        """
        Algorithmic implementation of the 'UT Bot Alerts' Pine Script indicator:
          a = key_value (sensitivity multiplier, default=1.0)
          c = atr_period (ATR length, default=10)
          xATR = atr(c)
          nLoss = a * xATR
          src = close (or Heikin-Ashi close)
          xATRTrailingStop tracking
          crossover(src, xATRTrailingStop) -> Buy Alert
          crossunder(src, xATRTrailingStop) -> Sell Alert
        """
        out = df.copy()
        high = pd.to_numeric(out["high"], errors="coerce").astype(np.float64)
        low = pd.to_numeric(out["low"], errors="coerce").astype(np.float64)
        close = pd.to_numeric(out["close"], errors="coerce").astype(np.float64)
        open_p = pd.to_numeric(out["open"], errors="coerce").astype(np.float64)

        src = (open_p + high + low + close) / 4.0 if use_heikin_ashi else close

        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        x_atr = tr.ewm(alpha=1.0 / max(atr_period, 1), adjust=False).mean()
        n_loss = float(key_value) * x_atr

        n = len(src)
        src_vals = src.to_numpy()
        n_loss_vals = n_loss.to_numpy()

        trailing_stop = np.zeros(n, dtype=np.float64)
        pos = np.zeros(n, dtype=np.int32)

        if n > 0:
            trailing_stop[0] = src_vals[0] - n_loss_vals[0]
            pos[0] = 1

        for t in range(1, n):
            prev_ts = trailing_stop[t - 1]
            curr_src = src_vals[t]
            prev_src = src_vals[t - 1]
            loss = n_loss_vals[t]

            if curr_src > prev_ts and prev_src > prev_ts:
                ts = max(prev_ts, curr_src - loss)
            elif curr_src < prev_ts and prev_src < prev_ts:
                ts = min(prev_ts, curr_src + loss)
            elif curr_src > prev_ts:
                ts = curr_src - loss
            else:
                ts = curr_src + loss
            trailing_stop[t] = ts

            if prev_src < prev_ts and curr_src > prev_ts:
                curr_pos = 1
            elif prev_src > prev_ts and curr_src < prev_ts:
                curr_pos = -1
            else:
                curr_pos = pos[t - 1]
            pos[t] = curr_pos

        out["ut_trailing_stop"] = pd.Series(trailing_stop, index=out.index)
        out["ut_pos"] = pd.Series(pos, index=out.index)

        prev_src_s = src.shift(1).fillna(src)
        prev_ts_s = out["ut_trailing_stop"].shift(1).fillna(out["ut_trailing_stop"])

        crossover = (prev_src_s <= prev_ts_s) & (src > out["ut_trailing_stop"])
        crossunder = (prev_src_s >= prev_ts_s) & (src < out["ut_trailing_stop"])

        out["ut_buy"] = (src > out["ut_trailing_stop"]) & crossover
        out["ut_sell"] = (src < out["ut_trailing_stop"]) & crossunder
        out["ut_trend"] = np.where(src > out["ut_trailing_stop"], 1, -1)
        return out

    @staticmethod
    def compute_raw_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators on OHLCV series prior to lookahead shifting."""
        out = df.copy()
        close = out["close"].astype(np.float64)
        high = out["high"].astype(np.float64)
        low = out["low"].astype(np.float64)
        volume = out["volume"].astype(np.float64)

        # Wilder's RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
        rsi = np.where((avg_loss == 0.0) & (avg_gain == 0.0), 50.0, rsi)
        out["rsi_14"] = pd.Series(rsi, index=out.index).fillna(50.0)

        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        out["macd"] = ema_12 - ema_26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        # Bollinger Bands (20, 2.0)
        bb_mid = close.rolling(window=20, min_periods=1).mean()
        bb_std = close.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
        out["bb_middle"] = bb_mid
        out["bb_upper"] = out["bb_middle"] + 2.0 * bb_std
        out["bb_lower"] = out["bb_middle"] - 2.0 * bb_std
        band_range = (out["bb_upper"] - out["bb_lower"]).replace(0.0, np.nan)
        out["bb_width"] = (band_range / out["bb_middle"].replace(0.0, np.nan)).fillna(0.0)
        out["bb_pct_b"] = ((close - out["bb_lower"]) / band_range).fillna(0.5)

        # Average True Range ATR(14)
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        out["atr_14"] = true_range.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()

        # Exponential & Simple Moving Averages
        out["ema_20"] = close.ewm(span=20, adjust=False).mean()
        out["ema_50"] = close.ewm(span=50, adjust=False).mean()
        out["ema_200"] = close.ewm(span=200, adjust=False).mean()
        out["sma_50"] = close.rolling(window=50, min_periods=1).mean()
        out["sma_200"] = close.rolling(window=200, min_periods=1).mean()

        # Rolling VWAP (20-period rolling typical price weighted by volume)
        typical_price = (high + low + close) / 3.0
        pv_rolling = (typical_price * volume).rolling(window=20, min_periods=1).sum()
        vol_rolling = volume.rolling(window=20, min_periods=1).sum().replace(0.0, np.nan)
        out["vwap"] = (pv_rolling / vol_rolling).fillna(typical_price)

        # Donchian Channels (20)
        out["donchian_upper_20"] = high.rolling(window=20, min_periods=1).max()
        out["donchian_lower_20"] = low.rolling(window=20, min_periods=1).min()
        out["donchian_mid_20"] = 0.5 * (out["donchian_upper_20"] + out["donchian_lower_20"])

        # Rolling returns, Return Z-score, Volume surge, High-Low range
        ret_1 = close.pct_change(1).fillna(0.0)
        out["rolling_return_1"] = ret_1
        out["rolling_return_5"] = close.pct_change(5).fillna(0.0)
        out["rolling_return_20"] = close.pct_change(20).fillna(0.0)

        ret_mean_20 = ret_1.rolling(window=20, min_periods=2).mean()
        ret_std_20 = ret_1.rolling(window=20, min_periods=2).std(ddof=0).replace(0.0, np.nan)
        out["return_zscore_20"] = ((ret_1 - ret_mean_20) / ret_std_20).fillna(0.0)

        vol_mean_20 = volume.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan)
        out["volume_surge_20"] = (volume / vol_mean_20).fillna(1.0)
        out["hl_range_pct"] = ((high - low) / close.replace(0.0, np.nan)).fillna(0.0)

        # Module 1: Candlestick Anatomy, Dimensionless Ratios & Pattern Quantification
        open_p = pd.to_numeric(out["open"], errors="coerce").astype(np.float64)
        spread = (high - low).clip(lower=0.0)
        body = (close - open_p).abs()
        upper_wick = (high - np.maximum(open_p, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_p, close) - low).clip(lower=0.0)
        direction = np.sign(close - open_p)
        eps = np.maximum(close.abs() * 1e-12, 1e-18)
        denom = spread + eps
        atr_safe = np.maximum(out["atr_14"].astype(np.float64), np.maximum(close.abs() * 1e-8, 1e-12))

        out["r_body"] = np.where(spread > eps, (body / denom).clip(0.0, 1.0), 0.0)
        out["r_wick"] = np.where(spread > eps, ((upper_wick - lower_wick) / denom).clip(-1.0, 1.0), 0.0)
        out["rel_spread"] = (spread / atr_safe).fillna(0.0)
        raw_pinbar = np.where(spread > eps, (lower_wick - 2.0 * body - upper_wick) / denom, 0.0)
        out["score_pinbar"] = np.clip(raw_pinbar, 0.0, 1.0)

        prev_body = body.shift(1).fillna(0.0)
        prev_dir = pd.Series(direction, index=out.index).shift(1).fillna(0.0)
        prev_high = high.shift(1).fillna(high)
        prev_low = low.shift(1).fillna(low)
        engulf_cond = (
            (direction != 0.0)
            & (prev_dir != 0.0)
            & (direction != prev_dir)
            & (body > prev_body)
            & (high >= prev_high)
            & (low <= prev_low)
        )
        out["score_engulf"] = np.where(engulf_cond, ((body - prev_body) / atr_safe).fillna(0.0), 0.0)
        # UT Bot Alerts Indicators
        ut_df = OHLCVPreprocessor.compute_ut_bot_indicators(out, key_value=1.0, atr_period=10)
        out["ut_trailing_stop"] = ut_df["ut_trailing_stop"]
        out["ut_trend"] = ut_df["ut_trend"].astype(np.float32)
        out["ut_buy"] = ut_df["ut_buy"].astype(np.float32)
        out["ut_sell"] = ut_df["ut_sell"].astype(np.float32)

        return out

    def engineer_features(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        enforce_lag: bool = True,
        drop_initial_na: bool = True,
    ) -> pd.DataFrame:
        """
        Clean OHLCV and compute all technical features.
        When enforce_lag=True (default), all feature columns are shifted by 1 period (t-1)
        so bar t's features strictly reflect information known prior to bar t (zero lookahead bias).
        """
        cleaned = self.clean_ohlcv(df, symbol=symbol, timeframe=timeframe)
        enriched = self.compute_raw_indicators(cleaned)

        if enforce_lag:
            for col in self.FEATURE_COLUMNS:
                enriched[col] = enriched[col].shift(1)
            if drop_initial_na and len(enriched) > 1:
                enriched = enriched.iloc[1:].copy()

        return self.feature_store.downcast_dataframe(enriched)

    def load_csv_file(
        self,
        csv_path: Path,
        symbol: Optional[str] = None,
        timeframe: str = "15m",
    ) -> pd.DataFrame:
        """Load local OHLCV CSV (handling comment lines such as XAUUSD_15m.csv) and clean."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        inferred_symbol = symbol or csv_path.stem.split("_")[0].upper()
        raw_df = pd.read_csv(csv_path, comment="#")
        return self.clean_ohlcv(raw_df, symbol=inferred_symbol, timeframe=timeframe)

    def process_and_export(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        output_name: str = "ohlcv",
        enforce_lag: bool = True,
    ) -> tuple[pd.DataFrame, Path]:
        """Run full OHLCV preprocessing pipeline and save to data/processed/ohlcv.parquet."""
        processed = self.engineer_features(
            df,
            symbol=symbol,
            timeframe=timeframe,
            enforce_lag=enforce_lag,
        )
        parquet_path = self.feature_store.save(processed, name=output_name, downcast=True)
        return processed, parquet_path
