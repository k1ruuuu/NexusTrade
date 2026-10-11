from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.data_pipeline.feature_store")


class FeatureStore:
    """Parquet-backed feature storage with memory downcasting and in-memory LRU cache."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.processed_dir = self.settings.processed_data_dir
        self.cache_dir = self.settings.cache_data_dir
        self._memory_cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 -> float32 and int64 -> int32 per AGENT.md memory guardrails."""
        optimized = df.copy()
        float_cols = optimized.select_dtypes(include=["float64"]).columns
        int_cols = optimized.select_dtypes(include=["int64"]).columns

        for col in float_cols:
            optimized[col] = optimized[col].astype(np.float32)
        for col in int_cols:
            col_min = optimized[col].min()
            col_max = optimized[col].max()
            if (
                pd.notna(col_min)
                and pd.notna(col_max)
                and col_min >= np.iinfo(np.int32).min
                and col_max <= np.iinfo(np.int32).max
            ):
                optimized[col] = optimized[col].astype(np.int32)

        return optimized

    def resolve_path(self, name: str) -> Path:
        filename = name if name.endswith(".parquet") else f"{name}.parquet"
        return self.processed_dir / filename

    def save(
        self,
        df: pd.DataFrame,
        name: str = "ohlcv",
        downcast: bool = True,
    ) -> Path:
        """Persist DataFrame as compressed Apache Parquet and update cache."""
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame to feature store '{name}'")

        prepared = self.downcast_dataframe(df) if downcast else df.copy()
        path = self.resolve_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_parquet(path, engine="pyarrow", compression="snappy", index=True)
        self._memory_cache[name] = prepared

        logger.info(
            "Saved feature dataset to Parquet",
            name=name,
            path=str(path),
            rows=len(prepared),
            columns=len(prepared.columns),
        )
        return path

    def load(
        self,
        name: str = "ohlcv",
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load feature dataset from memory cache or Parquet disk storage."""
        if use_cache and name in self._memory_cache:
            df = self._memory_cache[name]
        else:
            path = self.resolve_path(name)
            if not path.exists():
                raise FileNotFoundError(f"Feature store Parquet file not found: {path}")
            df = pd.read_parquet(path, engine="pyarrow")
            self._memory_cache[name] = df

        filtered = df
        if symbol is not None and "symbol" in filtered.columns:
            filtered = filtered[filtered["symbol"].str.upper() == symbol.upper()]
        if timeframe is not None and "timeframe" in filtered.columns:
            filtered = filtered[filtered["timeframe"] == timeframe]

        return filtered.copy()
