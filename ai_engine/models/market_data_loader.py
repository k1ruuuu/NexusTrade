from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings
from data_pipeline.market_symbols import normalize_symbol
from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor


def load_local_or_synthetic_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int,
    settings: Settings,
    preprocessor: OHLCVPreprocessor,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Resolve OHLCV series from local workspace datasets (XAUUSD, 50 Crypto),
    Parquet feature store, or deterministic synthetic series when offline.
    """
    clean_sym = normalize_symbol(symbol)
    cache_key = f"{clean_sym}_{timeframe}"
    if cache_key in cache:
        return cache[cache_key].tail(limit).copy()

    # 1. Check XAUUSD 15m dataset
    if "XAU" in clean_sym or "GOLD" in clean_sym:
        xau_path = settings.local_xauusd_path
        if xau_path.exists():
            df = preprocessor.load_csv_file(
                xau_path, symbol="XAUUSD", timeframe="15m"
            )
            cache[cache_key] = df
            return df.tail(limit).copy()

    # 2. Check 50 Crypto dataset directory
    crypto_dir = settings.local_crypto_dir
    if crypto_dir.exists():
        base_asset = (
            clean_sym.replace("USDT", "")
            .replace("USDC", "")
            .replace("USD", "")
            .lower()
        )
        candidates = [
            crypto_dir / f"{base_asset}_usdt_day.csv",
            crypto_dir / f"{base_asset}_usd_day.csv",
            crypto_dir / f"{base_asset}_usdc_day.csv",
            crypto_dir / "btc_usdt_day.csv",
            crypto_dir / "btc_usd_day.csv",
        ]
        for path in candidates:
            if path.exists():
                df = preprocessor.load_csv_file(
                    path, symbol=clean_sym, timeframe=timeframe
                )
                cache[cache_key] = df
                return df.tail(limit).copy()

    # 3. Fallback: check processed Parquet feature store
    parquet_path = preprocessor.feature_store.resolve_path("ohlcv")
    if parquet_path.exists():
        stored_df = preprocessor.feature_store.load("ohlcv", symbol=clean_sym)
        if not stored_df.empty:
            cache[cache_key] = stored_df
            return stored_df.tail(limit).copy()

    # 4. Deterministic synthetic OHLCV series if symbol not found in local CSVs
    rng = np.random.default_rng(abs(hash(clean_sym)) % (2**32))
    idx = pd.date_range("2026-01-01", periods=max(limit, 250), freq="15min", tz="UTC")
    steps = rng.normal(0.0006, 0.0045, size=len(idx))
    close = 100.0 * np.exp(np.cumsum(steps))
    open_p = np.roll(close, 1)
    open_p[0] = close[0] * 0.999
    high = np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.0015, 0.001, size=len(idx))))
    low = np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.0015, 0.001, size=len(idx))))
    vol = rng.uniform(1000.0, 8000.0, size=len(idx))
    synth_df = pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    cleaned_synth = preprocessor.clean_ohlcv(
        synth_df, symbol=clean_sym, timeframe=timeframe
    )
    cache[cache_key] = cleaned_synth
    return cleaned_synth.tail(limit).copy()
