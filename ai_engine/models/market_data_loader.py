from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings
from data_pipeline.market_symbols import normalize_symbol
from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor


TIMEFRAME_FREQ_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

TIMEFRAME_SECONDS_MAP: dict[str, float] = {
    "1m": 60.0,
    "5m": 300.0,
    "15m": 900.0,
    "30m": 1800.0,
    "1h": 3600.0,
    "4h": 14400.0,
    "1d": 86400.0,
}


def _align_dataframe_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Preserve original historical timestamps on `df` (R03 / NT-07):
    - Never overwrite original historical timestamps with fake 2026 dates.
    - If `df` has finer resolution than `timeframe`, downsample via proper OHLCV aggregation.
    - If `df` has coarser resolution than `timeframe`, preserve original timestamps and mark `timeframe_mismatch=True`.
    """
    out = df.copy()
    if len(out) == 0 or not isinstance(out.index, pd.DatetimeIndex):
        return out

    target_sec = TIMEFRAME_SECONDS_MAP.get(timeframe, 900.0)
    native_sec = (
        float(out.index.to_series().diff().dt.total_seconds().median())
        if len(out) >= 2
        else target_sec
    )
    if native_sec > target_sec * 1.05:
        # Coarser source cannot be upsampled into a finer timeframe without fabricating bars
        out.attrs["timeframe_mismatch"] = True
        out.attrs["native_interval_seconds"] = native_sec
        return out

    if native_sec < target_sec * 0.95 and all(c in out.columns for c in ("open", "high", "low", "close")):
        freq = TIMEFRAME_FREQ_MAP.get(timeframe, "15min")
        agg_map: dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in out.columns:
            agg_map["volume"] = "sum"
        downsampled = (
            out.resample(freq, closed="left", label="left")
            .agg(agg_map)
            .dropna(subset=["open", "high", "low", "close"])
        )
        if len(downsampled) > 0:
            downsampled["timestamp_iso"] = downsampled.index.strftime("%Y-%m-%dT%H:%M:%SZ")
            downsampled["timeframe"] = timeframe
            downsampled.attrs["timeframe_mismatch"] = False
            return downsampled

    out["timestamp_iso"] = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["timeframe"] = timeframe
    out.attrs["timeframe_mismatch"] = False
    return out


def load_local_or_synthetic_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int,
    settings: Settings,
    preprocessor: OHLCVPreprocessor,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Resolve OHLCV series from local workspace datasets (XAUUSD, 50 Crypto) ONLY when both
    the symbol AND native timeframe resolution are compatible, or generate an offline fixture series.
    """
    clean_sym = normalize_symbol(symbol)
    cache_key = f"{clean_sym}_{timeframe}"
    if cache_key in cache:
        return cache[cache_key].tail(limit).copy()

    target_sec = TIMEFRAME_SECONDS_MAP.get(timeframe, 900.0)

    # 1. Check XAUUSD 15m dataset (strictly for XAU / GOLD symbols; preserves original timestamps via _align_dataframe_timeframe)
    if "XAU" in clean_sym or "GOLD" in clean_sym:
        xau_path = settings.local_xauusd_path
        if xau_path.exists():
            df = preprocessor.load_csv_file(
                xau_path, symbol="XAUUSD", timeframe="15m"
            )
            df = _align_dataframe_timeframe(df, timeframe)
            df.attrs["data_source"] = "local_csv"
            cache[cache_key] = df
            return df.tail(limit).copy()

    # 2. Check 50 Crypto daily dataset directory (strictly matching base_asset; never fallback to BTC for unrelated symbols)
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
        ]
        for path in candidates:
            if path.exists():
                df = preprocessor.load_csv_file(
                    path, symbol=clean_sym, timeframe="1d"
                )
                df = _align_dataframe_timeframe(df, timeframe)
                df.attrs["data_source"] = "local_csv"
                cache[cache_key] = df
                return df.tail(limit).copy()
    # 3. Fallback: check processed Parquet feature store if resolution matches
    parquet_path = preprocessor.feature_store.resolve_path("ohlcv")
    if parquet_path.exists():
        stored_df = preprocessor.feature_store.load("ohlcv", symbol=clean_sym)
        if not stored_df.empty:
            aligned = _align_dataframe_timeframe(stored_df, timeframe)
            if not aligned.attrs.get("timeframe_mismatch", False):
                aligned.attrs["data_source"] = "feature_store"
                cache[cache_key] = aligned
                return aligned.tail(limit).copy()
    # 4. Deterministic symbol- and timeframe-specific synthetic OHLCV series when offline
    freq = TIMEFRAME_FREQ_MAP.get(timeframe, "15min")
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(clean_sym)) % (2**31)
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=max(limit, 250), freq=freq, tz="UTC")
    steps = rng.normal(0.0006, 0.0045, size=len(idx))
    base_price = 100.0 + (seed % 400)
    close = base_price * np.exp(np.cumsum(steps))
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
    cleaned_synth = _align_dataframe_timeframe(cleaned_synth, timeframe)
    cleaned_synth.attrs["data_source"] = "synthetic_offline"
    cache[cache_key] = cleaned_synth
    return cleaned_synth.tail(limit).copy()
