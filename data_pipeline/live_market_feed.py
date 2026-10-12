from __future__ import annotations

import time
from typing import Any, Optional

import httpx
import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import (
    BINANCE_INTERVAL_MAP,
    FOREX_CURRENCIES,
    KNOWN_CRYPTO_BASES,
    YAHOO_INTERVAL_RANGE_MAP,
    YAHOO_SYMBOL_MAP,
    is_crypto_symbol,
    normalize_symbol,
    resolve_yahoo_ticker,
)
from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

logger = get_logger("nexustrade.data_pipeline.live_feed")

__all__ = [
    "BINANCE_INTERVAL_MAP",
    "FOREX_CURRENCIES",
    "KNOWN_CRYPTO_BASES",
    "LiveMarketDataFeed",
    "YAHOO_INTERVAL_RANGE_MAP",
    "YAHOO_SYMBOL_MAP",
    "is_crypto_symbol",
    "normalize_symbol",
    "resolve_yahoo_ticker",
]


class LiveMarketDataFeed:
    """
    Real-time OHLCV and live spot price fetcher using Binance Public Market API
    (Crypto) and Yahoo Finance v8 Chart API (Gold/XAUUSD, Forex, Equities, Indices).
    """

    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        preprocessor: Optional[OHLCVPreprocessor] = None,
        ttl_seconds: int = 15,
    ) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = preprocessor or OHLCVPreprocessor(self.settings)
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    resolve_yahoo_ticker = staticmethod(resolve_yahoo_ticker)
    is_crypto_symbol = staticmethod(is_crypto_symbol)

    def fetch_binance_klines(
        self,
        binance_pair: str,
        canonical_symbol: str,
        timeframe: str = "15m",
        limit: int = 400,
    ) -> Optional[pd.DataFrame]:
        interval = BINANCE_INTERVAL_MAP.get(timeframe, "15m")
        clamped_limit = min(max(limit, 50), 1000)

        try:
            with httpx.Client(
                timeout=6.0,
                headers={"User-Agent": "NexusTradeAI/1.0"},
            ) as client:
                resp = client.get(
                    self.BINANCE_KLINES_URL,
                    params={
                        "symbol": binance_pair,
                        "interval": interval,
                        "limit": clamped_limit,
                    },
                )
                if resp.status_code != 200:
                    return None
                rows = resp.json()
                if not isinstance(rows, list) or len(rows) < 20:
                    return None

            records = [
                {
                    "timestamp": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                    "trade_count": float(r[8]),
                }
                for r in rows
            ]
            raw_df = pd.DataFrame(records)
            cleaned = self.preprocessor.clean_ohlcv(
                raw_df, symbol=canonical_symbol, timeframe=timeframe
            )
            logger.info(
                "Fetched real-time Binance OHLCV feed",
                symbol=canonical_symbol,
                binance_pair=binance_pair,
                timeframe=timeframe,
                latest_price=float(cleaned["close"].iloc[-1]),
                bars=len(cleaned),
            )
            return cleaned
        except Exception as exc:
            logger.debug(
                "Binance live feed unavailable; falling back",
                pair=binance_pair,
                error=str(exc),
            )
            return None

    def fetch_yahoo_chart(
        self,
        yahoo_ticker: str,
        canonical_symbol: str,
        timeframe: str = "15m",
        limit: int = 400,
    ) -> Optional[pd.DataFrame]:
        yf_interval, yf_range = YAHOO_INTERVAL_RANGE_MAP.get(timeframe, ("15m", "1mo"))
        url = self.YAHOO_CHART_URL.format(ticker=yahoo_ticker)

        try:
            with httpx.Client(
                timeout=6.0,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            ) as client:
                resp = client.get(
                    url,
                    params={
                        "interval": yf_interval,
                        "range": yf_range,
                        "includePrePost": "false",
                    },
                )
                if resp.status_code != 200:
                    return None
                payload: dict[str, Any] = resp.json()

            results = payload.get("chart", {}).get("result")
            if not results:
                return None

            result = results[0]
            meta = result.get("meta", {})
            timestamps = result.get("timestamp", [])
            quotes = result.get("indicators", {}).get("quote", [{}])[0]
            if not timestamps or not quotes:
                return None

            raw_df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                    "open": quotes.get("open", []),
                    "high": quotes.get("high", []),
                    "low": quotes.get("low", []),
                    "close": quotes.get("close", []),
                    "volume": quotes.get("volume", []),
                }
            ).dropna(subset=["open", "high", "low", "close"])

            if len(raw_df) < 20:
                return None

            live_price = meta.get("regularMarketPrice")
            if live_price is not None and float(live_price) > 0:
                lp = float(live_price)
                last_idx = raw_df.index[-1]
                raw_df.loc[last_idx, "close"] = lp
                raw_df.loc[last_idx, "high"] = max(float(raw_df.loc[last_idx, "high"]), lp)
                raw_df.loc[last_idx, "low"] = min(float(raw_df.loc[last_idx, "low"]), lp)

            if timeframe == "4h":
                raw_df = (
                    raw_df.set_index("timestamp")
                    .resample("4h")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                    .dropna(subset=["open", "high", "low", "close"])
                    .reset_index()
                )
            cleaned = self.preprocessor.clean_ohlcv(
                raw_df.tail(limit), symbol=canonical_symbol, timeframe=timeframe
            )
            cleaned.attrs["data_source"] = "live"
            logger.info(
                "Fetched real-time Yahoo Finance OHLCV feed",
                symbol=canonical_symbol,
                yahoo_ticker=yahoo_ticker,
                timeframe=timeframe,
                latest_price=float(cleaned["close"].iloc[-1]),
                bars=len(cleaned),
            )
            return cleaned
        except Exception as exc:
            logger.debug(
                "Yahoo Finance live feed unavailable; falling back",
                ticker=yahoo_ticker,
                error=str(exc),
            )
            return None

    def fetch_live_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 400,
    ) -> Optional[pd.DataFrame]:
        """Fetch real-time OHLCV DataFrame with short TTL memory caching."""
        clean_sym = normalize_symbol(symbol)
        cache_key = f"{clean_sym}_{timeframe}_{limit}"
        now = time.monotonic()

        if cache_key in self._cache:
            ts, cached_df = self._cache[cache_key]
            if now - ts <= self.ttl_seconds:
                return cached_df.copy()

        is_crypto, resolved = is_crypto_symbol(clean_sym)
        df: Optional[pd.DataFrame] = None

        if is_crypto:
            df = self.fetch_binance_klines(
                binance_pair=resolved,
                canonical_symbol=clean_sym,
                timeframe=timeframe,
                limit=limit,
            )

        if df is None:
            yahoo_ticker = resolve_yahoo_ticker(clean_sym)
            df = self.fetch_yahoo_chart(
                yahoo_ticker=yahoo_ticker,
                canonical_symbol=clean_sym,
                timeframe=timeframe,
                limit=limit,
            )

        if df is not None and not df.empty:
            self._cache[cache_key] = (now, df)
            return df.copy()

        return None

    def fetch_5y_daily_memory(
        self,
        symbol: str,
        max_bars: int = 2000,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch up to 5 years of historical daily (1d) OHLCV bars (from 5 years ago to today)
        and persist in `data/processed/memory_5y_<symbol>.parquet` for continual AI memory.
        """
        clean_sym = normalize_symbol(symbol)
        cache_key = f"5Y_MEM_{clean_sym}"
        now = time.monotonic()

        if cache_key in self._cache:
            ts, cached_df = self._cache[cache_key]
            if now - ts <= 600.0 and not cached_df.empty:
                return cached_df.copy()

        parquet_path = self.settings.processed_data_dir / f"memory_5y_{clean_sym.lower()}.parquet"
        if parquet_path.exists():
            try:
                mtime = parquet_path.stat().st_mtime
                if (time.time() - mtime) <= 3600.0:
                    disk_df = pd.read_parquet(parquet_path)
                    if len(disk_df) >= 100:
                        self._cache[cache_key] = (now, disk_df)
                        return disk_df.copy()
            except Exception:
                pass

        yahoo_ticker = resolve_yahoo_ticker(clean_sym)
        df_5y = self.fetch_yahoo_chart(
            yahoo_ticker=yahoo_ticker,
            canonical_symbol=clean_sym,
            timeframe="1d",
            limit=max_bars,
        )
        if df_5y is None or len(df_5y) < 100:
            is_crypto, binance_pair = is_crypto_symbol(clean_sym)
            if is_crypto:
                df_5y = self.fetch_binance_klines(
                    binance_pair=binance_pair,
                    canonical_symbol=clean_sym,
                    timeframe="1d",
                    limit=1000,
                )

        if df_5y is not None and not df_5y.empty:
            try:
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                df_5y.to_parquet(parquet_path, engine="pyarrow", index=True)
            except Exception:
                pass
            self._cache[cache_key] = (now, df_5y)
            return df_5y.copy()

        return None

    @staticmethod
    def compute_5y_macro_context(
        df_5y: Optional[pd.DataFrame],
        current_price: float,
    ) -> dict[str, Any]:
        """Compute 5-year price percentile, high/low bounds, and macro trend regime."""
        if df_5y is None or df_5y.empty or "close" not in df_5y.columns:
            return {
                "memory_bars_5y": 0,
                "five_year_high": current_price,
                "five_year_low": current_price,
                "five_year_percentile": 50.0,
                "macro_regime": "Data Jangka Pendek",
            }

        closes = pd.to_numeric(df_5y["close"], errors="coerce").dropna()
        highs = pd.to_numeric(df_5y.get("high", closes), errors="coerce").dropna()
        lows = pd.to_numeric(df_5y.get("low", closes), errors="coerce").dropna()
        if closes.empty:
            return {
                "memory_bars_5y": 0,
                "five_year_high": current_price,
                "five_year_low": current_price,
                "five_year_percentile": 50.0,
                "macro_regime": "Data Jangka Pendek",
            }

        high_5y = max(float(highs.max()), current_price)
        low_5y = min(float(lows.min()), current_price)
        span = max(high_5y - low_5y, 1e-6)
        pct_pos = float(min(max(((current_price - low_5y) / span) * 100.0, 0.0), 100.0))

        ema_50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
        ema_200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
        if current_price >= ema_50 >= ema_200:
            macro_regime = "Tren Naik Jangka Panjang (Bullish 5Y)"
        elif current_price <= ema_50 <= ema_200:
            macro_regime = "Tren Turun Jangka Panjang (Bearish 5Y)"
        else:
            macro_regime = "Fase Konsolidasi / Transisi (5Y)"

        return {
            "memory_bars_5y": int(len(closes)),
            "five_year_high": round(high_5y, 4),
            "five_year_low": round(low_5y, 4),
            "five_year_percentile": round(pct_pos, 1),
            "macro_regime": macro_regime,
        }
