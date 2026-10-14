from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.base_provider import BaseNewsProvider
from data_pipeline.news_providers.schemas import NewsSource, UnifiedNewsItem

logger = get_logger("nexustrade.news_providers.alphavantage")


class AlphaVantageProvider(BaseNewsProvider):
    """
    Alpha Vantage News & Sentiment REST API Provider.
    Extracts pre-scored institutional news & ticker-specific sentiment vectors.
    Aggressively cached (1800s - 3600s) to adhere strictly to free tier API constraints.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        super().__init__(
            source=NewsSource.ALPHA_VANTAGE,
            ttl_seconds=getattr(cfg, "ttl_alphavantage", 1800),
            settings=cfg,
        )
        self.api_key = (api_key or getattr(cfg, "alphavantage_api_key", "")).strip()

    def resolve_av_ticker(self, symbol: str) -> str:
        sym = symbol.upper().replace("/", "").replace("-", "")
        if "BTC" in sym:
            return "CRYPTO:BTC"
        if "ETH" in sym:
            return "CRYPTO:ETH"
        if "SOL" in sym:
            return "CRYPTO:SOL"
        if "XAU" in sym or "GOLD" in sym:
            return "FOREX:USD"
        if len(sym) == 6 and sym.isalpha():
            return f"FOREX:{sym[:3]}"
        return sym

    async def fetch(
        self,
        client: httpx.AsyncClient,
        ticker: str = "CRYPTO:BTC",
        limit: int = 15,
        **kwargs,
    ) -> List[UnifiedNewsItem]:
        if not self.api_key:
            return []

        resolved_ticker = self.resolve_av_ticker(ticker)
        cache_key = f"av_{resolved_ticker}_{limit}"
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        items: List[UnifiedNewsItem] = []
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": resolved_ticker,
                "limit": limit,
                "apikey": self.api_key,
            }
            resp = await client.get(self.BASE_URL, params=params, timeout=12.0)
            if resp.status_code != 200:
                logger.warning("Alpha Vantage returned non-200", status_code=resp.status_code)
                return self.get_cached(cache_key) or []

            payload = resp.json()
            if "Note" in payload or "Information" in payload:
                logger.warning("Alpha Vantage API rate limit notice", msg=payload.get("Note") or payload.get("Information"))
                return self.get_cached(cache_key) or []

            feed = payload.get("feed", [])
            for row in feed:
                title = str(row.get("title", "")).strip()
                summary = str(row.get("summary", "")).strip()
                if not title:
                    continue

                t_str = str(row.get("time_published", "")).strip()
                try:
                    pub_dt = datetime.strptime(t_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                try:
                    av_score = float(row.get("overall_sentiment_score", 0.0))
                except (ValueError, TypeError):
                    av_score = 0.0

                rel_score = 1.0
                for ts in row.get("ticker_sentiment", []):
                    if ts.get("ticker") == resolved_ticker:
                        try:
                            rel_score = float(ts.get("relevance_score", 1.0))
                        except (ValueError, TypeError):
                            rel_score = 1.0
                        break

                item = UnifiedNewsItem(
                    source=NewsSource.ALPHA_VANTAGE,
                    title=title,
                    summary=summary[:600],
                    url=str(row.get("url", "")).strip(),
                    published_at=pub_dt,
                    tickers=[resolved_ticker],
                    provider_sentiment=max(-1.0, min(1.0, av_score)),
                    relevance_score=max(0.0, min(1.0, rel_score)),
                    authority_weight=1.20,
                )
                item.compute_hash()
                items.append(item)

            self.set_cached(cache_key, items)
            logger.info("Fetched Alpha Vantage sentiment news", ticker=resolved_ticker, items_count=len(items))
            return items
        except Exception as exc:
            logger.warning("Failed to fetch Alpha Vantage news", error=str(exc))
            return self.get_cached(cache_key) or []
