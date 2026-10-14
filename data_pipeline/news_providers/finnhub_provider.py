from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.base_provider import BaseNewsProvider
from data_pipeline.news_providers.schemas import NewsSource, UnifiedNewsItem

logger = get_logger("nexustrade.news_providers.finnhub")


class FinnhubProvider(BaseNewsProvider):
    """
    Finnhub.io REST API Provider.
    Ingests real-time breaking market news across Forex, Crypto, and General categories.
    Rate limited to 60 calls/min (Free Tier), with short 120s TTL caching.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        super().__init__(
            source=NewsSource.FINNHUB,
            ttl_seconds=getattr(cfg, "ttl_finnhub", 120),
            settings=cfg,
        )
        self.api_key = (api_key or getattr(cfg, "finnhub_api_key", "")).strip()

    async def fetch(
        self,
        client: httpx.AsyncClient,
        category: str = "general",
        symbol: Optional[str] = None,
        **kwargs,
    ) -> List[UnifiedNewsItem]:
        if not self.api_key:
            return []

        cache_key = f"finnhub_{category}_{symbol or 'all'}"
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        items: List[UnifiedNewsItem] = []
        try:
            url = f"{self.BASE_URL}/news"
            params = {"category": category, "token": self.api_key}
            resp = await client.get(url, params=params, timeout=10.0)

            if resp.status_code == 429:
                logger.warning("Finnhub API rate limited (429)")
                return self.get_cached(cache_key) or []
            if resp.status_code != 200:
                logger.warning("Finnhub API returned status code", status_code=resp.status_code)
                return self.get_cached(cache_key) or []

            data = resp.json()
            if not isinstance(data, list):
                return []

            for row in data[:20]:
                headline = str(row.get("headline", "")).strip()
                summary = str(row.get("summary", "")).strip()
                if not headline:
                    continue

                ts_sec = row.get("datetime", 0)
                try:
                    pub_dt = datetime.fromtimestamp(int(ts_sec), tz=timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                item = UnifiedNewsItem(
                    source=NewsSource.FINNHUB,
                    title=headline,
                    summary=summary[:600],
                    url=str(row.get("url", "")).strip(),
                    published_at=pub_dt,
                    authority_weight=1.0,
                )
                item.compute_hash()
                items.append(item)

            self.set_cached(cache_key, items)
            logger.info("Fetched Finnhub real-time news", category=category, items_count=len(items))
            return items
        except Exception as exc:
            logger.warning("Failed to fetch Finnhub news", error=str(exc))
            return self.get_cached(cache_key) or []
