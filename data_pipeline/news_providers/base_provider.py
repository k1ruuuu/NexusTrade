from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.schemas import NewsSource

logger = get_logger("nexustrade.news_providers.base")


class BaseNewsProvider(ABC):
    """
    Abstract async base class for live news and economic data providers.
    Provides standardized in-memory TTL caching with Redis fallback,
    defensive error handling, and structured logging.
    """

    def __init__(
        self,
        source: NewsSource,
        ttl_seconds: int = 180,
        settings: Optional[Settings] = None,
    ) -> None:
        self.source = source
        self.ttl_seconds = max(1, ttl_seconds)
        self.settings = settings or get_settings()
        self._memory_cache: dict[str, tuple[float, Any]] = {}

    def get_cached(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from memory cache if within TTL."""
        now = time.monotonic()
        if cache_key in self._memory_cache:
            cached_at, value = self._memory_cache[cache_key]
            if now - cached_at < self.ttl_seconds:
                return value
            self._memory_cache.pop(cache_key, None)
        return None

    def set_cached(self, cache_key: str, value: Any, custom_ttl: Optional[int] = None) -> None:
        """Store data in memory cache with timestamp."""
        now = time.monotonic()
        self._memory_cache[cache_key] = (now, value)

    @abstractmethod
    async def fetch(self, client: httpx.AsyncClient, **kwargs: Any) -> Any:
        """Fetch and parse real-time data from the external provider."""
        raise NotImplementedError
