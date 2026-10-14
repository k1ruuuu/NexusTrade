from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.base_provider import BaseNewsProvider
from data_pipeline.news_providers.schemas import NewsSource, UnifiedNewsItem

logger = get_logger("nexustrade.news_providers.cryptopanic")


class CryptoPanicProvider(BaseNewsProvider):
    """
    CryptoPanic REST API Provider.
    Extracts real-time crypto news along with community crowd voting metrics
    (positive, negative, toxic, saved) to quantify crowd psychology.
    Rate limited to 5 calls/min (Free Tier), cached with 180s TTL.
    """

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key: Optional[str] = None, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        super().__init__(
            source=NewsSource.CRYPTOPANIC,
            ttl_seconds=getattr(cfg, "ttl_cryptopanic", 180),
            settings=cfg,
        )
        self.api_key = (api_key or getattr(cfg, "cryptopanic_api_key", "")).strip()

    @staticmethod
    def compute_crowd_sentiment(votes: dict[str, Any]) -> float:
        """
        PRD Section 4.5 Crowd Polarity Formulation:
        S_crowd = ((V_pos + 0.5 * V_saved) - (V_neg + 2.0 * V_toxic)) / (V_pos + V_neg + V_toxic + V_saved + 1.0)
        """
        try:
            v_pos = float(votes.get("positive", 0))
            v_neg = float(votes.get("negative", 0))
            v_tox = float(votes.get("toxic", 0))
            v_sav = float(votes.get("saved", 0))
            numerator = (v_pos + 0.5 * v_sav) - (v_neg + 2.0 * v_tox)
            denominator = v_pos + v_neg + v_tox + v_sav + 1.0
            return max(-1.0, min(1.0, numerator / denominator))
        except Exception:
            return 0.0

    async def fetch(
        self,
        client: httpx.AsyncClient,
        currencies: str = "BTC,ETH",
        limit: int = 15,
        **kwargs,
    ) -> List[UnifiedNewsItem]:
        if not self.api_key:
            return []

        clean_ccys = currencies.upper().replace("USDT", "").replace("USDC", "").strip()
        cache_key = f"cp_{clean_ccys}_{limit}"
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        items: List[UnifiedNewsItem] = []
        try:
            params = {
                "auth_token": self.api_key,
                "currencies": clean_ccys,
                "public": "true",
            }
            resp = await client.get(self.BASE_URL, params=params, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("CryptoPanic returned non-200", status_code=resp.status_code)
                return self.get_cached(cache_key) or []

            payload = resp.json()
            results = payload.get("results", [])[:limit]
            for row in results:
                title = str(row.get("title", "")).strip()
                if not title:
                    continue

                c_str = str(row.get("created_at", "")).strip()
                try:
                    pub_dt = datetime.fromisoformat(c_str.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                crowd_score = self.compute_crowd_sentiment(row.get("votes", {}))

                item = UnifiedNewsItem(
                    source=NewsSource.CRYPTOPANIC,
                    title=title,
                    summary=title,
                    url=str(row.get("url", "")).strip(),
                    published_at=pub_dt,
                    tickers=clean_ccys.split(","),
                    provider_sentiment=crowd_score,
                    authority_weight=0.95,
                )
                item.compute_hash()
                items.append(item)

            self.set_cached(cache_key, items)
            logger.info("Fetched CryptoPanic crowd sentiment news", currencies=clean_ccys, items_count=len(items))
            return items
        except Exception as exc:
            logger.warning("Failed to fetch CryptoPanic news", error=str(exc))
            return self.get_cached(cache_key) or []
