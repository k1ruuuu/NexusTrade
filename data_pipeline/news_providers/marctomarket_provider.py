from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.base_provider import BaseNewsProvider
from data_pipeline.news_providers.schemas import NewsSource, UnifiedNewsItem

logger = get_logger("nexustrade.news_providers.marctomarket")


class MarcToMarketProvider(BaseNewsProvider):
    """
    MarcToMarket (Blogger Atom/RSS JSON) Provider.
    Extracts institutional FX & global macro narrative analysis from Dr. Marc Chandler.
    Focuses on G10 currency policy, Treasury yields, central bank rhetoric, and dollar cycles.
    """

    MACRO_KEYWORDS = (
        "usd", "dollar", "fed", "federal reserve", "yields", "treasury",
        "eur", "ecb", "euro", "jpy", "boj", "yen", "gbp", "boe",
        "inflation", "cpi", "rate cut", "rate hike", "gold", "tariffs",
    )

    def __init__(self, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        super().__init__(
            source=NewsSource.MARC_TO_MARKET,
            ttl_seconds=getattr(cfg, "ttl_marctomarket", 1800),
            settings=cfg,
        )
        self.endpoint_url = getattr(
            cfg,
            "marctomarket_feed_url",
            "https://www.marctomarket.com/feeds/posts/default?alt=json",
        )

    async def fetch(self, client: httpx.AsyncClient, **kwargs) -> List[UnifiedNewsItem]:
        cache_key = "marctomarket_feed"
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        items: List[UnifiedNewsItem] = []
        try:
            resp = await client.get(self.endpoint_url, timeout=12.0)
            if resp.status_code != 200:
                logger.warning(
                    "MarcToMarket feed returned non-200",
                    status_code=resp.status_code,
                )
                return self.get_cached(cache_key) or []

            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])[:6]
            for entry in entries:
                title = str(entry.get("title", {}).get("$t", "")).strip()
                raw_html = str(entry.get("content", {}).get("$t", "") or entry.get("summary", {}).get("$t", ""))
                clean_text = re.sub(r"<[^>]+>", " ", html.unescape(raw_html))
                clean_text = re.sub(r"\s+", " ", clean_text).strip()

                # Filter macro paragraphs
                paragraphs = [p.strip() for p in clean_text.split(".") if len(p.strip()) > 20]
                relevant_paragraphs = [
                    p for p in paragraphs
                    if any(kw in p.lower() for kw in self.MACRO_KEYWORDS)
                ]
                summary = ". ".join(relevant_paragraphs[:4]) if relevant_paragraphs else clean_text[:600]

                pub_str = str(entry.get("published", {}).get("$t", "")).strip()
                try:
                    if pub_str.endswith("Z"):
                        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    else:
                        pub_dt = datetime.fromisoformat(pub_str)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    else:
                        pub_dt = pub_dt.astimezone(timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                item = UnifiedNewsItem(
                    source=NewsSource.MARC_TO_MARKET,
                    title=title,
                    summary=summary[:800],
                    published_at=pub_dt,
                    tickers=["DXY", "EURUSD", "USDJPY", "GBPUSD", "XAUUSD"],
                    authority_weight=1.15,
                )
                item.compute_hash()
                items.append(item)

            self.set_cached(cache_key, items)
            logger.info("Fetched MarcToMarket macro commentary", articles_count=len(items))
            return items
        except Exception as exc:
            logger.warning("Failed to fetch MarcToMarket feed", error=str(exc))
            return self.get_cached(cache_key) or []
