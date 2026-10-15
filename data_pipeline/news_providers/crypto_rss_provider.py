from __future__ import annotations

import asyncio
import email.utils
import html
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.base_provider import BaseNewsProvider
from data_pipeline.news_providers.schemas import NewsSource, UnifiedNewsItem

logger = get_logger("nexustrade.news_providers.crypto_rss")

TIER1_CRYPTO_RSS_ENDPOINTS: Dict[str, str] = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "TheBlock": "https://www.theblock.co/rss.xml",
}

ASSET_KEYWORDS: Dict[str, List[str]] = {
    "BTC": ["btc", "bitcoin", "satoshi", "halving", "spot etf"],
    "ETH": ["eth", "ethereum", "vitalik", "erc20", "layer 2", "pectra"],
    "SOL": ["sol", "solana", "firedancer"],
    "XRP": ["xrp", "ripple", "rlusd"],
    "BNB": ["bnb", "binance", "bsc"],
}


class Tier1CryptoRSSProvider(BaseNewsProvider):
    """
    Direct Tier-1 Crypto Media RSS Aggregator (Zero-Limit / Zero-Cost).
    Concurrently polls primary news feeds: CoinDesk, CoinTelegraph, Decrypt, and The Block.
    Provides zero-latency, zero-cost raw crypto news streaming.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        super().__init__(
            source=NewsSource.TIER1_CRYPTO_RSS,
            ttl_seconds=getattr(cfg, "ttl_crypto_rss", 60),
            settings=cfg,
        )
        self.endpoints = dict(TIER1_CRYPTO_RSS_ENDPOINTS)
        custom_urls = getattr(cfg, "crypto_rss_feeds", "")
        if custom_urls:
            urls = [u.strip() for u in custom_urls.split(",") if u.strip()]
            if len(urls) >= 1:
                names = ["CoinDesk", "CoinTelegraph", "Decrypt", "TheBlock"]
                for idx, url in enumerate(urls):
                    name = names[idx] if idx < len(names) else f"CryptoFeed{idx+1}"
                    self.endpoints[name] = url

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        clean = re.sub(r"<[^>]+>", " ", html.unescape(raw_html or ""))
        return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def _parse_rss_datetime(entry: Any) -> datetime:
        for field in ("published", "updated", "created"):
            val = entry.get(field) if hasattr(entry, "get") else None
            if val:
                try:
                    dt = email.utils.parsedate_to_datetime(str(val))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt
                except Exception:
                    pass
                try:
                    dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt
                except Exception:
                    pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _extract_crypto_tickers_and_relevance(
        text: str, target_symbol: str
    ) -> Tuple[List[str], float]:
        lower_text = text.lower()
        detected: List[str] = []
        for sym, keywords in ASSET_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", lower_text) for kw in keywords):
                detected.append(sym)

        clean_target = (
            target_symbol.upper()
            .replace("USDT", "")
            .replace("USDC", "")
            .replace("USD", "")
            .replace("PERP", "")
        )
        if clean_target in detected or (clean_target and clean_target.lower() in lower_text):
            return detected or [clean_target], 1.0
        return detected, 0.55

    async def _fetch_single_rss(
        self,
        client: httpx.AsyncClient,
        publisher: str,
        rss_url: str,
        target_symbol: str,
    ) -> List[UnifiedNewsItem]:
        items: List[UnifiedNewsItem] = []
        try:
            resp = await client.get(rss_url, timeout=8.0, follow_redirects=True)
            if resp.status_code != 200:
                logger.debug("RSS feed returned status", publisher=publisher, status=resp.status_code)
                return []

            parsed = feedparser.parse(resp.text)
            for entry in parsed.entries[:10]:
                title = self._clean_html(entry.get("title", ""))
                if not title:
                    continue
                summary = self._clean_html(
                    entry.get("summary", entry.get("description", ""))
                )[:500]
                pub_dt = self._parse_rss_datetime(entry)
                combined_text = f"{title}. {summary}"
                tickers, relevance = self._extract_crypto_tickers_and_relevance(
                    combined_text, target_symbol
                )

                item = UnifiedNewsItem(
                    source=NewsSource.TIER1_CRYPTO_RSS,
                    publisher=publisher,
                    title=title,
                    summary=summary,
                    url=str(entry.get("link", "")).strip(),
                    published_at=pub_dt,
                    tickers=tickers,
                    relevance_score=relevance,
                    authority_weight=1.10,
                )
                item.compute_hash()
                items.append(item)
        except Exception as exc:
            logger.debug("Failed to fetch/parse RSS feed", publisher=publisher, error=str(exc))
        return items

    async def fetch(
        self,
        client: httpx.AsyncClient,
        target_symbol: str = "BTC",
        **kwargs,
    ) -> List[UnifiedNewsItem]:
        """Fetch news from all Tier-1 Crypto RSS feeds in parallel with 60s TTL cache."""
        clean_target = (
            target_symbol.upper()
            .replace("USDT", "")
            .replace("USDC", "")
            .replace("USD", "")
            .replace("PERP", "")
            or "BTC"
        )
        cache_key = f"tier1_crypto_rss_{clean_target}"
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        tasks = [
            self._fetch_single_rss(client, pub, url, clean_target)
            for pub, url in self.endpoints.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged: List[UnifiedNewsItem] = []
        for res in results:
            if isinstance(res, list):
                merged.extend(res)

        self.set_cached(cache_key, merged)
        logger.info(
            "Fetched Tier-1 Crypto Media RSS news",
            target=clean_target,
            total_articles=len(merged),
            publishers=list(self.endpoints.keys()),
        )
        return merged
