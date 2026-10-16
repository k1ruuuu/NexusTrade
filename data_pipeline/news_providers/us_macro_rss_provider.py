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
from ai_engine.models.sentiment_lexicon import (
    compute_macro_monetary_bias,
    macro_monetary_to_gold_signal,
)

logger = get_logger("nexustrade.news_providers.us_macro")

US_MACRO_RSS_ENDPOINTS: Dict[str, str] = {
    "ForexLive": "https://www.forexlive.com/feed/",
    "CNBC_Economy": "https://search.cnbc.com/rs/search/combinedAsset/rss/rss.html?partnerId=wrss01&id=20910258",
    "Investing_Economic": "https://www.investing.com/rss/news_14.rss",
}

US_MACRO_KEYWORDS = (
    "cpi", "consumer price", "ppi", "producer price", "pce", "core pce",
    "nfp", "non-farm", "nonfarm", "payroll", "unemployment", "jobless claims",
    "retail sales", "inflation", "powell", "fomc", "federal reserve",
    "rate cut", "rate hike", "yields", "dxy", "dollar index",
)


class USMacroNewsProvider(BaseNewsProvider):
    """
    Real-Time US Macroeconomic & Squawk News Provider (Zero-Cost / Zero-Limit).
    Aggregates live feeds from ForexLive, CNBC Economy, and Investing.com.
    Specifically captures pre-news consensus and breaking post-release data for:
      - NFP (Non-Farm Payrolls & Unemployment)
      - CPI (Consumer Price Index)
      - PPI (Producer Price Index)
      - PCE (Personal Consumption Expenditures Inflation)
    Extracts institutional Hawkish vs Dovish monetary bias for Gold (XAUUSD).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        super().__init__(
            source=NewsSource.MARC_TO_MARKET,  # Macro news authority
            ttl_seconds=getattr(cfg, "ttl_macro_squawk", 20),
            settings=cfg,
        )
        self.endpoints = dict(US_MACRO_RSS_ENDPOINTS)

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
    def detect_macro_event_tag(text: str) -> Optional[str]:
        lower = text.lower()
        if any(k in lower for k in ("nfp", "non-farm", "nonfarm", "payroll")):
            return "NFP"
        if any(k in lower for k in ("cpi", "consumer price index")):
            return "CPI"
        if any(k in lower for k in ("ppi", "producer price index")):
            return "PPI"
        if any(k in lower for k in ("pce", "personal consumption expenditures")):
            return "PCE"
        if any(k in lower for k in ("fomc", "powell", "fed rate")):
            return "FOMC"
        return "MACRO"

    async def _fetch_single_macro_rss(
        self,
        client: httpx.AsyncClient,
        publisher: str,
        rss_url: str,
        target_event: Optional[str] = None,
    ) -> List[UnifiedNewsItem]:
        items: List[UnifiedNewsItem] = []
        try:
            resp = await client.get(rss_url, timeout=7.0, follow_redirects=True)
            if resp.status_code != 200:
                return []

            parsed = feedparser.parse(resp.text)
            for entry in parsed.entries[:15]:
                title = self._clean_html(entry.get("title", ""))
                summary = self._clean_html(entry.get("summary", entry.get("description", "")))[:600]
                combined = f"{title}. {summary}"
                lower_text = combined.lower()

                # Filter strictly for macroeconomic terms
                if not any(re.search(rf"\b{re.escape(kw)}\b", lower_text) for kw in US_MACRO_KEYWORDS):
                    continue

                event_tag = self.detect_macro_event_tag(combined)
                if target_event and event_tag != target_event.upper() and target_event.upper() not in lower_text:
                    # If user requested a specific event mode, prioritize matching articles
                    relevance = 0.50
                else:
                    relevance = 1.0

                pub_dt = self._parse_rss_datetime(entry)
                macro_score, stance = compute_macro_monetary_bias(combined)
                gold_score, gold_action = macro_monetary_to_gold_signal(macro_score, "XAUUSD")

                item = UnifiedNewsItem(
                    source=NewsSource.MARC_TO_MARKET,
                    publisher=f"USMacro_{publisher}",
                    title=f"[{stance}] {title}",
                    summary=summary,
                    url=str(entry.get("link", "")).strip(),
                    published_at=pub_dt,
                    tickers=["USD", "XAUUSD", "DXY", event_tag or "MACRO"],
                    provider_sentiment=gold_score,  # Direct transmission to Gold polarity
                    relevance_score=relevance,
                    authority_weight=1.25,  # High authority for US Macro breaking news
                )
                item.compute_hash()
                items.append(item)
        except Exception as exc:
            logger.debug("US Macro RSS fetch error", publisher=publisher, error=str(exc))
        return items

    async def fetch(
        self,
        client: httpx.AsyncClient,
        target_event: Optional[str] = None,
        **kwargs,
    ) -> List[UnifiedNewsItem]:
        """Fetch US Macro news from all fast RSS endpoints in parallel."""
        cache_key = f"us_macro_{target_event or 'all'}"
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        tasks = [
            self._fetch_single_macro_rss(client, pub, url, target_event)
            for pub, url in self.endpoints.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged: List[UnifiedNewsItem] = []
        for res in results:
            if isinstance(res, list):
                merged.extend(res)

        self.set_cached(cache_key, merged)
        logger.info(
            "Fetched breaking US Macro Squawk news",
            target_event=target_event,
            articles_count=len(merged),
        )
        return merged
