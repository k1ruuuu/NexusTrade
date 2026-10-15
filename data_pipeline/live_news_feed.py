from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import httpx
from data_pipeline.market_symbols import normalize_symbol

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.preprocessor_sentiment import SentimentNLPPreprocessor

logger = get_logger("nexustrade.data_pipeline.live_news")

SYMBOL_SEARCH_QUERIES: dict[str, tuple[str, str]] = {
    "BTC": ('"Bitcoin" OR "BTC" crypto market when:2d', "BTC-USD"),
    "BTCUSDT": ('"Bitcoin" OR "BTC" crypto market when:2d', "BTC-USD"),
    "BTCUSD": ('"Bitcoin" OR "BTC" crypto market when:2d', "BTC-USD"),
    "ETH": ('"Ethereum" OR "ETH" crypto market when:2d', "ETH-USD"),
    "ETHUSDT": ('"Ethereum" OR "ETH" crypto market when:2d', "ETH-USD"),
    "SOL": ('"Solana" OR "SOL" crypto market when:2d', "SOL-USD"),
    "SOLUSDT": ('"Solana" OR "SOL" crypto market when:2d', "SOL-USD"),
    "SOLUSD": ('"Solana" OR "SOL" crypto market when:2d', "SOL-USD"),
    "XAUUSD": ('XAUUSD OR "Gold price" bullion market when:2d', "GC=F"),
    "GOLD": ('XAUUSD OR "Gold price" bullion market when:2d', "GC=F"),
    "DOGE": ('"Dogecoin" OR "DOGE" crypto market when:2d', "DOGE-USD"),
    "DOGEUSDT": ('"Dogecoin" OR "DOGE" crypto market when:2d', "DOGE-USD"),
    "AVAX": ('"Avalanche" OR "AVAX" crypto market when:2d', "AVAX-USD"),
    "AVAXUSDT": ('"Avalanche" OR "AVAX" crypto market when:2d', "AVAX-USD"),
    "LINK": ('"Chainlink" OR "LINK" crypto market when:2d', "LINK-USD"),
    "LINKUSDT": ('"Chainlink" OR "LINK" crypto market when:2d', "LINK-USD"),
    "AAPL": ('"Apple stock" OR "AAPL" market when:2d', "AAPL"),
    "NVDA": ('"Nvidia stock" OR "NVDA" market when:2d', "NVDA"),
    "SPX": ('"S&P 500" OR "SPX" stock market when:2d', "^GSPC"),
}


@dataclass(frozen=True)
class LiveNewsItem:
    """Single breaking news headline harvested from the internet."""

    headline_id: str
    symbol: str
    headline: str
    source: str
    published_at: str
    url: str = ""


class LiveNewsFeed:
    """
    Real-time financial news harvester pulling breaking headlines from
    Google News RSS and Yahoo Finance News Search API with TTL caching.
    """

    GOOGLE_RSS_URL = "https://news.google.com/rss/search"
    YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        preprocessor: Optional[SentimentNLPPreprocessor] = None,
        ttl_seconds: int = 45,
    ) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = preprocessor or SentimentNLPPreprocessor(self.settings)
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, list[LiveNewsItem]]] = {}

    @staticmethod
    def compute_headline_id(headline: str) -> str:
        normalized = " ".join(headline.lower().strip().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def resolve_queries(symbol: str) -> tuple[str, str]:
        clean = normalize_symbol(symbol)
        if clean in SYMBOL_SEARCH_QUERIES:
            return SYMBOL_SEARCH_QUERIES[clean]
        for prefix, pair in SYMBOL_SEARCH_QUERIES.items():
            if clean.startswith(prefix):
                return pair
        return (f'"{clean}" financial market news when:2d', clean)

    @staticmethod
    def _is_relevant_headline(symbol: str, headline: str) -> bool:
        upper_sym = symbol.upper()
        lower_hl = headline.lower()
        keyword_map: dict[str, tuple[str, ...]] = {
            "XAU": ("gold", "xau", "bullion", "precious", "ounce"),
            "GOLD": ("gold", "xau", "bullion", "precious", "ounce"),
            "BTC": ("bitcoin", "btc", "crypto", "satoshi", "etf"),
            "ETH": ("ethereum", "eth", "crypto", "ether", "defi", "layer-2"),
            "SOL": ("solana", "sol", "crypto", "defi"),
            "DOGE": ("dogecoin", "doge", "crypto", "meme"),
            "AVAX": ("avalanche", "avax", "crypto"),
            "LINK": ("chainlink", "link", "oracle", "crypto"),
            "AAPL": ("apple", "aapl", "iphone", "mac", "tim cook"),
            "NVDA": ("nvidia", "nvda", "gpu", "chip", "ai"),
            "SPX": ("s&p", "spx", "wall street", "stocks", "equities", "fed"),
        }
        for prefix, keywords in keyword_map.items():
            if prefix in upper_sym:
                return any(kw in lower_hl for kw in keywords)
        return True

    def fetch_google_rss(self, symbol: str, query: str, max_items: int = 8) -> list[LiveNewsItem]:
        items: list[LiveNewsItem] = []
        try:
            with httpx.Client(
                timeout=6.0,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            ) as client:
                resp = client.get(
                    self.GOOGLE_RSS_URL,
                    params={
                        "q": query,
                        "hl": "en-US",
                        "gl": "US",
                        "ceid": "US:en",
                    },
                )
                if resp.status_code != 200:
                    return []

            root = ET.fromstring(resp.text)
            for node in root.findall("./channel/item")[:max_items]:
                raw_title = (node.findtext("title") or "").strip()
                if not raw_title:
                    continue
                # Strip trailing " - Source Name" if present while preserving source
                source_name = (node.findtext("source") or "Google News").strip()
                if " - " in raw_title:
                    parts = raw_title.rsplit(" - ", 1)
                    if len(parts[0]) >= 15:
                        raw_title = parts[0].strip()
                        if source_name == "Google News":
                            source_name = parts[1].strip()

                cleaned = self.preprocessor.clean_text(raw_title)
                if len(cleaned) < 12:
                    continue

                pub_raw = (node.findtext("pubDate") or "").strip()
                try:
                    dt = parsedate_to_datetime(pub_raw).astimezone(timezone.utc)
                    iso_ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                link = (node.findtext("link") or "").strip()
                items.append(
                    LiveNewsItem(
                        headline_id=self.compute_headline_id(cleaned),
                        symbol=symbol.upper(),
                        headline=cleaned,
                        source=source_name,
                        published_at=iso_ts,
                        url=link,
                    )
                )
        except Exception as exc:
            logger.debug("Google RSS news fetch failed", symbol=symbol, error=str(exc))

        return items

    def fetch_yahoo_news(self, symbol: str, yahoo_query: str, max_items: int = 8) -> list[LiveNewsItem]:
        items: list[LiveNewsItem] = []
        try:
            with httpx.Client(
                timeout=6.0,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            ) as client:
                resp = client.get(
                    self.YAHOO_SEARCH_URL,
                    params={
                        "q": yahoo_query,
                        "newsCount": max_items,
                        "quotesCount": 0,
                    },
                )
                if resp.status_code != 200:
                    return []
                payload: dict[str, Any] = resp.json()

            for entry in payload.get("news", [])[:max_items]:
                raw_title = str(entry.get("title") or "").strip()
                cleaned = self.preprocessor.clean_text(raw_title)
                if len(cleaned) < 12 or not self._is_relevant_headline(symbol, cleaned):
                    continue
                pub_epoch = entry.get("providerPublishTime")
                if pub_epoch:
                    dt = datetime.fromtimestamp(int(pub_epoch), tz=timezone.utc)
                    iso_ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                publisher = str(entry.get("publisher") or "Yahoo Finance").strip()
                link = str(entry.get("link") or "").strip()
                items.append(
                    LiveNewsItem(
                        headline_id=self.compute_headline_id(cleaned),
                        symbol=symbol.upper(),
                        headline=cleaned,
                        source=publisher,
                        published_at=iso_ts,
                        url=link,
                    )
                )
        except Exception as exc:
            logger.debug("Yahoo Finance news fetch failed", symbol=symbol, error=str(exc))

        return items

    def fetch_live_news(
        self,
        symbol: str = "BTCUSDT",
        max_items: int = 30,
        force_refresh: bool = False,
        lookback_days: int = 90,
    ) -> list[LiveNewsItem]:
        """
        Fetch deduplicated news headlines from the latest breaking news back to `lookback_days`
        (default 90 days / 3 months) across Google News RSS (2d, 30d, 90d windows) and Yahoo Finance.
        """
        clean_sym = normalize_symbol(symbol)
        now = time.monotonic()

        if not force_refresh and clean_sym in self._cache:
            ts, cached_items = self._cache[clean_sym]
            if now - ts <= self.ttl_seconds and len(cached_items) >= min(max_items, 12):
                return list(cached_items[:max_items])

        now_utc = datetime.now(timezone.utc)
        cutoff_30d = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")
        cutoff_90d = (now_utc - timedelta(days=max(lookback_days, 90))).strftime("%Y-%m-%d")

        rss_q, yf_q = self.resolve_queries(clean_sym)
        q_30d = rss_q.replace("when:2d", f"after:{cutoff_30d}")
        q_90d = rss_q.replace("when:2d", f"after:{cutoff_90d}")

        per_window = max(12, max_items // 2)
        rss_breaking = self.fetch_google_rss(clean_sym, rss_q, max_items=per_window)
        yf_breaking = self.fetch_yahoo_news(clean_sym, yf_q, max_items=per_window)
        rss_30d = self.fetch_google_rss(clean_sym, q_30d, max_items=per_window)
        rss_90d = self.fetch_google_rss(clean_sym, q_90d, max_items=max(18, max_items))

        deduped: dict[str, LiveNewsItem] = {}
        for item in [*rss_breaking, *yf_breaking, *rss_30d, *rss_90d]:
            if item.headline_id not in deduped:
                deduped[item.headline_id] = item

        sorted_items = sorted(
            deduped.values(),
            key=lambda x: x.published_at,
            reverse=True,
        )[: max(max_items, 30)]

        if sorted_items:
            self._cache[clean_sym] = (now, sorted_items)
            logger.info(
                "Harvested live + 3-month (90d) financial news from internet",
                symbol=clean_sym,
                headlines_count=len(sorted_items),
                latest_published=sorted_items[0].published_at,
                oldest_published=sorted_items[-1].published_at,
            )

        return sorted_items
