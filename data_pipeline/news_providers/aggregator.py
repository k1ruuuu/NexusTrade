from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.alphavantage_provider import AlphaVantageProvider
from data_pipeline.news_providers.cryptopanic_provider import CryptoPanicProvider
from data_pipeline.news_providers.finnhub_provider import FinnhubProvider
from data_pipeline.news_providers.forexfactory_provider import ForexFactoryProvider
from data_pipeline.news_providers.marctomarket_provider import MarcToMarketProvider
from data_pipeline.news_providers.schemas import (
    EconomicCalendarEvent,
    ImpactLevel,
    NewsSource,
    UnifiedNewsItem,
)

logger = get_logger("nexustrade.news_providers.aggregator")


class MultiSourceNewsAggregator:
    """
    Asynchronous 5-Provider News, Macro Commentary & Economic Calendar Engine.
    Coordinates ForexFactory, MarcToMarket, Finnhub, Alpha Vantage, and CryptoPanic.
    Applies SHA-256 deduplication, exponential time-decay weighting, provider authority weights,
    hybrid polarity scoring, and high-impact macroeconomic blackout checks.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        finbert_scorer: Optional[Callable[[str], float]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.finbert_scorer = finbert_scorer or (lambda text: 0.0)

        # Initialize providers
        self.forexfactory = ForexFactoryProvider(self.settings)
        self.marctomarket = MarcToMarketProvider(self.settings)
        self.finnhub = FinnhubProvider(
            api_key=getattr(self.settings, "finnhub_api_key", ""),
            settings=self.settings,
        )
        self.alphavantage = AlphaVantageProvider(
            api_key=getattr(self.settings, "alphavantage_api_key", ""),
            settings=self.settings,
        )
        self.cryptopanic = CryptoPanicProvider(
            api_key=getattr(self.settings, "cryptopanic_api_key", ""),
            settings=self.settings,
        )

    @staticmethod
    def classify_asset_category(symbol: str) -> tuple[str, list[str]]:
        """Identify asset class (crypto, forex, commodity, index) and relevant currency tags."""
        clean = symbol.upper().replace("/", "").replace("-", "").strip()
        currencies: list[str] = ["USD"]

        if any(c in clean for c in ("BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "BNB", "XRP")):
            base_coin = clean.replace("USDT", "").replace("USDC", "").replace("USD", "")
            return "crypto", [base_coin or "BTC", "USD"]

        if clean in ("XAUUSD", "GOLD", "XAU"):
            return "commodity", ["USD"]
        if clean in ("XAGUSD", "SILVER", "XAG"):
            return "commodity", ["USD"]
        if clean in ("WTI", "USOIL", "BRENT", "UKOIL"):
            return "commodity", ["USD"]

        if len(clean) == 6 and clean.isalpha():
            return "forex", [clean[:3], clean[3:]]

        return "general", ["USD"]

    async def get_composite_intelligence(
        self,
        symbol: str = "BTCUSDT",
        as_of: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Fetch from all active providers in parallel, deduplicate via SHA-256,
        score hybrid polarity, apply exponential time-decay, and compute
        composite sentiment score along with macro blackout gate status.
        """
        now = as_of or datetime.now(timezone.utc)
        asset_class, target_ccys = self.classify_asset_category(symbol)

        async with httpx.AsyncClient(
            headers={"User-Agent": "NexusTradeAI/2.0 InstitutionalEngine"},
            timeout=15.0,
        ) as client:
            # Query relevant providers concurrently
            tasks = [
                self.forexfactory.fetch_calendar(client),
                self.marctomarket.fetch(client),
                self.finnhub.fetch(
                    client,
                    category="crypto" if asset_class == "crypto" else "forex",
                    symbol=symbol,
                ),
            ]
            if self.alphavantage.api_key:
                tasks.append(self.alphavantage.fetch(client, ticker=symbol))
            if asset_class == "crypto" and self.cryptopanic.api_key:
                tasks.append(self.cryptopanic.fetch(client, currencies=target_ccys[0]))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        calendar_events: List[EconomicCalendarEvent] = (
            results[0] if isinstance(results[0], list) else []
        )

        raw_news: List[UnifiedNewsItem] = []
        for res in results[1:]:
            if isinstance(res, list):
                raw_news.extend(res)

        # Deduplicate by SHA-256 title hash
        deduped: Dict[str, UnifiedNewsItem] = {}
        for item in raw_news:
            h = item.compute_hash()
            if h not in deduped:
                deduped[h] = item
            else:
                # If duplicate exists, keep item with higher authority weight
                if item.authority_weight > deduped[h].authority_weight:
                    deduped[h] = item

        # Score items and apply exponential time-decay
        weighted_sum = 0.0
        total_weight = 0.0
        provider_counts: dict[str, int] = {}

        for item in deduped.values():
            provider_counts[item.source.value] = provider_counts.get(item.source.value, 0) + 1
            text_payload = f"{item.title}. {item.summary}".strip()
            fb_score = float(self.finbert_scorer(text_payload))
            item.finbert_sentiment = fb_score

            # Hybrid polarity fusion (PRD Section 5.2)
            if item.provider_sentiment is not None:
                if item.source == NewsSource.ALPHA_VANTAGE:
                    item.final_polarity = 0.60 * float(item.provider_sentiment) + 0.40 * fb_score
                elif item.source == NewsSource.CRYPTOPANIC:
                    item.final_polarity = 0.50 * float(item.provider_sentiment) + 0.50 * fb_score
                else:
                    item.final_polarity = fb_score
            else:
                item.final_polarity = fb_score

            item.final_polarity = max(-1.0, min(1.0, item.final_polarity))

            # Exponential time-decay (PRD Section 5.1)
            age_hours = max(0.0, (now - item.published_at).total_seconds() / 3600.0)
            half_life = 12.0 if item.source == NewsSource.MARC_TO_MARKET else 2.0
            lambda_val = math.log(2) / half_life
            decay = math.exp(-lambda_val * age_hours)

            w = float(item.authority_weight) * float(item.relevance_score) * decay
            weighted_sum += w * item.final_polarity
            total_weight += w

        composite_sentiment = (
            weighted_sum / (total_weight + 1e-8) if total_weight > 0 else 0.0
        )
        composite_sentiment = round(float(max(-1.0, min(1.0, composite_sentiment))), 4)

        # Economic Blackout Gatekeeper (PRD Section 5.4)
        window_mins = getattr(self.settings, "macro_blackout_window_minutes", 15)
        blackout_enabled = getattr(self.settings, "macro_blackout_enabled", True)
        is_blackout = False
        blackout_reason = None

        if blackout_enabled and calendar_events:
            is_blackout, blackout_reason = self.forexfactory.is_in_high_impact_blackout(
                currencies=target_ccys,
                events=calendar_events,
                window_minutes=window_mins,
                as_of=now,
            )

        top_headlines = [
            {
                "source": i.source.value,
                "title": i.title,
                "summary": i.summary[:140],
                "published_at": i.published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "polarity": round(i.final_polarity, 3),
            }
            for i in sorted(
                deduped.values(), key=lambda x: x.published_at, reverse=True
            )[:8]
        ]

        logger.info(
            "Computed multi-source news composite intelligence",
            symbol=symbol,
            composite_sentiment=composite_sentiment,
            article_count=len(deduped),
            is_macro_blackout=is_blackout,
            blackout_event=blackout_reason,
            providers=provider_counts,
        )

        return {
            "symbol": symbol.upper(),
            "composite_sentiment": composite_sentiment,
            "is_macro_blackout": is_blackout,
            "blackout_event": blackout_reason,
            "article_count": len(deduped),
            "provider_breakdown": provider_counts,
            "calendar_events_count": len(calendar_events),
            "top_headlines": top_headlines,
        }
    def get_composite_intelligence_sync(
        self,
        symbol: str = "BTCUSDT",
        as_of: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for get_composite_intelligence safe across any event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.get_composite_intelligence(symbol, as_of)
                ).result(timeout=18.0)
        else:
            return asyncio.run(self.get_composite_intelligence(symbol, as_of))
