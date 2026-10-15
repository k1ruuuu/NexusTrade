from __future__ import annotations

from data_pipeline.news_providers.aggregator import MultiSourceNewsAggregator
from data_pipeline.news_providers.alphavantage_provider import AlphaVantageProvider
from data_pipeline.news_providers.base_provider import BaseNewsProvider
from data_pipeline.news_providers.crypto_rss_provider import Tier1CryptoRSSProvider
from data_pipeline.news_providers.finnhub_provider import FinnhubProvider
from data_pipeline.news_providers.forexfactory_provider import ForexFactoryProvider
from data_pipeline.news_providers.marctomarket_provider import MarcToMarketProvider
from data_pipeline.news_providers.schemas import (
    EconomicCalendarEvent,
    ImpactLevel,
    NewsSource,
    UnifiedNewsItem,
)

__all__ = [
    "AlphaVantageProvider",
    "BaseNewsProvider",
    "EconomicCalendarEvent",
    "FinnhubProvider",
    "ForexFactoryProvider",
    "ImpactLevel",
    "MarcToMarketProvider",
    "MultiSourceNewsAggregator",
    "NewsSource",
    "Tier1CryptoRSSProvider",
    "UnifiedNewsItem",
]
