from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.alphavantage_provider import AlphaVantageProvider
from data_pipeline.news_providers.crypto_rss_provider import Tier1CryptoRSSProvider
from data_pipeline.news_providers.finnhub_provider import FinnhubProvider
from data_pipeline.news_providers.forexfactory_provider import ForexFactoryProvider
from data_pipeline.news_providers.marctomarket_provider import MarcToMarketProvider
from data_pipeline.news_providers.us_macro_rss_provider import USMacroNewsProvider
from data_pipeline.news_providers.schemas import (
    EconomicCalendarEvent,
    ImpactLevel,
    NewsSource,
    UnifiedNewsItem,
)
from ai_engine.models.sentiment_lexicon import (
    compute_macro_monetary_bias,
    macro_monetary_to_gold_signal,
)

logger = get_logger("nexustrade.news_providers.aggregator")

BULLISH_CATALYSTS = (
    "etf approved",
    "institutional inflow",
    "strategic reserve",
    "rate cut",
    "mainnet launch",
    "short squeeze",
    "all-time high",
    "adoption",
)

BEARISH_CATALYSTS = (
    "hacked",
    "exploit",
    "bridge drained",
    "sec lawsuit",
    "insolvency",
    "withdrawals halted",
    "bankruptcy",
    "delisted",
    "rug pull",
)


class MultiSourceNewsAggregator:
    """
    Asynchronous 6-Provider News, Macro Commentary & Economic Calendar Engine (v2.5).
    Integrates ForexFactory, MarcToMarket, Finnhub, Alpha Vantage, Tier-1 Crypto Media RSS,
    and Real-Time US Macro Squawk Feeds (ForexLive, CNBC, Investing.com).
    Features institutional Hawkish vs Dovish monetary transmission to Gold (XAUUSD).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        finbert_scorer: Optional[Callable[[str], float]] = None,
        finnhub_key: Optional[str] = None,
        alphavantage_key: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.finbert_scorer = finbert_scorer or (lambda text: 0.0)

        # Provider instances
        self.forexfactory = ForexFactoryProvider(self.settings)
        self.marctomarket = MarcToMarketProvider(self.settings)
        self.finnhub = FinnhubProvider(
            api_key=finnhub_key if finnhub_key is not None else getattr(self.settings, "finnhub_api_key", ""),
            settings=self.settings,
        )
        self.alphavantage = AlphaVantageProvider(
            api_key=alphavantage_key if alphavantage_key is not None else getattr(self.settings, "alphavantage_api_key", ""),
            settings=self.settings,
        )
        self.crypto_rss = Tier1CryptoRSSProvider(self.settings)
        self.us_macro = USMacroNewsProvider(self.settings)

    @staticmethod
    def compute_lexical_modifier(text: str) -> float:
        """
        PRD Section 5.2 Deterministic Lexical Black-Swan / Catalyst Modifier.
        Returns a boost/penalty in [-0.40, +0.40].
        """
        lower = text.lower()
        mod = 0.0
        for kw in BULLISH_CATALYSTS:
            if kw in lower:
                mod += 0.25
        for kw in BEARISH_CATALYSTS:
            if kw in lower:
                mod -= 0.35
        return max(-0.40, min(0.40, mod))

    @staticmethod
    def classify_asset_category(symbol: str) -> Tuple[str, List[str]]:
        """Identify asset class and extract currency tags."""
        clean = (
            symbol.upper()
            .replace("/", "")
            .replace("-", "")
            .replace("PERP", "")
            .strip()
        )
        currencies: List[str] = ["USD"]

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
        event_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Poll providers concurrently, deduplicate by alphanumeric SHA-256,
        apply lexical black-swan modifier, compute exponential time-decay, and
        calculate composite sentiment and economic calendar blackout gate.
        If event_mode is provided (e.g. 'CPI', 'PPI', 'NFP'), blackout is bypassed
        for that event and US Macro Squawk feeds establish Hawkish vs Dovish stance.
        """
        now = as_of or datetime.now(timezone.utc)
        asset_class, target_ccys = self.classify_asset_category(symbol)
        clean_base = target_ccys[0]

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; NexusTradeAI/2.0)"},
            timeout=15.0,
        ) as client:
            av_ticker = f"CRYPTO:{clean_base}" if asset_class == "crypto" else f"FOREX:{clean_base}"
            fh_cat = "crypto" if asset_class == "crypto" else "forex"

            tasks = [
                self.forexfactory.fetch_calendar(client),
                self.marctomarket.fetch(client),
                self.finnhub.fetch(client, category=fh_cat, symbol=symbol),
            ]
            if self.alphavantage.api_key:
                tasks.append(self.alphavantage.fetch(client, ticker=av_ticker))
            if asset_class == "crypto":
                tasks.append(self.crypto_rss.fetch(client, target_symbol=clean_base))
            if asset_class != "crypto" or event_mode or any(k in symbol.upper() for k in ("XAU", "GOLD")):
                tasks.append(self.us_macro.fetch(client, target_event=event_mode))
            results = await asyncio.gather(*tasks, return_exceptions=True)

        calendar_events: List[EconomicCalendarEvent] = (
            results[0] if isinstance(results[0], list) else []
        )
        raw_news: List[UnifiedNewsItem] = []
        for res in results[1:]:
            if isinstance(res, list):
                raw_news.extend(res)

        # Deduplicate syndicated headlines across RSS & APIs by SHA-256 hash
        deduped: Dict[str, UnifiedNewsItem] = {}
        for item in raw_news:
            h = item.compute_hash()
            if h not in deduped:
                deduped[h] = item
            else:
                if item.authority_weight > deduped[h].authority_weight:
                    deduped[h] = item

        # Score with FinBERT + Lexical Modifier and apply exponential time-decay
        weighted_sum = 0.0
        total_weight = 0.0
        provider_counts: Dict[str, int] = {}

        for item in deduped.values():
            pub_tag = item.publisher or item.source.value
            provider_counts[pub_tag] = provider_counts.get(pub_tag, 0) + 1

            text_payload = f"{item.title}. {item.summary}".strip()
            fb_raw = float(self.finbert_scorer(text_payload))
            lex_mod = self.compute_lexical_modifier(text_payload)
            fb_adjusted = max(-1.0, min(1.0, fb_raw + lex_mod))

            item.finbert_sentiment = fb_raw
            item.lexical_modifier = lex_mod

            if item.provider_sentiment is not None and "USMacro" in pub_tag:
                # Institutional Hawkish vs Dovish transmission to Gold
                item.final_polarity = float(item.provider_sentiment)
            elif item.provider_sentiment is not None and item.source == NewsSource.ALPHA_VANTAGE:
                item.final_polarity = 0.60 * float(item.provider_sentiment) + 0.40 * fb_adjusted
            else:
                item.final_polarity = fb_adjusted

            item.final_polarity = max(-1.0, min(1.0, item.final_polarity))
            # Exponential time-decay
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

        # Economic Blackout Gatekeeper
        window_mins = getattr(self.settings, "macro_blackout_window_minutes", 15)
        blackout_enabled = getattr(self.settings, "macro_blackout_enabled", False)
        is_blackout = False
        blackout_reason = None

        if blackout_enabled and calendar_events:
            is_blackout, blackout_reason = self.forexfactory.is_in_high_impact_blackout(
                currencies=target_ccys,
                events=calendar_events,
                window_minutes=window_mins,
                as_of=now,
                bypassed_event_mode=event_mode,
            )

        # Institutional Macro Stance Calculation (Hawkish vs Dovish)
        macro_items = [i for i in deduped.values() if "USMacro" in (i.publisher or "")]
        if macro_items:
            avg_macro_gold = sum(i.final_polarity for i in macro_items) / len(macro_items)
        else:
            avg_macro_gold = composite_sentiment

        if avg_macro_gold <= -0.15:
            macro_stance = "HAWKISH"
            gold_action_bias = "SELL"
        elif avg_macro_gold >= 0.15:
            macro_stance = "DOVISH"
            gold_action_bias = "BUY"
        else:
            macro_stance = "NEUTRAL"
            gold_action_bias = "HOLD"
        top_headlines = [
            {
                "source": i.publisher or i.source.value,
                "title": i.title,
                "summary": i.summary[:140],
                "published_at": i.published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "polarity": round(i.final_polarity, 3),
                "relevance": round(i.relevance_score, 2),
                "lexical_modifier": round(i.lexical_modifier, 2),
            }
            for i in sorted(
                deduped.values(), key=lambda x: x.published_at, reverse=True
            )[:8]
        ]

        logger.info(
            "Computed 5-provider composite intelligence (v2.0 Tier-1 RSS)",
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
            "macro_stance": macro_stance,
            "gold_action_bias": gold_action_bias,
            "event_mode": event_mode.upper() if event_mode else None,
            "article_count": len(deduped),
            "provider_breakdown": provider_counts,
            "calendar_events_count": len(calendar_events),
            "top_headlines": top_headlines,
        }

    def get_composite_intelligence_sync(
        self,
        symbol: str = "BTCUSDT",
        as_of: Optional[datetime] = None,
        event_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for get_composite_intelligence safe across running event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.get_composite_intelligence(symbol, as_of, event_mode=event_mode)
                ).result(timeout=18.0)
        else:
            return asyncio.run(self.get_composite_intelligence(symbol, as_of, event_mode=event_mode))
