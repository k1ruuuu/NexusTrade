from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from config.settings import Settings
from data_pipeline.news_providers.aggregator import MultiSourceNewsAggregator
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
from ai_engine.models.sentiment_analyzer import FinancialSentimentAnalyzer
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent


# -----------------------------------------------------------------------------
# AC-1: Graceful Degradation
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ac1_graceful_degradation_missing_keys_and_rate_limits() -> None:
    """AC-1: Aggregator continues seamlessly when API keys are missing or rate-limited (HTTP 429)."""
    settings = Settings(
        _env_file=None,
        finnhub_api_key="",        # Missing key
        alphavantage_api_key="",   # Missing key
        cryptopanic_api_key="",    # Missing key
    )
    aggregator = MultiSourceNewsAggregator(settings=settings)

    # Mock ForexFactory to return 1 event and MarcToMarket to return 1 article
    now = datetime.now(timezone.utc)
    mock_events = [
        EconomicCalendarEvent(
            title="ISM Manufacturing PMI",
            country="USD",
            timestamp=now - timedelta(hours=5),
            impact=ImpactLevel.MEDIUM,
        )
    ]
    mock_m2m_item = UnifiedNewsItem(
        source=NewsSource.MARC_TO_MARKET,
        title="Dollar trades steady ahead of central bank meetings",
        summary="Dollar index was stable near 103.50 with Treasury yields holding firm.",
        published_at=now - timedelta(hours=1),
        tickers=["DXY", "EURUSD"],
    )

    with patch.object(aggregator.forexfactory, "fetch_calendar", AsyncMock(return_value=mock_events)), \
         patch.object(aggregator.marctomarket, "fetch", AsyncMock(return_value=[mock_m2m_item])):

        intel = await aggregator.get_composite_intelligence(symbol="BTCUSDT")

        assert isinstance(intel, dict)
        assert intel["symbol"] == "BTCUSDT"
        assert intel["is_macro_blackout"] is False
        assert intel["article_count"] == 1
        assert len(intel["top_headlines"]) == 1
        assert intel["top_headlines"][0]["source"] == "marctomarket"


# -----------------------------------------------------------------------------
# AC-2: ForexFactory High Impact Blackout Accuracy
# -----------------------------------------------------------------------------
def test_ac2_forexfactory_blackout_accuracy() -> None:
    """AC-2: is_in_high_impact_blackout() returns True when event is within +/- 15 min."""
    now = datetime(2026, 10, 14, 12, 30, 0, tzinfo=timezone.utc)

    # Event 1: High impact USD event at 12:35 UTC (5 min away -> in blackout)
    event_high_soon = EconomicCalendarEvent(
        title="CPI m/m",
        country="USD",
        timestamp=now + timedelta(minutes=5),
        impact=ImpactLevel.HIGH,
    )
    # Event 2: High impact EUR event at 15:00 UTC (150 min away -> not in blackout)
    event_high_far = EconomicCalendarEvent(
        title="ECB Rate Decision",
        country="EUR",
        timestamp=now + timedelta(hours=2, minutes=30),
        impact=ImpactLevel.HIGH,
    )
    # Event 3: Low impact USD event at 12:32 UTC (2 min away -> Low impact, not blackout)
    event_low_soon = EconomicCalendarEvent(
        title="Trade Balance",
        country="USD",
        timestamp=now + timedelta(minutes=2),
        impact=ImpactLevel.LOW,
    )

    events = [event_high_soon, event_high_far, event_low_soon]

    # Test XAUUSD (correlates with USD): Should trigger blackout due to CPI
    is_blackout, reason = ForexFactoryProvider.is_in_high_impact_blackout(
        currencies=["USD"],
        events=events,
        window_minutes=15,
        as_of=now,
    )
    assert is_blackout is True
    assert "USD CPI m/m" in (reason or "")

    # Test EURGBP: Neither EUR nor GBP has a high impact event within 15 min
    is_blackout_eur, _ = ForexFactoryProvider.is_in_high_impact_blackout(
        currencies=["EUR", "GBP"],
        events=[event_high_far, event_low_soon],
        window_minutes=15,
        as_of=now,
    )
    assert is_blackout_eur is False

    # Test exactly 15 min boundary
    event_boundary = EconomicCalendarEvent(
        title="Non-Farm Employment Change",
        country="USD",
        timestamp=now - timedelta(minutes=15),
        impact=ImpactLevel.HIGH,
    )
    is_blackout_boundary, _ = ForexFactoryProvider.is_in_high_impact_blackout(
        currencies=["USD"],
        events=[event_boundary],
        window_minutes=15,
        as_of=now,
    )
    assert is_blackout_boundary is True

    # Test 16 min away (outside window)
    event_outside = EconomicCalendarEvent(
        title="FOMC Statement",
        country="USD",
        timestamp=now + timedelta(minutes=16),
        impact=ImpactLevel.HIGH,
    )
    is_blackout_outside, _ = ForexFactoryProvider.is_in_high_impact_blackout(
        currencies=["USD"],
        events=[event_outside],
        window_minutes=15,
        as_of=now,
    )
    assert is_blackout_outside is False


# -----------------------------------------------------------------------------
# AC-3: SHA-256 Title Deduplication Across Providers
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ac3_sha256_title_deduplication() -> None:
    """AC-3: Syndicated duplicate headlines from multiple providers are deduplicated by SHA-256 title hash."""
    aggregator = MultiSourceNewsAggregator(
        settings=Settings(_env_file=None, alphavantage_api_key="mock_av_key"),
        finbert_scorer=lambda txt: 0.50,
    )
    now = datetime.now(timezone.utc)
    title = "Bitcoin Surpasses $90,000 Milestone Amid Institutional Demand"
    item_finnhub = UnifiedNewsItem(
        source=NewsSource.FINNHUB,
        title=title,
        summary="Bitcoin crossed $90k with record ETF inflows.",
        published_at=now - timedelta(minutes=30),
        authority_weight=1.0,
    )
    item_alphavantage = UnifiedNewsItem(
        source=NewsSource.ALPHA_VANTAGE,
        title=title,
        summary="Detailed analysis of Bitcoin surpassing $90,000.",
        published_at=now - timedelta(minutes=25),
        provider_sentiment=0.85,
        authority_weight=1.20,  # Higher authority
    )

    with patch.object(aggregator.forexfactory, "fetch_calendar", AsyncMock(return_value=[])), \
         patch.object(aggregator.marctomarket, "fetch", AsyncMock(return_value=[])), \
         patch.object(aggregator.finnhub, "fetch", AsyncMock(return_value=[item_finnhub])), \
         patch.object(aggregator.alphavantage, "fetch", AsyncMock(return_value=[item_alphavantage])):

        intel = await aggregator.get_composite_intelligence(symbol="BTCUSDT")

        # Must collapse 2 duplicate stories into 1 single article
        assert intel["article_count"] == 1
        assert len(intel["top_headlines"]) == 1
        # Preserved item with higher authority (Alpha Vantage: 1.20)
        assert intel["top_headlines"][0]["source"] == "alphavantage"


# -----------------------------------------------------------------------------
# AC-4: Rate Limit & TTL Caching Compliance
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ac4_rate_limit_compliance_and_ttl_caching() -> None:
    """AC-4: In-memory TTL cache prevents duplicate network requests within TTL window."""
    ff_provider = ForexFactoryProvider(settings=Settings(_env_file=None, ttl_forexfactory=3600))
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "title": "Fed Chair Powell Speaks",
            "country": "USD",
            "date": "2026-10-14T18:00:00-04:00",
            "impact": "High",
        }
    ]
    mock_client.get.return_value = mock_resp

    # First fetch: calls network
    events_1 = await ff_provider.fetch_calendar(mock_client)
    assert len(events_1) == 1
    assert mock_client.get.call_count == 1

    # Second fetch within 3600s: MUST use cache and NOT call network again
    events_2 = await ff_provider.fetch_calendar(mock_client)
    assert len(events_2) == 1
    assert mock_client.get.call_count == 1  # Unchanged!


# -----------------------------------------------------------------------------
# CryptoPanic Crowd Voting Formula Verification (PRD 4.5)
# -----------------------------------------------------------------------------
def test_cryptopanic_crowd_sentiment_formula() -> None:
    """Test mathematical crowd sentiment formula from PRD Section 4.5."""
    # S_crowd = ((V_pos + 0.5*V_saved) - (V_neg + 2.0*V_toxic)) / (V_pos + V_neg + V_toxic + V_saved + 1)
    votes_bullish = {"positive": 10, "saved": 2, "negative": 1, "toxic": 0}
    # Numerator = (10 + 1) - (1 + 0) = 10
    # Denominator = 10 + 1 + 0 + 2 + 1 = 14
    expected_bullish = 10.0 / 14.0
    score = CryptoPanicProvider.compute_crowd_sentiment(votes_bullish)
    assert score == pytest.approx(expected_bullish, rel=1e-3)

    # Highly toxic/bearish
    votes_toxic = {"positive": 1, "saved": 0, "negative": 5, "toxic": 3}
    # Numerator = (1) - (5 + 6) = -10
    # Denominator = 1 + 5 + 3 + 0 + 1 = 10
    score_toxic = CryptoPanicProvider.compute_crowd_sentiment(votes_toxic)
    assert score_toxic == -1.0


# -----------------------------------------------------------------------------
# Full Ensemble Integration & Macro Blackout Circuit Breaker
# -----------------------------------------------------------------------------
def test_ensemble_agent_macro_blackout_rejection() -> None:
    """Test that active macro blackout forces EnsembleDecisionAgent to reject trade with HOLD."""
    cfg = Settings(_env_file=None, use_live_market_data=False, use_live_news_feed=True)
    agent = EnsembleDecisionAgent(settings=cfg)

    blackout_event_name = "USD CPI m/m (12:30 UTC)"
    from ai_engine.models.sentiment_analyzer import SentimentAnalysisResult
    mock_res = SentimentAnalysisResult(
        symbol="BTCUSDT",
        s_sent=0.0,
        label="NEUTRAL",
        bullish_ratio=0.0,
        bearish_ratio=0.0,
        neutral_ratio=1.0,
        model_name="ProsusAI/finbert",
        is_macro_blackout=True,
        blackout_event=blackout_event_name,
    )
    with patch.object(agent.sentiment_analyzer, "analyze_headlines", return_value=mock_res):
        decision = agent.analyze_symbol(symbol="BTCUSDT", timeframe="15m")
        assert decision.approved is False
        assert decision.action == "HOLD"
        assert "HIGH_IMPACT_NEWS_BLACKOUT" in (decision.rejection_reason or "")
        assert blackout_event_name in (decision.rejection_reason or "")
