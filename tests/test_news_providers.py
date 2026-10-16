from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from config.settings import Settings
from data_pipeline.news_providers.aggregator import (
    MultiSourceNewsAggregator,
    BULLISH_CATALYSTS,
    BEARISH_CATALYSTS,
)
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
from ai_engine.models.sentiment_analyzer import (
    FinancialSentimentAnalyzer,
    SentimentAnalysisResult,
)
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent


# -----------------------------------------------------------------------------
# AC-1: Zero Paywall / Zero API Key for Tier-1 Crypto RSS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ac1_zero_paywall_tier1_crypto_rss() -> None:
    """AC-1: fetch() concurrently fetches and parses XML feeds with zero API keys required."""
    provider = Tier1CryptoRSSProvider(settings=Settings(_env_file=None))
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # Simulated valid RSS XML from CoinDesk & CoinTelegraph
    sample_rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>CoinDesk News</title>
        <item>
          <title>Bitcoin Breaches Record High as Institutional Inflow Surges</title>
          <description>Spot Bitcoin ETF inflows accelerated today with heavy volume.</description>
          <link>https://coindesk.com/article1</link>
          <pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_rss_xml
    mock_client.get.return_value = mock_resp

    items = await provider.fetch(mock_client, target_symbol="BTC")

    assert len(items) >= 1
    item = items[0]
    assert item.source == NewsSource.TIER1_CRYPTO_RSS
    assert "Bitcoin" in item.title
    assert "BTC" in item.tickers
    assert item.relevance_score == 1.0
    assert item.authority_weight == 1.10


# -----------------------------------------------------------------------------
# AC-2: Sub-2.5-Second Parallel Execution
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ac2_sub_2_5_second_parallel_execution() -> None:
    """AC-2: All providers resolve concurrently in < 2.5 seconds via asyncio.gather."""
    aggregator = MultiSourceNewsAggregator(
        settings=Settings(_env_file=None),
        finbert_scorer=lambda txt: 0.10,
    )

    async def simulated_delayed_fetch(*args, **kwargs):
        await asyncio.sleep(0.08)  # Each provider simulates ~80ms network latency
        return []

    with patch.object(aggregator.forexfactory, "fetch_calendar", side_effect=simulated_delayed_fetch), \
         patch.object(aggregator.marctomarket, "fetch", side_effect=simulated_delayed_fetch), \
         patch.object(aggregator.finnhub, "fetch", side_effect=simulated_delayed_fetch), \
         patch.object(aggregator.crypto_rss, "fetch", side_effect=simulated_delayed_fetch):

        start_time = time.perf_counter()
        intel = await aggregator.get_composite_intelligence(symbol="BTCUSDT")
        duration = time.perf_counter() - start_time

        assert isinstance(intel, dict)
        assert duration < 2.5, f"Parallel execution took {duration:.2f}s, expected < 2.5s"


# -----------------------------------------------------------------------------
# AC-3: Lexical Black-Swan & Bullish Catalyst Detection
# -----------------------------------------------------------------------------
def test_ac3_lexical_black_swan_and_catalyst_detection() -> None:
    """AC-3: High-impact keywords trigger deterministic lexical modifier (e.g., -0.35 for hacked/exploit)."""
    # 1. Bearish Black Swan Catalysts (-0.35)
    bearish_sample = "Major DeFi protocol hacked and bridge drained of $150M in flash loan exploit"
    mod_bear = MultiSourceNewsAggregator.compute_lexical_modifier(bearish_sample)
    assert mod_bear <= -0.35

    sec_sample = "SEC lawsuit alleges unregistered security offering against major exchange"
    mod_sec = MultiSourceNewsAggregator.compute_lexical_modifier(sec_sample)
    assert mod_sec <= -0.35

    # 2. Bullish Catalysts (+0.25)
    bullish_sample = "SEC confirms spot ETF approved with historic institutional inflow"
    mod_bull = MultiSourceNewsAggregator.compute_lexical_modifier(bullish_sample)
    assert mod_bull >= 0.25

    # 3. Neutral / General headline (0.0)
    neutral_sample = "Markets open mixed in European morning trading"
    mod_neutral = MultiSourceNewsAggregator.compute_lexical_modifier(neutral_sample)
    assert mod_neutral == 0.0


# -----------------------------------------------------------------------------
# AC-4: SHA-256 Cross-Feed Deduplication
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ac4_sha256_cross_feed_deduplication() -> None:
    """AC-4: Syndicated stories appearing simultaneously on CoinDesk and Finnhub are deduplicated."""
    aggregator = MultiSourceNewsAggregator(
        settings=Settings(_env_file=None, finnhub_api_key="mock_key"),
        finbert_scorer=lambda txt: 0.50,
    )
    now = datetime.now(timezone.utc)

    # Identical syndicated headline on CoinDesk (Tier-1 RSS) and Finnhub (REST)
    title = "Bitcoin Surpasses $90,000 Milestone Amid Historic Inflows"
    item_coindesk = UnifiedNewsItem(
        source=NewsSource.TIER1_CRYPTO_RSS,
        publisher="CoinDesk",
        title=title,
        summary="Detailed report on Bitcoin surpassing 90k.",
        published_at=now - timedelta(minutes=20),
        authority_weight=1.10,
    )
    item_finnhub = UnifiedNewsItem(
        source=NewsSource.FINNHUB,
        publisher="Finnhub",
        title=title,
        summary="Wire report on Bitcoin surpassing 90k.",
        published_at=now - timedelta(minutes=25),
        authority_weight=1.00,
    )

    with patch.object(aggregator.forexfactory, "fetch_calendar", AsyncMock(return_value=[])), \
         patch.object(aggregator.marctomarket, "fetch", AsyncMock(return_value=[])), \
         patch.object(aggregator.finnhub, "fetch", AsyncMock(return_value=[item_finnhub])), \
         patch.object(aggregator.crypto_rss, "fetch", AsyncMock(return_value=[item_coindesk])):

        intel = await aggregator.get_composite_intelligence(symbol="BTCUSDT")

        # Must collapse duplicate stories into 1 single article
        assert intel["article_count"] == 1
        assert len(intel["top_headlines"]) == 1
        # Kept CoinDesk due to higher authority (1.10 > 1.00)
        assert intel["top_headlines"][0]["source"] == "CoinDesk"


# -----------------------------------------------------------------------------
# ForexFactory High-Impact Blackout & Circuit Breaker Tests
# -----------------------------------------------------------------------------
def test_forexfactory_blackout_logic() -> None:
    """Verify is_in_high_impact_blackout triggers within +/- 15 min of High impact events."""
    now = datetime(2026, 10, 14, 12, 30, 0, tzinfo=timezone.utc)

    event_high_cpi = EconomicCalendarEvent(
        title="CPI m/m",
        country="USD",
        timestamp=now + timedelta(minutes=5),  # 5 min away
        impact=ImpactLevel.HIGH,
    )
    event_low = EconomicCalendarEvent(
        title="Wholesale Inventories",
        country="USD",
        timestamp=now + timedelta(minutes=2),
        impact=ImpactLevel.LOW,
    )

    is_blackout, reason = ForexFactoryProvider.is_in_high_impact_blackout(
        currencies=["USD"],
        events=[event_high_cpi, event_low],
        window_minutes=15,
        as_of=now,
    )
    assert is_blackout is True
    assert "USD CPI m/m" in (reason or "")


def test_ensemble_agent_blackout_rejection() -> None:
    """Verify active macro blackout forces EnsembleDecisionAgent to reject trade with HOLD."""
    cfg = Settings(_env_file=None, use_live_market_data=False, use_live_news_feed=True)
    agent = EnsembleDecisionAgent(settings=cfg)

    blackout_event_name = "USD CPI m/m (12:30 UTC)"
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


# -----------------------------------------------------------------------------
# Tests for US Macro News, Hawkish vs Dovish Engine, and Event Sniper
# -----------------------------------------------------------------------------
from ai_engine.models.sentiment_lexicon import (
    compute_macro_monetary_bias,
    macro_monetary_to_gold_signal,
)
from data_pipeline.news_providers.us_macro_rss_provider import USMacroNewsProvider
from ai_engine.auto_trader.session_models import parse_macro_event_mode_token


def test_hawkish_vs_dovish_monetary_transmission_to_gold() -> None:
    """
    Verify institutional macro transmission logic:
      - Strong US Inflation / Jobs (Hawkish) -> USD rallies -> Gold (XAUUSD) drops (SELL)
      - Weak US Inflation / Jobs (Dovish) -> USD drops -> Gold (XAUUSD) rallies (BUY)
    """
    # Test 1: Hot PPI / CPI inflation beat -> Hawkish -> SELL Gold
    hawkish_headline = "US Core PPI jumps 0.4% MoM beating forecasts on persistent sticky inflation, yields surge"
    score_h, stance_h = compute_macro_monetary_bias(hawkish_headline)
    gold_score_h, action_h = macro_monetary_to_gold_signal(score_h, "XAUUSD")
    assert stance_h == "HAWKISH"
    assert score_h > 0.0
    assert gold_score_h < 0.0
    assert action_h == "SELL"

    # Test 2: Cool CPI / NFP disappointment -> Dovish -> BUY Gold
    dovish_headline = "US Consumer Price Index cools sharply as jobless claims jump, fueling aggressive Fed rate cuts"
    score_d, stance_d = compute_macro_monetary_bias(dovish_headline)
    gold_score_d, action_d = macro_monetary_to_gold_signal(score_d, "XAUUSD")
    assert stance_d == "DOVISH"
    assert score_d < 0.0
    assert gold_score_d > 0.0
    assert action_d == "BUY"


@pytest.mark.asyncio
async def test_us_macro_rss_provider_parsing_and_tagging() -> None:
    """Verify USMacroNewsProvider parses breaking RSS feeds and tags macro events (CPI, PPI, NFP)."""
    provider = USMacroNewsProvider(settings=Settings(_env_file=None))
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    sample_forexlive_rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>ForexLive US Macro News</title>
        <item>
          <title>US Core PPI jumps 0.3% vs 0.1% expected as producer price inflation accelerates</title>
          <description>Hot producer prices indicate stubborn wholesale price pressures for the Fed.</description>
          <link>https://www.forexlive.com/news/us-ppi-heat-2026</link>
          <pubDate>Thu, 15 Oct 2026 12:30:00 +0000</pubDate>
        </item>
        <item>
          <title>Non-farm payrolls miss expectations with 95k added as labor market cools</title>
          <description>US employment slowdown prompts markets to price in 50bp rate cut.</description>
          <link>https://www.forexlive.com/news/nfp-miss-2026</link>
          <pubDate>Thu, 15 Oct 2026 12:31:00 +0000</pubDate>
        </item>
      </channel>
    </rss>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_forexlive_rss
    mock_client.get.return_value = mock_resp

    items = await provider.fetch(mock_client, target_event="PPI")
    assert len(items) >= 2
    # First item is PPI
    ppi_item = next(it for it in items if "PPI" in it.tickers)
    assert ppi_item.publisher == "USMacro_ForexLive"
    assert "[HAWKISH]" in ppi_item.title
    # Transmitted polarity to Gold should be negative (SELL Gold)
    assert ppi_item.provider_sentiment is not None and ppi_item.provider_sentiment < 0


def test_event_sniper_blackout_bypass_logic() -> None:
    """Verify is_in_high_impact_blackout bypasses blackout when user targets specific event mode."""
    now = datetime.now(timezone.utc)
    event_ppi = EconomicCalendarEvent(
        title="Core PPI m/m",
        country="USD",
        timestamp=now + timedelta(minutes=5),  # 5 min away
        impact=ImpactLevel.HIGH,
    )
    # Standard call without bypass -> True (Blackout active)
    is_blackout, reason = ForexFactoryProvider.is_in_high_impact_blackout(
        currencies=["USD"],
        events=[event_ppi],
        window_minutes=15,
        as_of=now,
    )
    assert is_blackout is True
    assert "Core PPI" in (reason or "")

    # Event Sniper call with bypassed_event_mode="PPI" -> False (Blackout Bypassed!)
    is_blackout_sniper, reason_sniper = ForexFactoryProvider.is_in_high_impact_blackout(
        currencies=["USD"],
        events=[event_ppi],
        window_minutes=15,
        as_of=now,
        bypassed_event_mode="PPI",
    )
    assert is_blackout_sniper is False
    assert reason_sniper is None


def test_parse_macro_event_mode_tokens() -> None:
    """Verify parsing /autotrade event arguments (NFP, CPI, PPI, PCE)."""
    assert parse_macro_event_mode_token("nfp") == "NFP"
    assert parse_macro_event_mode_token("CPI") == "CPI"
    assert parse_macro_event_mode_token("ppi") == "PPI"
    assert parse_macro_event_mode_token("event:pce") == "PCE"
    assert parse_macro_event_mode_token("BTCUSDT") is None
    assert parse_macro_event_mode_token("0.05") is None
