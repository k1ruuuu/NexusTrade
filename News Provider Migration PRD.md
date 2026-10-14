# PRODUCT REQUIREMENTS DOCUMENT (PRD): Multi-Source Real-Time News & Macroeconomic Sentiment Ingestion Engine

## 1. Document Metadata & Executive Summary

* **Project Name:** NexusTrade AI
* **Feature / Module:** Real-Time Multi-Provider News, Macro Commentary & Economic Calendar Engine
* **Target Directory:** `data_pipeline/news_providers/` & `ai_engine/models/sentiment_analyzer.py`
* **Objective:** Deprecate static Kaggle-only news text ingestion during live inference and replace it with an asynchronous, fault-tolerant, 5-provider real-time ingestion pipeline (**ForexFactory**, **MarcToMarket**, **Finnhub.io**, **Alpha Vantage**, and **CryptoPanic**).

---

## 2. Background & Problem Statement

### 2.1 Current State (Legacy Architecture)
Currently, `NexusTrade AI` relies on static historical Kaggle datasets (`ankurzing/sentiment-analysis-for-financial-news` and `miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests`) for training and simulating sentiment scores. While sufficient for offline FinBERT fine-tuning, static datasets provide zero visibility into live market catalysts, breaking geopolitical headlines, or scheduled macroeconomic volatility events (e.g., CPI, NFP, FOMC rate decisions).

### 2.2 Target State (New Architecture)
Implement a unified, asynchronous `MultiSourceNewsAggregator` that continuously polls, normalizes, deduplicates, and scores real-time data from 5 specialized providers:
1. **ForexFactory (Fair Economy JSON):** Scheduled macroeconomic calendar & volatility blackout windows.
2. **MarcToMarket (Atom/RSS JSON Feed):** Institutional FX & global macro narrative analysis.
3. **Finnhub.io REST API:** Real-time breaking headlines across Forex, Crypto, and General markets.
4. **Alpha Vantage News & Sentiment API:** Pre-scored institutional news & ticker-specific sentiment vectors.
5. **CryptoPanic REST API:** Real-time crypto-native aggregator with community bull/bear voting metrics.

---

## 3. System Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        5 LIVE EXTERNAL PROVIDERS                           │
├──────────────┬───────────────┬──────────────┬───────────────┬──────────────┤
│ ForexFactory │ MarcToMarket  │  Finnhub.io  │ Alpha Vantage │ CryptoPanic  │
│ (JSON Cal)   │ (RSS/JSON)    │  (REST API)  │  (REST API)   │ (REST API)   │
└──────┬───────┴───────┬───────┴──────┬───────┴───────┬───────┴──────┬───────┘
       │               │              │               │              │
       ▼               ▼              ▼               ▼              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│             Async Ingestion & Rate-Limiting Layer (httpx + Redis)          │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│        Schema Normalization & SHA-256 Deduplication (UnifiedNewsItem)      │
└──────────────────┬──────────────────────────────────┬──────────────────────┘
                   │                                  │
                   ▼                                  ▼
┌────────────────────────────────────┐ ┌─────────────────────────────────────┐
│   Economic Blackout Gatekeeper     │ │   Hybrid Sentiment Scoring Engine   │
│ (Halts M1 Trading ±15m High Impact)│ │ (FinBERT + Provider Native Scores)  │
└──────────────────┬─────────────────┘ └──────────────┬──────────────────────┘
                   │                                  │
                   └──────────────────┬───────────────┘
                                      ▼
                    [ AI Ensemble Agent & Telegram Bot ]
```

---

## 4. Detailed Provider Specifications

### 4.1 Provider 1: ForexFactory (Fair Economy Calendar)
* **Role:** Macroeconomic Event Gatekeeper & Volatility Filter.
* **Endpoint:** `https://nfs.faireconomy.media/ff_calendar_thisweek.json`
* **Authentication:** None (Public CDN endpoint).
* **Rate Limit Constraint:** Strictly poll maximum once every 60 minutes (`TTL = 3600s` in Redis). Exceeding this causes IP bans.
* **Key Fields Extracted:** `title`, `country` (Currency code: `USD`, `EUR`, `GBP`, `JPY`, etc.), `date` (ISO-8601), `impact` (`High`, `Medium`, `Low`), `forecast`, `previous`.
* **Trading Logic:** Triggers a **Hard Circuit Breaker** (Execution Blackout) on correlated pairs $\pm 15$ minutes around any event where `impact == "High"`.

### 4.2 Provider 2: MarcToMarket (Macro & FX Commentary)
* **Role:** Global Macro Regime & DXY (US Dollar Index) Bias Context.
* **Endpoint:** `https://www.marctomarket.com/feeds/posts/default?alt=json` (Fallback: `?alt=rss`).
* **Authentication:** None.
* **Polling Interval:** Every 30 minutes (`TTL = 1800s`).
* **Processing Logic:** Strip HTML tags from `entry.content.$t`, split into paragraphs, filter paragraphs mentioning target currencies (`USD`, `EUR`, `JPY`, `Fed`, `ECB`, `BOJ`, `tariffs`, `yields`), and pass through **FinBERT** (`ProsusAI/finbert`) to derive a daily Macro Regime Bias score.

### 4.3 Provider 3: Finnhub.io API
* **Role:** Low-latency breaking market news (General, Forex, Crypto).
* **Endpoints:**
  * Market News: `GET https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}` (where `category` $\in \{\text{general}, \text{forex}, \text{crypto}\}$).
  * Company News: `GET https://finnhub.io/api/v1/company-news?symbol={symbol}&from={YYYY-MM-DD}&to={YYYY-MM-DD}&token={FINNHUB_API_KEY}`
* **Rate Limit:** 60 calls / minute (Free Tier). Poll every 2 minutes (`TTL = 120s`).
* **Processing Logic:** Extract `headline` and `summary`, score locally via **FinBERT**.

### 4.4 Provider 4: Alpha Vantage (News & Sentiment API)
* **Role:** Pre-computed AI Sentiment & Relevance Scoring.
* **Endpoint:** `GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&topics={topics}&apikey={ALPHAVANTAGE_API_KEY}`
  * Supported `tickers`: `AAPL`, `TSLA`, `CRYPTO:BTC`, `FOREX:EUR`, `FOREX:USD`, etc.
  * Supported `topics`: `economy_monetary`, `economy_macro`, `financial_markets`, `blockchain`.
* **Rate Limit:** 25 requests / day on standard free tier (or 75+ req/min on premium). Cache responses aggressively (`TTL = 1800s` to `3600s`).
* **Key Fields Extracted:** `overall_sentiment_score` $\in [-1.0, 1.0]$, `ticker_sentiment_score`, and `relevance_score` $\in [0.0, 1.0]$.

### 4.5 Provider 5: CryptoPanic API
* **Role:** Real-time Cryptocurrency News & Crowd Sentiment Votes.
* **Endpoint:** `GET https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&currencies={BTC,ETH,SOL}&filter=important&public=true`
* **Rate Limit:** 5 requests / minute (Free Tier). Poll every 3 minutes (`TTL = 180s`).
* **Sentiment Extraction:** Compute crowd polarity directly from the `votes` object (`positive`, `negative`, `important`, `toxic`, `lol`, `saved`):
  $$S_{\text{crowd}} = \frac{(V_{\text{pos}} + 0.5 \cdot V_{\text{saved}}) - (V_{\text{neg}} + 2.0 \cdot V_{\text{toxic}})}{V_{\text{pos}} + V_{\text{neg}} + V_{\text{toxic}} + V_{\text{saved}} + 1}$$

---

## 5. Mathematical Formulation for Multi-Source Sentiment Fusion

### 5.1 Exponential Time-Decay Weighting
News impact decays exponentially over time. For any news item $i$ published at timestamp $t_i$ evaluated at current time $t_{\text{now}}$:
$$\Delta t_i = \frac{t_{\text{now}} - t_i}{3600} \quad (\text{in hours})$$
$$w_{\text{decay}}(\Delta t_i) = \exp(-\lambda \cdot \Delta t_i), \quad \text{where } \lambda = \frac{\ln(2)}{\tau_{1/2}}$$
* For intraday scalping ($M1/M15$), half-life $\tau_{1/2} = 2.0\text{ hours}$ ($\lambda \approx 0.3466$).
* For macro commentary (`MarcToMarket`), half-life $\tau_{1/2} = 12.0\text{ hours}$ ($\lambda \approx 0.0578$).

### 5.2 Provider Reliability & Hybrid Polarity Score
Each news item $i$ receives a unified polarity score $p_i \in [-1.0, +1.0]$:
* If sourced from **Alpha Vantage**:
  $$p_i = 0.60 \cdot S_{\text{AV}, i} + 0.40 \cdot S_{\text{FinBERT}, i}$$
* If sourced from **CryptoPanic**:
  $$p_i = 0.50 \cdot S_{\text{crowd}, i} + 0.50 \cdot S_{\text{FinBERT}, i}$$
* If sourced from **Finnhub** or **MarcToMarket**:
  $$p_i = S_{\text{FinBERT}, i} = P(\text{Positive}) - P(\text{Negative})$$

### 5.3 Composite Asset Sentiment Score ($S_{\text{sent}}$)
Given $N$ deduplicated news items for a target asset over the last 24 hours, with provider authority weights $\omega_s$ ($\text{AlphaVantage}=1.2$, $\text{MarcToMarket}=1.15$, $\text{Finnhub}=1.0$, $\text{CryptoPanic}=0.95$):
$$S_{\text{sent}}(t) = \frac{\sum_{i=1}^{N} \omega_{s(i)} \cdot r_i \cdot w_{\text{decay}}(\Delta t_i) \cdot p_i}{\sum_{i=1}^{N} \omega_{s(i)} \cdot r_i \cdot w_{\text{decay}}(\Delta t_i) + \epsilon} \in [-1.0, +1.0]$$
Where $r_i \in [0, 1]$ is the asset relevance score.

### 5.4 Macroeconomic Event Blackout Gate ($G_{\text{macro}}$)
Let $\mathcal{E}_{\text{high}}$ be the set of ForexFactory events where `impact == "High"` and currency matches either base or quote currency of the traded pair:
$$G_{\text{macro}}(t) = \begin{cases} 0 \ (\text{BLOCK EXECUTION}), & \text{if } \exists e \in \mathcal{E}_{\text{high}} \text{ s.t. } -15\text{ min} \le (t - t_e) \le +15\text{ min} \\ 1 \ (\text{ALLOW EXECUTION}), & \text{otherwise} \end{cases}$$

---

## 6. Directory & File Migration Plan

The AI Coding Agent must execute the following structural changes:

### 6.1 Files to Deprecate / Refactor
* **Modify** `data_pipeline/kaggle_downloader.py`: Keep `ankurzing/...` and `miguelaenlle/...` strictly under an `--offline-training` CLI flag for FinBERT weight calibration only. Remove them from live runtime inference.
* **Refactor** `ai_engine/models/sentiment_analyzer.py`: Replace static CSV lookup with live `MultiSourceNewsAggregator` calls.

### 6.2 New Files to Create
```
nexustrade-ai/
├── config/
│   └── settings.py                        # Add API keys & polling TTL configs
├── data_pipeline/
│   └── news_providers/
│       ├── __init__.py
│       ├── schemas.py                     # UnifiedNewsItem & EconomicEvent Pydantic models
│       ├── base_provider.py               # Abstract async base class with retry & Redis cache
│       ├── forexfactory_provider.py       # Fair Economy JSON calendar client
│       ├── marctomarket_provider.py       # Atom/RSS JSON macro commentary parser
│       ├── finnhub_provider.py            # Finnhub REST client
│       ├── alphavantage_provider.py       # Alpha Vantage News & Sentiment client
│       ├── cryptopanic_provider.py        # CryptoPanic REST & vote polarity client
│       └── aggregator.py                  # MultiSourceNewsAggregator & deduplicator
```

---

## 7. Production Python Implementation Blueprint

### 7.1 Environment Variables (`.env.example` Update)
```env
# --- Real-Time News & Macro Providers ---
FINNHUB_API_KEY="your_finnhub_api_key_here"
ALPHAVANTAGE_API_KEY="your_alphavantage_api_key_here"
CRYPTOPANIC_API_KEY="your_cryptopanic_api_key_here"

# Public Feeds (No API Key Required)
FOREXFACTORY_CALENDAR_URL="https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MARCTOMARKET_FEED_URL="https://www.marctomarket.com/feeds/posts/default?alt=json"

# Polling & Cache TTLs (Seconds)
TTL_FOREXFACTORY=3600
TTL_MARCTOMARKET=1800
TTL_FINNHUB=120
TTL_ALPHAVANTAGE=1800
TTL_CRYPTOPANIC=180
```

### 7.2 Unified Data Schemas (`data_pipeline/news_providers/schemas.py`)
```python
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import hashlib


class NewsSource(str, Enum):
    FOREX_FACTORY = "forexfactory"
    MARC_TO_MARKET = "marctomarket"
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alphavantage"
    CRYPTOPANIC = "cryptopanic"


class ImpactLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    HOLIDAY = "Holiday"


class EconomicCalendarEvent(BaseModel):
    title: str
    country: str
    timestamp: datetime
    impact: ImpactLevel
    forecast: Optional[str] = None
    previous: Optional[str] = None


class UnifiedNewsItem(BaseModel):
    item_id: str = Field(default="")
    source: NewsSource
    title: str
    summary: str = ""
    url: str = ""
    published_at: datetime
    tickers: List[str] = Field(default_factory=list)
    provider_sentiment: Optional[float] = None  # [-1.0, 1.0] if native score exists
    finbert_sentiment: Optional[float] = None   # [-1.0, 1.0] computed locally
    final_polarity: float = 0.0                 # Combined score
    relevance_score: float = 1.0                # [0.0, 1.0]
    authority_weight: float = 1.0

    def compute_hash(self) -> str:
        normalized = f"{self.title.strip().lower()[:80]}"
        self.item_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.item_id
```

### 7.3 Multi-Provider Aggregator (`data_pipeline/news_providers/aggregator.py`)
```python
import asyncio
import html
import math
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
import httpx

from data_pipeline.news_providers.schemas import (
    EconomicCalendarEvent,
    ImpactLevel,
    NewsSource,
    UnifiedNewsItem,
)


class MultiSourceNewsAggregator:
    """
    Asynchronous 5-Provider News & Economic Calendar Engine.
    Fetches, normalizes, deduplicates, and computes time-decayed sentiment.
    """

    def __init__(
        self,
        finnhub_key: str,
        alphavantage_key: str,
        cryptopanic_key: str,
        finbert_scorer: Optional[Callable[[str], float]] = None,
    ):
        self.finnhub_key = finnhub_key
        self.alphavantage_key = alphavantage_key
        self.cryptopanic_key = cryptopanic_key
        self.finbert_scorer = finbert_scorer or (lambda text: 0.0)
        self._calendar_cache: List[EconomicCalendarEvent] = []
        self._calendar_last_fetch: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # 1. FOREXFACTORY CALENDAR (Fair Economy JSON)
    # -------------------------------------------------------------------------
    async def fetch_forexfactory_calendar(
        self, client: httpx.AsyncClient
    ) -> List[EconomicCalendarEvent]:
        now = datetime.now(timezone.utc)
        if (
            self._calendar_last_fetch
            and (now - self._calendar_last_fetch).total_seconds() < 3600
            and self._calendar_cache
        ):
            return self._calendar_cache

        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            raw_events = resp.json()
            events: List[EconomicCalendarEvent] = []
            for ev in raw_events:
                impact_str = ev.get("impact", "Low")
                impact = (
                    ImpactLevel(impact_str)
                    if impact_str in ImpactLevel._value2member_map_
                    else ImpactLevel.LOW
                )
                dt = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
                events.append(
                    EconomicCalendarEvent(
                        title=ev.get("title", ""),
                        country=ev.get("country", "").upper(),
                        timestamp=dt,
                        impact=impact,
                        forecast=ev.get("forecast"),
                        previous=ev.get("previous"),
                    )
                )
            self._calendar_cache = events
            self._calendar_last_fetch = now
            return events
        except Exception:
            return self._calendar_cache

    def is_in_high_impact_blackout(
        self,
        currencies: List[str],
        events: List[EconomicCalendarEvent],
        window_minutes: int = 15,
    ) -> Tuple[bool, Optional[str]]:
        """Checks if current UTC time is within ±window_minutes of a High Impact event."""
        now = datetime.now(timezone.utc)
        target_ccys = {c.upper() for c in currencies}
        for ev in events:
            if ev.impact == ImpactLevel.HIGH and ev.country in target_ccys:
                diff_mins = abs((now - ev.timestamp).total_seconds()) / 60.0
                if diff_mins <= window_minutes:
                    return True, f"{ev.country} {ev.title} ({ev.timestamp.strftime('%H:%M UTC')})"
        return False, None

    # -------------------------------------------------------------------------
    # 2. MARCTOMARKET (Atom/JSON Macro Feed)
    # -------------------------------------------------------------------------
    async def fetch_marctomarket(
        self, client: httpx.AsyncClient
    ) -> List[UnifiedNewsItem]:
        url = "https://www.marctomarket.com/feeds/posts/default?alt=json"
        items: List[UnifiedNewsItem] = []
        try:
            resp = await client.get(url, timeout=12.0)
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])[:5]
            for entry in entries:
                title = entry.get("title", {}).get("$t", "")
                raw_html = entry.get("content", {}).get("$t", "")
                clean_text = re.sub(r"<[^>]+>", " ", html.unescape(raw_html))
                clean_text = re.sub(r"\s+", " ", clean_text).strip()[:1000]
                pub_str = entry.get("published", {}).get("$t", "")
                pub_dt = datetime.fromisoformat(pub_str).astimezone(timezone.utc)

                item = UnifiedNewsItem(
                    source=NewsSource.MARC_TO_MARKET,
                    title=title,
                    summary=clean_text,
                    published_at=pub_dt,
                    tickers=["DXY", "EURUSD", "USDJPY", "GBPUSD"],
                    authority_weight=1.15,
                )
                item.compute_hash()
                items.append(item)
        except Exception:
            pass
        return items

    # -------------------------------------------------------------------------
    # 3. FINNHUB.IO REST API
    # -------------------------------------------------------------------------
    async def fetch_finnhub(
        self, client: httpx.AsyncClient, category: str = "general"
    ) -> List[UnifiedNewsItem]:
        if not self.finnhub_key:
            return []
        url = "https://finnhub.io/api/v1/news"
        params = {"category": category, "token": self.finnhub_key}
        items: List[UnifiedNewsItem] = []
        try:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            for row in resp.json()[:15]:
                pub_dt = datetime.fromtimestamp(row.get("datetime", 0), tz=timezone.utc)
                item = UnifiedNewsItem(
                    source=NewsSource.FINNHUB,
                    title=row.get("headline", ""),
                    summary=row.get("summary", "")[:500],
                    url=row.get("url", ""),
                    published_at=pub_dt,
                    authority_weight=1.0,
                )
                item.compute_hash()
                items.append(item)
        except Exception:
            pass
        return items

    # -------------------------------------------------------------------------
    # 4. ALPHA VANTAGE NEWS & SENTIMENT API
    # -------------------------------------------------------------------------
    async def fetch_alphavantage(
        self, client: httpx.AsyncClient, ticker: str = "CRYPTO:BTC"
    ) -> List[UnifiedNewsItem]:
        if not self.alphavantage_key:
            return []
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": 15,
            "apikey": self.alphavantage_key,
        }
        items: List[UnifiedNewsItem] = []
        try:
            resp = await client.get(url, params=params, timeout=12.0)
            resp.raise_for_status()
            feed = resp.json().get("feed", [])
            for row in feed:
                t_str = row.get("time_published", "")
                pub_dt = datetime.strptime(t_str, "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                av_score = float(row.get("overall_sentiment_score", 0.0))
                item = UnifiedNewsItem(
                    source=NewsSource.ALPHA_VANTAGE,
                    title=row.get("title", ""),
                    summary=row.get("summary", "")[:500],
                    url=row.get("url", ""),
                    published_at=pub_dt,
                    tickers=[ticker],
                    provider_sentiment=av_score,
                    authority_weight=1.20,
                )
                item.compute_hash()
                items.append(item)
        except Exception:
            pass
        return items

    # -------------------------------------------------------------------------
    # 5. CRYPTOPANIC API
    # -------------------------------------------------------------------------
    async def fetch_cryptopanic(
        self, client: httpx.AsyncClient, currencies: str = "BTC,ETH"
    ) -> List[UnifiedNewsItem]:
        if not self.cryptopanic_key:
            return []
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.cryptopanic_key,
            "currencies": currencies,
            "public": "true",
        }
        items: List[UnifiedNewsItem] = []
        try:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            results = resp.json().get("results", [])[:15]
            for row in results:
                pub_dt = datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                votes = row.get("votes", {})
                v_pos = float(votes.get("positive", 0))
                v_neg = float(votes.get("negative", 0))
                v_tox = float(votes.get("toxic", 0))
                v_sav = float(votes.get("saved", 0))
                crowd_score = (
                    (v_pos + 0.5 * v_sav) - (v_neg + 2.0 * v_tox)
                ) / (v_pos + v_neg + v_tox + v_sav + 1.0)

                item = UnifiedNewsItem(
                    source=NewsSource.CRYPTOPANIC,
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    published_at=pub_dt,
                    tickers=currencies.split(","),
                    provider_sentiment=max(-1.0, min(1.0, crowd_score)),
                    authority_weight=0.95,
                )
                item.compute_hash()
                items.append(item)
        except Exception:
            pass
        return items

    # -------------------------------------------------------------------------
    # AGGREGATION, DEDUPLICATION & COMPOSITE SCORING
    # -------------------------------------------------------------------------
    async def get_composite_intelligence(
        self, symbol: str, asset_class: str = "crypto"
    ) -> Dict[str, object]:
        async with httpx.AsyncClient(
            headers={"User-Agent": "NexusTradeAI/2.0 InstitutionalEngine"}
        ) as client:
            av_ticker = f"CRYPTO:{symbol}" if asset_class == "crypto" else f"FOREX:{symbol}"
            fh_category = "crypto" if asset_class == "crypto" else "forex"

            tasks = [
                self.fetch_forexfactory_calendar(client),
                self.fetch_marctomarket(client),
                self.fetch_finnhub(client, category=fh_category),
                self.fetch_alphavantage(client, ticker=av_ticker),
            ]
            if asset_class == "crypto":
                tasks.append(self.fetch_cryptopanic(client, currencies=symbol))

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
            if item.item_id not in deduped:
                deduped[item.item_id] = item

        # Score and apply exponential time decay
        now = datetime.now(timezone.utc)
        weighted_sum = 0.0
        total_weight = 0.0

        for item in deduped.values():
            text_payload = f"{item.title}. {item.summary}".strip()
            fb_score = self.finbert_scorer(text_payload)
            item.finbert_sentiment = fb_score

            if item.provider_sentiment is not None:
                if item.source == NewsSource.ALPHA_VANTAGE:
                    item.final_polarity = 0.60 * item.provider_sentiment + 0.40 * fb_score
                else:
                    item.final_polarity = 0.50 * item.provider_sentiment + 0.50 * fb_score
            else:
                item.final_polarity = fb_score

            age_hours = max(0.0, (now - item.published_at).total_seconds() / 3600.0)
            half_life = 12.0 if item.source == NewsSource.MARC_TO_MARKET else 2.0
            decay = math.exp(-(math.log(2) / half_life) * age_hours)

            w = item.authority_weight * item.relevance_score * decay
            weighted_sum += w * item.final_polarity
            total_weight += w

        composite_sentiment = weighted_sum / (total_weight + 1e-8)
        is_blackout, blackout_reason = self.is_in_high_impact_blackout(
            currencies=["USD", symbol[:3], symbol[-3:]],
            events=calendar_events,
        )

        return {
            "symbol": symbol,
            "composite_sentiment": round(float(composite_sentiment), 4),
            "is_macro_blackout": is_blackout,
            "blackout_event": blackout_reason,
            "article_count": len(deduped),
            "top_headlines": [
                {
                    "source": i.source.value,
                    "title": i.title,
                    "polarity": round(i.final_polarity, 3),
                }
                for i in sorted(
                    deduped.values(), key=lambda x: x.published_at, reverse=True
                )[:5]
            ],
        }
```

---

## 8. Integration with Ensemble Agent & Telegram Bot

### 8.1 Ensemble Decision Matrix Update (`ai_engine/models/ensemble_agent.py`)
1. Before generating any trade order on $M1$, query `aggregator.get_composite_intelligence(symbol)`.
2. **Rule 1 (Blackout Circuit Breaker):** If `is_macro_blackout == True`, immediately return `SignalAction.HOLD` with reason `"HIGH_IMPACT_NEWS_BLACKOUT"`.
3. **Rule 2 (Sentiment Confluence Boost / Penalty):**
   * If Technical Signal is `BUY` and `composite_sentiment < -0.35`, veto the trade or reduce position size by $50\%$.
   * If Technical Signal is `SELL` and `composite_sentiment > +0.35`, veto the trade or reduce position size by $50\%$.

### 8.2 Telegram `/sentiment <symbol>` Command Upgrade
Update `telegram_bot/handlers/signals.py` to display:
* **Composite Sentiment Score:** $(-1.00 \text{ to } +1.00)$ with visual gauge.
* **Macro Calendar Status:** `🟢 CLEAR` or `🔴 BLACKOUT ACTIVE (Event Name)`.
* **Source Breakdown:** Latest headlines tagged with `[Finnhub]`, `[AlphaVantage]`, `[CryptoPanic]`, or `[MarcToMarket]` alongside their individual polarity badges.

---

## 9. Acceptance Criteria & Verification Tests

- [ ] **AC-1 (Graceful Degradation):** If 1 or more API keys are missing or rate-limited (HTTP 429), `MultiSourceNewsAggregator` continues operating seamlessly using the remaining active providers without raising unhandled exceptions.
- [ ] **AC-2 (ForexFactory Blackout Accuracy):** Unit test confirms `is_in_high_impact_blackout()` returns `True` when a simulated `High` impact USD event is within $\pm 15$ minutes of current UTC time.
- [ ] **AC-3 (SHA-256 Deduplication):** Identical breaking headlines syndicated across Finnhub and Alpha Vantage are collapsed into a single `UnifiedNewsItem` prior to FinBERT inference.
- [ ] **AC-4 (Rate Limit Compliance):** Redis or in-memory TTL cache prevents ForexFactory from being queried more than once per hour.