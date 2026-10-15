# PRODUCT REQUIREMENTS DOCUMENT (PRD): Multi-Source Real-Time News & Macroeconomic Sentiment Ingestion Engine (v2.0 - Zero-Cost Tier-1 RSS Edition)

## 1. Document Metadata & Executive Summary

* **Project Name:** NexusTrade AI
* **Feature / Module:** Real-Time Multi-Provider News, Macro Commentary & Economic Calendar Engine
* **Target Directory:** `data_pipeline/news_providers/` & `ai_engine/models/sentiment_analyzer.py`
* **Objective:** Deprecate static Kaggle-only news text ingestion and replace the paid/delayed CryptoPanic API with an asynchronous, zero-cost, fault-tolerant 5-provider real-time ingestion pipeline:
  1. **ForexFactory (Fair Economy JSON):** Scheduled macroeconomic calendar & volatility blackout windows.
  2. **MarcToMarket (Atom/RSS JSON Feed):** Institutional FX & global macro narrative analysis.
  3. **Finnhub.io REST API:** Real-time breaking headlines across Forex, Crypto, and General markets.
  4. **Alpha Vantage News & Sentiment API:** Pre-scored institutional news & ticker-specific sentiment vectors.
  5. **Direct Tier-1 Crypto Media RSS Aggregator (CoinDesk, CoinTelegraph, Decrypt, The Block):** Zero-limit, zero-latency primary-source crypto news feeds scored locally via FinBERT.

---

## 2. Background & Problem Statement

### 2.1 Why Deprecate Static Datasets & CryptoPanic API?
1. **Static Kaggle Datasets (`ankurzing/...`, `miguelaenlle/...`):** Essential for offline FinBERT weight calibration and historical backtesting, but provide zero visibility into live market catalysts or scheduled macroeconomic releases.
2. **CryptoPanic API Limitations:** The free tier of CryptoPanic imposes severe rate limits, restricts sentiment vote metadata to paid PRO tiers, and introduces a 15-to-30-minute publication delay—rendering it unsuitable for $M1/M5$ algorithmic execution.

### 2.2 The Direct Tier-1 RSS Solution (Provider #5 Replacement)
Aggregator platforms like CryptoPanic scrape primary publishers before redistributing links. By bypassing middlemen and polling **Tier-1 Primary Crypto Publishers** directly via XML/RSS feeds (`CoinDesk`, `CoinTelegraph`, `Decrypt`, `The Block`) concurrently using `httpx` and `feedparser`:
* **Zero Latency:** Headlines arrive the exact second articles are published.
* **Zero API Cost & No Paywalls:** Public syndication feeds require no API keys or paid subscriptions.
* **Institutional Sentiment Scoring:** Raw headlines and article summaries are piped directly into our local **FinBERT** (`ProsusAI/finbert`) transformer model alongside keyword-boosted urgency heuristics.

---

## 3. System Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                            5 LIVE EXTERNAL PROVIDERS                               │
├──────────────┬───────────────┬──────────────┬───────────────┬──────────────────────┤
│ ForexFactory │ MarcToMarket  │  Finnhub.io  │ Alpha Vantage │ Tier-1 Crypto RSS    │
│ (JSON Cal)   │ (Macro Feed)  │  (REST API)  │  (REST API)   │ (CD / CT / DC / TB)  │
└──────┬───────┴───────┬───────┴──────┬───────┴───────┬───────┴──────────┬───────────┘
       │               │              │               │                  │
       ▼               ▼              ▼               ▼                  ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│               Async Ingestion & Caching Layer (httpx + feedparser)                 │
└─────────────────────────────────────────┬──────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│            Schema Normalization & SHA-256 Deduplication (UnifiedNewsItem)          │
└──────────────────────┬──────────────────────────────────────┬──────────────────────┘
                       │                                      │
                       ▼                                      ▼
┌────────────────────────────────────────┐ ┌─────────────────────────────────────────┐
│     Economic Blackout Gatekeeper       │ │     Hybrid Sentiment Scoring Engine     │
│   (Halts M1 Trading ±15m High Impact)  │ │   (FinBERT + Keyword Urgency + Native)  │
└──────────────────────┬─────────────────┘ └──────────────────┬──────────────────────┘
                       │                                      │
                       └──────────────────┬───────────────────┘
                                          ▼
                        [ AI Ensemble Agent & Telegram Bot ]
```

---

## 4. Detailed 5-Provider Specifications

### 4.1 Provider 1: ForexFactory (Fair Economy Calendar)
* **Role:** Macroeconomic Event Gatekeeper & Volatility Circuit Breaker.
* **Endpoint:** `https://nfs.faireconomy.media/ff_calendar_thisweek.json`
* **Authentication:** None (Public CDN endpoint).
* **Rate Limit Constraint:** Poll maximum once every 60 minutes (`TTL = 3600s`).
* **Trading Logic:** Triggers a **Hard Circuit Breaker** (Execution Blackout) on correlated pairs $\pm 15$ minutes around any event where `impact == "High"` (e.g., CPI, Non-Farm Payrolls, FOMC, Rate Decisions).

### 4.2 Provider 2: MarcToMarket (Macro & FX Commentary)
* **Role:** Global Macro Regime & DXY (US Dollar Index) Bias Context.
* **Endpoint:** `https://www.marctomarket.com/feeds/posts/default?alt=json`
* **Authentication:** None.
* **Polling Interval:** Every 30 minutes (`TTL = 1800s`).
* **Processing Logic:** Strip HTML tags, extract commentary mentioning central banks (`Fed`, `ECB`, `BOJ`, `BOE`), yields, and currency pairs, and score via **FinBERT** to establish a 12-hour macro directional bias.

### 4.3 Provider 3: Finnhub.io API
* **Role:** Low-latency breaking market news (General, Forex, Crypto).
* **Endpoint:** `GET https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}` (`category` $\in \{\text{general}, \text{forex}, \text{crypto}\}$).
* **Rate Limit:** 60 calls / minute (Free Tier). Poll every 2 minutes (`TTL = 120s`).
* **Processing Logic:** Extract `headline` and `summary`, score locally via **FinBERT**.

### 4.4 Provider 4: Alpha Vantage (News & Sentiment API)
* **Role:** Pre-computed AI Sentiment & Relevance Scoring.
* **Endpoint:** `GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&apikey={ALPHAVANTAGE_API_KEY}`
* **Rate Limit:** 25 requests / day on standard free tier. Cache responses (`TTL = 1800s`).
* **Key Fields Extracted:** `overall_sentiment_score` $\in [-1.0, 1.0]$ and `relevance_score` $\in [0.0, 1.0]$.

### 4.5 Provider 5: Direct Tier-1 Crypto Media RSS Aggregator (Replaces CryptoPanic)
* **Role:** Ultra-low-latency, zero-limit primary cryptocurrency news stream.
* **Authentication:** None (100% Free Public XML/RSS Syndication).
* **Polling Interval:** Every 60 seconds (`TTL = 60s`) concurrently via `asyncio.gather`.
* **Primary Syndication Endpoints:**
  1. **CoinDesk:** `https://www.coindesk.com/arc/outboundfeeds/rss/`
  2. **CoinTelegraph:** `https://cointelegraph.com/rss`
  3. **Decrypt:** `https://decrypt.co/feed`
  4. **The Block:** `https://www.theblock.co/rss.xml`
* **Processing & Asset Tagging Logic:**
  1. Parse XML via `feedparser` and strip HTML tags from `<title>` and `<description>`.
  2. Match regex asset patterns (e.g., `\b(BTC|BITCOIN)\b`, `\b(ETH|ETHEREUM)\b`, `\b(SOL|SOLANA)\b`, `\b(SEC|ETF|HACK|EXPLOIT|FOMC|BINANCE)\b`) to assign target ticker tags and relevance multipliers.
  3. Score combined `title + summary` through **FinBERT** + **High-Impact Keyword Heuristic Modifier**.

---

## 5. Mathematical Formulation for Multi-Source Sentiment Fusion

### 5.1 Exponential Time-Decay Weighting
News impact decays exponentially over time. For any news item $i$ published at UTC timestamp $t_i$ evaluated at current time $t_{\text{now}}$:

$$\Delta t_i = \frac{t_{\text{now}} - t_i}{3600} \quad (\text{in hours})$$

$$w_{\text{decay}}(\Delta t_i) = \exp(-\lambda \cdot \Delta t_i), \quad \text{where } \lambda = \frac{\ln(2)}{\tau_{1/2}}$$

* For breaking news (`Tier-1 Crypto RSS`, `Finnhub`, `Alpha Vantage`), half-life $\tau_{1/2} = 2.0\text{ hours}$ ($\lambda \approx 0.3466$).
* For macro commentary (`MarcToMarket`), half-life $\tau_{1/2} = 12.0\text{ hours}$ ($\lambda \approx 0.0578$).

### 5.2 High-Impact Keyword Heuristic Modifier ($M_{\text{lex}, i}$)
Because general financial language models can under-react to crypto-specific black swan events (e.g., protocol exploits, exchange insolvency, ETF approvals), apply a deterministic lexical modifier $M_{\text{lex}, i} \in [-0.40, +0.40]$ to raw FinBERT scores:

$$S_{\text{FinBERT}}^*(i) = \operatorname{clip}\left(S_{\text{FinBERT}}(i) + M_{\text{lex}, i}, -1.0, +1.0\right)$$

* **Bullish Catalysts ($+0.25$):** `"etf approved"`, `"institutional inflow"`, `"strategic reserve"`, `"rate cut"`, `"mainnet launch"`.
* **Bearish Catalysts ($-0.35$):** `"hacked"`, `"exploit"`, `"bridge drained"`, `"sec lawsuit"`, `"insolvency"`, `"withdrawals halted"`, `"delisted"`.

### 5.3 Unified Item Polarity Score ($p_i$)
Each news item $i$ receives a final normalized polarity score $p_i \in [-1.0, +1.0]$:
* **If sourced from Alpha Vantage:**
  $$p_i = 0.60 \cdot S_{\text{AV}, i} + 0.40 \cdot S_{\text{FinBERT}}^*(i)$$
* **If sourced from Tier-1 Crypto RSS, Finnhub, or MarcToMarket:**
  $$p_i = S_{\text{FinBERT}}^*(i) = \operatorname{clip}\left(P(\text{Positive}) - P(\text{Negative}) + M_{\text{lex}, i}, -1.0, +1.0\right)$$

### 5.4 Composite Asset Sentiment Score ($S_{\text{sent}}$)
Given $N$ deduplicated news items for a target asset over the last 24 hours, with provider authority weights $\omega_s$ ($\text{AlphaVantage}=1.20$, $\text{MarcToMarket}=1.15$, $\text{Tier1\_Crypto\_RSS}=1.10$, $\text{Finnhub}=1.00$):

$$S_{\text{sent}}(t) = \frac{\sum_{i=1}^{N} \omega_{s(i)} \cdot r_i \cdot w_{\text{decay}}(\Delta t_i) \cdot p_i}{\sum_{i=1}^{N} \omega_{s(i)} \cdot r_i \cdot w_{\text{decay}}(\Delta t_i) + \epsilon} \in [-1.0, +1.0]$$

Where $r_i \in [0, 1]$ is the asset relevance score ($r_i = 1.0$ if the target symbol is explicitly mentioned in the title; $r_i = 0.5$ for general market-wide news).

### 5.5 Macroeconomic Event Blackout Gate ($G_{\text{macro}}$)
Let $\mathcal{E}_{\text{high}}$ be the set of ForexFactory events where `impact == "High"` and currency matches either base or quote currency of the traded pair:

$$G_{\text{macro}}(t) = \begin{cases} 0 \ (\text{BLOCK EXECUTION}), & \text{if } \exists e \in \mathcal{E}_{\text{high}} \text{ s.t. } -15\text{ min} \le (t - t_e) \le +15\text{ min} \\ 1 \ (\text{ALLOW EXECUTION}), & \text{otherwise} \end{cases}$$

---

## 6. Directory & File Migration Plan

The AI Coding Agent must execute the following structural changes:

### 6.1 Files to Deprecate / Refactor
* **Remove** `cryptopanic_provider.py` and any `CRYPTOPANIC_API_KEY` references in `.env` and `config/settings.py`.
* **Modify** `data_pipeline/kaggle_downloader.py`: Keep `ankurzing/...` and `miguelaenlle/...` strictly under an `--offline-training` CLI flag for FinBERT weight calibration only.
* **Refactor** `ai_engine/models/sentiment_analyzer.py`: Connect live inference directly to `MultiSourceNewsAggregator`.

### 6.2 Target Package Structure
```
nexustrade-ai/
├── config/
│   └── settings.py                        # Updated settings without CryptoPanic
├── data_pipeline/
│   └── news_providers/
│       ├── __init__.py
│       ├── schemas.py                     # UnifiedNewsItem & EconomicCalendarEvent
│       ├── forexfactory_provider.py       # Provider 1: Fair Economy JSON calendar
│       ├── marctomarket_provider.py       # Provider 2: Macro FX RSS/JSON parser
│       ├── finnhub_provider.py            # Provider 3: Finnhub REST client
│       ├── alphavantage_provider.py       # Provider 4: Alpha Vantage REST client
│       ├── crypto_rss_provider.py         # Provider 5: Tier-1 Crypto Media RSS Aggregator
│       └── aggregator.py                  # MultiSourceNewsAggregator & deduplicator
```

---

## 7. Production Python Implementation Blueprint

### 7.1 Environment Variables (`.env.example` Update)
```env
# --- Real-Time News & Macro Providers ---
FINNHUB_API_KEY="your_finnhub_api_key_here"
ALPHAVANTAGE_API_KEY="your_alphavantage_api_key_here"

# Public Zero-Key Feeds (ForexFactory, MarcToMarket & Tier-1 Crypto RSS)
FOREXFACTORY_CALENDAR_URL="https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MARCTOMARKET_FEED_URL="https://www.marctomarket.com/feeds/posts/default?alt=json"
CRYPTO_RSS_FEEDS="https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss,https://decrypt.co/feed,https://www.theblock.co/rss.xml"

# Polling & Cache TTLs (Seconds)
TTL_FOREXFACTORY=3600
TTL_MARCTOMARKET=1800
TTL_FINNHUB=120
TTL_ALPHAVANTAGE=1800
TTL_CRYPTO_RSS=60
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
    TIER1_CRYPTO_RSS = "tier1_crypto_rss"


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
    publisher: str = ""
    title: str
    summary: str = ""
    url: str = ""
    published_at: datetime
    tickers: List[str] = Field(default_factory=list)
    provider_sentiment: Optional[float] = None  # [-1.0, 1.0] if native score exists
    finbert_sentiment: Optional[float] = None   # [-1.0, 1.0] computed locally
    lexical_modifier: float = 0.0               # [-0.4, 0.4] black-swan/catalyst boost
    final_polarity: float = 0.0                 # Combined score [-1.0, 1.0]
    relevance_score: float = 1.0                # [0.0, 1.0]
    authority_weight: float = 1.0

    def compute_hash(self) -> str:
        normalized = re.sub(r"[^a-z0-9]", "", self.title.strip().lower())[:75]
        self.item_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.item_id


import re
```

### 7.3 Multi-Provider Aggregator with Tier-1 Crypto RSS (`data_pipeline/news_providers/aggregator.py`)
```python
import asyncio
import email.utils
import html
import math
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
import feedparser
import httpx

from data_pipeline.news_providers.schemas import (
    EconomicCalendarEvent,
    ImpactLevel,
    NewsSource,
    UnifiedNewsItem,
)


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
    Asynchronous 5-Provider News & Economic Calendar Engine.
    Integrates ForexFactory, MarcToMarket, Finnhub, AlphaVantage, and Tier-1 Crypto RSS.
    """

    def __init__(
        self,
        finnhub_key: str = "",
        alphavantage_key: str = "",
        finbert_scorer: Optional[Callable[[str], float]] = None,
    ):
        self.finnhub_key = finnhub_key
        self.alphavantage_key = alphavantage_key
        self.finbert_scorer = finbert_scorer or (lambda text: 0.0)
        self._calendar_cache: List[EconomicCalendarEvent] = []
        self._calendar_last_fetch: Optional[datetime] = None

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        clean = re.sub(r"<[^>]+>", " ", html.unescape(raw_html or ""))
        return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def _parse_rss_datetime(entry: dict) -> datetime:
        for field in ("published", "updated", "created"):
            val = entry.get(field)
            if val:
                try:
                    dt = email.utils.parsedate_to_datetime(val)
                    return dt.astimezone(timezone.utc)
                except Exception:
                    pass
                try:
                    dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    return dt.astimezone(timezone.utc)
                except Exception:
                    pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _compute_lexical_modifier(text: str) -> float:
        lower_text = text.lower()
        mod = 0.0
        for kw in BULLISH_CATALYSTS:
            if kw in lower_text:
                mod += 0.25
        for kw in BEARISH_CATALYSTS:
            if kw in lower_text:
                mod -= 0.35
        return max(-0.40, min(0.40, mod))

    @staticmethod
    def _extract_crypto_tickers_and_relevance(
        text: str, target_symbol: str
    ) -> Tuple[List[str], float]:
        lower_text = text.lower()
        detected: List[str] = []
        for sym, keywords in ASSET_KEYWORDS.items():
            if any(re.search(rf"\b{ re.escape(kw) }\b", lower_text) for kw in keywords):
                detected.append(sym)

        clean_target = (
            target_symbol.upper()
            .replace("USDT", "")
            .replace("USD", "")
            .replace("PERP", "")
        )
        if clean_target in detected or clean_target.lower() in lower_text:
            return detected or [clean_target], 1.0
        return detected, 0.55

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
        now = datetime.now(timezone.utc)
        target_ccys = {c.upper() for c in currencies if c}
        for ev in events:
            if ev.impact == ImpactLevel.HIGH and ev.country in target_ccys:
                diff_mins = abs((now - ev.timestamp).total_seconds()) / 60.0
                if diff_mins <= window_minutes:
                    return (
                        True,
                        f"{ev.country} {ev.title} ({ev.timestamp.strftime('%H:%M UTC')})",
                    )
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
                clean_text = self._clean_html(raw_html)[:1000]
                pub_str = entry.get("published", {}).get("$t", "")
                pub_dt = datetime.fromisoformat(pub_str).astimezone(timezone.utc)

                item = UnifiedNewsItem(
                    source=NewsSource.MARC_TO_MARKET,
                    publisher="MarcToMarket",
                    title=title,
                    summary=clean_text,
                    published_at=pub_dt,
                    tickers=["DXY", "EURUSD", "USDJPY", "GBPUSD"],
                    authority_weight=1.15,
                    relevance_score=0.85,
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
                    publisher=row.get("source", "Finnhub"),
                    title=row.get("headline", ""),
                    summary=self._clean_html(row.get("summary", ""))[:500],
                    url=row.get("url", ""),
                    published_at=pub_dt,
                    authority_weight=1.00,
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
                    publisher=row.get("source", "AlphaVantage"),
                    title=row.get("title", ""),
                    summary=self._clean_html(row.get("summary", ""))[:500],
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
    # 5. DIRECT TIER-1 CRYPTO MEDIA RSS AGGREGATOR (Zero-Limit / Zero-Cost)
    # -------------------------------------------------------------------------
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
            resp.raise_for_status()
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
                    url=entry.get("link", ""),
                    published_at=pub_dt,
                    tickers=tickers,
                    relevance_score=relevance,
                    authority_weight=1.10,
                )
                item.compute_hash()
                items.append(item)
        except Exception:
            pass
        return items

    async def fetch_tier1_crypto_rss(
        self, client: httpx.AsyncClient, target_symbol: str = "BTC"
    ) -> List[UnifiedNewsItem]:
        """Concurrently polls CoinDesk, CoinTelegraph, Decrypt, and TheBlock RSS feeds."""
        tasks = [
            self._fetch_single_rss(client, pub, url, target_symbol)
            for pub, url in TIER1_CRYPTO_RSS_ENDPOINTS.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged: List[UnifiedNewsItem] = []
        for res in results:
            if isinstance(res, list):
                merged.extend(res)
        return merged

    # -------------------------------------------------------------------------
    # AGGREGATION, DEDUPLICATION & COMPOSITE SCORING
    # -------------------------------------------------------------------------
    async def get_composite_intelligence(
        self, symbol: str, asset_class: str = "crypto"
    ) -> Dict[str, object]:
        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; NexusTradeAI/2.0)"}
        ) as client:
            clean_sym = (
                symbol.upper()
                .replace("USDT", "")
                .replace("USD", "")
                .replace("PERP", "")
            )
            av_ticker = (
                f"CRYPTO:{clean_sym}"
                if asset_class == "crypto"
                else f"FOREX:{clean_sym}"
            )
            fh_category = "crypto" if asset_class == "crypto" else "forex"

            tasks = [
                self.fetch_forexfactory_calendar(client),
                self.fetch_marctomarket(client),
                self.fetch_finnhub(client, category=fh_category),
                self.fetch_alphavantage(client, ticker=av_ticker),
            ]
            if asset_class == "crypto":
                tasks.append(
                    self.fetch_tier1_crypto_rss(client, target_symbol=clean_sym)
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

        calendar_events: List[EconomicCalendarEvent] = (
            results[0] if isinstance(results[0], list) else []
        )
        raw_news: List[UnifiedNewsItem] = []
        for res in results[1:]:
            if isinstance(res, list):
                raw_news.extend(res)

        # Deduplicate syndicated headlines across RSS & APIs by SHA-256 title hash
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
            fb_raw = float(self.finbert_scorer(text_payload))
            lex_mod = self._compute_lexical_modifier(text_payload)
            fb_adjusted = max(-1.0, min(1.0, fb_raw + lex_mod))

            item.finbert_sentiment = fb_raw
            item.lexical_modifier = lex_mod

            if (
                item.provider_sentiment is not None
                and item.source == NewsSource.ALPHA_VANTAGE
            ):
                item.final_polarity = (
                    0.60 * item.provider_sentiment + 0.40 * fb_adjusted
                )
            else:
                item.final_polarity = fb_adjusted

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
                    "source": i.publisher or i.source.value,
                    "title": i.title,
                    "polarity": round(i.final_polarity, 3),
                    "relevance": round(i.relevance_score, 2),
                }
                for i in sorted(
                    deduped.values(), key=lambda x: x.published_at, reverse=True
                )[:6]
            ],
        }
```

---

## 8. Integration with Ensemble Agent & Telegram Bot

### 8.1 Ensemble Decision Matrix Update (`ai_engine/models/ensemble_agent.py`)
1. Before generating any trade order on $M1$, query `aggregator.get_composite_intelligence(symbol, asset_class)`.
2. **Rule 1 (Blackout Circuit Breaker):** If `is_macro_blackout == True`, immediately return `SignalAction.HOLD` with reason `"HIGH_IMPACT_NEWS_BLACKOUT"`.
3. **Rule 2 (Sentiment Confluence Filter):**
   * If Technical Signal is `BUY` and `composite_sentiment < -0.35`, veto the trade or slash position size by $50\%$.
   * If Technical Signal is `SELL` and `composite_sentiment > +0.35`, veto the trade or slash position size by $50\%$.

### 8.2 Telegram `/sentiment <symbol>` Command Output
Update `telegram_bot/handlers/signals.py` to format the output card with publisher badges:
* **Composite Sentiment Score:** $(-1.00 \text{ to } +1.00)$ with visual gauge.
* **Macro Calendar Status:** `🟢 CLEAR` or `🔴 BLACKOUT ACTIVE (Event Name)`.
* **Live Feed Breakdown:** Displays top headlines tagged with `[CoinDesk]`, `[CoinTelegraph]`, `[Decrypt]`, `[TheBlock]`, `[Finnhub]`, `[AlphaVantage]`, or `[MarcToMarket]`.

---

## 9. Acceptance Criteria & Verification Tests

- [ ] **AC-1 (Zero Paywall / Zero API Key for Crypto RSS):** `fetch_tier1_crypto_rss()` concurrently fetches and parses XML feeds from CoinDesk, CoinTelegraph, Decrypt, and The Block with zero API keys required.
- [ ] **AC-2 (Sub-2-Second Parallel Execution):** All 4 Tier-1 Crypto RSS feeds plus ForexFactory and MarcToMarket resolve in `< 2.5 seconds` via `asyncio.gather`.
- [ ] **AC-3 (Lexical Black-Swan Detection):** Headlines containing `"hacked"`, `"exploit"`, or `"sec lawsuit"` trigger a negative lexical modifier ($-0.35$) to immediately protect the bot from buying falling knives.
- [ ] **AC-4 (SHA-256 Cross-Feed Deduplication):** Syndicated stories appearing simultaneously on CoinDesk and Finnhub are deduplicated prior to FinBERT inference.