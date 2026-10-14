from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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
