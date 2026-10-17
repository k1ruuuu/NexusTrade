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
    TIER1_CRYPTO_RSS = "tier1_crypto_rss"
    US_MACRO_SQUAWK = "us_macro_squawk"

class ImpactLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    HOLIDAY = "Holiday"


class EconomicCalendarEvent(BaseModel):
    title: str
    country: str
    timestamp: datetime
    impact: ImpactLevel = ImpactLevel.LOW
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    def matches_event_mode(self, mode: Optional[str]) -> bool:
        """
        Check if this economic event matches the requested mode
        (e.g. FOMC, IMF, NFP, CPI, PPI, PCE, GDP, RETAIL, PMI, JOBLESS).
        """
        if not mode:
            return True
        m = mode.strip().upper()
        title_lower = self.title.lower()
        if m == "FOMC":
            return any(k in title_lower for k in (
                "fomc", "federal open market", "federal reserve", "interest rate",
                "funds rate", "powell", "rate statement", "monetary policy", "dot plot"
            ))
        if m == "IMF":
            return any(k in title_lower for k in (
                "imf", "international monetary fund", "world economic outlook", "global economic"
            ))
        if m == "CPI":
            return "cpi" in title_lower or "consumer price" in title_lower
        if m == "PPI":
            return "ppi" in title_lower or "producer price" in title_lower
        if m == "NFP":
            return any(k in title_lower for k in ("non-farm", "nonfarm", "payroll", "unemployment", "labor force"))
        if m == "PCE":
            return "pce" in title_lower or "personal consumption" in title_lower
        if m == "GDP":
            return "gdp" in title_lower or "gross domestic product" in title_lower
        if m == "RETAIL":
            return "retail sales" in title_lower or "core retail" in title_lower
        if m == "PMI":
            return any(k in title_lower for k in ("pmi", "ism manufacturing", "ism services", "purchasing managers"))
        if m == "JOBLESS":
            return any(k in title_lower for k in ("jobless claims", "initial claims", "continuing claims"))
        return m.lower() in title_lower
    def compute_surprise(self) -> Optional[float]:
        """
        Compute economic surprise = Actual - Forecast.
        Returns float or None if either number cannot be parsed.
        """
        if not self.actual or not self.forecast:
            return None
        import re
        def _clean_num(val: str) -> Optional[float]:
            cleaned = re.sub(r"[^\d.-]", "", val)
            try:
                return float(cleaned)
            except ValueError:
                return None
        act = _clean_num(self.actual)
        fc = _clean_num(self.forecast)
        if act is None or fc is None:
            return None
        return round(act - fc, 4)

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
    lexical_modifier: float = 0.0               # [-0.4, 0.4] black-swan / catalyst boost
    final_polarity: float = 0.0                 # Combined score [-1.0, 1.0]
    relevance_score: float = 1.0                # [0.0, 1.0]
    authority_weight: float = 1.0

    def compute_hash(self) -> str:
        import re
        normalized = re.sub(r"[^a-z0-9]", "", self.title.strip().lower())[:75]
        self.item_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.item_id
