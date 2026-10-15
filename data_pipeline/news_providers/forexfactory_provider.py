from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple
import httpx

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.news_providers.base_provider import BaseNewsProvider
from data_pipeline.news_providers.schemas import (
    EconomicCalendarEvent,
    ImpactLevel,
    NewsSource,
)

logger = get_logger("nexustrade.news_providers.forexfactory")


class ForexFactoryProvider(BaseNewsProvider):
    """
    ForexFactory (Fair Economy JSON Calendar) Provider.
    Extracts scheduled macroeconomic releases (NFP, CPI, FOMC, Rate Decisions)
    and enforces high-impact volatility blackout windows.
    Strictly cached for 3600s (1 hour) to respect Fair Economy CDN rate limits.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        super().__init__(
            source=NewsSource.FOREX_FACTORY,
            ttl_seconds=getattr(cfg, "ttl_forexfactory", 3600),
            settings=cfg,
        )
        self.endpoint_url = getattr(
            cfg,
            "forexfactory_calendar_url",
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        )

    async def fetch(self, client: httpx.AsyncClient, **kwargs) -> List[EconomicCalendarEvent]:
        """Fetch economic calendar events with TTL cache protection."""
        return await self.fetch_calendar(client)

    async def fetch_calendar(self, client: httpx.AsyncClient) -> List[EconomicCalendarEvent]:
        cache_key = "forexfactory_calendar"
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        events: List[EconomicCalendarEvent] = []
        try:
            resp = await client.get(self.endpoint_url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning(
                    "ForexFactory calendar endpoint returned non-200",
                    status_code=resp.status_code,
                )
                return self.get_cached(cache_key) or []

            raw_events = resp.json()
            if not isinstance(raw_events, list):
                logger.warning("ForexFactory response is not a JSON list")
                return []

            for ev in raw_events:
                impact_raw = str(ev.get("impact", "Low")).strip()
                impact = (
                    ImpactLevel(impact_raw)
                    if impact_raw in ImpactLevel._value2member_map_
                    else ImpactLevel.LOW
                )
                date_str = str(ev.get("date", "")).strip()
                try:
                    if date_str.endswith("Z"):
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromisoformat(date_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                except Exception:
                    continue

                events.append(
                    EconomicCalendarEvent(
                        title=str(ev.get("title", "")).strip(),
                        country=str(ev.get("country", "")).strip().upper(),
                        timestamp=dt,
                        impact=impact,
                        forecast=str(ev.get("forecast")) if ev.get("forecast") is not None else None,
                        previous=str(ev.get("previous")) if ev.get("previous") is not None else None,
                    )
                )

            self.set_cached(cache_key, events)
            logger.info(
                "Fetched ForexFactory macroeconomic calendar",
                total_events=len(events),
                high_impact_count=sum(1 for e in events if e.impact == ImpactLevel.HIGH),
            )
            return events
        except Exception as exc:
            logger.warning("Failed to fetch ForexFactory calendar", error=str(exc))
            return self.get_cached(cache_key) or []

    @staticmethod
    def is_in_high_impact_blackout(
        currencies: List[str],
        events: List[EconomicCalendarEvent],
        window_minutes: int = 15,
        as_of: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if current time (or as_of) is within +/- window_minutes of any
        High Impact event matching the target currencies (e.g. USD, EUR, GBP).
        """
        now = as_of or datetime.now(timezone.utc)
        target_ccys = {str(c).upper().strip() for c in currencies if c}
        for ev in events:
            if ev.impact == ImpactLevel.HIGH and ev.country in target_ccys:
                diff_mins = abs((now - ev.timestamp).total_seconds()) / 60.0
                if diff_mins <= window_minutes:
                    event_time = ev.timestamp.strftime("%H:%M UTC")
                    return True, f"{ev.country} {ev.title} ({event_time})"
        return False, None
