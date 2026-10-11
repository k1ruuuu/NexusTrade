from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config.settings import Settings, get_settings
from data_pipeline.market_symbols import normalize_symbol

TICKER_SANITIZE_PATTERN = re.compile(r"^[A-Z0-9_\-\.\:\/]{2,32}$")
NormalizedAction = Literal["BUY", "SELL", "HOLD"]


class AlertIndicators(BaseModel):
    """Technical indicator snapshot embedded in a TradingView webhook payload."""

    model_config = ConfigDict(extra="allow")

    rsi: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    macd: Optional[float] = Field(default=None)
    atr: Optional[float] = Field(default=None, ge=0.0)
    ema_20: Optional[float] = Field(default=None, gt=0.0)
    ema_50: Optional[float] = Field(default=None, gt=0.0)
    ema_200: Optional[float] = Field(default=None, gt=0.0)
    vwap: Optional[float] = Field(default=None, gt=0.0)

    def to_float_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.model_dump().items():
            if v is not None:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    continue
        return out


class TradingViewAlertPayload(BaseModel):
    """Validated and sanitized TradingView Webhook Alert payload schema."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    secret: Optional[str] = Field(default=None, exclude=True, repr=False)
    ticker: str = Field(..., min_length=2, max_length=32)
    interval: str = Field(default="15m", min_length=1, max_length=10)
    action: str = Field(..., min_length=3, max_length=12)
    price: float = Field(..., gt=0.0)
    strategy: str = Field(default="NexusTrade_Pine_v5", max_length=64)
    indicators: AlertIndicators = Field(default_factory=AlertIndicators)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    canonical_symbol: str = Field(default="")
    normalized_action: NormalizedAction = Field(default="HOLD")

    @field_validator("ticker", mode="before")
    @classmethod
    def sanitize_ticker(cls, value: Any) -> str:
        raw = str(value).strip().upper()
        if not TICKER_SANITIZE_PATTERN.match(raw):
            raise ValueError(f"Invalid ticker format: {value!r}")
        return raw

    @field_validator("interval", mode="before")
    @classmethod
    def sanitize_interval(cls, value: Any) -> str:
        raw = str(value).strip()
        numeric_map = {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "D": "1d",
            "1D": "1d",
            "W": "1W",
        }
        return numeric_map.get(raw, raw)

    @field_validator("timestamp", mode="before")
    @classmethod
    def sanitize_timestamp(cls, value: Any) -> str:
        if not value:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw = str(value).strip()
        try:
            if raw.endswith("Z"):
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc

    @model_validator(mode="after")
    def resolve_mappings(self) -> TradingViewAlertPayload:
        settings = get_settings()
        tv_cfg = settings.load_tradingview_config()
        action_map: dict[str, str] = tv_cfg.get(
            "action_normalization",
            {
                "BUY": "BUY",
                "LONG": "BUY",
                "SELL": "SELL",
                "SHORT": "SELL",
                "HOLD": "HOLD",
                "CLOSE": "HOLD",
            },
        )
        upper_action = self.action.upper().strip()
        if upper_action not in action_map:
            raise ValueError(
                f"Unsupported alert action '{self.action}'. Supported: {list(action_map)}"
            )
        self.normalized_action = action_map[upper_action]  # type: ignore[assignment]

        ticker_mappings: dict[str, dict[str, Any]] = tv_cfg.get("ticker_mappings", {})
        if self.ticker in ticker_mappings:
            self.canonical_symbol = str(ticker_mappings[self.ticker]["canonical_symbol"]).upper()
        else:
            self.canonical_symbol = normalize_symbol(self.ticker)

        return self


class AlertParser:
    """Service wrapper for validating and normalizing raw TradingView webhook JSON payloads."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def parse(self, raw_payload: dict[str, Any]) -> TradingViewAlertPayload:
        """Validate raw dictionary into TradingViewAlertPayload."""
        return TradingViewAlertPayload.model_validate(raw_payload)
