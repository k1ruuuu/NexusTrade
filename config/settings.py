from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, type-validated configuration for NexusTrade AI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Environment
    app_name: str = Field(default="NexusTrade AI", description="Application display name")
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development"
    )
    log_level: str = Field(default="INFO")
    log_json_format: bool = Field(default=False)
    data_dir: Path = Field(default=Path("data"))
    local_xauusd_path: Path = Field(default=Path("XAUUSD/XAUUSD_15m.csv"))
    local_crypto_dir: Path = Field(
        default=Path("50 Crypto/Top50PlusCryptoCurrenciesDailyMarketData")
    )
    use_live_market_data: bool = Field(
        default=True,
        description="Fetch real-time market prices and OHLCV from Binance/Yahoo Finance before local CSV fallback",
    )
    live_market_cache_ttl_seconds: int = Field(default=15, ge=1)
    use_live_news_feed: bool = Field(
        default=True,
        description="Automatically harvest breaking financial news from Yahoo Finance & Google News RSS",
    )
    online_learning_enabled: bool = Field(
        default=True,
        description="Enable continuous Learn-by-Doing online training from live news and market price feedback",
    )
    online_learning_interval_seconds: int = Field(default=180, ge=30)

    # Telegram Bot Configuration
    telegram_bot_token: str = Field(
        default="1234567890:AAH_nexustrade_placeholder_token"
    )
    telegram_allowed_user_ids: Union[list[int], str, int] = Field(
        default_factory=lambda: [123456789]
    )
    telegram_admin_chat_id: Optional[int] = Field(default=123456789)
    telegram_admin_user_ids: Union[list[int], str, int] = Field(
        default_factory=list,
        description="Telegram user IDs with Admin role allowed to fall back to .env credentials",
    )
    telegram_open_access: bool = Field(
        default=False,
        description="Allow any Telegram user to interact with the bot (non-admins must still run /setup for auto-trade)",
    )
    telegram_rate_limit_calls: int = Field(default=5, ge=1)
    telegram_rate_limit_period: float = Field(default=10.0, gt=0.0)

    # TradingView Webhook & Charting Configuration
    tradingview_webhook_secret: str = Field(
        default="nexustrade_default_webhook_secret", min_length=8
    )
    tradingview_config_path: Path = Field(
        default=Path("config/tradingview_config.json")
    )
    webhook_host: str = Field(default="0.0.0.0")
    webhook_port: int = Field(default=8000, ge=1, le=65535)
    chart_width: int = Field(default=1200, ge=400)
    chart_height: int = Field(default=800, ge=300)
    webhook_max_age_seconds: int = Field(default=300, ge=0)

    # Kaggle API Credentials
    kaggle_username: Optional[str] = Field(default=None)
    kaggle_key: Optional[str] = Field(default=None)

    # AI Engine & Ensemble Weights
    ensemble_tech_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    ensemble_sent_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    signal_confidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    finbert_model_name: str = Field(default="ProsusAI/finbert")
    use_transformers_pipeline: bool = Field(default=False)

    # Real-Time Multi-Source News & Macro Providers
    finnhub_api_key: str = Field(default="", description="Finnhub.io REST API key")
    alphavantage_api_key: str = Field(default="", description="Alpha Vantage News & Sentiment API key")
    forexfactory_calendar_url: str = Field(
        default="https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        description="Fair Economy ForexFactory JSON calendar endpoint",
    )
    marctomarket_feed_url: str = Field(
        default="https://www.marctomarket.com/feeds/posts/default?alt=json",
        description="MarcToMarket Atom/JSON macro feed endpoint",
    )
    crypto_rss_feeds: str = Field(
        default="https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss,https://decrypt.co/feed,https://www.theblock.co/rss.xml",
        description="Tier-1 Crypto Media RSS feed URLs comma separated",
    )
    ttl_forexfactory: int = Field(default=3600, ge=60, description="ForexFactory cache TTL (seconds)")
    ttl_marctomarket: int = Field(default=1800, ge=60, description="MarcToMarket cache TTL (seconds)")
    ttl_finnhub: int = Field(default=120, ge=30, description="Finnhub cache TTL (seconds)")
    ttl_alphavantage: int = Field(default=1800, ge=60, description="Alpha Vantage cache TTL (seconds)")
    ttl_crypto_rss: int = Field(default=60, ge=15, description="Tier-1 Crypto RSS cache TTL (seconds)")
    macro_blackout_window_minutes: int = Field(
        default=15, ge=1, description="Execution blackout window in minutes (+/-) around High Impact events"
    )
    macro_blackout_enabled: bool = Field(
        default=True, description="Enforce execution blackout on correlated pairs around High Impact events"
    )
    allow_news_auto_reverse: bool = Field(
        default=True, description="Allow automatic counter-trade execution after an emergency news exit"
    )

    # Institutional Risk Guardrails
    portfolio_initial_capital: float = Field(default=100000.0, gt=0.0)
    max_account_risk_per_trade: float = Field(default=0.015, gt=0.0, le=0.10)
    atr_stop_loss_multiplier: float = Field(default=2.0, gt=0.0)
    min_risk_reward_ratio: float = Field(default=2.0, ge=1.0)
    max_daily_drawdown_limit: float = Field(default=0.04, gt=0.0, le=0.50)
    max_volatility_threshold: float = Field(default=0.85, gt=0.0, le=1.0)

    # Admin-Only Binance API & Multi-User Vault Configuration (.env applies strictly to Admin role)
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = Field(default=True)
    vault_encryption_secret: str = Field(
        default="nexustrade_vault_master_secret_key_2026", min_length=16
    )

    @field_validator("telegram_allowed_user_ids", "telegram_admin_user_ids", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, value: Any) -> list[int]:
        if value is None or value == "":
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                return [int(item) for item in parsed]
            return [
                int(part.strip())
                for part in stripped.split(",")
                if part.strip()
            ]
        raise ValueError(f"Unsupported format for telegram_allowed_user_ids: {value!r}")

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = str(value).upper().strip()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            raise ValueError(f"Invalid log_level '{value}'. Expected one of {valid_levels}")
        return normalized

    @model_validator(mode="after")
    def validate_ensemble_weights(self) -> Settings:
        total = self.ensemble_tech_weight + self.ensemble_sent_weight
        if total <= 0:
            raise ValueError("Sum of ensemble_tech_weight and ensemble_sent_weight must be positive")
        if abs(total - 1.0) > 1e-6:
            self.ensemble_tech_weight = round(self.ensemble_tech_weight / total, 6)
            self.ensemble_sent_weight = round(self.ensemble_sent_weight / total, 6)
        if self.environment == "production":
            if (
                self.tradingview_webhook_secret == "nexustrade_default_webhook_secret"
                or "your_encrypted_webhook_secret" in self.tradingview_webhook_secret
                or len(self.tradingview_webhook_secret) < 16
            ):
                raise ValueError(
                    "Production environment requires a strong, non-default TRADINGVIEW_WEBHOOK_SECRET (>=16 chars)"
                )
            if (
                "placeholder" in self.telegram_bot_token
                or "your_telegram_bot_token" in self.telegram_bot_token
            ):
                raise ValueError("Production environment requires a valid TELEGRAM_BOT_TOKEN")
            if (
                self.vault_encryption_secret == "nexustrade_vault_master_secret_key_2026"
                or "change_this" in self.vault_encryption_secret.lower()
                or len(self.vault_encryption_secret.strip()) < 32
            ):
                raise ValueError(
                    "Production environment requires a strong, non-default VAULT_ENCRYPTION_SECRET (>=32 chars)"
                )
        return self

    def is_admin_user(self, user_id: Optional[int]) -> bool:
        """Return True strictly if user_id has Admin role via TELEGRAM_ADMIN_USER_IDS or TELEGRAM_ADMIN_CHAT_ID."""
        if user_id is None:
            return False
        uid = int(user_id)
        admin_ids = {int(x) for x in self.telegram_admin_user_ids}
        if self.telegram_admin_chat_id is not None:
            admin_ids.add(int(self.telegram_admin_chat_id))
        return uid in admin_ids
    @property
    def raw_data_dir(self) -> Path:
        path = self.data_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_data_dir(self) -> Path:
        path = self.data_dir / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_data_dir(self) -> Path:
        path = self.data_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_tradingview_config(self) -> dict[str, Any]:
        if self.tradingview_config_path.exists():
            return json.loads(self.tradingview_config_path.read_text(encoding="utf-8"))
        return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
