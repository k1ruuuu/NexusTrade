from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from config.settings import Settings, get_settings


def setup_logging(
    settings: Optional[Settings] = None,
    log_level: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> structlog.stdlib.BoundLogger:
    """Configure structured logging with Rich console output or JSON rendering."""
    cfg = settings or get_settings()
    effective_level_name = (log_level or cfg.log_level).upper()
    effective_level = getattr(logging, effective_level_name, logging.INFO)
    use_json = json_format if json_format is not None else cfg.log_json_format

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(effective_level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if use_json:
        handler = logging.StreamHandler(sys.stdout)
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        console = Console(stderr=False, force_terminal=False)
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=False,
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )
        rich_handler.setFormatter(formatter)
        root_logger.addHandler(rich_handler)

    # Silence noisy third-party loggers
    for noisy_logger in ("httpx", "httpcore", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = get_logger("nexustrade.startup")
    logger.info(
        "NexusTrade AI configuration initialized",
        app_name=cfg.app_name,
        environment=cfg.environment,
        log_level=effective_level_name,
        confidence_threshold=cfg.signal_confidence_threshold,
        max_risk_per_trade=cfg.max_account_risk_per_trade,
        daily_drawdown_limit=cfg.max_daily_drawdown_limit,
    )
    return logger


def get_logger(name: str = "nexustrade") -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to the given module name."""
    return structlog.get_logger(name)
