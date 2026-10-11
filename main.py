from __future__ import annotations

import argparse
import asyncio
import socket
from typing import Optional

import uvicorn

from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.logging_config import get_logger, setup_logging
from config.settings import Settings, get_settings
from telegram_bot.bot import NexusTradeBotService
from tradingview.chart_generator import ChartSnapshotGenerator
from tradingview.webhook_server import create_webhook_app


def is_port_available(host: str, port: int) -> bool:
    """Return True if (host, port) can be bound on TCP."""
    bind_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
            return True
        except OSError:
            return False


def resolve_webhook_port(host: str, preferred_port: int, max_probes: int = 20) -> int:
    """Return preferred_port if free, otherwise find the next available TCP port."""
    if is_port_available(host, preferred_port):
        return preferred_port
    for candidate in range(preferred_port + 1, preferred_port + max_probes + 1):
        if is_port_available(host, candidate):
            return candidate
    return preferred_port


def has_placeholder_bot_token(token: str) -> bool:
    lowered = token.strip().lower()
    return (
        not lowered
        or "placeholder" in lowered
        or "your_telegram_bot_token" in lowered
    )


async def run_services(
    settings: Settings,
    webhook_only: bool = False,
    port_override: Optional[int] = None,
) -> None:
    """Launch integrated FastAPI TradingView Webhook Server and Aiogram v3 Telegram Bot."""
    logger = get_logger("nexustrade.runtime")

    ensemble_agent = EnsembleDecisionAgent(settings)
    chart_generator = ChartSnapshotGenerator(settings)
    bot_service = NexusTradeBotService(
        settings=settings,
        ensemble_agent=ensemble_agent,
        chart_generator=chart_generator,
    )

    fastapi_app = create_webhook_app(
        settings=settings,
        ensemble_agent=ensemble_agent,
        chart_generator=chart_generator,
        dispatch_callback=bot_service.dispatch_webhook_alert,
    )

    requested_port = port_override or settings.webhook_port
    effective_port = resolve_webhook_port(settings.webhook_host, requested_port)
    if effective_port != requested_port:
        logger.warning(
            "Requested webhook port is already in use; selected next available port",
            requested_port=requested_port,
            effective_port=effective_port,
            host=settings.webhook_host,
        )

    uvicorn_config = uvicorn.Config(
        app=fastapi_app,
        host=settings.webhook_host,
        port=effective_port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(uvicorn_config)

    try:
        if webhook_only or has_placeholder_bot_token(settings.telegram_bot_token):
            logger.info(
                "Starting NexusTrade AI FastAPI Webhook server (webhook-only mode)",
                host=settings.webhook_host,
                port=effective_port,
            )
            await server.serve()
        else:
            logger.info(
                "Starting NexusTrade AI FastAPI Webhook server + Telegram Bot polling",
                host=settings.webhook_host,
                port=effective_port,
            )

            stop_event = asyncio.Event()

            async def _run_uvicorn() -> None:
                try:
                    await server.serve()
                finally:
                    stop_event.set()
                    try:
                        await bot_service.dp.stop_polling()
                    except RuntimeError:
                        pass

            async def _run_bot() -> None:
                try:
                    await bot_service.start_polling(handle_signals=False)
                finally:
                    stop_event.set()
                    server.should_exit = True

            async def _run_online_learner() -> None:
                if not settings.online_learning_enabled:
                    return
                watched_symbols = ("BTCUSDT", "XAUUSD", "ETHUSDT")
                while not stop_event.is_set():
                    for sym in watched_symbols:
                        if stop_event.is_set():
                            break
                        try:
                            report = await asyncio.to_thread(
                                ensemble_agent.learn_online_cycle, sym, "15m"
                            )
                            if int(report.get("newly_learned_headlines", 0)) > 0:
                                logger.info(
                                    "Background Online Learner trained on new breaking news",
                                    symbol=sym,
                                    newly_learned=report["newly_learned_headlines"],
                                    corpus_size=report["total_corpus_size"],
                                    s_sent=report["s_sent"],
                                    action=report["action"],
                                )
                        except Exception as exc:
                            logger.debug(
                                "Online learning cycle skipped for symbol",
                                symbol=sym,
                                error=str(exc),
                            )
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=float(settings.online_learning_interval_seconds),
                        )
                    except asyncio.TimeoutError:
                        continue

            await asyncio.gather(_run_uvicorn(), _run_bot(), _run_online_learner())
    finally:
        await bot_service.bot.session.close()

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NexusTrade AI Institutional Trading Bot")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration, initialize AI ensemble, and exit.",
    )
    parser.add_argument(
        "--webhook-only",
        action="store_true",
        help="Run FastAPI TradingView webhook server without Telegram long-polling.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override FastAPI webhook port (default: WEBHOOK_PORT from .env).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logger = setup_logging(settings)

    if args.check:
        agent = EnsembleDecisionAgent(settings)
        decision = agent.analyze_symbol("BTCUSDT", "15m")
        logger.info(
            "System self-check passed",
            symbol=decision.symbol,
            action=decision.action,
            confidence=decision.confidence,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit_1=decision.take_profit_1,
        )
        return 0

    try:
        asyncio.run(
            run_services(
                settings=settings,
                webhook_only=args.webhook_only,
                port_override=args.port,
            )
        )
    except KeyboardInterrupt:
        logger.info("NexusTrade AI stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
