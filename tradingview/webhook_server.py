from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
import hmac
import json
from pathlib import Path
import time
from typing import Any, Awaitable, Callable, Optional

from fastapi import BackgroundTasks, FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from tradingview.alert_parser import AlertParser, TradingViewAlertPayload
from tradingview.chart_generator import ChartSnapshotGenerator

logger = get_logger("nexustrade.tradingview.webhook")

MAX_WEBHOOK_BODY_BYTES = 65_536
WEBHOOK_RATE_LIMIT_CALLS = 30
WEBHOOK_RATE_LIMIT_WINDOW = 60.0

WebhookDispatchCallback = Callable[
    [TradingViewAlertPayload, SignalDecision, Optional[Path]],
    Awaitable[bool],
]


def verify_webhook_secret(
    expected_secret: str,
    header_secret: Optional[str],
    payload_secret: Optional[str],
) -> bool:
    """
    Constant-time verification of TradingView webhook secret.
    If X-TradingView-Secret header is provided, it must match expected_secret;
    if payload also provides secret, it must also match expected_secret.
    """
    if not expected_secret:
        return False

    expected_bytes = expected_secret.encode("utf-8")

    if header_secret is not None:
        header_ok = hmac.compare_digest(header_secret.strip().encode("utf-8"), expected_bytes)
        if not header_ok:
            return False
        if payload_secret is not None:
            return hmac.compare_digest(payload_secret.strip().encode("utf-8"), expected_bytes)
        return True

    if payload_secret is not None:
        return hmac.compare_digest(payload_secret.strip().encode("utf-8"), expected_bytes)

    return False


def create_webhook_app(
    settings: Optional[Settings] = None,
    ensemble_agent: Optional[EnsembleDecisionAgent] = None,
    chart_generator: Optional[ChartSnapshotGenerator] = None,
    dispatch_callback: Optional[WebhookDispatchCallback] = None,
) -> FastAPI:
    """Create and configure the FastAPI TradingView Webhook receiver application."""
    cfg = settings or get_settings()
    agent = ensemble_agent or EnsembleDecisionAgent(cfg)
    parser = AlertParser(cfg)
    charts = chart_generator or ChartSnapshotGenerator(cfg)
    is_prod = cfg.environment == "production"

    app = FastAPI(
        title="NexusTrade AI - TradingView Webhook Server",
        version="1.0.0",
        description="Institutional real-time TradingView alert receiver and AI Ensemble validator",
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
    )

    app.state.settings = cfg
    app.state.ensemble_agent = agent
    app.state.alert_parser = parser
    app.state.chart_generator = charts
    app.state.dispatch_callback = dispatch_callback
    app.state.processed_alerts = deque(maxlen=500)

    ip_rate_windows: dict[str, deque[float]] = defaultdict(deque)

    def _check_ip_rate_limit(client_ip: str) -> bool:
        now = time.monotonic()
        history = ip_rate_windows[client_ip]
        cutoff = now - WEBHOOK_RATE_LIMIT_WINDOW
        while history and history[0] <= cutoff:
            history.popleft()
        if len(history) >= WEBHOOK_RATE_LIMIT_CALLS:
            return False
        history.append(now)
        return True

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "Webhook payload validation failed",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "error",
                "code": "INVALID_PAYLOAD",
                "detail": "Payload failed schema validation",
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception in webhook server",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "code": "INTERNAL_ERROR",
                "detail": "Webhook processing encountered an internal error",
            },
        )

    @app.get("/health")
    async def healthcheck() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": cfg.app_name,
            "environment": cfg.environment,
            "processed_alerts": len(app.state.processed_alerts),
        }

    async def _dispatch_signal_task(
        alert: TradingViewAlertPayload,
        decision: SignalDecision,
    ) -> None:
        try:
            df = agent.load_market_series(alert.canonical_symbol, timeframe=alert.interval)
            chart_path = await charts.generate_chart_async(
                df=df,
                symbol=alert.canonical_symbol,
                timeframe=alert.interval,
                decision=decision,
            )
            callback: Optional[WebhookDispatchCallback] = app.state.dispatch_callback
            if callback is not None:
                await callback(alert, decision, chart_path)
        except Exception as exc:
            logger.error(
                "Failed to dispatch webhook signal to Telegram",
                symbol=alert.canonical_symbol,
                error=str(exc),
            )

    @app.post("/api/v1/tradingview/webhook")
    async def receive_tradingview_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_tradingview_secret: Optional[str] = Header(
            default=None, alias="X-TradingView-Secret"
        ),
    ) -> JSONResponse:
        client_host = request.client.host if request.client else "unknown"
        if not _check_ip_rate_limit(client_host):
            logger.warning("Throttled excessive webhook requests", client_host=client_host)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "code": "RATE_LIMITED",
                    "detail": "Too many webhook requests from this host",
                },
            )

        raw_bytes = await request.body()
        if len(raw_bytes) > MAX_WEBHOOK_BODY_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "status": "error",
                    "code": "PAYLOAD_TOO_LARGE",
                    "detail": "Webhook payload exceeds maximum allowed size",
                },
            )

        try:
            raw_body = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "code": "MALFORMED_JSON",
                    "detail": "Request body must be valid JSON",
                },
            )

        if not isinstance(raw_body, dict):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "code": "INVALID_JSON_TYPE",
                    "detail": "Webhook JSON payload must be an object",
                },
            )

        payload_secret = raw_body.get("secret")
        if not verify_webhook_secret(
            expected_secret=cfg.tradingview_webhook_secret,
            header_secret=x_tradingview_secret,
            payload_secret=str(payload_secret) if payload_secret is not None else None,
        ):
            logger.warning(
                "Rejected unauthorized TradingView webhook request",
                ticker=raw_body.get("ticker"),
                client_host=client_host,
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "status": "error",
                    "code": "UNAUTHORIZED",
                    "detail": "Invalid or missing TradingView webhook secret",
                },
            )

        try:
            alert = parser.parse(raw_body)
        except Exception as exc:
            logger.warning("Payload schema validation failed", error=str(exc))
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "status": "error",
                    "code": "VALIDATION_ERROR",
                    "detail": "Payload failed schema validation",
                },
            )

        if is_prod and cfg.webhook_max_age_seconds > 0:
            alert_dt = datetime.fromisoformat(alert.timestamp.replace("Z", "+00:00"))
            age_seconds = abs((datetime.now(timezone.utc) - alert_dt).total_seconds())
            if age_seconds > cfg.webhook_max_age_seconds:
                logger.warning(
                    "Rejected stale TradingView webhook alert",
                    ticker=alert.ticker,
                    age_seconds=round(age_seconds, 1),
                )
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content={
                        "status": "error",
                        "code": "STALE_TIMESTAMP",
                        "detail": "Alert timestamp is outside the allowed freshness window",
                    },
                )

        decision = agent.analyze_symbol(
            symbol=alert.canonical_symbol,
            timeframe=alert.interval,
            price_override=alert.price,
            action_hint=alert.normalized_action,
            indicator_overrides=alert.indicators.to_float_dict(),
        )

        should_dispatch = (
            decision.approved
            and decision.action in ("BUY", "SELL")
            and decision.confidence >= cfg.signal_confidence_threshold
        )

        record = {
            "alert": alert.model_dump(),
            "decision": asdict(decision),
            "dispatched": should_dispatch,
        }
        app.state.processed_alerts.append(record)

        if should_dispatch:
            background_tasks.add_task(_dispatch_signal_task, alert, decision)

        logger.info(
            "Processed TradingView webhook alert",
            ticker=alert.ticker,
            canonical_symbol=alert.canonical_symbol,
            action=decision.action,
            confidence=decision.confidence,
            dispatched=should_dispatch,
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "queued": should_dispatch,
                "symbol": alert.canonical_symbol,
                "interval": alert.interval,
                "strategy": alert.strategy,
                "decision": {
                    "action": decision.action,
                    "badge": decision.badge,
                    "confidence": decision.confidence,
                    "risk_multiplier": decision.risk_multiplier,
                    "entry_price": decision.entry_price,
                    "stop_loss": decision.stop_loss,
                    "take_profit_1": decision.take_profit_1,
                    "take_profit_2": decision.take_profit_2,
                    "position_units": decision.position_units,
                    "approved": decision.approved,
                    "rejection_reason": decision.rejection_reason,
                },
            },
        )

    return app


app = create_webhook_app()
