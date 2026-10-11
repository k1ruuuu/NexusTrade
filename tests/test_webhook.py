from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx
import pytest
from PIL import Image

from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from config.settings import Settings
from tradingview.alert_parser import AlertParser, TradingViewAlertPayload
from tradingview.chart_generator import ChartSnapshotGenerator
from tradingview.webhook_server import create_webhook_app


def test_alert_parser_validation_and_mapping() -> None:
    parser = AlertParser()
    raw = {
        "secret": "test_secret_12345",
        "ticker": "BINANCE:BTCUSDT",
        "interval": "15m",
        "action": "BUY",
        "price": 67420.50,
        "strategy": "TrendBreakout_v2",
        "indicators": {"rsi": 62.4, "macd": 142.1, "atr": 450.0},
        "timestamp": "2026-10-11T04:50:00Z",
    }
    parsed = parser.parse(raw)
    assert parsed.canonical_symbol == "BTCUSDT"
    assert parsed.normalized_action == "BUY"
    assert parsed.price == 67420.50
    assert parsed.indicators.rsi == 62.4


def test_chart_generator_renders_1200x800_png(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", chart_width=1200, chart_height=800)
    agent = EnsembleDecisionAgent(settings=settings)
    df = agent.load_market_series("XAUUSD", "15m", limit=100)
    decision = agent.analyze_symbol("XAUUSD", "15m", df=df)

    generator = ChartSnapshotGenerator(settings=settings)
    chart_path = generator.generate_chart(
        df=df,
        symbol="XAUUSD",
        timeframe="15m",
        decision=decision,
        output_path=tmp_path / "xauusd_snapshot.png",
    )

    assert chart_path.exists()
    with Image.open(chart_path) as img:
        assert img.size == (1200, 800)


@pytest.mark.asyncio
async def test_webhook_server_auth_and_dispatch(tmp_path: Path) -> None:
    secret = "institutional_webhook_secret_2026"
    settings = Settings(
        data_dir=tmp_path / "data",
        tradingview_webhook_secret=secret,
        signal_confidence_threshold=0.70,
    )
    agent = EnsembleDecisionAgent(settings=settings)
    charts = ChartSnapshotGenerator(settings=settings)
    dispatched_events: list[tuple[TradingViewAlertPayload, SignalDecision, Optional[Path]]] = []

    async def fake_dispatcher(
        alert: TradingViewAlertPayload,
        decision: SignalDecision,
        chart_path: Optional[Path],
    ) -> bool:
        dispatched_events.append((alert, decision, chart_path))
        return True

    app = create_webhook_app(
        settings=settings,
        ensemble_agent=agent,
        chart_generator=charts,
        dispatch_callback=fake_dispatcher,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Healthcheck endpoint
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        valid_payload = {
            "secret": secret,
            "ticker": "BINANCE:BTCUSDT",
            "interval": "15m",
            "action": "BUY",
            "price": 67420.50,
            "strategy": "TrendBreakout_v2",
            "indicators": {
                "rsi": 62.4,
                "macd": 142.1,
                "atr": 450.0,
            },
            "timestamp": "2026-10-11T04:50:00Z",
        }

        # 2. Invalid secret returns 401 Unauthorized
        bad_resp = await client.post(
            "/api/v1/tradingview/webhook",
            headers={"X-TradingView-Secret": "wrong_secret"},
            json=valid_payload,
        )
        assert bad_resp.status_code == 401
        assert bad_resp.json()["code"] == "UNAUTHORIZED"

        # 3. Valid secret returns 200 OK and dispatches signal
        ok_resp = await client.post(
            "/api/v1/tradingview/webhook",
            headers={"X-TradingView-Secret": secret},
            json=valid_payload,
        )
        assert ok_resp.status_code == 200
        body = ok_resp.json()
        assert body["status"] == "ok"
        assert body["symbol"] == "BTCUSDT"
        assert body["decision"]["stop_loss"] > 0
        assert body["decision"]["take_profit_1"] > body["decision"]["entry_price"]
        assert body["queued"] is True
        assert len(dispatched_events) == 1
        assert dispatched_events[0][2] is not None and dispatched_events[0][2].exists()
