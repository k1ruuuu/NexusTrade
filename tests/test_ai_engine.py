from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ai_engine.backtester.engine import EventDrivenBacktester
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from ai_engine.models.ohlcv_predictor import OHLCVPredictor
from ai_engine.models.sentiment_analyzer import FinancialSentimentAnalyzer
from ai_engine.models.volatility_estimator import RealizedVolatilityEstimator
from ai_engine.risk_management.stop_loss_manager import StopLossManager
from config.settings import Settings


def test_ohlcv_predictor_roc_auc_and_bounds() -> None:
    agent = EnsembleDecisionAgent()
    btc_df = agent.load_market_series("BTCUSDT", "1d", limit=500)

    predictor = OHLCVPredictor()
    roc_auc = predictor.train(btc_df, symbol="BTCUSDT", timeframe="1d")
    assert roc_auc > 0.58, f"Expected validation ROC-AUC > 0.58, got {roc_auc}"

    pred = predictor.predict(btc_df, symbol="BTCUSDT", timeframe="1d")
    assert -1.0 <= pred.s_tech <= 1.0
    assert pred.action in ("BUY", "SELL", "HOLD")
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-3
    assert pred.atr_14 > 0.0


def test_realized_volatility_estimator_bounds() -> None:
    agent = EnsembleDecisionAgent()
    xau_df = agent.load_market_series("XAUUSD", "15m", limit=300)

    estimator = RealizedVolatilityEstimator()
    forecast = estimator.estimate(xau_df, symbol="XAUUSD")
    assert 0.0 <= forecast.s_vol <= 1.0
    assert 0.0 <= forecast.percentile_rank <= 1.0
    assert forecast.regime in ("LOW", "NORMAL", "ELEVATED", "EXTREME")
    assert forecast.current_rv >= 0.0
    assert forecast.predicted_rv > 0.0


def test_finbert_sentiment_engine_polarity(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    analyzer = FinancialSentimentAnalyzer(settings=settings, use_transformers=False)
    bullish = analyzer.score_headline(
        "Bitcoin institutional ETF inflows surge to record high on strong adoption"
    )
    bearish = analyzer.score_headline(
        "Exchange hack triggers massive liquidation selloff and regulatory lawsuit"
    )
    neutral = analyzer.score_headline(
        "Scheduled maintenance window completed during overnight session"
    )

    assert bullish.polarity > 0.25
    assert bullish.label == "POSITIVE"
    assert bearish.polarity < -0.25
    assert bearish.label == "NEGATIVE"
    assert neutral.label == "NEUTRAL"

    agg = analyzer.analyze_headlines(
        [
            "Institutional inflows surge on strong earnings breakout",
            "Profit growth beats expectations with record upside",
        ],
        symbol="BTCUSDT",
    )
    assert -1.0 <= agg.s_sent <= 1.0
    assert agg.s_sent > 0.30
    assert agg.bullish_ratio == 1.0


def test_ensemble_decision_and_risk_guardrails() -> None:
    settings = Settings(
        ensemble_tech_weight=0.65,
        ensemble_sent_weight=0.35,
        max_account_risk_per_trade=0.015,
        atr_stop_loss_multiplier=2.0,
        min_risk_reward_ratio=2.0,
        max_daily_drawdown_limit=0.04,
        max_volatility_threshold=0.85,
    )
    agent = EnsembleDecisionAgent(settings=settings)

    # 1. Standard high-confidence BUY decision matching TASKS.md 3.4 criteria
    decision = agent.synthesize_scores(
        s_tech=0.88,
        s_vol=0.25,
        s_sent=0.71,
        symbol="BTCUSDT",
        timeframe="15m",
        entry_price=67420.50,
        atr_14=450.0,
    )
    assert decision.action == "BUY"
    assert round(decision.confidence, 2) == 0.82
    assert decision.risk_multiplier == 0.75
    assert decision.approved is True
    assert decision.stop_loss == round(67420.50 - 2.0 * 450.0, 4)
    assert decision.take_profit_1 == round(67420.50 + 4.0 * 450.0, 4)
    assert decision.account_risk_pct <= 0.015 + 1e-6

    # 2. Extreme volatility safety suppression (Task 6.2)
    high_vol_decision = agent.synthesize_scores(
        s_tech=0.90,
        s_vol=0.92,
        s_sent=0.80,
        symbol="BTCUSDT",
        entry_price=67420.50,
        atr_14=450.0,
    )
    assert high_vol_decision.approved is False
    assert high_vol_decision.action == "HOLD"
    assert high_vol_decision.position_units == 0.0
    assert "volatility" in (high_vol_decision.rejection_reason or "").lower()

    # 3. Daily drawdown circuit breaker (Task 6.2: > 4% daily loss halts signals)
    cb_decision = agent.synthesize_scores(
        s_tech=0.90,
        s_vol=0.20,
        s_sent=0.85,
        symbol="BTCUSDT",
        entry_price=67420.50,
        atr_14=450.0,
        daily_drawdown_override=0.045,
    )
    assert cb_decision.approved is False
    assert cb_decision.action == "HOLD"
    assert "circuit breaker" in (cb_decision.rejection_reason or "").lower()

    # 4. Trailing stop ratcheting (Task 6.1)
    sl_mgr = StopLossManager(settings)
    new_sl = sl_mgr.update_trailing_stop(
        action="BUY",
        current_price=69000.0,
        current_stop_loss=decision.stop_loss,
        trailing_distance=900.0,
    )
    assert new_sl == 68100.0
    # Price pullback must never loosen trailing stop
    assert sl_mgr.update_trailing_stop("BUY", 67500.0, new_sl, 900.0) == new_sl


def test_event_driven_backtester_report_and_plot(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    agent = EnsembleDecisionAgent(settings=settings)
    btc_df = agent.load_market_series("BTCUSDT", "1d", limit=180)

    backtester = EventDrivenBacktester(settings=settings)
    report = backtester.run(btc_df, symbol="BTCUSDT", days=90, generate_plot=True)
    report_dict = report.to_dict()

    assert report_dict["symbol"] == "BTCUSDT"
    assert len(report.equity_curve) > 10
    assert report.chart_path is not None and report.chart_path.exists()
    assert "sharpe_ratio" in report_dict["metrics"]
    assert "max_drawdown_pct" in report_dict["metrics"]


def test_concurrent_train_and_predict_thread_safety(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        use_live_market_data=False,
        use_live_news_feed=False,
    )
    agent = EnsembleDecisionAgent(settings=settings)

    def _worker(idx: int) -> str:
        sym = "BTCUSDT" if idx % 2 == 0 else "XAUUSD"
        if idx % 3 == 0:
            return str(agent.learn_online_cycle(sym, "15m")["action"])
        return agent.analyze_symbol(sym, "15m").action

    with ThreadPoolExecutor(max_workers=6) as pool:
        actions = list(pool.map(_worker, range(12)))

    assert len(actions) == 12
    assert all(a in ("BUY", "SELL", "HOLD") for a in actions)
