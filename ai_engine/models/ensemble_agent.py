from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ai_engine.models.ohlcv_predictor import OHLCVPredictor, TechnicalPrediction
from ai_engine.models.market_data_loader import load_local_or_synthetic_ohlcv
from ai_engine.models.sentiment_analyzer import (
    FinancialSentimentAnalyzer,
    SentimentAnalysisResult,
)
from ai_engine.models.sentiment_lexicon import classify_sentiment_polarity
from ai_engine.models.volatility_estimator import (
    RealizedVolatilityEstimator,
    VolatilityForecast,
)
from ai_engine.risk_management.position_sizer import PositionSizer, PositionSizeResult
from ai_engine.risk_management.stop_loss_manager import StopLossManager, StopLossTargets
from ai_engine.risk_management.support_resistance import (
    SupportResistanceDetector,
    SupportResistanceLevels,
)
from config.logging_config import get_logger
from data_pipeline.live_market_feed import LiveMarketDataFeed
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import normalize_symbol
from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

logger = get_logger("nexustrade.ai_engine.ensemble")

ActionType = Literal["BUY", "SELL", "HOLD"]
BadgeType = Literal["STRONG BUY", "BUY", "NEUTRAL / HOLD", "SELL", "STRONG SELL"]


@dataclass
class SignalDecision:
    """
    Unified multimodal AI ensemble trading decision combining Technical (S_tech),
    Realized Volatility (S_vol), FinBERT Sentiment (S_sent), and Risk Guardrails.
    """

    action: ActionType
    confidence: float
    risk_multiplier: float
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    badge: BadgeType = "NEUTRAL / HOLD"
    s_tech: float = 0.0
    s_vol: float = 0.0
    s_sent: float = 0.0
    signed_composite: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    atr_14: float = 0.0
    position_units: float = 0.0
    position_notional: float = 0.0
    account_risk_pct: float = 0.0
    kelly_fraction: float = 0.0
    approved: bool = True
    rejection_reason: Optional[str] = None
    volatility_regime: str = "NORMAL"
    sentiment_label: str = "NEUTRAL"
    indicators: dict[str, float] = field(default_factory=dict)
    headlines: list[str] = field(default_factory=list)
    newly_learned_headlines: int = 0
    total_corpus_size: int = 0
    validation_accuracy: float = 0.0
    validation_roc_auc: float = 0.0
    memory_bars_count: int = 0
    five_year_percentile: float = 50.0
    macro_regime: str = "Normal"
    support_1: float = 0.0
    support_2: float = 0.0
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    pivot_point: float = 0.0
    dist_to_r1_pct: float = 0.0
    dist_to_s1_pct: float = 0.0
    sr_status: str = "Tengah Jalur S1–R1 (Ruang Gerak Lega)"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    @property
    def meets_confidence_threshold(self) -> bool:
        return self.approved and self.action in ("BUY", "SELL") and self.confidence >= 0.72


class EnsembleDecisionAgent:
    """
    Meta-learner synthesizing Technical Predictor, Realized Volatility Estimator,
    and FinBERT Sentiment Engine into an institutional risk-managed SignalDecision.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        ohlcv_predictor: Optional[OHLCVPredictor] = None,
        volatility_estimator: Optional[RealizedVolatilityEstimator] = None,
        sentiment_analyzer: Optional[FinancialSentimentAnalyzer] = None,
        position_sizer: Optional[PositionSizer] = None,
        stop_loss_manager: Optional[StopLossManager] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ohlcv_preprocessor = OHLCVPreprocessor(self.settings)
        self.ohlcv_predictor = ohlcv_predictor or OHLCVPredictor(
            self.settings, self.ohlcv_preprocessor
        )
        self.volatility_estimator = volatility_estimator or RealizedVolatilityEstimator(
            self.settings
        )
        self.sentiment_analyzer = sentiment_analyzer or FinancialSentimentAnalyzer(
            self.settings
        )
        self.live_feed = LiveMarketDataFeed(
            settings=self.settings,
            preprocessor=self.ohlcv_preprocessor,
            ttl_seconds=self.settings.live_market_cache_ttl_seconds,
        )
        self.position_sizer = position_sizer or PositionSizer(self.settings)
        self.stop_loss_manager = stop_loss_manager or StopLossManager(self.settings)
        self.sr_detector = SupportResistanceDetector()
        self._market_data_cache: dict[str, pd.DataFrame] = {}
        self._last_trained_signature: dict[str, tuple[str, float]] = {}

    @staticmethod
    def resolve_badge(action: ActionType, confidence: float) -> BadgeType:
        if action == "BUY":
            return "STRONG BUY" if confidence >= 0.72 else "BUY"
        if action == "SELL":
            return "STRONG SELL" if confidence >= 0.72 else "SELL"
        return "NEUTRAL / HOLD"

    def synthesize_scores(
        self,
        s_tech: float,
        s_vol: float,
        s_sent: float,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        entry_price: float = 67420.50,
        atr_14: Optional[float] = None,
        action_hint: Optional[ActionType] = None,
        portfolio_equity: Optional[float] = None,
        daily_drawdown_override: Optional[float] = None,
        indicators: Optional[dict[str, float]] = None,
        headlines: Optional[list[str]] = None,
        newly_learned_headlines: int = 0,
        total_corpus_size: int = 0,
        validation_accuracy: float = 0.0,
        validation_roc_auc: float = 0.0,
        memory_bars_count: int = 0,
        five_year_percentile: float = 50.0,
        macro_regime: str = "Normal",
        sr_levels: Optional[SupportResistanceLevels] = None,
    ) -> SignalDecision:
        """
        Apply AGENT.md Section 5.A Ensemble Decision Matrix:
          Signal Confidence = |w_1 * S_tech + w_2 * S_sent|
          Position Size = Base Size * (1 - S_vol) * Confidence
        """
        w1 = self.settings.ensemble_tech_weight
        w2 = self.settings.ensemble_sent_weight

        clamped_tech = float(np.clip(s_tech, -1.0, 1.0))
        clamped_sent = float(np.clip(s_sent, -1.0, 1.0))
        clamped_vol = float(np.clip(s_vol, 0.0, 1.0))

        signed_composite = w1 * clamped_tech + w2 * clamped_sent
        confidence = round(float(np.clip(abs(signed_composite), 0.0, 1.0)), 4)
        risk_multiplier = round(1.0 - clamped_vol, 4)

        if signed_composite >= 0.15:
            raw_action: ActionType = "BUY"
        elif signed_composite <= -0.15:
            raw_action = "SELL"
        elif action_hint in ("BUY", "SELL") and abs(signed_composite) >= 0.08:
            raw_action = action_hint
        else:
            raw_action = "HOLD"

        effective_atr = atr_14 if (atr_14 is not None and atr_14 > 0) else entry_price * 0.012
        resolved_sr = sr_levels or self.sr_detector.detect_levels(
            None, current_price=entry_price, atr_14=effective_atr
        )
        targets: StopLossTargets = self.stop_loss_manager.calculate_targets(
            entry_price=entry_price,
            atr_14=effective_atr,
            action=raw_action,
        )

        sizing: PositionSizeResult = self.position_sizer.calculate_position(
            entry_price=entry_price,
            stop_loss=targets.stop_loss,
            confidence=confidence,
            s_vol=clamped_vol,
            portfolio_equity=portfolio_equity,
            reward_risk_ratio=targets.reward_risk_ratio_1,
            daily_drawdown_override=daily_drawdown_override,
        )

        final_action: ActionType = raw_action if sizing.approved else "HOLD"
        badge = self.resolve_badge(final_action, confidence)

        vol_regime = self.volatility_estimator.classify_regime(
            clamped_vol, self.settings.max_volatility_threshold
        )
        sent_label = classify_sentiment_polarity(clamped_sent)

        return SignalDecision(
            action=final_action,
            confidence=confidence,
            risk_multiplier=risk_multiplier,
            symbol=symbol.upper(),
            timeframe=timeframe,
            badge=badge,
            s_tech=round(clamped_tech, 4),
            s_vol=round(clamped_vol, 4),
            s_sent=round(clamped_sent, 4),
            signed_composite=round(signed_composite, 4),
            entry_price=targets.entry_price,
            stop_loss=targets.stop_loss,
            take_profit_1=targets.take_profit_1,
            take_profit_2=targets.take_profit_2,
            atr_14=targets.atr_14,
            position_units=sizing.position_units,
            position_notional=sizing.position_notional,
            account_risk_pct=sizing.account_risk_pct,
            kelly_fraction=sizing.kelly_fraction,
            approved=sizing.approved,
            rejection_reason=sizing.rejection_reason,
            volatility_regime=vol_regime,
            sentiment_label=sent_label,
            indicators=indicators or {},
            headlines=headlines or [],
            newly_learned_headlines=newly_learned_headlines,
            total_corpus_size=total_corpus_size,
            validation_accuracy=validation_accuracy,
            validation_roc_auc=validation_roc_auc,
            memory_bars_count=memory_bars_count,
            five_year_percentile=five_year_percentile,
            macro_regime=macro_regime,
            support_1=resolved_sr.support_1,
            support_2=resolved_sr.support_2,
            resistance_1=resolved_sr.resistance_1,
            resistance_2=resolved_sr.resistance_2,
            pivot_point=resolved_sr.pivot_point,
            dist_to_r1_pct=resolved_sr.dist_to_r1_pct,
            dist_to_s1_pct=resolved_sr.dist_to_s1_pct,
            sr_status=resolved_sr.status_label,
        )

    def load_market_series(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        limit: int = 400,
    ) -> pd.DataFrame:
        """Load real-time OHLCV series from live market APIs with fallback to local datasets."""
        clean_sym = normalize_symbol(symbol)
        if self.settings.use_live_market_data:
            live_df = self.live_feed.fetch_live_ohlcv(
                symbol=clean_sym, timeframe=timeframe, limit=limit
            )
            if live_df is not None and len(live_df) >= 30:
                return live_df.tail(limit).copy()

        return load_local_or_synthetic_ohlcv(
            symbol=clean_sym,
            timeframe=timeframe,
            limit=limit,
            settings=self.settings,
            preprocessor=self.ohlcv_preprocessor,
            cache=self._market_data_cache,
        )

    def analyze_symbol(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        df: Optional[pd.DataFrame] = None,
        headlines: Optional[list[str]] = None,
        price_override: Optional[float] = None,
        action_hint: Optional[ActionType] = None,
        indicator_overrides: Optional[dict[str, float]] = None,
        daily_drawdown_override: Optional[float] = None,
    ) -> SignalDecision:
        """
        Execute end-to-end multimodal inference (LightGBM Technical + Optiver Volatility
        + FinBERT Sentiment + Kelly/ATR Risk Engine) for a symbol or live webhook alert.
        """
        clean_sym = normalize_symbol(symbol)
        market_df = (
            df.copy()
            if df is not None and not df.empty
            else self.load_market_series(clean_sym, timeframe=timeframe)
        )
        df_5y: Optional[pd.DataFrame] = None
        if self.settings.use_live_market_data and df is None:
            df_5y = self.live_feed.fetch_5y_daily_memory(clean_sym)

        sent_result: SentimentAnalysisResult = self.sentiment_analyzer.analyze_headlines(
            headlines=headlines,
            symbol=clean_sym,
            market_df=market_df,
        )
        sig_key = f"{clean_sym}_{timeframe}"
        latest_bar_ts = str(market_df.index[-1]) if len(market_df) > 0 else ""
        latest_bar_close = float(market_df["close"].iloc[-1]) if len(market_df) > 0 else 0.0
        current_sig = (latest_bar_ts, round(latest_bar_close, 4))
        data_changed = self._last_trained_signature.get(sig_key) != current_sig

        if (sent_result.newly_learned_count > 0 or data_changed) and len(market_df) >= 60:
            self.ohlcv_predictor.train(
                market_df, symbol=clean_sym, timeframe=timeframe, df_5y=df_5y
            )
            self.volatility_estimator.fit(market_df)
            self._last_trained_signature[sig_key] = current_sig

        tech_pred: TechnicalPrediction = self.ohlcv_predictor.predict(
            market_df,
            symbol=clean_sym,
            timeframe=timeframe,
            indicator_overrides=indicator_overrides,
            df_5y=df_5y,
        )
        vol_forecast: VolatilityForecast = self.volatility_estimator.estimate(
            market_df,
            symbol=clean_sym,
        )
        s_tech = tech_pred.s_tech
        s_sent = sent_result.s_sent
        # If webhook alert supplies an explicit action + constructive RSI/MACD, align conviction
        if action_hint == "BUY" and indicator_overrides:
            rsi_val = float(indicator_overrides.get("rsi", 55.0))
            macd_val = float(indicator_overrides.get("macd", 1.0))
            if rsi_val >= 50.0 and macd_val >= 0.0:
                s_tech = max(s_tech, 0.90)
                if s_sent > -0.35:
                    s_sent = max(s_sent, 0.50)
        elif action_hint == "SELL" and indicator_overrides:
            rsi_val = float(indicator_overrides.get("rsi", 45.0))
            macd_val = float(indicator_overrides.get("macd", -1.0))
            if rsi_val <= 50.0 and macd_val <= 0.0:
                s_tech = min(s_tech, -0.90)
                if s_sent < 0.35:
                    s_sent = min(s_sent, -0.50)
        entry_price = (
            float(price_override)
            if price_override is not None and price_override > 0
            else tech_pred.latest_price
        )
        atr_val = (
            float(indicator_overrides["atr"])
            if indicator_overrides and indicator_overrides.get("atr")
            else tech_pred.atr_14
        )

        macro_ctx = self.live_feed.compute_5y_macro_context(df_5y, entry_price)

        sr_levels = self.sr_detector.detect_levels(
            market_df, current_price=entry_price, atr_14=atr_val
        )
        if action_hint is None:
            s_tech = float(np.clip(s_tech + sr_levels.sr_bias, -1.0, 1.0))
        return self.synthesize_scores(
            s_tech=s_tech,
            s_vol=vol_forecast.s_vol,
            s_sent=s_sent,
            symbol=clean_sym,
            timeframe=timeframe,
            entry_price=entry_price,
            atr_14=atr_val,
            action_hint=action_hint,
            daily_drawdown_override=daily_drawdown_override,
            indicators=tech_pred.indicators,
            headlines=[h.headline for h in sent_result.headline_scores],
            newly_learned_headlines=sent_result.newly_learned_count,
            total_corpus_size=sent_result.total_corpus_size,
            validation_accuracy=tech_pred.validation_accuracy,
            validation_roc_auc=tech_pred.validation_roc_auc,
            memory_bars_count=tech_pred.memory_bars_trained or len(market_df),
            five_year_percentile=float(macro_ctx["five_year_percentile"]),
            macro_regime=str(macro_ctx["macro_regime"]),
            sr_levels=sr_levels,
        )

    def learn_online_cycle(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
    ) -> dict[str, Any]:
        """
        Execute a full 'Learn-by-Doing' online learning cycle from the internet:
        1. Fetch live OHLCV market candles from Binance / Yahoo Finance.
        2. Harvest breaking news from Google News RSS & Yahoo Finance News.
        3. Label unseen headlines using teacher NLP + realized market return feedback
           and incrementally update the online SGDClassifier via partial_fit().
        4. Retrain LightGBM OHLCVPredictor and Optiver HAR-RV VolatilityEstimator on live bars.
        """
        clean_sym = normalize_symbol(symbol)
        market_df = self.load_market_series(clean_sym, timeframe=timeframe, limit=400)
        df_5y: Optional[pd.DataFrame] = None
        if self.settings.use_live_market_data:
            df_5y = self.live_feed.fetch_5y_daily_memory(clean_sym)

        sent_result = self.sentiment_analyzer.analyze_headlines(
            symbol=clean_sym,
            market_df=market_df,
        )
        newly_learned = sent_result.newly_learned_count

        roc_auc = 0.0
        if len(market_df) >= 60:
            roc_auc = self.ohlcv_predictor.train(
                market_df, symbol=clean_sym, timeframe=timeframe, df_5y=df_5y
            )
            self.volatility_estimator.fit(market_df)

        tech_pred = self.ohlcv_predictor.predict(
            market_df, symbol=clean_sym, timeframe=timeframe, df_5y=df_5y
        )
        vol_forecast = self.volatility_estimator.estimate(
            market_df, symbol=clean_sym
        )
        macro_ctx = self.live_feed.compute_5y_macro_context(df_5y, tech_pred.latest_price)
        sr_levels = self.sr_detector.detect_levels(
            market_df, current_price=tech_pred.latest_price, atr_14=tech_pred.atr_14
        )
        decision = self.synthesize_scores(
            s_tech=tech_pred.s_tech,
            s_vol=vol_forecast.s_vol,
            s_sent=sent_result.s_sent,
            symbol=clean_sym,
            timeframe=timeframe,
            entry_price=tech_pred.latest_price,
            atr_14=tech_pred.atr_14,
            indicators=tech_pred.indicators,
            headlines=[h.headline for h in sent_result.headline_scores],
            newly_learned_headlines=newly_learned,
            total_corpus_size=sent_result.total_corpus_size,
            validation_accuracy=tech_pred.validation_accuracy,
            validation_roc_auc=tech_pred.validation_roc_auc,
            memory_bars_count=tech_pred.memory_bars_trained or len(market_df),
            five_year_percentile=float(macro_ctx["five_year_percentile"]),
            macro_regime=str(macro_ctx["macro_regime"]),
            sr_levels=sr_levels,
        )

        return {
            "symbol": clean_sym,
            "timeframe": timeframe,
            "latest_price": tech_pred.latest_price,
            "live_headlines_fetched": len(sent_result.headline_scores),
            "newly_learned_headlines": newly_learned,
            "total_corpus_size": sent_result.total_corpus_size,
            "online_updates_count": self.sentiment_analyzer.online_updates_count,
            "validation_roc_auc": roc_auc or tech_pred.validation_roc_auc,
            "validation_accuracy": tech_pred.validation_accuracy,
            "memory_bars_count": tech_pred.memory_bars_trained or len(market_df),
            "five_year_percentile": macro_ctx["five_year_percentile"],
            "macro_regime": macro_ctx["macro_regime"],
            "s_tech": tech_pred.s_tech,
            "s_sent": sent_result.s_sent,
            "s_vol": vol_forecast.s_vol,
            "action": decision.action,
            "confidence": decision.confidence,
            "sample_headlines": [
                {
                    "headline": h.headline,
                    "polarity": h.polarity,
                    "source": h.source,
                    "published_at": h.published_at,
                }
                for h in sent_result.headline_scores[:4]
            ],
            "timestamp": decision.timestamp,
        }
