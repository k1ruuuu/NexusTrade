from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ai_engine.models.ohlcv_predictor import OHLCVPredictor, TechnicalPrediction
from ai_engine.models.confluence_analyzer import (
    ConfluenceReport,
    MultiFactorConfluenceAnalyzer,
)
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
from data_pipeline.candlestick_reader import CandlestickReader
from src.features.structure_extractor import CandleStructuralExtractor
from src.models.macro_regime import MacroRegimeDetector
from ai_engine.models.meta_label_engine import (
    AdvancedFeatureExtractor,
    MacroRiskFilter,
    TwoStageMetaLabelEngine,
)

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
    confluence_pillars: int = 5
    cvd_status: str = "Seimbang (0.00)"
    relative_volume: float = 1.0
    expected_move_1d: float = 0.0
    expectancy_r: float = 0.50
    adx_14: float = 25.0
    is_sideways: bool = False
    market_structure_status: str = "Trending Normal (ADX 25.0)"
    mtf_summary: str = "1D: — | 1H: — | 15M: — | 5M: — ➔ Action 1M: SIAP"
    mtf_aligned_count: int = 4
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    meta_probability: float = 0.0
    dxy_filter_triggered: bool = False
    macro_stance: str = "NEUTRAL"
    gold_action_bias: str = "HOLD"
    event_mode: Optional[str] = None
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
        self.structural_extractor = CandleStructuralExtractor()
        self.macro_regime_detector = MacroRegimeDetector()
        self._market_data_cache: dict[str, pd.DataFrame] = {}
        self._last_trained_signature: dict[str, tuple[str, float]] = {}
        self.meta_engine = TwoStageMetaLabelEngine(
            gate_threshold=self.settings.signal_confidence_threshold
        )
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
        market_df: Optional[pd.DataFrame] = None,
        annualized_volatility: float = 0.45,
        macro_stance: str = "NEUTRAL",
        gold_action_bias: str = "HOLD",
        event_mode: Optional[str] = None,
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
        # Calibrated weight redistribution when news is neutral (never mutates s_sent)
        if (
            abs(clamped_tech) >= 0.72
            and abs(clamped_sent) < 0.15
            and headlines is not None
            and len(headlines) > 0
        ):
            eff_w1 = min(0.82, w1 + 0.15)
            eff_w2 = max(0.18, 1.0 - eff_w1)
            signed_composite = eff_w1 * clamped_tech + eff_w2 * clamped_sent
        else:
            signed_composite = w1 * clamped_tech + w2 * clamped_sent
        confidence = round(float(np.clip(abs(signed_composite), 0.0, 1.0)), 4)
        risk_multiplier = round(1.0 - clamped_vol, 4)

        if signed_composite >= 0.15:
            raw_action: ActionType = "BUY"
        elif signed_composite <= -0.15:
            raw_action = "SELL"
        elif action_hint in ("BUY", "SELL") and abs(signed_composite) >= 0.08:
            raw_action = action_hint
        elif event_mode and gold_action_bias in ("BUY", "SELL"):
            # Institutional Macro Monetary Transmission in Event Sniper Mode
            raw_action = gold_action_bias
            confidence = max(confidence, 0.76)
        else:
            raw_action = "HOLD"
        effective_atr = atr_14 if (atr_14 is not None and atr_14 > 0) else entry_price * 0.012
        resolved_sr = sr_levels or self.sr_detector.detect_levels(
            None, current_price=entry_price, atr_14=effective_atr
        )
        oc_sig_val = int(indicators.get("oc_signal", 0)) if indicators else 0
        oc_sl_val = float(indicators.get("oc_sl", 0.0)) if indicators else 0.0
        oc_tp_val = float(indicators.get("oc_tp", 0.0)) if indicators else 0.0
        if event_mode:
            targets: StopLossTargets = self.stop_loss_manager.calculate_event_mode_targets(
                entry_price=entry_price,
                action=raw_action,
                account_balance=portfolio_equity or self.settings.portfolio_initial_capital,
                lot_size=0.01,
                symbol=symbol,
                risk_pct=self.settings.max_account_risk_per_trade,
            )
        elif oc_sig_val != 0 and oc_sl_val > 0 and (
            (raw_action == "BUY" and oc_sig_val == 1) or (raw_action == "SELL" and oc_sig_val == -1)
        ):
            # Dynamic RRR anchoring: Calibrate Stop Loss and Take Profit to current entry_price maintaining strict 1:2 RRR
            risk_dist = max(abs(entry_price - oc_sl_val), entry_price * 0.001)
            dyn_tp1 = entry_price + (risk_dist * 2.0) if raw_action == "BUY" else max(entry_price * 0.01, entry_price - (risk_dist * 2.0))
            dyn_tp2 = entry_price + (risk_dist * 3.0) if raw_action == "BUY" else max(entry_price * 0.01, entry_price - (risk_dist * 3.0))
            targets = StopLossTargets(
                action=raw_action,
                entry_price=round(entry_price, 4),
                atr_14=round(effective_atr, 4),
                stop_loss=round(oc_sl_val, 4),
                take_profit_1=round(dyn_tp1, 4),
                take_profit_2=round(dyn_tp2, 4),
                trailing_stop_distance=round(risk_dist * 0.60, 4),
                risk_per_unit=round(risk_dist, 4),
                reward_risk_ratio_1=2.0,
                reward_risk_ratio_2=3.0,
            )
        elif sr_levels is not None:
            targets = self.stop_loss_manager.calculate_realistic_targets(
                entry_price=entry_price,
                atr_14=effective_atr,
                action=raw_action,
                support_1=resolved_sr.support_1,
                support_2=resolved_sr.support_2,
                resistance_1=resolved_sr.resistance_1,
                resistance_2=resolved_sr.resistance_2,
            )
        else:
            targets = self.stop_loss_manager.calculate_targets(
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

        conf_report: ConfluenceReport = MultiFactorConfluenceAnalyzer.evaluate_confluence(
            df=market_df,
            action=final_action,
            s_tech=clamped_tech,
            s_sent=clamped_sent,
            s_vol=clamped_vol,
            entry_price=targets.entry_price,
            stop_loss=targets.stop_loss,
            take_profit_1=targets.take_profit_1,
            support_1=resolved_sr.support_1,
            resistance_1=resolved_sr.resistance_1,
            annualized_volatility=annualized_volatility,
            risk_capital_usd=sizing.adjusted_risk_capital,
        )
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
            confluence_pillars=conf_report.pillars_passed,
            cvd_status=conf_report.cvd_status,
            relative_volume=conf_report.relative_volume,
            expected_move_1d=conf_report.expected_move_1d,
            expectancy_r=conf_report.expectancy_r,
            adx_14=conf_report.adx_14,
            is_sideways=conf_report.is_sideways,
            market_structure_status=conf_report.market_structure_status,
            macro_stance=macro_stance,
            gold_action_bias=gold_action_bias,
            event_mode=event_mode.upper() if event_mode else None,
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
        event_mode: Optional[str] = None,
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
        if self.settings.use_live_market_data:
            df_5y = self.live_feed.fetch_5y_daily_memory(clean_sym)
        sent_result: SentimentAnalysisResult = self.sentiment_analyzer.analyze_headlines(
            symbol=clean_sym,
            headlines=headlines,
            market_df=market_df,
            event_mode=event_mode,
        )
        sig_key = f"{clean_sym}_{timeframe}"
        # Only train synchronously on initial cold-start if model is not yet fitted.
        # Periodic retraining is handled asynchronously in the background learner thread.
        with self.ohlcv_predictor._train_lock:
            if sig_key not in self.ohlcv_predictor._models_by_key and len(market_df) >= 60:
                self.ohlcv_predictor.train(
                    market_df, symbol=clean_sym, timeframe=timeframe, df_5y=df_5y
                )
                self.volatility_estimator.fit(market_df)
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
                s_tech = float(np.clip(0.70 * s_tech + 0.30 * 0.90, -1.0, 1.0))
                if s_sent > -0.35:
                    s_sent = float(np.clip(0.70 * s_sent + 0.30 * 0.50, -1.0, 1.0))
        elif action_hint == "SELL" and indicator_overrides:
            rsi_val = float(indicator_overrides.get("rsi", 45.0))
            macd_val = float(indicator_overrides.get("macd", -1.0))
            if rsi_val <= 50.0 and macd_val <= 0.0:
                s_tech = float(np.clip(0.70 * s_tech + 0.30 * (-0.90), -1.0, 1.0))
                if s_sent < 0.35:
                    s_sent = float(np.clip(0.70 * s_sent + 0.30 * (-0.50), -1.0, 1.0))
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
        hmm_regime = self.macro_regime_detector.evaluate_latest(market_df)
        tech_pred.indicators["regime_prob_bull"] = hmm_regime.prob_bull_expansion
        tech_pred.indicators["regime_prob_chop"] = hmm_regime.prob_consolidation
        tech_pred.indicators["regime_prob_bear"] = hmm_regime.prob_crisis_bear
        tech_pred.indicators["roro_index"] = hmm_regime.roro_index

        sr_levels = self.sr_detector.detect_levels(
            market_df, current_price=entry_price, atr_14=atr_val
        )
        if action_hint is None:
            s_tech = float(np.clip(s_tech + sr_levels.sr_bias, -1.0, 1.0))
        decision = self.synthesize_scores(
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
            market_df=market_df,
            annualized_volatility=vol_forecast.annualized_volatility,
            macro_stance=sent_result.macro_stance,
            gold_action_bias=sent_result.gold_action_bias,
            event_mode=event_mode,
        )
        # Stage 2: Secondary Meta-Model & Macro Risk Filtering (ADVANCED_AI_TRADING_ENGINE_SPEC)
        meta_feat = AdvancedFeatureExtractor.compute_features(
            market_df, sr_level=sr_levels.support_1 if s_tech > 0 else sr_levels.resistance_1
        )
        if len(meta_feat) > 0:
            meta_prob = self.meta_engine.predict_meta_probability(
                meta_feat.iloc[-1], primary_confidence=decision.confidence
            )
        else:
            meta_prob = decision.confidence
        decision.meta_probability = round(meta_prob, 4)

        # Section 5.1: DXY 1-hour trend filter
        if self.settings.use_live_market_data and clean_sym in ("BTCUSDT", "BTCUSD", "XAUUSD", "ETHUSDT"):
            dxy_df = self.live_feed.fetch_live_ohlcv("DXY", timeframe="15m", limit=10)
            dxy_suppressed, dxy_reason = MacroRiskFilter.evaluate_dxy_trend_filter(dxy_df, decision.action)
            if dxy_suppressed:
                decision.dxy_filter_triggered = True
                self._reject(decision, dxy_reason)

        if getattr(self.settings, "macro_blackout_enabled", False) and getattr(sent_result, "is_macro_blackout", False):
            self._reject(decision, f"HIGH_IMPACT_NEWS_BLACKOUT ({sent_result.blackout_event or 'High Impact Event'})")
        return decision

    @staticmethod
    def _reject(decision: SignalDecision, reason: str) -> None:
        decision.action = "HOLD"
        decision.badge = "NEUTRAL / HOLD"
        decision.approved = False
        decision.rejection_reason = reason
        decision.position_units = decision.position_notional = decision.account_risk_pct = 0.0

    def _score_timeframe_trend(
        self,
        df: Optional[pd.DataFrame],
    ) -> tuple[float, float, str]:
        """
        Compute directional trend & movement score in [-1.0, +1.0], ATR(14), and badge
        for a single timeframe DataFrame (`1d`, `1h`, `15m`, `5m`, or `1m`).
        """
        if df is None or len(df) < 15:
            return 0.0, 0.0, "🟡 FLAT"
        try:
            ind = self.ohlcv_preprocessor.compute_raw_indicators(df.tail(120))
            ind = MultiFactorConfluenceAnalyzer.compute_confluence_series(ind)
            last = ind.iloc[-1]
            close_p = float(last["close"])
            ema20 = float(last["ema_20"])
            ema50 = float(last["ema_50"])
            ema200 = float(last["ema_200"])
            rsi14 = float(last["rsi_14"])
            macd_h = float(last["macd_hist"])
            atr14 = float(last["atr_14"])
            cvd20 = float(last.get("cvd_norm_20", 0.0))

            # Define momentum and order-flow scores before blending
            rsi_comp = np.tanh((rsi14 - 50.0) / 15.0)
            macd_comp = np.sign(macd_h) * min(1.0, abs(macd_h) / max(atr14 * 0.25, 1e-6))
            mom_score = float(np.clip(0.60 * rsi_comp + 0.40 * macd_comp, -1.0, 1.0))
            flow_score = float(np.clip(cvd20 * 4.0, -1.0, 1.0))
            ema_score = (
                (1.0 if close_p > ema20 else -1.0) * 0.40
                + (1.0 if ema20 > ema50 else -1.0) * 0.35
                + (1.0 if ema50 > ema200 else -1.0) * 0.25
            )
            ut_trend = float(last.get("ut_trend", 0.0))
            ut_buy = bool(last.get("ut_buy", False))
            ut_sell = bool(last.get("ut_sell", False))
            ut_score = 0.85 if ut_buy else (-0.85 if ut_sell else (0.50 * ut_trend))

            st_dir = float(last.get("supertrend_dir", 0.0))
            st_buy = bool(last.get("supertrend_buy", False))
            st_sell = bool(last.get("supertrend_sell", False))
            st_score = 0.85 if st_buy else (-0.85 if st_sell else (0.50 * st_dir))
            oc_sig = float(last.get("oc_signal", 0.0))
            oc_score = 0.85 if oc_sig == 1 else (-0.85 if oc_sig == -1 else 0.0)

            strategy_score = 0.40 * ut_score + 0.40 * st_score + 0.20 * oc_score
            score = float(np.clip(0.30 * ema_score + 0.35 * strategy_score + 0.20 * mom_score + 0.15 * flow_score, -1.0, 1.0))
            lbl = "🟢 BULL" if score >= 0.12 else ("🔴 BEAR" if score <= -0.12 else "🟡 FLAT")
            return round(score, 3), max(atr14, close_p * 0.0015), lbl
        except Exception:
            return 0.0, 0.0, "🟡 FLAT"

    def analyze_multi_timeframe_autotrade(
        self,
        symbol: str = "BTCUSDT",
        event_mode: Optional[str] = None,
    ) -> SignalDecision:
        """
        1. Reads latest breaking news + 3 months (90 days) historical news in detail.
        2. Analyzes market movement & trend across `1D`, `4H`, `1H`, `15M`, and `5M`.
        3. Executes open-position timing, entry price, and realistic 1:2 / 1:3 SL/TP on `1M` (`1m`).
        """
        clean_sym = normalize_symbol(symbol)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            f_1d = pool.submit(self.load_market_series, clean_sym, "1d", 160)
            f_4h = pool.submit(self.load_market_series, clean_sym, "4h", 160)
            f_1h = pool.submit(self.load_market_series, clean_sym, "1h", 160)
            f_15m = pool.submit(self.load_market_series, clean_sym, "15m", 300)
            f_5m = pool.submit(self.load_market_series, clean_sym, "5m", 160)
            f_1m = pool.submit(self.load_market_series, clean_sym, "1m", 160)
            df_1d = f_1d.result()
            df_4h = f_4h.result()
            df_1h = f_1h.result()
            df_15m = f_15m.result()
            df_5m = f_5m.result()
            df_1m = f_1m.result()

        sent_90d = self.sentiment_analyzer.analyze_symbol(
            symbol=clean_sym,
            market_df=df_1d if (df_1d is not None and len(df_1d) > 0) else df_15m,
            event_mode=event_mode,
        )
        # Step 2: Multi-Timeframe (1D, 4H, 1H, 15M, 5M) Market & Trend Analysis
        s_1d, _, lbl_1d = self._score_timeframe_trend(df_1d)
        s_4h, atr_4h, lbl_4h = self._score_timeframe_trend(df_4h)
        s_1h, atr_1h, lbl_1h = self._score_timeframe_trend(df_1h)
        s_5m, atr_5m, lbl_5m = self._score_timeframe_trend(df_5m)
        s_1m, atr_1m, lbl_1m = self._score_timeframe_trend(df_1m)

        base_15m = self.analyze_symbol(
            clean_sym,
            timeframe="15m",
            df=df_15m,
            headlines=[h.headline for h in sent_90d.headline_scores] or None,
            event_mode=event_mode,
        )
        s_15m_raw, atr_15m, _ = self._score_timeframe_trend(df_15m)
        s_15m = float(np.clip(0.65 * base_15m.s_tech + 0.35 * s_15m_raw, -1.0, 1.0))
        lbl_15m = "🟢 BULL" if s_15m >= 0.12 else ("🔴 BEAR" if s_15m <= -0.12 else "🟡 FLAT")

        # Step 2: Strict Asymmetric HTF Gate (1D, 4H, 1H) + LTF Execution Trigger (15M, 5M, 1M)
        htf_anchor_score = 0.30 * s_1d + 0.35 * s_4h + 0.35 * s_1h
        ltf_exec_score = 0.60 * s_15m + 0.40 * s_5m
        mtf_trend_score = 0.55 * htf_anchor_score + 0.45 * ltf_exec_score

        dom_sign = 1.0 if mtf_trend_score >= 0 else -1.0
        tf_scores = [s_1d, s_4h, s_1h, s_15m, s_5m]
        aligned_count = sum(1 for sc in tf_scores if (sc * dom_sign) >= 0.08)

        # Asymmetric HTF Lock: Never allow LTF counter-trend bounce to override strong HTF trend
        htf_bull_locked = htf_anchor_score >= 0.18 and s_1h >= 0.10
        htf_bear_locked = htf_anchor_score <= -0.18 and s_1h <= -0.10

        if htf_bull_locked and mtf_trend_score < 0:
            blended_s_tech = 0.0
        elif htf_bear_locked and mtf_trend_score > 0:
            blended_s_tech = 0.0
        elif (base_15m.s_tech * dom_sign) < -0.15:
            blended_s_tech = float(0.50 * mtf_trend_score + 0.50 * base_15m.s_tech)
        elif aligned_count >= 3:
            aligned_15m = abs(base_15m.s_tech) if (base_15m.s_tech * dom_sign) > 0 else 0.0
            alignment_boost = 1.0 + 0.10 * (aligned_count - 2)
            raw_mag = max(aligned_15m, abs(mtf_trend_score) * alignment_boost)
            blended_s_tech = float(dom_sign * np.clip(raw_mag, 0.0, 0.96))
        elif aligned_count == 2 and abs(base_15m.s_tech) >= 0.72 and (s_15m * s_5m) > 0:
            blended_s_tech = float(base_15m.s_tech)
        else:
            blended_s_tech = float(mtf_trend_score * 0.55)
        # Step 3: Rekomendasi 1 & 3 — Eksekusi Aksi di 5M, ATR Robust (bebas noise spread), & Filter Sideways/Chop
        entry_px_5m = (
            float(df_5m["close"].iloc[-1])
            if (df_5m is not None and len(df_5m) > 0)
            else (float(df_1m["close"].iloc[-1]) if (df_1m is not None and len(df_1m) > 0) else base_15m.entry_price)
        )
        # ATR Robust: Bebas dari batas 0.12% yang mencekik SL, memberikan ruang napas wajar di 5M/15M
        raw_robust_atr = max(
            atr_5m * 1.20,
            atr_15m * 0.70,
            entry_px_5m * 0.0030,
        )
        blended_atr = raw_robust_atr

        # UT Bot Alerts & Supertrend Strategy on 5M Execution Timeframe
        ut_source_df = df_5m if (df_5m is not None and len(df_5m) >= 10) else (df_15m if df_15m is not None else df_1d)
        ut_5m_df = self.ohlcv_preprocessor.compute_ut_bot_indicators(ut_source_df.tail(40), key_value=1.0, atr_period=10)
        last_ut = ut_5m_df.iloc[-1]
        ut_ts_level = float(last_ut.get("ut_trailing_stop", 0.0))
        ut_trend_dir = int(last_ut.get("ut_trend", 0))
        ut_buy_trigger = bool(last_ut.get("ut_buy", False))
        ut_sell_trigger = bool(last_ut.get("ut_sell", False))

        prev_ut = ut_5m_df.iloc[-2] if len(ut_5m_df) >= 2 else last_ut
        ut_recent_buy = ut_buy_trigger or bool(prev_ut.get("ut_buy", False))
        ut_recent_sell = ut_sell_trigger or bool(prev_ut.get("ut_sell", False))

        st_5m_df = self.ohlcv_preprocessor.compute_supertrend_indicators(ut_source_df.tail(40), period=10, multiplier=3.0)
        last_st = st_5m_df.iloc[-1]
        st_level = float(last_st.get("supertrend", 0.0))
        st_dir = int(last_st.get("supertrend_dir", 0))
        st_buy_trigger = bool(last_st.get("supertrend_buy", False))
        st_sell_trigger = bool(last_st.get("supertrend_sell", False))

        prev_st = st_5m_df.iloc[-2] if len(st_5m_df) >= 2 else last_st
        st_recent_buy = st_buy_trigger or bool(prev_st.get("supertrend_buy", False))
        st_recent_sell = st_sell_trigger or bool(prev_st.get("supertrend_sell", False))
        # OC Buy/Sell Signals (Pine Script v6) on 5M Execution Timeframe
        oc_5m_df = self.ohlcv_preprocessor.compute_oc_signals(ut_source_df.tail(40), body_len=20, body_mult=1.5)
        last_oc = oc_5m_df.iloc[-1]
        oc_sig = int(last_oc.get("oc_signal", 0))
        oc_entry = float(last_oc.get("oc_entry", 0.0))
        oc_sl = float(last_oc.get("oc_sl", 0.0))
        oc_tp = float(last_oc.get("oc_tp", 0.0))
        if oc_sig == 0 and len(oc_5m_df) >= 2:
            prev_oc = oc_5m_df.iloc[-2]
            if int(prev_oc.get("oc_signal", 0)) != 0:
                oc_sig = int(prev_oc.get("oc_signal", 0))
                oc_entry = float(prev_oc.get("oc_entry", 0.0))
                oc_sl = float(prev_oc.get("oc_sl", 0.0))
                oc_tp = float(prev_oc.get("oc_tp", 0.0))

        oc_tag = f" • OC: {('BUY' if oc_sig == 1 else 'SELL')} (SL: ${oc_sl:,.2f} | TP: ${oc_tp:,.2f})" if oc_sig != 0 else ""

        # Candlestick Pattern Identification & 3-Pillar Price Action on 5M Timeframe
        cp_5m_df = self.ohlcv_preprocessor.identify_candlestick_patterns(ut_source_df.tail(40), trend_bars=5)
        cs_analysis = CandlestickReader.analyze_series_and_swings(ut_source_df.tail(40))
        last_cp = cp_5m_df.iloc[-1]
        active_pattern = str(last_cp.get("cp_pattern_name", "")).strip() or cs_analysis.latest_candle.pattern_label
        cp_score = float(last_cp.get("cp_score", 0.0))
        if cp_score == 0.0:
            if cs_analysis.latest_candle.color == "PUTIH":
                cp_score = round(cs_analysis.latest_candle.body_ratio * 0.8, 2)
            elif cs_analysis.latest_candle.color == "MERAH":
                cp_score = round(-cs_analysis.latest_candle.body_ratio * 0.8, 2)
        if not active_pattern and len(cp_5m_df) >= 2:
            prev_cp = cp_5m_df.iloc[-2]
            prev_pname = str(prev_cp.get("cp_pattern_name", "")).strip()
            if prev_pname:
                active_pattern = f"{prev_pname} (Konfirmasi)"
                cp_score = float(prev_cp.get("cp_score", 0.0))
        pat_prefix = f"[🕯️ {active_pattern}] " if active_pattern else ""

        # Market Chop filter only triggers when higher timeframes lack a clear trend (aligned_count < 3)
        has_htf_trend = (aligned_count >= 3) or (htf_bull_locked or htf_bear_locked)
        is_market_chop = bool(
            not has_htf_trend
            and (base_15m.is_sideways or (abs(s_1h) < 0.08 and abs(s_15m) < 0.08 and abs(s_5m) < 0.08))
        )

        # Dual Strategy Confluence (UT Bot + Supertrend + Candlestick Pattern) Acuan Eksekusi:
        is_dual_buy = (ut_recent_buy or ut_trend_dir == 1) and (st_recent_buy or st_dir == 1)
        is_dual_sell = (ut_recent_sell or ut_trend_dir == -1) and (st_recent_sell or st_dir == -1)

        if is_dual_buy and dom_sign > 0 and not is_market_chop:
            act_5m_tag = f"🟢 {pat_prefix}UT BOT + SUPERTREND BUY (ST: ${st_level:,.2f} | UT: ${ut_ts_level:,.2f}){oc_tag}"
            if blended_s_tech < 0.28 and dom_sign > 0:
                blended_s_tech = max(0.32, blended_s_tech + 0.40)
            if cp_score > 0:
                blended_s_tech = min(0.96, blended_s_tech + 0.10)
        elif is_dual_sell and dom_sign < 0 and not is_market_chop:
            act_5m_tag = f"🔴 {pat_prefix}UT BOT + SUPERTREND SELL (ST: ${st_level:,.2f} | UT: ${ut_ts_level:,.2f}){oc_tag}"
            if blended_s_tech > -0.28 and dom_sign < 0:
                blended_s_tech = min(-0.32, blended_s_tech - 0.40)
            if cp_score < 0:
                blended_s_tech = max(-0.96, blended_s_tech - 0.10)
        elif ut_recent_buy or (ut_trend_dir == 1 and dom_sign > 0 and not is_market_chop):
            act_5m_tag = f"🟢 {pat_prefix}UT BOT BUY (Trailing Stop: ${ut_ts_level:,.2f}){oc_tag}"
            if blended_s_tech < 0.25 and dom_sign > 0:
                blended_s_tech = max(0.28, blended_s_tech + 0.35)
        elif ut_recent_sell or (ut_trend_dir == -1 and dom_sign < 0 and not is_market_chop):
            act_5m_tag = f"🔴 {pat_prefix}UT BOT SELL (Trailing Stop: ${ut_ts_level:,.2f}){oc_tag}"
            if blended_s_tech > -0.25 and dom_sign < 0:
                blended_s_tech = min(-0.28, blended_s_tech - 0.35)
        elif is_market_chop:
            blended_s_tech *= 0.35
            act_5m_tag = "⏳ PASAR SIDEWAYS / CHOP (Rawan Fakeout • Tahan Entri)"
        elif blended_s_tech > 0.15 and s_5m < -0.25:
            blended_s_tech *= 0.45
            act_5m_tag = f"⏳ TUNGGU PULLBACK 5M SELESAI ({s_5m:+.2f})"
        elif blended_s_tech < -0.15 and s_5m > 0.25:
            blended_s_tech *= 0.45
            act_5m_tag = f"⏳ TUNGGU LONJAKAN 5M REDA ({s_5m:+.2f})"
        else:
            act_5m_tag = f"⚡ {pat_prefix}UT: ${ut_ts_level:,.2f} | ST: ${st_level:,.2f} ({lbl_5m}){oc_tag}"
        sr_levels = self.sr_detector.detect_levels(
            df_15m if (df_15m is not None and len(df_15m) >= 20) else df_1h,
            current_price=entry_px_5m,
            atr_14=blended_atr,
        )
        decision = self.synthesize_scores(
            blended_s_tech,
            s_vol=base_15m.s_vol,
            s_sent=sent_90d.s_sent,
            symbol=clean_sym,
            timeframe="5m",
            entry_price=entry_px_5m,
            atr_14=blended_atr,
            indicators={
                **dict(base_15m.indicators),
                "ut_trailing_stop": round(ut_ts_level, 4),
                "ut_trend": float(ut_trend_dir),
                "ut_buy": 1.0 if ut_recent_buy else 0.0,
                "ut_sell": 1.0 if ut_recent_sell else 0.0,
                "supertrend": round(st_level, 4),
                "supertrend_dir": float(st_dir),
                "supertrend_buy": 1.0 if st_recent_buy else 0.0,
                "supertrend_sell": 1.0 if st_recent_sell else 0.0,
                "candlestick_pattern": active_pattern,
                "candlestick_score": float(cp_score),
                "oc_signal": float(oc_sig),
                "oc_entry": round(oc_entry, 4),
                "oc_sl": round(oc_sl, 4),
                "oc_tp": round(oc_tp, 4),
                "candle_color": cs_analysis.latest_candle.color,
                "candle_winner": cs_analysis.latest_candle.winner,
                "candle_rejection": cs_analysis.latest_candle.rejection_site,
                "candle_trend_series": cs_analysis.trend,
                "candle_swing_high": cs_analysis.latest_swing_high,
                "candle_swing_low": cs_analysis.latest_swing_low,
                "candle_three_pillars": cs_analysis.three_pillars_summary,
            },
            headlines=[h.headline for h in sent_90d.headline_scores],
            newly_learned_headlines=sent_90d.newly_learned_count,
            total_corpus_size=sent_90d.total_corpus_size,
            validation_accuracy=base_15m.validation_accuracy,
            validation_roc_auc=base_15m.validation_roc_auc,
            memory_bars_count=base_15m.memory_bars_count,
            five_year_percentile=base_15m.five_year_percentile,
            macro_regime=base_15m.macro_regime,
            sr_levels=sr_levels,
            market_df=df_15m,
            macro_stance=sent_90d.macro_stance,
            gold_action_bias=sent_90d.gold_action_bias,
            event_mode=event_mode,
        )
        if is_market_chop:
            decision.is_sideways = True
            decision.market_structure_status = "Sideways / Chop (ADX Rendah • Rawan Fakeout)"
        # Multi-Timeframe Alignment Confidence Synchronization:
        # If higher timeframes confirm the 15M signal in the same direction (aligned_count >= 3),
        # confidence should not be diluted below the single-timeframe baseline, and receives
        # an institutional consensus boost for 4/5 or 5/5 unanimity.
        if (base_15m.s_tech * dom_sign) > 0 and not is_market_chop:
            mtf_bonus = 0.04 if aligned_count >= 4 else (0.02 if aligned_count >= 3 else 0.0)
            target_confidence = max(decision.confidence, base_15m.confidence) + mtf_bonus
            decision.confidence = round(float(np.clip(target_confidence, 0.0, 0.98)), 4)
            decision.badge = self.resolve_badge(decision.action, decision.confidence)
        decision.macro_stance = getattr(sent_90d, "macro_stance", "NEUTRAL")
        decision.gold_action_bias = getattr(sent_90d, "gold_action_bias", "HOLD")
        decision.event_mode = event_mode.upper() if event_mode else getattr(sent_90d, "event_mode", None)
        if event_mode and sent_90d.gold_action_bias in ("BUY", "SELL"):
            # Macro Monetary Policy Transmission override in Event Sniper Mode:
            # Hawkish (US inflation/jobs strong -> Sell Gold)
            # Dovish (US inflation/jobs weak -> Buy Gold)
            if sent_90d.gold_action_bias == "SELL" and decision.action != "BUY":
                decision.action = "SELL"
                decision.approved = True
                decision.confidence = round(max(decision.confidence, 0.76), 4)
                decision.badge = self.resolve_badge(decision.action, decision.confidence)
            elif sent_90d.gold_action_bias == "BUY" and decision.action != "SELL":
                decision.action = "BUY"
                decision.approved = True
                decision.confidence = round(max(decision.confidence, 0.76), 4)
                decision.badge = self.resolve_badge(decision.action, decision.confidence)
        if getattr(self.settings, "macro_blackout_enabled", False) and getattr(sent_90d, "is_macro_blackout", False):
            self._reject(decision, f"HIGH_IMPACT_NEWS_BLACKOUT ({sent_90d.blackout_event or 'High Impact Event'})")
        elif decision.action == "BUY" and decision.s_sent < -0.35:
            self._reject(decision, f"Sentiment Conflict: Composite sentiment ({decision.s_sent:+.2f}) vetoes BUY")
        elif decision.action == "SELL" and decision.s_sent > 0.35:
            self._reject(decision, f"Sentiment Conflict: Composite sentiment ({decision.s_sent:+.2f}) vetoes SELL")
        decision.mtf_aligned_count = aligned_count
        decision.mtf_summary = (
            f"News 3Bln({len(sent_90d.headline_scores)} berita: {sent_90d.s_sent:+.2f}) • "
            f"1D:{lbl_1d}({s_1d:+.2f}) • 4H:{lbl_4h}({s_4h:+.2f}) • 1H:{lbl_1h}({s_1h:+.2f}) • "
            f"15M:{lbl_15m}({s_15m:+.2f}) • 5M:{lbl_5m}({s_5m:+.2f}) ➔ Action 5M: {act_5m_tag}"
        )
        return decision

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
        roc_auc = 0.0
        if len(market_df) >= 60:
            roc_auc = self.ohlcv_predictor.train(
                market_df, symbol=clean_sym, timeframe=timeframe, df_5y=df_5y
            )
            self.volatility_estimator.fit(market_df)
            self._last_trained_signature[f"{clean_sym}_{timeframe}"] = (
                str(market_df.index[-1]),
                round(float(market_df["close"].iloc[-1]), 4),
            )

        decision = self.analyze_symbol(
            symbol=clean_sym,
            timeframe=timeframe,
            df=market_df,
            headlines=[h.headline for h in sent_result.headline_scores] or None,
        )
        return {
            "symbol": clean_sym,
            "timeframe": timeframe,
            "latest_price": decision.entry_price,
            "live_headlines_fetched": len(sent_result.headline_scores),
            "newly_learned_headlines": sent_result.newly_learned_count,
            "total_corpus_size": sent_result.total_corpus_size,
            "online_updates_count": self.sentiment_analyzer.online_updates_count,
            "validation_roc_auc": roc_auc or decision.validation_roc_auc,
            "validation_accuracy": decision.validation_accuracy,
            "memory_bars_count": decision.memory_bars_count,
            "five_year_percentile": decision.five_year_percentile,
            "macro_regime": decision.macro_regime,
            "s_tech": decision.s_tech,
            "s_sent": sent_result.s_sent,
            "s_vol": decision.s_vol,
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
