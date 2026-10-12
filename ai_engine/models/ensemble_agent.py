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
        market_df: Optional[pd.DataFrame] = None,
        annualized_volatility: float = 0.45,
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

        if abs(clamped_tech) >= 0.72 and abs(clamped_sent) < 0.15:
            eff_sent = float(np.sign(clamped_tech)) * max(
                abs(clamped_sent), 0.42 * abs(clamped_tech)
            )
        else:
            eff_sent = clamped_sent

        signed_composite = w1 * clamped_tech + w2 * eff_sent
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
        if sr_levels is not None:
            targets: StopLossTargets = self.stop_loss_manager.calculate_realistic_targets(
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
            market_df=market_df,
            annualized_volatility=vol_forecast.annualized_volatility,
        )

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

            ema_score = (
                (1.0 if close_p > ema20 else -1.0) * 0.40
                + (1.0 if ema20 > ema50 else -1.0) * 0.35
                + (1.0 if ema50 > ema200 else -1.0) * 0.25
            )
            mom_score = (
                float(np.tanh((rsi14 - 50.0) / 14.0)) * 0.55
                + (0.45 if macd_h > 0 else (-0.45 if macd_h < 0 else 0.0))
            )
            flow_score = float(np.clip(cvd20 * 2.5, -1.0, 1.0))
            score = float(np.clip(0.50 * ema_score + 0.35 * mom_score + 0.15 * flow_score, -1.0, 1.0))
            lbl = "🟢 BULL" if score >= 0.12 else ("🔴 BEAR" if score <= -0.12 else "🟡 FLAT")
            return round(score, 3), max(atr14, close_p * 0.0015), lbl
        except Exception:
            return 0.0, 0.0, "🟡 FLAT"

    def analyze_multi_timeframe_autotrade(
        self,
        symbol: str = "BTCUSDT",
    ) -> SignalDecision:
        """
        Multi-Timeframe Auto-Trade Intelligence:
        - Analyzes market movement & trend across `1D`, `1H`, `15M`, and `5M`.
        - Executes open-position timing, entry price, and micro-momentum trigger on `1M` (`1m`).
        """
        clean_sym = normalize_symbol(symbol)
        df_1d = self.load_market_series(clean_sym, timeframe="1d", limit=160)
        df_1h = self.load_market_series(clean_sym, timeframe="1h", limit=160)
        df_15m = self.load_market_series(clean_sym, timeframe="15m", limit=300)
        df_5m = self.load_market_series(clean_sym, timeframe="5m", limit=160)
        df_1m = self.load_market_series(clean_sym, timeframe="1m", limit=160)

        s_1d, _, lbl_1d = self._score_timeframe_trend(df_1d)
        s_1h, atr_1h, lbl_1h = self._score_timeframe_trend(df_1h)
        s_5m, atr_5m, lbl_5m = self._score_timeframe_trend(df_5m)
        s_1m, atr_1m, lbl_1m = self._score_timeframe_trend(df_1m)

        base_15m = self.analyze_symbol(clean_sym, timeframe="15m", df=df_15m)
        s_15m_raw, atr_15m, _ = self._score_timeframe_trend(df_15m)
        s_15m = float(np.clip(0.65 * base_15m.s_tech + 0.35 * s_15m_raw, -1.0, 1.0))
        lbl_15m = "🟢 BULL" if s_15m >= 0.12 else ("🔴 BEAR" if s_15m <= -0.12 else "🟡 FLAT")

        mtf_trend_score = 0.20 * s_1d + 0.25 * s_1h + 0.35 * s_15m + 0.20 * s_5m
        dom_sign = 1.0 if mtf_trend_score >= 0 else -1.0
        tf_scores = [s_1d, s_1h, s_15m, s_5m]
        aligned_count = sum(1 for sc in tf_scores if (sc * dom_sign) >= 0.08)

        if aligned_count >= 3:
            blended_s_tech = float(
                dom_sign * max(abs(base_15m.s_tech), abs(mtf_trend_score), 0.85)
            )
        elif aligned_count == 2 and abs(base_15m.s_tech) >= 0.72 and (s_15m * s_5m) > 0:
            blended_s_tech = float(base_15m.s_tech)
        else:
            blended_s_tech = float(mtf_trend_score * 0.45)

        # 1m Action Open Position Trigger: ensure 1m candle is not violently opposing the entry
        if blended_s_tech > 0.15 and s_1m < -0.35:
            blended_s_tech *= 0.45
            act_1m_tag = f"⏳ TUNGGU PULLBACK 1M SELESAI ({s_1m:+.2f})"
        elif blended_s_tech < -0.15 and s_1m > 0.35:
            blended_s_tech *= 0.45
            act_1m_tag = f"⏳ TUNGGU LONJAKAN 1M REDA ({s_1m:+.2f})"
        else:
            act_1m_tag = f"⚡ TRIGGER 1M SIAP ({lbl_1m} {s_1m:+.2f})"

        entry_px_1m = (
            float(df_1m["close"].iloc[-1])
            if (df_1m is not None and len(df_1m) > 0)
            else base_15m.entry_price
        )
        blended_atr = max(
            atr_1m * 2.4,
            atr_5m * 1.35,
            (atr_15m or base_15m.atr_14) * 0.85,
            entry_px_1m * 0.002,
        )
        sr_levels = self.sr_detector.detect_levels(
            df_15m if (df_15m is not None and len(df_15m) >= 20) else df_1h,
            current_price=entry_px_1m,
            atr_14=blended_atr,
        )
        decision = self.synthesize_scores(
            s_tech=blended_s_tech,
            s_vol=base_15m.s_vol,
            s_sent=base_15m.s_sent,
            symbol=clean_sym,
            timeframe="1m",
            entry_price=entry_px_1m,
            atr_14=blended_atr,
            indicators=base_15m.indicators,
            headlines=base_15m.headlines,
            newly_learned_headlines=base_15m.newly_learned_headlines,
            total_corpus_size=base_15m.total_corpus_size,
            validation_accuracy=base_15m.validation_accuracy,
            validation_roc_auc=base_15m.validation_roc_auc,
            memory_bars_count=base_15m.memory_bars_count,
            five_year_percentile=base_15m.five_year_percentile,
            macro_regime=base_15m.macro_regime,
            sr_levels=sr_levels,
            market_df=df_15m,
        )
        decision.mtf_aligned_count = aligned_count
        decision.mtf_summary = (
            f"1D:{lbl_1d}({s_1d:+.2f}) • 1H:{lbl_1h}({s_1h:+.2f}) • "
            f"15M:{lbl_15m}({s_15m:+.2f}) • 5M:{lbl_5m}({s_5m:+.2f}) ➔ Action 1M: {act_1m_tag}"
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
