from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Optional
import numpy as np
import pandas as pd

from ai_engine.auto_trader.mt5_symbols import (
    calculate_ea_sl_tp,
    get_mt5_pip_size,
)

from ai_engine.auto_trader.mt5_exness_client import (
    BinanceOrderResult,
    MT5ExnessClient,
    anchor_targets_to_fill_price,
    calculate_mt5_lots,
    estimate_mt5_pnl_usdt,
    get_mt5_contract_size,
)
from ai_engine.auto_trader.session_manager import (
    AutoTradeSessionManager,
    AutoTradeSessionState,
    NotifyCallback,
    PositionMonitorResult,
    format_position_closed_notification,
)
from ai_engine.auto_trader.user_vault import (
    UserBinanceCredentials,
    UserPositionRecord,
    UserVault,
    refresh_live_exness_balance,
)
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe
logger = get_logger("nexustrade.auto_trader.engine")


@dataclass
class AutoTradeGateCheck:
    """Single validation gate result in the 6-Gate Pre-Trade Checklist."""

    gate_number: int
    name: str
    passed: bool
    detail: str


@dataclass
class AutoTradeExecutionReport:
    """Comprehensive 6-Gate evaluation & order execution report for a specific Telegram user."""

    user_id: int
    symbol: str
    timeframe: str
    setup_required: bool
    executed: bool
    decision: Optional[SignalDecision] = None
    credentials: Optional[UserBinanceCredentials] = None
    gates: list[AutoTradeGateCheck] = field(default_factory=list)
    order_result: Optional[BinanceOrderResult] = None
    position_record: Optional[UserPositionRecord] = None
    rejection_summary: Optional[str] = None
    trade_mode: bool = False
    ea_mode: Optional[str] = None
class SmartAutoTraderEngine:
    """
    Institutional 6-Gate Smart Auto-Trader Engine.
    Never opens a position blindly: every trade must pass all 6 pre-trade gates:
      Gate 1: Mandatory Personal /setup Account Verification (Admin vs User isolation)
      Gate 2: Breaking News & FinBERT Sentiment Alignment
      Gate 3: 5-Year + Live Technical Conviction (Confidence & Directional Agreement)
      Gate 4: Support & Resistance Structural Room (No buying into R1 wall / selling into S1 floor)
      Gate 5: Volatility Regime, Daily Drawdown Circuit Breaker & Duplicate Position Guard
      Gate 6: Live Binance Order Execution + Locked 2xATR Stop-Loss & TP1/TP2 Targets
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        ensemble_agent: Optional[EnsembleDecisionAgent] = None,
        user_vault: Optional[UserVault] = None,
        mt5_client: Optional[MT5ExnessClient] = None,
        notify_user_callback: Optional[NotifyCallback] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ensemble_agent = ensemble_agent or EnsembleDecisionAgent(self.settings)
        self.vault = user_vault or UserVault(self.settings)
        self.vault._linked_position_sizer = self.ensemble_agent.position_sizer
        self.mt5_client = mt5_client or MT5ExnessClient()
        self.notify_user_callback: Optional[NotifyCallback] = notify_user_callback
        self._last_broker_closed: dict[tuple[int, str], PositionMonitorResult] = {}
        self._notified_closed_tickets: set[str] = set()
        self._execution_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._last_broker_request_ts: dict[str, float] = {}
        self.session_manager = AutoTradeSessionManager(self)

    async def _enforce_broker_throttle(
        self,
        acct_key: str,
        min_ms: int = 500,
        max_ms: int = 1500,
    ) -> float:
        """
        Anti-Broker-Ban Protection 1: Request Throttle (InpThrottleMs = 500ms to 1500ms)
        Enforces a minimum jitter delay between outbound broker requests per account
        to prevent overwhelming broker servers or triggering anti-bot / rate-limit bans.
        """
        now = time.time()
        last_ts = self._last_broker_request_ts.get(acct_key, 0.0)
        target_delay = random.uniform(min_ms / 1000.0, max_ms / 1000.0)
        elapsed = now - last_ts
        if elapsed < target_delay:
            wait_s = target_delay - elapsed
            await asyncio.sleep(wait_s)
        self._last_broker_request_ts[acct_key] = time.time()
        return target_delay

    async def start_timed_session(
        self,
        user_id: int,
        symbol: str,
        duration_code: str = "24/7",
        timeframe: str = "1m",
        poll_interval_seconds: float = 5.0,
        notify_callback: Optional[NotifyCallback] = None,
        custom_lot: Optional[float] = None,
        live_card_callback: Optional[NotifyCallback] = None,
        macro_event_mode: Optional[str] = None,
        max_positions: int = 1,
        is_trade_mode: bool = False,
        ea_mode: Optional[str] = None,
    ) -> tuple[AutoTradeSessionState, object]:
        """Start a 24/7 autonomous trading session on 1m action + 1D/1H/15M/5M analysis."""
        return await self.session_manager.start_timed_session(
            user_id=user_id,
            symbol=symbol,
            duration_code=duration_code,
            timeframe=timeframe,
            poll_interval_seconds=poll_interval_seconds,
            notify_callback=notify_callback,
            custom_lot=custom_lot,
            live_card_callback=live_card_callback,
            macro_event_mode=macro_event_mode,
            max_positions=max_positions,
            is_trade_mode=is_trade_mode,
            ea_mode=ea_mode,
        )

    def stop_session_by_id(
        self,
        user_id: int,
        target_id: str,
    ) -> Optional[AutoTradeSessionState]:
        return self.session_manager.stop_session_by_id(user_id, target_id)

    def get_user_sessions(self, user_id: int) -> list[AutoTradeSessionState]:
        return self.session_manager.get_user_sessions(user_id)

    def stop_user_sessions(
        self,
        user_id: int,
        symbol: Optional[str] = None,
    ) -> list[AutoTradeSessionState]:
        return self.session_manager.stop_user_sessions(user_id, symbol=symbol)

    async def monitor_and_manage_open_position(
        self,
        user_id: int,
        symbol: str,
        timeframe: str = "1m",
        price_override: Optional[float] = None,
        high_override: Optional[float] = None,
        low_override: Optional[float] = None,
        headlines_override: Optional[list[str]] = None,
    ) -> PositionMonitorResult:
        return await self.session_manager.monitor_and_manage_open_position(
            user_id=user_id,
            symbol=symbol,
            timeframe=timeframe,
            price_override=price_override,
            high_override=high_override,
            low_override=low_override,
            headlines_override=headlines_override,
        )

    async def sync_live_mt5_positions(
        self,
        user_id: int,
        notify_closures: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Reconcile SQLite OPEN positions against real-time Exness MT5 OpenedOrders & ClosedOrders.
        When a position was closed on Exness MT5 (hit TP/SL or closed in Exness Go), records the exact
        `closePrice` and `profit` from `/ClosedOrders` and triggers a Telegram notification.
        """
        creds = self.vault.resolve_credentials(user_id)
        if (
            creds is None
            or self.mt5_client._is_exness_trial_demo(creds.api_key, creds.testnet)
        ):
            return {}
        live_orders = await self.mt5_client.get_live_open_orders(
            creds.api_key, creds.api_secret, creds.server_name
        )
        if live_orders is None:
            return {}
        live_ticket_map: dict[str, dict[str, Any]] = {}
        for o in live_orders:
            if isinstance(o, dict):
                t_id = str(o.get("ticket") or o.get("order") or "")
                if t_id:
                    live_ticket_map[t_id] = o

        open_in_db = self.vault.get_open_positions(user_id)
        missing_positions = [p for p in open_in_db if p.order_id not in live_ticket_map]
        if not missing_positions:
            return live_ticket_map

        closed_map = await self.mt5_client.get_closed_orders_map(
            creds.api_key, creds.api_secret, creds.server_name
        )
        for pos in missing_positions:
            c_ord = closed_map.get(pos.order_id)
            if not isinstance(c_ord, dict):
                # F08: Do NOT guess closure or zero PnL when ticket is not confirmed in /ClosedOrders
                logger.debug(
                    "Position missing from OpenedOrders but not yet in ClosedOrders (RECONCILIATION_PENDING)",
                    user_id=user_id,
                    order_id=pos.order_id,
                )
                continue
            exit_px = float(c_ord.get("closePrice") or pos.entry_price)
            net_pnl = round(
                float(c_ord.get("profit") or 0.0)
                + float(c_ord.get("swap") or 0.0)
                + float(c_ord.get("commission") or 0.0),
                2,
            )
            tp_hit = (
                net_pnl > 0.0
                or (
                    pos.take_profit_1 > 0
                    and pos.stop_loss > 0
                    and abs(exit_px - pos.take_profit_1) <= abs(exit_px - pos.stop_loss)
                )
            )
            if tp_hit:
                action_type = "CLOSED_TP"
                reason = f"Target Profit (TP @ ${exit_px:,.2f}) Tercapai di Exness MT5"
            else:
                is_profit_stop = (
                    pos.stop_loss > pos.entry_price
                    if pos.side == "BUY"
                    else (0 < pos.stop_loss < pos.entry_price)
                )
                action_type = "CLOSED_TP" if is_profit_stop else "CLOSED_SL"
                reason = (
                    f"Trailing Stop Profit Terkunci (@ ${exit_px:,.2f}) di Exness MT5"
                    if is_profit_stop
                    else f"Batas Rugi Stop-Loss (SL @ ${exit_px:,.2f}) Tersentuh di Exness MT5"
                )

            closed_rec = self.vault.close_position(pos.position_id, exit_px, net_pnl)
            pnl_pct = round((net_pnl / max(pos.notional_usdt, 1e-6)) * 100.0, 2)
            mon_res = PositionMonitorResult(
                action_taken=action_type,  # type: ignore[arg-type]
                position=closed_rec or pos,
                current_price=exit_px,
                unrealized_pnl_usdt=0.0,
                unrealized_pnl_pct=pnl_pct,
                exit_reason=reason,
                realized_pnl_usdt=net_pnl,
            )
            self._last_broker_closed[(int(user_id), normalize_symbol(pos.symbol))] = mon_res

            if (
                notify_closures
                and self.notify_user_callback is not None
                and pos.order_id not in self._notified_closed_tickets
            ):
                self._notified_closed_tickets.add(pos.order_id)
                try:
                    await self.notify_user_callback(
                        int(user_id),
                        format_position_closed_notification(
                            mon_res,
                            header_title=f"AUTO-TRADE MENUTUP POSISI (TP/SL EXNESS) — {pos.symbol}",
                            pnl_label="Hasil PnL Bersih",
                        ),
                    )
                except Exception as exc:
                    logger.debug("Could not send TP/SL closure notification", error=str(exc))

        return live_ticket_map
    async def evaluate_and_execute(
        self,
        user_id: int,
        symbol: str = "BTCUSDT",
        timeframe: str = "1m",
        decision_override: Optional[SignalDecision] = None,
        require_autotrade_toggle: bool = True,
        custom_lot: Optional[float] = None,
        macro_event_mode: Optional[str] = None,
        max_positions: int = 1,
        trade_mode: bool = False,
        ea_mode: Optional[str] = None,
    ) -> AutoTradeExecutionReport:
        """
        Run the complete 6-Gate Pre-Trade Checklist for `user_id` on `symbol` (`timeframe`)
        and execute the order on the user's Binance account only if all 6 gates pass.
        """
        clean_sym = normalize_symbol(symbol)
        clean_tf = normalize_timeframe(timeframe)
        gates: list[AutoTradeGateCheck] = []

        # Gate 1: Mandatory /setup Credential Resolution (Strict Admin vs User Isolation)
        creds = self.vault.resolve_credentials(user_id)
        if creds is None:
            gates.append(
                AutoTradeGateCheck(
                    gate_number=1,
                    name="Verifikasi Akun Exness Go / MT5 (/setup)",
                    passed=False,
                    detail="Akun Exness Go / MT5 pribadi belum diatur. Wajib jalankan /setup terlebih dahulu.",
                )
            )
            return AutoTradeExecutionReport(
                user_id=int(user_id),
                symbol=clean_sym,
                timeframe=clean_tf,
                setup_required=True,
                executed=False,
                gates=gates,
                rejection_summary="Wajib mengatur akun Exness Go / MT5 pribadi melalui /setup sebelum memakai fitur Auto-Trade.",
                trade_mode=trade_mode,
                ea_mode=ea_mode,
            )
        if require_autotrade_toggle and not creds.autotrade_enabled:
            gates.append(
                AutoTradeGateCheck(
                    gate_number=1,
                    name="Status Saklar Auto-Trade",
                    passed=False,
                    detail="Saklar Auto-Trade Anda sedang OFF (/autotrade on untuk mengaktifkan).",
                )
            )
            return AutoTradeExecutionReport(
                user_id=int(user_id),
                symbol=clean_sym,
                timeframe=clean_tf,
                setup_required=False,
                executed=False,
                credentials=creds,
                gates=gates,
                rejection_summary="Saklar Auto-Trade sedang OFF.",
                trade_mode=trade_mode,
                ea_mode=ea_mode,
            )

        await refresh_live_exness_balance(self.vault, self.mt5_client, user_id, creds)
        if creds.verified_usdt_balance < 10.0:
            gates.append(
                AutoTradeGateCheck(
                    gate_number=1,
                    name="Kecukupan Saldo USD/USDT",
                    passed=False,
                    detail=f"DITOLAK: Saldo Akun (${creds.verified_usdt_balance:,.2f}) kurang dari batas minimum $10.00 USD.",
                )
            )
            return AutoTradeExecutionReport(
                user_id=int(user_id),
                symbol=clean_sym,
                timeframe=clean_tf,
                setup_required=False,
                executed=False,
                credentials=creds,
                gates=gates,
                rejection_summary=f"Saldo Akun (${creds.verified_usdt_balance:,.2f}) di bawah batas minimum $10.00 USD.",
                trade_mode=trade_mode,
                ea_mode=ea_mode,
            )
        mode_str = f"EXNESS MT5 ({creds.server_name})"
        src_str = "Setup Pribadi" if creds.source == "user_setup" else "Admin (.env)"
        gates.append(
            AutoTradeGateCheck(
                gate_number=1,
                name="Verifikasi Akun Exness Go / MT5 (/setup)",
                passed=True,
                detail=f"Terhubung ({mode_str} • {src_str} • Login {creds.masked_api_key} • Saldo ${creds.verified_usdt_balance:,.2f})",
            )
        )

        # Run Multi-Timeframe AI analysis (Trend: 1D, 1H, 15M, 5M -> Action Open Position: 1M)
        if decision_override is not None:
            decision = decision_override
        elif ea_mode == "EA_EXPERT_ADVISOR":
            market_df = self.ensemble_agent.load_market_series(clean_sym, timeframe="1m", limit=100)
            closes = pd.to_numeric(market_df["close"], errors="coerce").ffill()
            delta = closes.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            rsi_series = 100.0 - (100.0 / (1.0 + rs))
            latest_rsi = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 and not np.isnan(rsi_series.iloc[-1]) else 50.0
            latest_price = float(closes.iloc[-1])

            if latest_rsi >= 70.0:
                ea_action = "SELL"
                ea_conf = min(0.95, 0.70 + (latest_rsi - 70.0) * 0.015)
                ea_detail = f"RSI 14 Overbought ({latest_rsi:.1f} ≥ 70) ➔ Sinyal SELL (SL 50 Pips | TP 150 Pips)"
            elif latest_rsi <= 30.0:
                ea_action = "BUY"
                ea_conf = min(0.95, 0.70 + (30.0 - latest_rsi) * 0.015)
                ea_detail = f"RSI 14 Oversold ({latest_rsi:.1f} ≤ 30) ➔ Sinyal BUY (SL 50 Pips | TP 150 Pips)"
            else:
                ea_action = "HOLD"
                ea_conf = 0.50
                ea_detail = f"Menunggu RSI 14 ({latest_rsi:.1f}) mencapai Overbought (≥70) atau Oversold (≤30)"

            ea_sl, ea_tp, _ = calculate_ea_sl_tp(
                symbol=clean_sym,
                side=ea_action if ea_action in ("BUY", "SELL") else "BUY",
                entry_price=latest_price,
                sl_pips=50.0,
                tp_pips=150.0,
                stops_level_pips=3.0,
            )
            decision = SignalDecision(
                symbol=clean_sym,
                timeframe="1m",
                action=ea_action,
                badge=f"EA-EA {ea_action}" if ea_action in ("BUY", "SELL") else "NEUTRAL / HOLD",
                confidence=round(ea_conf, 4),
                s_tech=0.60 if ea_action == "BUY" else (-0.60 if ea_action == "SELL" else 0.0),
                s_vol=0.10,
                s_sent=0.0,
                risk_multiplier=1.0,
                entry_price=latest_price,
                stop_loss=ea_sl,
                take_profit_1=ea_tp,
                take_profit_2=ea_tp,
                expectancy_r=1.5,
                confluence_pillars=5,
                is_sideways=False,
                resistance_1=latest_price * 1.05,
                resistance_2=latest_price * 1.10,
                support_1=latest_price * 0.95,
                support_2=latest_price * 0.90,
                indicators={"rsi": latest_rsi, "ea_mode": "EA_EXPERT_ADVISOR", "ea_detail": ea_detail},
            )
        elif ea_mode == "EA_SCALPING":
            market_df = self.ensemble_agent.load_market_series(clean_sym, timeframe="1m", limit=100)
            closes = pd.to_numeric(market_df["close"], errors="coerce").ffill()
            ema9 = closes.ewm(span=9, adjust=False).mean()
            ema21 = closes.ewm(span=21, adjust=False).mean()
            latest_price = float(closes.iloc[-1])
            e9 = float(ema9.iloc[-1])
            e21 = float(ema21.iloc[-1])

            bull_cross = (e9 > e21) and (latest_price >= e9 * 0.9990)
            bear_cross = (e9 < e21) and (latest_price <= e9 * 1.0010)

            if bull_cross:
                ea_action = "BUY"
                ea_conf = 0.85
                ea_detail = f"EMA Ribbon Bullish (EMA9 {e9:,.2f} > EMA21 {e21:,.2f}) ➔ Scalping BUY (SL 20 Pips | TP 40 Pips)"
            elif bear_cross:
                ea_action = "SELL"
                ea_conf = 0.85
                ea_detail = f"EMA Ribbon Bearish (EMA9 {e9:,.2f} < EMA21 {e21:,.2f}) ➔ Scalping SELL (SL 20 Pips | TP 40 Pips)"
            else:
                ea_action = "HOLD"
                ea_conf = 0.50
                ea_detail = f"Menunggu Momentum Scalping M1 (EMA9: {e9:,.2f} vs EMA21: {e21:,.2f})"

            ea_sl, ea_tp, _ = calculate_ea_sl_tp(
                symbol=clean_sym,
                side=ea_action if ea_action in ("BUY", "SELL") else "BUY",
                entry_price=latest_price,
                sl_pips=20.0,
                tp_pips=40.0,
                stops_level_pips=3.0,
            )
            decision = SignalDecision(
                symbol=clean_sym,
                timeframe="1m",
                action=ea_action,
                badge=f"EA-SCP {ea_action}" if ea_action in ("BUY", "SELL") else "NEUTRAL / HOLD",
                confidence=round(ea_conf, 4),
                s_tech=0.60 if ea_action == "BUY" else (-0.60 if ea_action == "SELL" else 0.0),
                s_vol=0.10,
                s_sent=0.0,
                risk_multiplier=1.0,
                entry_price=latest_price,
                stop_loss=ea_sl,
                take_profit_1=ea_tp,
                take_profit_2=ea_tp,
                expectancy_r=1.0,
                confluence_pillars=5,
                is_sideways=False,
                resistance_1=latest_price * 1.05,
                resistance_2=latest_price * 1.10,
                support_1=latest_price * 0.95,
                support_2=latest_price * 0.90,
                indicators={"ema9": e9, "ema21": e21, "ea_mode": "EA_SCALPING", "ea_detail": ea_detail},
            )
        else:
            decision = self.ensemble_agent.analyze_multi_timeframe_autotrade(
                symbol=clean_sym,
                event_mode=macro_event_mode,
            )
        # Gate 2: Berita 90 Hari & Terkini Sebagai Data Pendukung Acuan Tren
        n_news = len(decision.headlines)
        is_event_sniper = bool(macro_event_mode and decision.gold_action_bias in ("BUY", "SELL"))
        if getattr(self.settings, "macro_blackout_enabled", False) and "HIGH_IMPACT_NEWS_BLACKOUT" in (decision.rejection_reason or ""):
            news_ok = False
            news_detail = f"DITOLAK (BLACKOUT KALENDER MAKRO): {decision.rejection_reason}"
        elif is_event_sniper:
            news_ok = True
            news_detail = (
                f"🎯 MODE EVENT MAKRO ({macro_event_mode.upper()} SNIPER): Sikap Kebijakan {decision.macro_stance} "
                f"(Bias Aksi: {decision.gold_action_bias} Emas) • Berita Squawk & Kalender Selaras ({n_news} berita)"
            )
        elif decision.action == "BUY" and decision.s_sent < -0.40:
            news_ok = False
            news_detail = f"DITOLAK (KONFLIK SENTIMEN EKSTREM): Berita sangat negatif ({decision.s_sent:+.2f} dari {n_news} berita), berbahaya untuk BELI"
        elif decision.action == "SELL" and decision.s_sent > 0.40:
            news_ok = False
            news_detail = f"DITOLAK (KONFLIK SENTIMEN EKSTREM): Berita sangat positif ({decision.s_sent:+.2f} dari {n_news} berita), berbahaya untuk JUAL"
        else:
            news_ok = True
            bias_tag = "Netral (Aman)" if abs(decision.s_sent) <= 0.15 else ("Bullish" if decision.s_sent > 0 else "Bearish")
            news_detail = (
                f"Data Berita 90 Hari Terolah ({decision.s_sent:+.2f} [{bias_tag}] dari {n_news} berita) ➔ "
                f"Data sentimen selaras sebagai acuan pendukung arah tren"
            )
        gates.append(
            AutoTradeGateCheck(
                gate_number=2,
                name="Analisis Berita Terbaru s/d 3 Bulan (90 Hari) Kebelakang",
                passed=news_ok,
                detail=news_detail,
            )
        )

        # Gate 3: 5-Year + Live Technical Conviction & Anti-Sideways Chop Guard
        min_conf = self.settings.signal_confidence_threshold
        tech_aligned = (
            (decision.action == "BUY" and decision.s_tech > 0.15)
            or (decision.action == "SELL" and decision.s_tech < -0.15)
        )
        ut_ts = float(decision.indicators.get("ut_trailing_stop", 0.0))
        st_val = float(decision.indicators.get("supertrend", 0.0))
        st_dir = float(decision.indicators.get("supertrend_dir", 0.0))
        oc_sig = int(decision.indicators.get("oc_signal", 0))
        oc_sl = float(decision.indicators.get("oc_sl", 0.0))
        oc_tp = float(decision.indicators.get("oc_tp", 0.0))
        ut_trend = float(decision.indicators.get("ut_trend", 0.0))
        cp_score = float(decision.indicators.get("candlestick_score", 0.0))
        cp_pat = str(decision.indicators.get("candlestick_pattern", "")).strip()

        if ea_mode == "EA_EXPERT_ADVISOR":
            ea_ok = decision.action in ("BUY", "SELL")
            ea_detail_str = decision.indicators.get("ea_detail", "")
            if ea_ok:
                tech_detail = f"🤖 EA EXPERT ADVISOR MT5 (RSI 14): {ea_detail_str} • SL 50 Pips | TP 150 Pips (1:3 RRR)"
            else:
                tech_detail = f"DITOLAK (EA-EA): {ea_detail_str}"
            conf_ok = ea_ok
        elif ea_mode == "EA_SCALPING":
            ea_ok = decision.action in ("BUY", "SELL")
            ea_detail_str = decision.indicators.get("ea_detail", "")
            if ea_ok:
                tech_detail = f"⚡ EA SCALPING M1/M5: {ea_detail_str} • SL 20 Pips | TP 40 Pips (1:2 RRR)"
            else:
                tech_detail = f"DITOLAK (EA-SCP): {ea_detail_str}"
            conf_ok = ea_ok
        elif trade_mode:
            if decision.action == "BUY":
                quad_ok = (ut_trend >= 0) and (st_dir >= 0) and (oc_sig >= 0) and (cp_score >= -0.5)
            elif decision.action == "SELL":
                quad_ok = (ut_trend <= 0) and (st_dir <= 0) and (oc_sig <= 0) and (cp_score <= 0.5)
            else:
                quad_ok = False

            conf_ok = (
                decision.action in ("BUY", "SELL")
                and quad_ok
                and not decision.is_sideways
                and decision.confidence >= min_conf
                and tech_aligned
                and decision.expectancy_r > 0.0
            )
            if not quad_ok:
                tech_detail = (
                    f"DITOLAK (/trade): Menunggu Konfluensi 4 Indikator Murni "
                    f"(UT Bot: {int(ut_trend):+d}, Supertrend: {int(st_dir):+d}, OC: {oc_sig:+d}, Candle: {cp_pat or 'Netral'})"
                )
            elif conf_ok:
                pat_str = f"[{cp_pat}] " if cp_pat else ""
                oc_info = f" • OC: {('BUY' if oc_sig == 1 else 'SELL')}" if oc_sig != 0 else ""
                tech_detail = (
                    f"💎 4 INDIKATOR MURNI TERKONFIRMASI ({pat_str}{decision.badge}) ➔ "
                    f"UT TS: ${ut_ts:,.2f} | ST: ${st_val:,.2f}{oc_info} (Keyakinan {decision.confidence * 100:.1f}%)"
                )
            elif decision.is_sideways:
                tech_detail = (
                    f"DITOLAK (/trade): Pasar Sedang Sideways / Konsolidasi ({decision.market_structure_status}) — "
                    f"Menunggu Konfirmasi Arah Tren ({decision.mtf_summary})"
                )
            elif not tech_aligned:
                tech_detail = f"DITOLAK (/trade): Sinyal Teknis Belum Selaras (S_tech={decision.s_tech:+.2f})"
            elif decision.expectancy_r <= 0.0:
                tech_detail = f"DITOLAK (/trade): Ekspektansi Nilai R Negatif ({decision.expectancy_r:+.2f}R)"
            else:
                tech_detail = f"DITOLAK (/trade): Keyakinan {decision.confidence * 100:.1f}% < {min_conf * 100:.0f}%"
        elif is_event_sniper:
            conf_ok = (
                decision.action in ("BUY", "SELL")
                and decision.confidence >= min_conf
                and decision.expectancy_r >= 0.0
            )
            tech_detail = (
                f"🎯 EVENT SNIPER ({macro_event_mode.upper()}): Menargetkan Volatilitas Rilis Berita AS • "
                f"Sikap Moneter {decision.macro_stance} ➔ Eksekusi {decision.action} ({decision.confidence * 100:.1f}%)"
            )
        else:
            conf_ok = (
                decision.action in ("BUY", "SELL")
                and not decision.is_sideways
                and decision.confidence >= min_conf
                and tech_aligned
                and decision.expectancy_r > 0.0
                and decision.confluence_pillars >= 4
            )
            if decision.is_sideways:
                tech_detail = (
                    f"DITOLAK (PASAR SIDEWAYS): {decision.market_structure_status} — "
                    f"Menunggu Breakout Valid ({decision.mtf_summary})"
                )
            elif conf_ok:
                ut_ts = float(decision.indicators.get("ut_trailing_stop", 0.0))
                st_val = float(decision.indicators.get("supertrend", 0.0))
                st_dir = float(decision.indicators.get("supertrend_dir", 0.0))
                oc_sig = int(decision.indicators.get("oc_signal", 0))
                oc_sl = float(decision.indicators.get("oc_sl", 0.0))
                oc_tp = float(decision.indicators.get("oc_tp", 0.0))
                oc_info = f" • OC Signals: {('BUY' if oc_sig == 1 else 'SELL')} (SL: ${oc_sl:,.2f} | TP: ${oc_tp:,.2f})" if oc_sig != 0 else ""
                st_tag = " [Supertrend Bull]" if st_dir > 0 else (" [Supertrend Bear]" if st_dir < 0 else "")
                strat_info = f" • UT TS: ${ut_ts:,.2f} | ST: ${st_val:,.2f}{oc_info}" if ut_ts > 0 else oc_info
                tech_detail = (
                    f"Strategi UT Bot, Supertrend & OC Signals Selaras ({decision.badge}{st_tag} • Keyakinan {decision.confidence * 100:.1f}%, "
                    f"MTF [{decision.mtf_aligned_count}/4]: {decision.mtf_summary}{strat_info}, "
                    f"Struktur: {decision.market_structure_status}, Konfluensi {decision.confluence_pillars}/6 Pilar)"
                )
            else:
                tech_detail = (
                    f"DITOLAK: Keyakinan {decision.confidence * 100:.1f}% (Min {min_conf * 100:.0f}%), "
                    f"MTF [{decision.mtf_aligned_count}/4]: {decision.mtf_summary}, "
                    f"Struktur: {decision.market_structure_status}, Konfluensi {decision.confluence_pillars}/6 Pilar"
                )
        gates.append(
            AutoTradeGateCheck(
                gate_number=3,
                name=(
                    "EA Expert Advisor MT5 (RSI 14)"
                    if ea_mode == "EA_EXPERT_ADVISOR"
                    else (
                        "EA Scalping M1/M5 (Micro Momentum)"
                        if ea_mode == "EA_SCALPING"
                        else ("Strategi 4 Indikator Murni (/trade)" if trade_mode else "Strategi UT Bot + Supertrend + OC Signals (1D•1H•15M•5M ➔ 1M)")
                    )
                ),
                passed=conf_ok,
                detail=tech_detail,
            )
        )

        # Gate 4: Support & Resistance Structural Room Check
        if decision.action == "BUY":
            if decision.resistance_1 > 0 and decision.entry_price >= decision.resistance_1:
                # Bullish Breakout above R1: Measure room towards Resistance R2
                dist_to_r2_pct = ((decision.resistance_2 - decision.entry_price) / max(decision.entry_price, 1e-6)) * 100.0
                sr_ok = dist_to_r2_pct >= 0.15
                sr_detail = (
                    f"Breakout R1 Terkonfirmasi: Ruang naik ke Resistance R2 ({decision.resistance_2:,.2f}) terbuka lega (+{dist_to_r2_pct:.2f}%)"
                    if sr_ok
                    else f"DITOLAK: Breakout R1 tapi terlalu mepet atap Resistance R2 ({decision.resistance_2:,.2f}, sisa +{dist_to_r2_pct:.2f}%)"
                )
            else:
                sr_ok = decision.dist_to_r1_pct >= 0.15
                sr_detail = (
                    f"Ruang Naik ke Resistance R1 ({decision.resistance_1:,.2f}) masih lega (+{decision.dist_to_r1_pct:.2f}%)"
                    if sr_ok
                    else f"DITOLAK: Harga terlalu mepet atap Resistance R1 ({decision.resistance_1:,.2f}, sisa +{decision.dist_to_r1_pct:.2f}%)"
                )
        elif decision.action == "SELL":
            if decision.support_1 > 0 and decision.entry_price <= decision.support_1:
                # Bearish Breakdown below S1: Measure room towards Support S2
                dist_to_s2_pct = ((decision.entry_price - decision.support_2) / max(decision.entry_price, 1e-6)) * 100.0
                sr_ok = dist_to_s2_pct >= 0.15
                sr_detail = (
                    f"Breakdown S1 Terkonfirmasi: Ruang turun ke Support S2 ({decision.support_2:,.2f}) terbuka lega (-{dist_to_s2_pct:.2f}%)"
                    if sr_ok
                    else f"DITOLAK: Breakdown S1 tapi terlalu mepet lantai Support S2 ({decision.support_2:,.2f}, sisa -{dist_to_s2_pct:.2f}%)"
                )
            else:
                sr_ok = abs(decision.dist_to_s1_pct) >= 0.15
                sr_detail = (
                    f"Ruang Turun ke Support S1 ({decision.support_1:,.2f}) masih lega ({decision.dist_to_s1_pct:.2f}%)"
                    if sr_ok
                    else f"DITOLAK: Harga terlalu mepet lantai Support S1 ({decision.support_1:,.2f}, sisa {decision.dist_to_s1_pct:.2f}%)"
                )
        else:
            sr_ok = False
            sr_detail = f"Menunggu sinyal arah di antara Support S1 ({decision.support_1:,.2f}) & Resistance R1 ({decision.resistance_1:,.2f})"

        gates.append(
            AutoTradeGateCheck(
                gate_number=4,
                name="Ruang Gerak Support & Resistance",
                passed=sr_ok,
                detail=sr_detail,
            )
        )

        # Gate 5 & 6: Atomic per-account/symbol lock prevents duplicate concurrent entries (F07)
        acct_key = creds.account_id or creds.api_key.strip() or str(user_id)
        exec_lock = self._execution_locks.setdefault((acct_key, clean_sym), asyncio.Lock())
        async with exec_lock:
            await self.sync_live_mt5_positions(user_id)
            open_positions = self.vault.get_open_positions(user_id, symbol=clean_sym, account_id=creds.account_id or None)
            effective_max_pos = max(1, int(max_positions))
            has_duplicate = len(open_positions) >= effective_max_pos
            user_dd, start_eq, curr_eq = self.vault.get_user_daily_drawdown(user_id, account_id=creds.account_id or None)
            eff_dd = user_dd
            is_micro_account = 0.0 < float(creds.verified_usdt_balance) < 100.0
            eff_dd_limit = (
                max(float(self.settings.max_daily_drawdown_limit), 0.20)
                if is_micro_account
                else float(self.settings.max_daily_drawdown_limit)
            )
            cb_tripped = (
                eff_dd >= eff_dd_limit
                if is_micro_account
                else self.ensemble_agent.position_sizer.is_circuit_breaker_tripped(eff_dd)
            )
            risk_ok = (
                decision.approved
                and not cb_tripped
                and decision.volatility_regime != "EXTREME"
                and not has_duplicate
            )
            if has_duplicate:
                risk_detail = (
                    f"DITOLAK (BATAS LAYER TERCAPAI): Sudah ada {len(open_positions)}/{effective_max_pos} "
                    f"posisi {clean_sym} yang sedang OPEN di akun Anda"
                )
            elif cb_tripped:
                risk_detail = (
                    f"DITOLAK: Circuit Breaker Rugi Harian aktif "
                    f"(Drawdown {eff_dd * 100:.2f}% >= batas {eff_dd_limit * 100:.1f}%)"
                )
            elif not decision.approved or decision.volatility_regime == "EXTREME":
                risk_detail = f"DITOLAK: {decision.rejection_reason or 'Volatilitas pasar terlalu ekstrem'}"
            else:
                risk_detail = (
                    f"Aman (Gejolak {decision.volatility_regime}, S_vol={decision.s_vol:.2f}, "
                    f"Drawdown Harian {eff_dd * 100:.2f}%, Layer {len(open_positions)}/{effective_max_pos})"
                )

            gates.append(
                AutoTradeGateCheck(
                    gate_number=5,
                    name="Filter Gejolak, Circuit Breaker & Kontrol Layer Posisi",
                    passed=risk_ok,
                    detail=risk_detail,
                )
            )

            failed_gates = [g for g in gates if not g.passed]
            if failed_gates:
                first_fail = failed_gates[0]
                return AutoTradeExecutionReport(
                    user_id=int(user_id),
                    symbol=clean_sym,
                    timeframe=clean_tf,
                    setup_required=False,
                    executed=False,
                    decision=decision,
                    credentials=creds,
                    gates=gates,
                    rejection_summary=f"Gerbang {first_fail.gate_number} ({first_fail.name}) belum terpenuhi: {first_fail.detail}",
                    trade_mode=trade_mode,
                    ea_mode=ea_mode,
                )

            # Gate 6: Strict min(user_limit, global_limit) risk cap & NO SL narrowing (R01, R02, NT-02)
            user_equity = float(creds.verified_usdt_balance)
            user_sizing = self.ensemble_agent.position_sizer.calculate_position(
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                confidence=decision.confidence,
                s_vol=decision.s_vol,
                portfolio_equity=user_equity,
                reward_risk_ratio=self.settings.min_risk_reward_ratio,
                daily_drawdown_override=(
                    eff_dd if not is_micro_account else (eff_dd if eff_dd >= eff_dd_limit else 0.0)
                ),
            )
            if not user_sizing.approved:
                gates.append(
                    AutoTradeGateCheck(
                        gate_number=6,
                        name="Validasi Risiko & Ukuran Lot",
                        passed=False,
                        detail=f"DITOLAK: {user_sizing.rejection_reason or 'Tidak lolos validasi risiko posisi.'}",
                    )
                )
                return AutoTradeExecutionReport(
                    user_id=int(user_id),
                    symbol=clean_sym,
                    timeframe=clean_tf,
                    setup_required=False,
                    executed=False,
                    decision=decision,
                    credentials=creds,
                    gates=gates,
                    rejection_summary=user_sizing.rejection_reason or "Ditolak oleh PositionSizer.",
                    trade_mode=trade_mode,
                    ea_mode=ea_mode,
                )

            order_side = "BUY" if decision.action == "BUY" else "SELL"
            qty = user_sizing.position_units
            candidate_lots = (
                round(max(0.01, min(float(custom_lot), 50.0)), 2)
                if (custom_lot is not None and custom_lot > 0)
                else calculate_mt5_lots(clean_sym, qty, decision.entry_price)
            )
            eff_risk_rate = (
                0.01
                if ea_mode
                else min(float(creds.risk_per_trade_pct), float(self.settings.max_account_risk_per_trade))
            )
            trade_budget = user_equity * eff_risk_rate
            if is_micro_account:
                # Rekomendasi 1 & Saldo Mikro: Dukung jarak Stop Loss 5M yang sehat pada lot minimum broker (0.01)
                micro_min_lot_budget = min(user_equity * 0.45, 4.50) if candidate_lots <= 0.01 else min(user_equity * 0.12, 1.20)
                trade_budget = max(trade_budget, micro_min_lot_budget)
            daily_headroom = (
                max(0.0, start_eq * eff_dd_limit - max(0.0, start_eq - curr_eq))
                if start_eq > 0
                else trade_budget
            )
            max_risk_budget = min(trade_budget, daily_headroom)
            if macro_event_mode:
                event_targets = self.ensemble_agent.stop_loss_manager.calculate_event_mode_targets(
                    entry_price=decision.entry_price,
                    action=decision.action,
                    account_balance=user_equity,
                    lot_size=candidate_lots,
                    symbol=clean_sym,
                    risk_pct=eff_risk_rate,
                )
                eff_stop_loss = event_targets.stop_loss
                eff_take_profit = 0.0
                decision.stop_loss = eff_stop_loss
                decision.take_profit_1 = 0.0
                decision.take_profit_2 = 0.0
            else:
                eff_stop_loss = decision.stop_loss
                eff_take_profit = decision.take_profit_1

            loss_at_stop = abs(
                estimate_mt5_pnl_usdt(
                    clean_sym, order_side, decision.entry_price, eff_stop_loss, candidate_lots
                )
            )
            if not macro_event_mode and loss_at_stop > max_risk_budget + 1e-6:
                loss_per_lot = abs(
                    estimate_mt5_pnl_usdt(
                        clean_sym, order_side, decision.entry_price, eff_stop_loss, 1.0
                    )
                )
                affordable_lots = (
                    math.floor((max_risk_budget / max(loss_per_lot, 1e-9)) * 100.0 + 1e-9) / 100.0
                )
                if custom_lot is None and affordable_lots >= 0.01:
                    candidate_lots = round(affordable_lots, 2)
                else:
                    # R02 / NT-02: NEVER narrow stop_loss to force 0.01 lot; reject with MIN_LOT_EXCEEDS_RISK
                    rej_code = "CUSTOM_LOT_EXCEEDS_RISK" if custom_lot is not None else "MIN_LOT_EXCEEDS_RISK"
                    gates.append(
                        AutoTradeGateCheck(
                            gate_number=6,
                            name=f"Batas Risiko Lot ({rej_code})",
                            passed=False,
                            detail=(
                                f"DITOLAK ({rej_code}): Risiko {candidate_lots:.2f} Lot ke SL (${loss_at_stop:,.2f}) "
                                f"melampaui anggaran risiko efektif (${max_risk_budget:,.2f})."
                            ),
                        )
                    )
                    return AutoTradeExecutionReport(
                        user_id=int(user_id),
                        symbol=clean_sym,
                        timeframe=clean_tf,
                        setup_required=False,
                        executed=False,
                        decision=decision,
                        credentials=creds,
                        gates=gates,
                        rejection_summary=f"{rej_code}: Risiko {candidate_lots:.2f} Lot (${loss_at_stop:,.2f}) > budget (${max_risk_budget:,.2f}).",
                        trade_mode=trade_mode,
                        ea_mode=ea_mode,
                    )

            # Anti-Broker-Ban Protection 1: Request Throttle (InpThrottleMs = 500ms to 1500ms)
            await self._enforce_broker_throttle(acct_key, 500, 1500)
            order_res = await self.mt5_client.place_market_order(
                account_id_or_login=creds.api_key,
                token_or_password=creds.api_secret,
                symbol=clean_sym,
                side=order_side,
                quantity=qty,
                reference_price=decision.entry_price,
                stop_loss=eff_stop_loss,
                take_profit=eff_take_profit,
                server_name=creds.server_name,
                testnet=creds.testnet,
                custom_lot=candidate_lots,
            )

            if not order_res.executed:
                unprotected_rec: Optional[UserPositionRecord] = None
                # R05 / NT-05: If order filled on broker but SL/TP modify AND emergency close failed, record FILLED_UNPROTECTED
                if order_res.status == "FILLED_UNPROTECTED" and order_res.executed_qty > 0:
                    unprotected_rec = self.vault.record_open_position(
                        user_id=int(user_id),
                        symbol=clean_sym,
                        side=order_side,
                        entry_price=order_res.avg_price,
                        quantity=order_res.executed_qty,
                        notional_usdt=order_res.notional_usdt,
                        stop_loss=0.0,
                        take_profit_1=eff_take_profit,
                        take_profit_2=0.0 if macro_event_mode else decision.take_profit_2,
                        confidence=decision.confidence,
                        order_id=order_res.order_id,
                        testnet=creds.testnet,
                        account_id=creds.account_id,
                        status="FILLED_UNPROTECTED",
                    )
                gates.append(
                    AutoTradeGateCheck(
                        gate_number=6,
                        name="Eksekusi Order ke Exness Go / MT5",
                        passed=False,
                        detail=order_res.error_message or "Order gagal dieksekusi oleh Exness MT5.",
                    )
                )
                return AutoTradeExecutionReport(
                    user_id=int(user_id),
                    symbol=clean_sym,
                    timeframe=clean_tf,
                    setup_required=False,
                    executed=False,
                    decision=decision,
                    credentials=creds,
                    gates=gates,
                    order_result=order_res,
                    position_record=unprotected_rec,
                    rejection_summary=order_res.error_message or "Eksekusi order ditolak oleh Exness MT5.",
                    trade_mode=trade_mode,
                    ea_mode=ea_mode,
                )

            anchored_sl, anchored_tp1, anchored_tp2 = anchor_targets_to_fill_price(
                side=order_side,
                fill_price=order_res.avg_price,
                ref_entry=decision.entry_price,
                ref_sl=eff_stop_loss,
                ref_tp1=decision.take_profit_1,
                ref_tp2=decision.take_profit_2,
            )

            pos_record = self.vault.record_open_position(
                user_id=int(user_id),
                symbol=clean_sym,
                side=order_side,
                entry_price=order_res.avg_price,
                quantity=order_res.executed_qty,
                notional_usdt=order_res.notional_usdt,
                stop_loss=anchored_sl,
                take_profit_1=anchored_tp1,
                take_profit_2=0.0 if macro_event_mode else anchored_tp2,
                confidence=decision.confidence,
                order_id=order_res.order_id,
                testnet=creds.testnet,
                account_id=creds.account_id,
                status="OPEN",
            )
        gates.append(
            AutoTradeGateCheck(
                gate_number=6,
                name="Eksekusi Order & Kunci SL/TP",
                passed=True,
                detail=(
                    f"BERHASIL: {order_side} {order_res.executed_qty} {order_res.symbol} @ ${order_res.avg_price:,.2f} "
                    f"(Order ID: {order_res.order_id})"
                ),
            )
        )

        return AutoTradeExecutionReport(
            user_id=int(user_id),
            symbol=clean_sym,
            timeframe=clean_tf,
            setup_required=False,
            executed=True,
            decision=decision,
            credentials=creds,
            gates=gates,
            order_result=order_res,
            position_record=pos_record,
            trade_mode=trade_mode,
            ea_mode=ea_mode,
        )
