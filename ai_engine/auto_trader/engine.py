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
    GridScalperConfig,
    build_pending_stop_grid_levels,
    calculate_compounded_grid_lot,
    calculate_ea_sl_tp,
    get_mt5_digits_and_point,
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
from ai_engine.auto_trader.nx_ea_engine import NXEAEngine, NXSignalResult
from ai_engine.auto_trader.ea_scalping_engine import EAScalpingEngine, EAScalpingSignal
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
from ai_engine.risk_management.risk_policy import (
    AccountRiskSnapshot,
    PreflightRiskResult,
    RiskPolicy,
    SymbolTradingSpec,
)
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

    async def evaluate_and_execute_grid_cycle(self, sess: AutoTradeSessionState) -> Any:
        """
        Execute one cycle of EA_GRID_SCALPER:
        1. Query live balance and equity.
        2. Check Target Equity ($5,000) or Max Drawdown -> Mass Close if reached.
        3. Calculate compounded lot size.
        4. Deploy/verify pending stop grid trap orders.
        """
        creds = self.vault.resolve_credentials(sess.user_id)
        if creds is None:
            return None
        acct_key = creds.account_id or creds.api_key.strip() or str(sess.user_id)
        await refresh_live_exness_balance(self.vault, self.mt5_client, sess.user_id, creds)
        live_eq = float(creds.verified_usdt_balance)

        cfg = GridScalperConfig()

        # Hard Target Equity Auto-Stop ($5,000 target from video)
        if live_eq >= cfg.target_equity_usd:
            logger.info("Target Equity reached in Grid EA cycle", equity=live_eq)
            await self.mt5_client.close_all_and_delete_pending(creds.api_key, creds.api_secret, sess.symbol, creds.server_name)
            sess.status = "COMPLETED"
            if self.notify_user_callback:
                await self.notify_user_callback(
                    sess.user_id,
                    f"🎉 <b>TARGET EQUITY TERCAPAI (${live_eq:,.2f} >= ${cfg.target_equity_usd:,.2f})</b>\n"
                    f"Seluruh posisi dan pending order {sess.symbol} berhasil ditutup & dibersihkan."
                )
            return None

        comp_lot = calculate_compounded_grid_lot(live_eq, cfg)
        sess.current_lot = comp_lot

        ref_px = sess.live_current_price if sess.live_current_price > 0 else 2650.0
        pip_sz = get_mt5_pip_size(sess.symbol, ref_px)
        spread = 15 * (pip_sz * 0.10)
        grid_orders = build_pending_stop_grid_levels(sess.symbol, ref_px, ref_px + spread, cfg)

        sess.last_check_summary = f"Trap Grid M1 Aktif (Lot {comp_lot:.2f} | Equity ${live_eq:,.2f})"
        return sess
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
        last_exit_reason: Optional[str] = None,
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
            g_val = float(avg_gain.iloc[-1])
            l_val = float(avg_loss.iloc[-1])
            if l_val <= 1e-12:
                latest_rsi = 100.0 if g_val > 0 else 50.0
            elif g_val <= 1e-12:
                latest_rsi = 0.0
            else:
                rs = g_val / l_val
                latest_rsi = round(100.0 - (100.0 / (1.0 + rs)), 2)
            latest_price = float(closes.iloc[-1])

            # NT-08: Require closed-bar reversal confirmation, not blind overbought/oversold counter-trend entries
            candle_reversal_sell = (len(closes) >= 2 and float(closes.iloc[-1]) < float(closes.iloc[-2]))
            candle_reversal_buy = (len(closes) >= 2 and float(closes.iloc[-1]) > float(closes.iloc[-2]))

            if latest_rsi >= 70.0 and candle_reversal_sell:
                ea_action = "SELL"
                rsi_excess = (latest_rsi - 70.0) / 30.0
                ea_conf = round(min(0.88, 0.70 + (rsi_excess * 0.15)), 4)
                ea_detail = f"RSI 14 Overbought ({latest_rsi:.1f} ≥ 70) + Konfirmasi Reversal Bearish ➔ Sinyal SELL"
            elif latest_rsi <= 30.0 and candle_reversal_buy:
                ea_action = "BUY"
                rsi_deficit = (30.0 - latest_rsi) / 30.0
                ea_conf = round(min(0.88, 0.70 + (rsi_deficit * 0.15)), 4)
                ea_detail = f"RSI 14 Oversold ({latest_rsi:.1f} ≤ 30) + Konfirmasi Reversal Bullish ➔ Sinyal BUY"
            elif latest_rsi >= 70.0 or latest_rsi <= 30.0:
                ea_action = "HOLD"
                ea_conf = 0.50
                ea_detail = f"RSI 14 Ekstrem ({latest_rsi:.1f}), Menunggu Konfirmasi Penutupan Candle Reversal"
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
                position_units=round(0.01 * get_mt5_contract_size(clean_sym), 4),
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
            latest_price = float(market_df["close"].iloc[-1]) if len(market_df) > 0 else 2650.0

            # Evaluasi EA Scalping: Trend reading + Candlestick reading + Forward prediction + Tight SL/TP
            scalp_res = EAScalpingEngine.evaluate_scalping(
                df=market_df,
                symbol=clean_sym,
                sl_pips=12.0,
                tp_pips=14.0,
                last_exit_reason=last_exit_reason,
            )
            ea_action = scalp_res.action
            ea_conf = scalp_res.confidence
            ea_detail = scalp_res.details

            decision = SignalDecision(
                symbol=clean_sym,
                timeframe="1m",
                action=ea_action,
                badge=f"EA-SCALPING {ea_action}" if ea_action in ("BUY", "SELL") else "NEUTRAL / HOLD",
                confidence=round(ea_conf, 4),
                s_tech=0.60 if ea_action == "BUY" else (-0.60 if ea_action == "SELL" else 0.0),
                s_vol=0.10,
                s_sent=0.0,
                risk_multiplier=1.0,
                entry_price=latest_price,
                position_units=round(0.01 * get_mt5_contract_size(clean_sym), 4),
                stop_loss=scalp_res.stop_loss,
                take_profit_1=scalp_res.take_profit,
                take_profit_2=scalp_res.take_profit,
                expectancy_r=1.2,
                confluence_pillars=5,
                is_sideways=False,
                resistance_1=latest_price * 1.05,
                resistance_2=latest_price * 1.10,
                support_1=latest_price * 0.95,
                support_2=latest_price * 0.90,
                indicators={
                    "ea_mode": "EA_SCALPING",
                    "trend": scalp_res.trend_direction,
                    "candle": scalp_res.candle_pattern,
                    "prediction": scalp_res.forward_prediction,
                    "ea_detail": ea_detail,
                },
            )
        elif ea_mode == "EA_GRID_SCALPER":
            market_df = self.ensemble_agent.load_market_series(clean_sym, timeframe="1m", limit=100)
            closes = pd.to_numeric(market_df["close"], errors="coerce").ffill()
            latest_price = float(closes.iloc[-1])
            grid_cfg = GridScalperConfig()
            comp_lot = calculate_compounded_grid_lot(user_equity, grid_cfg)
            ea_sl, ea_tp, _ = calculate_ea_sl_tp(
                symbol=clean_sym,
                side="BUY",
                entry_price=latest_price,
                sl_pips=grid_cfg.stop_loss_pips,
                tp_pips=grid_cfg.take_profit_pips,
                stops_level_pips=3.0,
            )
            decision = SignalDecision(
                symbol=clean_sym,
                timeframe="1m",
                action="BUY",
                badge="EA-GRID SCALPER",
                confidence=0.90,
                s_tech=0.60,
                s_vol=0.10,
                s_sent=0.0,
                risk_multiplier=1.0,
                entry_price=latest_price,
                position_units=round(comp_lot * get_mt5_contract_size(clean_sym), 4),
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
                indicators={"ea_mode": "EA_GRID_SCALPER", "comp_lot": comp_lot, "ea_detail": f"Trap Grid M1 Aktif (Lot {comp_lot:.2f})"},
            )
        elif ea_mode in ("EA_NX_BREAK", "EA_NX_PARSIAL", "EA_NX_REVERSE", "EA_NX_POC"):
            market_df = self.ensemble_agent.load_market_series(clean_sym, timeframe="1m", limit=120)
            closes = pd.to_numeric(market_df["close"], errors="coerce").ffill()
            latest_price = float(closes.iloc[-1])
            if ea_mode == "EA_NX_BREAK":
                nx_res = NXEAEngine.evaluate_nx_break(market_df, clean_sym)
            elif ea_mode == "EA_NX_PARSIAL":
                nx_res = NXEAEngine.evaluate_nx_parsial(market_df, clean_sym)
            elif ea_mode == "EA_NX_REVERSE":
                nx_res = NXEAEngine.evaluate_nx_reverse(market_df, clean_sym)
            else:
                nx_res = NXEAEngine.evaluate_nx_poc(market_df, clean_sym)

            ea_action = nx_res.action
            ea_conf = 0.88 if ea_action in ("BUY", "SELL") else 0.50
            sl_val = nx_res.stop_loss if nx_res.stop_loss > 0 else (latest_price * 0.98 if ea_action == "BUY" else latest_price * 1.02)
            tp1_val = nx_res.take_profit_levels[0] if nx_res.take_profit_levels else nx_res.take_profit
            tp2_val = nx_res.take_profit_levels[1] if len(nx_res.take_profit_levels) > 1 else nx_res.take_profit
            decision = SignalDecision(
                symbol=clean_sym,
                timeframe="1m",
                action=ea_action,
                badge=f"{nx_res.strategy.upper()} {ea_action}" if ea_action in ("BUY", "SELL") else "NEUTRAL / HOLD",
                confidence=ea_conf,
                s_tech=0.60 if ea_action == "BUY" else (-0.60 if ea_action == "SELL" else 0.0),
                s_vol=0.10,
                s_sent=0.0,
                risk_multiplier=1.0,
                entry_price=latest_price,
                position_units=round(0.01 * get_mt5_contract_size(clean_sym), 4),
                stop_loss=sl_val,
                take_profit_1=tp1_val,
                take_profit_2=tp2_val,
                expectancy_r=1.5,
                confluence_pillars=5,
                is_sideways=False,
                resistance_1=latest_price * 1.05,
                resistance_2=latest_price * 1.10,
                support_1=latest_price * 0.95,
                support_2=latest_price * 0.90,
                indicators={
                    "ea_mode": ea_mode,
                    "nx_detail": nx_res.details,
                    "nx_strategy": nx_res.strategy,
                    "nx_levels": nx_res.take_profit_levels,
                },
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
                tech_detail = f"⚡ EA SCALPING (TREND + CANDLE): {ea_detail_str}"
            else:
                tech_detail = f"DITOLAK (EA-SCALPING): {ea_detail_str}"
            conf_ok = ea_ok
        elif ea_mode == "EA_GRID_SCALPER":
            ea_ok = True
            tech_detail = f"🪤 EA GRID/TRAP SCALPER M1: Pending Stop Trap Grid Aktif (Lot {decision.indicators.get('comp_lot', 0.01):.2f})"
            conf_ok = True
        elif ea_mode and ea_mode.startswith("EA_NX_"):
            ea_ok = decision.action in ("BUY", "SELL")
            nx_detail_str = decision.indicators.get("nx_detail", "")
            if ea_ok:
                tech_detail = f"🎯 {decision.badge}: {nx_detail_str}"
            else:
                tech_detail = f"DITOLAK ({ea_mode}): {nx_detail_str}"
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
                    f"EA {decision.indicators.get('nx_strategy', ea_mode)}"
                    if ea_mode and ea_mode.startswith("EA_NX_")
                    else (
                        "EA Grid/Trap Scalper M1"
                        if ea_mode == "EA_GRID_SCALPER"
                        else (
                            "EA Expert Advisor MT5 (RSI 14)"
                            if ea_mode == "EA_EXPERT_ADVISOR"
                            else (
                                "EA Scalping (Trend + Candle + Prediksi)"
                                if ea_mode == "EA_SCALPING"
                                else ("Strategi 4 Indikator Murni (/trade)" if trade_mode else "Strategi UT Bot + Supertrend + OC Signals (1D•1H•15M•5M ➔ 1M)")
                            )
                        )
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
            user_equity = float(creds.verified_usdt_balance)

            # Persistent Account Risk Snapshot (NT-04, NT-11, R-16)
            acct_snapshot = self.vault.get_account_risk_snapshot(
                account_id=acct_key, user_id=int(user_id), live_balance=user_equity
            )

            # Check persistent latched circuit breaker
            cb_tripped = (
                acct_snapshot.is_circuit_breaker_latched
                or (eff_dd >= self.settings.max_daily_drawdown_limit)
            )
            if eff_dd >= self.settings.max_daily_drawdown_limit and not acct_snapshot.is_circuit_breaker_latched:
                self.vault.latch_circuit_breaker(
                    acct_key, reason=f"Drawdown {eff_dd*100:.2f}% >= batas {self.settings.max_daily_drawdown_limit*100:.1f}%"
                )
                acct_snapshot.is_circuit_breaker_latched = True

            all_acct_positions = self.vault.get_open_positions(user_id, symbol=None, account_id=creds.account_id or None)
            has_unprotected = any(p.status == "FILLED_UNPROTECTED" for p in all_acct_positions)
            risk_ok = (
                decision.approved
                and not cb_tripped
                and not acct_snapshot.kill_switch_active
                and not has_unprotected
                and decision.volatility_regime != "EXTREME"
                and not has_duplicate
            )
            if acct_snapshot.kill_switch_active:
                risk_detail = "DITOLAK: Master Kill Switch aktif. Eksekusi dibekukan."
            elif has_unprotected:
                risk_detail = "DITOLAK (INCIDENT_UNPROTECTED): Terdapat posisi terbuka tanpa proteksi Stop Loss di broker."
            elif has_duplicate:
                risk_detail = (
                    f"DITOLAK (BATAS LAYER TERCAPAI): Sudah ada {len(open_positions)}/{effective_max_pos} "
                    f"posisi {clean_sym} yang sedang OPEN di akun Anda"
                )
            elif cb_tripped:
                risk_detail = (
                    f"DITOLAK: Circuit Breaker Rugi Harian aktif (terkunci persisten) "
                    f"(Drawdown {eff_dd * 100:.2f}% >= batas {self.settings.max_daily_drawdown_limit * 100:.1f}%)"
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

            # Gate 6: Centralized Non-Bypassable Risk Policy Preflight (NT-01, NT-02, NT-03, NT-11)
            order_side = "BUY" if decision.action == "BUY" else "SELL"
            eff_risk_rate = (
                0.01
                if ea_mode
                else min(float(creds.risk_per_trade_pct), float(self.settings.max_account_risk_per_trade))
            )

            if macro_event_mode:
                event_targets = self.ensemble_agent.stop_loss_manager.calculate_event_mode_targets(
                    entry_price=decision.entry_price,
                    action=decision.action,
                    account_balance=user_equity,
                    lot_size=0.01,
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

            digits, pt = get_mt5_digits_and_point(clean_sym, decision.entry_price)
            spec = SymbolTradingSpec(
                symbol=clean_sym,
                digits=digits,
                point=pt,
                contract_size=get_mt5_contract_size(clean_sym),
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                spread_points=15,
            )

            policy = RiskPolicy(
                max_risk_per_trade_pct=eff_risk_rate,
                max_daily_drawdown_limit=self.settings.max_daily_drawdown_limit,
                max_total_drawdown_limit=self.settings.max_total_drawdown_limit,
                max_open_risk_fraction=self.settings.max_open_risk_fraction,
                live_trading_enabled=self.settings.live_trading_enabled,
                allow_micro_min_lot_override=bool(ea_mode),
            )

            preflight = policy.evaluate_preflight(
                spec=spec,
                account=acct_snapshot,
                side=order_side,
                entry_price=decision.entry_price,
                stop_loss=eff_stop_loss,
                take_profit=eff_take_profit,
                requested_lots=float(custom_lot) if custom_lot is not None and custom_lot > 0 else None,
                is_live_account=not creds.testnet,
            )

            if not preflight.approved:
                rej_code = preflight.rejection_code or "RISK_REJECTED"
                gates.append(
                    AutoTradeGateCheck(
                        gate_number=6,
                        name=f"Batas Risiko Lot ({rej_code})",
                        passed=False,
                        detail=f"DITOLAK ({rej_code}): {preflight.rejection_reason}",
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
                    rejection_summary=f"{rej_code}: {preflight.rejection_reason}",
                    trade_mode=trade_mode,
                    ea_mode=ea_mode,
                )

            candidate_lots = preflight.allowed_lots

            # NT-04: Persisted Order Lifecycle Intent & Atomic Risk Reservation
            intent_id = f"INT-{int(user_id)}-{clean_sym}-{int(time.time()*1000)}"
            self.vault.record_order_intent(
                intent_id=intent_id,
                user_id=int(user_id),
                account_id=acct_key,
                symbol=clean_sym,
                side=order_side,
                requested_lots=candidate_lots,
                price=decision.entry_price,
                stop_loss=eff_stop_loss,
                take_profit=eff_take_profit,
                state="RISK_RESERVED",
            )
            self.vault.reserve_risk(
                intent_id=intent_id,
                account_id=acct_key,
                user_id=int(user_id),
                symbol=clean_sym,
                reserved_loss_usd=preflight.estimated_risk_usd,
            )
            self.vault.update_order_intent_state(intent_id, state="SUBMITTING")

            try:
                # Anti-Broker-Ban Protection 1: Request Throttle (InpThrottleMs = 500ms to 1500ms)
                await self._enforce_broker_throttle(acct_key, 500, 1500)
                order_res = await self.mt5_client.place_market_order(
                    account_id_or_login=creds.api_key,
                    token_or_password=creds.api_secret,
                    symbol=clean_sym,
                    side=order_side,
                    quantity=candidate_lots,
                    reference_price=decision.entry_price,
                    stop_loss=eff_stop_loss,
                    take_profit=eff_take_profit,
                    server_name=creds.server_name,
                    testnet=creds.testnet,
                    custom_lot=candidate_lots,
                )
            finally:
                self.vault.release_risk_reservation(intent_id)

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
            # NT-04 / NT-05: Update intent state machine
            self.vault.update_order_intent_state(
                intent_id, state="PROTECTION_VERIFIED", broker_ticket=order_res.order_id, filled_lots=order_res.executed_qty
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
