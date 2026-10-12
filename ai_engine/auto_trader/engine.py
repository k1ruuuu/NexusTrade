from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
from typing import Any, Optional

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
        self.session_manager = AutoTradeSessionManager(self)
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
        )

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
        decision = decision_override or self.ensemble_agent.analyze_multi_timeframe_autotrade(
            symbol=clean_sym
        )
        # Gate 2: Mandatory Breaking + 3-Month (90-Day) Deep News Sentiment Alignment
        n_news = len(decision.headlines)
        if decision.action == "BUY":
            news_ok = decision.s_sent > -0.15
            news_detail = (
                f"Berita Terbaru s/d 3 Bulan Selaras ({decision.s_sent:+.2f} dari {n_news} berita 90 hari mendukung/netral untuk BELI)"
                if news_ok
                else f"DITOLAK: Berita Terbaru s/d 3 Bulan negatif ({decision.s_sent:+.2f} dari {n_news} berita), berbahaya untuk BELI"
            )
        elif decision.action == "SELL":
            news_ok = decision.s_sent < 0.15
            news_detail = (
                f"Berita Terbaru s/d 3 Bulan Selaras ({decision.s_sent:+.2f} dari {n_news} berita 90 hari mendukung/netral untuk JUAL)"
                if news_ok
                else f"DITOLAK: Berita Terbaru s/d 3 Bulan positif ({decision.s_sent:+.2f} dari {n_news} berita), berbahaya untuk JUAL"
            )
        else:
            news_ok = False
            news_detail = f"Sentimen berita 3 bulan {decision.s_sent:+.2f} ({n_news} berita), menunggu pemicu arah kuat"

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
            tech_detail = (
                f"Sinyal {decision.badge} Kuat (Keyakinan {decision.confidence * 100:.1f}%, "
                f"MTF [{decision.mtf_aligned_count}/4 Selaras]: {decision.mtf_summary}, "
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
                name="Tren Multi-Timeframe (1D•1H•15M•5M) ➔ Eksekusi 1M & Anti-Sideways",
                passed=conf_ok,
                detail=tech_detail,
            )
        )

        # Gate 4: Support & Resistance Structural Room Check
        if decision.action == "BUY":
            sr_ok = decision.dist_to_r1_pct >= 0.15
            sr_detail = (
                f"Ruang Naik ke Resistance R1 ({decision.resistance_1:,.2f}) masih lega (+{decision.dist_to_r1_pct:.2f}%)"
                if sr_ok
                else f"DITOLAK: Harga terlalu mepet atap Resistance R1 ({decision.resistance_1:,.2f}, sisa +{decision.dist_to_r1_pct:.2f}%)"
            )
        elif decision.action == "SELL":
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
            has_duplicate = len(open_positions) > 0
            user_dd, start_eq, curr_eq = self.vault.get_user_daily_drawdown(user_id, account_id=creds.account_id or None)
            eff_dd = user_dd
            cb_tripped = self.ensemble_agent.position_sizer.is_circuit_breaker_tripped(eff_dd)

            risk_ok = (
                decision.approved
                and not cb_tripped
                and decision.volatility_regime != "EXTREME"
                and not has_duplicate
            )
            if has_duplicate:
                risk_detail = f"DITOLAK: Masih ada {len(open_positions)} posisi {clean_sym} yang sedang OPEN di akun Anda"
            elif cb_tripped:
                risk_detail = (
                    f"DITOLAK: Circuit Breaker Rugi Harian aktif "
                    f"(Drawdown {eff_dd * 100:.2f}% >= batas {self.settings.max_daily_drawdown_limit * 100:.1f}%)"
                )
            elif not decision.approved or decision.volatility_regime == "EXTREME":
                risk_detail = f"DITOLAK: {decision.rejection_reason or 'Volatilitas pasar terlalu ekstrem'}"
            else:
                risk_detail = (
                    f"Aman (Gejolak {decision.volatility_regime}, S_vol={decision.s_vol:.2f}, "
                    f"Drawdown Harian {eff_dd * 100:.2f}%, Tidak ada posisi ganda)"
                )

            gates.append(
                AutoTradeGateCheck(
                    gate_number=5,
                    name="Filter Gejolak, Circuit Breaker & Anti-Posisi Ganda",
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
                daily_drawdown_override=eff_dd,
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
                )

            order_side = "BUY" if decision.action == "BUY" else "SELL"
            eff_risk_rate = min(
                float(creds.risk_per_trade_pct),
                float(self.settings.max_account_risk_per_trade),
            )
            trade_budget = user_equity * eff_risk_rate
            daily_headroom = (
                max(0.0, start_eq * self.settings.max_daily_drawdown_limit - max(0.0, start_eq - curr_eq))
                if start_eq > 0
                else trade_budget
            )
            max_risk_budget = min(trade_budget, daily_headroom)
            qty = user_sizing.position_units
            candidate_lots = (
                round(max(0.01, min(float(custom_lot), 50.0)), 2)
                if (custom_lot is not None and custom_lot > 0)
                else calculate_mt5_lots(clean_sym, qty, decision.entry_price)
            )
            eff_stop_loss = decision.stop_loss
            loss_at_stop = abs(
                estimate_mt5_pnl_usdt(
                    clean_sym, order_side, decision.entry_price, eff_stop_loss, candidate_lots
                )
            )
            if loss_at_stop > max_risk_budget + 1e-6:
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
                    )

            order_res = await self.mt5_client.place_market_order(
                account_id_or_login=creds.api_key,
                token_or_password=creds.api_secret,
                symbol=clean_sym,
                side=order_side,
                quantity=qty,
                reference_price=decision.entry_price,
                stop_loss=eff_stop_loss,
                take_profit=decision.take_profit_1,
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
                        take_profit_1=decision.take_profit_1,
                        take_profit_2=decision.take_profit_2,
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
                take_profit_2=anchored_tp2,
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
        )
