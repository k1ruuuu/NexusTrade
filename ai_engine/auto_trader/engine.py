from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ai_engine.auto_trader.binance_client import BinanceOrderResult, BinanceSpotClient
from ai_engine.auto_trader.user_vault import (
    UserBinanceCredentials,
    UserPositionRecord,
    UserVault,
)
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent, SignalDecision
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.market_symbols import is_crypto_symbol, normalize_symbol, normalize_timeframe

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
        binance_client: Optional[BinanceSpotClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ensemble_agent = ensemble_agent or EnsembleDecisionAgent(self.settings)
        self.vault = user_vault or UserVault(self.settings)
        self.binance_client = binance_client or BinanceSpotClient()

    async def evaluate_and_execute(
        self,
        user_id: int,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        decision_override: Optional[SignalDecision] = None,
        require_autotrade_toggle: bool = True,
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
                    name="Verifikasi Akun Binance (/setup)",
                    passed=False,
                    detail="Akun Binance pribadi belum diatur. Wajib jalankan /setup terlebih dahulu.",
                )
            )
            return AutoTradeExecutionReport(
                user_id=int(user_id),
                symbol=clean_sym,
                timeframe=clean_tf,
                setup_required=True,
                executed=False,
                gates=gates,
                rejection_summary="Wajib mengatur akun Binance pribadi melalui /setup sebelum memakai fitur Auto-Trade.",
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

        is_crypto, binance_pair = is_crypto_symbol(clean_sym)
        if not is_crypto:
            gates.append(
                AutoTradeGateCheck(
                    gate_number=1,
                    name="Kompatibilitas Simbol Binance",
                    passed=False,
                    detail=f"DITOLAK: {clean_sym} bukan aset Crypto yang diperdagangkan di Binance Spot.",
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
                rejection_summary=f"Simbol {clean_sym} bukan pasangan Crypto Binance Spot.",
            )

        if creds.verified_usdt_balance < 10.0:
            gates.append(
                AutoTradeGateCheck(
                    gate_number=1,
                    name="Kecukupan Saldo USDT",
                    passed=False,
                    detail=f"DITOLAK: Saldo USDT (${creds.verified_usdt_balance:,.2f}) kurang dari batas minimum $10.00 USDT.",
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
                rejection_summary=f"Saldo USDT (${creds.verified_usdt_balance:,.2f}) di bawah batas minimum $10.00 USDT.",
            )
        mode_str = "TESTNET" if creds.testnet else "LIVE"
        src_str = "Setup Pribadi" if creds.source == "user_setup" else "Admin (.env)"
        gates.append(
            AutoTradeGateCheck(
                gate_number=1,
                name="Verifikasi Akun Binance (/setup)",
                passed=True,
                detail=f"Terhubung ({mode_str} • {src_str} • Key {creds.masked_api_key} • Saldo ${creds.verified_usdt_balance:,.2f})",
            )
        )

        # Run full multimodal AI analysis (Live News + 5Y/Live OHLCV + S/R + Volatility)
        decision = decision_override or self.ensemble_agent.analyze_symbol(
            symbol=clean_sym, timeframe=clean_tf
        )

        # Gate 2: Breaking News & FinBERT Sentiment Alignment
        n_news = len(decision.headlines)
        if decision.action == "BUY":
            news_ok = decision.s_sent >= -0.05
            news_detail = (
                f"Sentimen Berita Selaras ({decision.s_sent:+.2f} dari {n_news} berita mendukung/netral untuk BELI)"
                if news_ok
                else f"DITOLAK: Berita sedang negatif ({decision.s_sent:+.2f}), berbahaya untuk membuka posisi BELI"
            )
        elif decision.action == "SELL":
            news_ok = decision.s_sent <= 0.05
            news_detail = (
                f"Sentimen Berita Selaras ({decision.s_sent:+.2f} dari {n_news} berita mendukung/netral untuk JUAL)"
                if news_ok
                else f"DITOLAK: Berita sedang positif ({decision.s_sent:+.2f}), berbahaya untuk membuka posisi JUAL"
            )
        else:
            news_ok = False
            news_detail = f"Sentimen berita {decision.s_sent:+.2f} ({n_news} berita), namun belum ada pemicu arah kuat"

        gates.append(
            AutoTradeGateCheck(
                gate_number=2,
                name="Baca Berita & Sentimen Pasar",
                passed=news_ok,
                detail=news_detail,
            )
        )

        # Gate 3: 5-Year + Live Technical Conviction
        min_conf = self.settings.signal_confidence_threshold
        tech_aligned = (
            (decision.action == "BUY" and decision.s_tech > 0.15)
            or (decision.action == "SELL" and decision.s_tech < -0.15)
        )
        conf_ok = decision.action in ("BUY", "SELL") and decision.confidence >= min_conf and tech_aligned
        if conf_ok:
            tech_detail = (
                f"Sinyal {decision.badge} Kuat (Keyakinan {decision.confidence * 100:.1f}% >= {min_conf * 100:.0f}%, "
                f"Skor Teknikal {decision.s_tech:+.2f})"
            )
        else:
            tech_detail = (
                f"DITOLAK: Keyakinan AI baru {decision.confidence * 100:.1f}% (Syarat minimal {min_conf * 100:.0f}% "
                f"dengan arah teknikal searah, saat ini {decision.action} / {decision.s_tech:+.2f})"
            )
        gates.append(
            AutoTradeGateCheck(
                gate_number=3,
                name="Keyakinan Analisis Teknikal 5Thn+Live",
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

        # Gate 5: Volatility, Drawdown Circuit Breaker & Anti-Duplicate Position Guard
        open_positions = self.vault.get_open_positions(user_id, symbol=clean_sym)
        has_duplicate = len(open_positions) > 0
        risk_ok = (
            decision.approved
            and decision.volatility_regime != "EXTREME"
            and not has_duplicate
        )
        if has_duplicate:
            risk_detail = f"DITOLAK: Masih ada {len(open_positions)} posisi {clean_sym} yang sedang OPEN di akun Anda"
        elif not decision.approved or decision.volatility_regime == "EXTREME":
            risk_detail = f"DITOLAK: {decision.rejection_reason or 'Volatilitas pasar terlalu ekstrem'}"
        else:
            risk_detail = f"Aman (Gejolak {decision.volatility_regime}, S_vol={decision.s_vol:.2f}, Tidak ada posisi ganda)"

        gates.append(
            AutoTradeGateCheck(
                gate_number=5,
                name="Filter Gejolak & Anti-Posisi Ganda",
                passed=risk_ok,
                detail=risk_detail,
            )
        )

        # Check if all pre-execution gates (1..5) passed
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

        # Gate 6: Calculate user-specific position size & execute on user's Binance account
        user_equity = float(creds.verified_usdt_balance)
        user_sizing = self.ensemble_agent.position_sizer.calculate_position(
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            confidence=decision.confidence,
            s_vol=decision.s_vol,
            portfolio_equity=user_equity,
            reward_risk_ratio=self.settings.min_risk_reward_ratio,
        )
        risk_scale = creds.risk_per_trade_pct / max(self.settings.max_account_risk_per_trade, 1e-6)
        max_notional_by_user_risk = user_sizing.position_notional * min(1.0, risk_scale)
        target_notional = min(user_sizing.position_notional, user_equity * 0.95, max_notional_by_user_risk)
        if target_notional < 10.0:
            target_notional = min(user_equity * 0.95, 10.0)
        qty = target_notional / max(decision.entry_price, 1e-9)

        order_side = "BUY" if decision.action == "BUY" else "SELL"
        order_res = await self.binance_client.place_market_order(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            symbol=clean_sym,
            side=order_side,
            quantity=qty,
            reference_price=decision.entry_price,
            testnet=creds.testnet,
        )

        if not order_res.executed:
            gates.append(
                AutoTradeGateCheck(
                    gate_number=6,
                    name="Eksekusi Order ke Binance",
                    passed=False,
                    detail=order_res.error_message or "Order gagal dieksekusi oleh Binance.",
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
                rejection_summary=order_res.error_message or "Eksekusi order ditolak oleh Binance.",
            )

        pos_record = self.vault.record_open_position(
            user_id=int(user_id),
            symbol=clean_sym,
            side=order_side,
            entry_price=order_res.avg_price,
            quantity=order_res.executed_qty,
            notional_usdt=order_res.notional_usdt,
            stop_loss=decision.stop_loss,
            take_profit_1=decision.take_profit_1,
            take_profit_2=decision.take_profit_2,
            confidence=decision.confidence,
            order_id=order_res.order_id,
            testnet=creds.testnet,
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
