from __future__ import annotations

from typing import Any, Literal, Optional

from ai_engine.auto_trader.mt5_exness_client import (
    anchor_targets_to_fill_price,
    estimate_mt5_pnl_usdt,
)
from ai_engine.auto_trader.session_models import PositionMonitorResult
from ai_engine.auto_trader.user_vault import UserPositionRecord
from config.logging_config import get_logger
from data_pipeline.market_symbols import normalize_symbol

logger = get_logger("nexustrade.auto_trader.position_monitor")


def format_position_closed_notification(
    mon: PositionMonitorResult,
    header_title: str,
    pnl_label: str = "Hasil PnL Bersih",
    remaining_human: Optional[str] = None,
) -> str:
    """
    Shared HTML formatter for position closure & Emergency News Auto-Reverse notifications
    used by both timed session loops and manual `/autotrade <SYMBOL>` checks.
    """
    icon = (
        "🔄"
        if mon.re_entry_position is not None
        else ("🎯" if mon.action_taken == "CLOSED_TP" else "🛡️")
    )
    re_line = (
        f"\n• <b>Posisi Baru (Sesuai Berita):</b> <b>{mon.re_entry_position.side} {mon.re_entry_position.symbol}</b> "
        f"@ <code>${mon.re_entry_position.entry_price:,.2f}</code> "
        f"(SL: <code>${mon.re_entry_position.stop_loss:,.2f}</code> | TP1: <code>${mon.re_entry_position.take_profit_1:,.2f}</code>)"
        if mon.re_entry_position is not None
        else ""
    )
    rem_line = (
        f"\n• <b>Sisa Waktu Sesi:</b> <code>{remaining_human}</code>"
        if remaining_human is not None
        else ""
    )
    return (
        f"{icon} <b>{header_title}</b>\n"
        f"• <b>Alasan:</b> {mon.exit_reason}\n"
        f"• <b>Harga Keluar:</b> <code>${mon.current_price:,.2f}</code>\n"
        f"• <b>{pnl_label}:</b> <code>${mon.realized_pnl_usdt:+.2f} USDT</code>"
        f"{re_line}"
        f"{rem_line}"
    )


class PositionMonitorMixin:
    """
    Manages open position monitoring, dynamic Trailing Stop ratcheting,
    and In-Position Emergency Breaking News Exit & Auto-Reverse on Exness MT5.
    """

    engine: Any

    async def monitor_and_manage_open_position(
        self,
        user_id: int,
        symbol: str,
        timeframe: str = "15m",
        price_override: Optional[float] = None,
        high_override: Optional[float] = None,
        low_override: Optional[float] = None,
        headlines_override: Optional[list[str]] = None,
    ) -> PositionMonitorResult:
        """
        Inspect active OPEN position for `(user_id, symbol)`:
        - Does NOT close prematurely on small fluctuations.
        - Checks for sudden adverse Breaking News while holding position (Emergency Exit & Auto-Reverse
          on strong opposing news |s_sent| >= 0.30, or automatic Stop-Loss tightening on moderate news).
        - Ratchets Trailing Stop into profit once price moves >= +0.60 * ATR in favor.
        - Closes position on broker & vault when TP1/TP2, Stop-Loss/Trailing-Stop, or Emergency News Exit triggers.
        """
        live_ticket_map: dict[str, dict[str, Any]] = {}
        if hasattr(self.engine, "sync_live_mt5_positions"):
            sync_res = await self.engine.sync_live_mt5_positions(user_id)
            if isinstance(sync_res, dict):
                live_ticket_map = sync_res
        clean_sym = normalize_symbol(symbol)
        open_positions = self.engine.vault.get_open_positions(user_id, symbol=clean_sym)
        if not open_positions:
            return PositionMonitorResult(action_taken="NONE")

        pos = open_positions[0]
        live_ord = live_ticket_map.get(pos.order_id)
        live_close_px = float(live_ord.get("closePrice") or 0.0) if isinstance(live_ord, dict) else 0.0

        if price_override is None and live_close_px > 0:
            curr_price = live_close_px
            bar_high = float(high_override if high_override is not None else curr_price)
            bar_low = float(low_override if low_override is not None else curr_price)
        else:
            df = self.engine.ensemble_agent.load_market_series(clean_sym, timeframe=timeframe, limit=30)
            if df is not None and len(df) > 0:
                last_row = df.iloc[-1]
                curr_price = float(price_override if price_override is not None else last_row["close"])
                bar_high = float(
                    high_override if high_override is not None else max(curr_price, float(last_row["high"]))
                )
                bar_low = float(
                    low_override if low_override is not None else min(curr_price, float(last_row["low"]))
                )
            else:
                curr_price = float(price_override or pos.entry_price)
                bar_high = float(high_override or curr_price)
                bar_low = float(low_override or curr_price)

        initial_stop_dist = max(abs(pos.entry_price - pos.stop_loss), pos.entry_price * 0.002)
        est_atr = initial_stop_dist / 0.85
        trail_dist = max(0.65 * est_atr, initial_stop_dist * 0.75)

        if (
            price_override is None
            and isinstance(live_ord, dict)
            and live_ord.get("profit") is not None
        ):
            unr_pnl = float(live_ord.get("profit") or 0.0)
        else:
            unr_pnl = estimate_mt5_pnl_usdt(
                pos.symbol, pos.side, pos.entry_price, curr_price, pos.quantity
            )

        news_s_sent = self._evaluate_live_position_news_sentiment(clean_sym, headlines_override)

        if pos.side == "BUY":
            unr_pct = ((curr_price - pos.entry_price) / max(pos.entry_price, 1e-9)) * 100.0
            if pos.take_profit_2 > 0 and bar_high >= pos.take_profit_2:
                exit_px = max(curr_price, pos.take_profit_2)
                return await self._close_position_on_broker(
                    pos, exit_px, "CLOSED_TP", f"Target Profit 2 (TP2 @ {pos.take_profit_2:,.2f}) Tercapai"
                )
            if pos.take_profit_1 > 0 and bar_high >= pos.take_profit_1:
                exit_px = max(curr_price, pos.take_profit_1)
                return await self._close_position_on_broker(
                    pos, exit_px, "CLOSED_TP", f"Target Profit 1 (TP1 @ {pos.take_profit_1:,.2f}) Tercapai"
                )
            if pos.stop_loss > 0 and bar_low <= pos.stop_loss:
                exit_px = pos.stop_loss
                is_profit_stop = pos.stop_loss > pos.entry_price
                reason = (
                    f"Trailing Stop Profit Terkunci (@ {pos.stop_loss:,.2f})"
                    if is_profit_stop
                    else f"Batas Rugi Stop-Loss (@ {pos.stop_loss:,.2f}) Tersentuh"
                )
                return await self._close_position_on_broker(
                    pos, exit_px, "CLOSED_TP" if is_profit_stop else "CLOSED_SL", reason
                )

            if news_s_sent is not None and news_s_sent <= -0.30:
                return await self._execute_emergency_news_exit_and_reverse(
                    pos=pos,
                    new_side="SELL",
                    curr_price=curr_price,
                    est_atr=est_atr,
                    news_s_sent=news_s_sent,
                    unr_pnl=unr_pnl,
                    catalyst_label="Berita Buruk",
                )

            candidate_sl = pos.stop_loss
            if news_s_sent is not None and -0.30 < news_s_sent <= -0.18:
                if curr_price >= pos.entry_price + 0.30 * est_atr:
                    candidate_sl = max(candidate_sl, round(pos.entry_price, 4))
                else:
                    candidate_sl = max(candidate_sl, round(curr_price - 0.40 * est_atr, 4))
            if bar_high >= pos.entry_price + 0.60 * est_atr:
                be_plus = round(pos.entry_price + 0.20 * est_atr, 4)
                candidate_sl = max(candidate_sl, be_plus)
                trailed = self.engine.ensemble_agent.stop_loss_manager.update_trailing_stop(
                    "BUY", bar_high, candidate_sl, trail_dist
                )
                candidate_sl = max(candidate_sl, trailed)

            if candidate_sl > pos.stop_loss + 1e-6:
                return await self._apply_trailing_stop_update(
                    pos, candidate_sl, curr_price, unr_pnl, unr_pct
                )
            return PositionMonitorResult(
                action_taken="HOLDING",
                position=pos,
                current_price=curr_price,
                unrealized_pnl_usdt=round(unr_pnl, 2),
                unrealized_pnl_pct=round(unr_pct, 2),
            )

        # SELL (Short) position management
        unr_pct = ((pos.entry_price - curr_price) / max(pos.entry_price, 1e-9)) * 100.0
        if pos.take_profit_2 > 0 and bar_low <= pos.take_profit_2:
            exit_px = min(curr_price, pos.take_profit_2)
            return await self._close_position_on_broker(
                pos, exit_px, "CLOSED_TP", f"Target Profit 2 (TP2 @ {pos.take_profit_2:,.2f}) Tercapai"
            )
        if pos.take_profit_1 > 0 and bar_low <= pos.take_profit_1:
            exit_px = min(curr_price, pos.take_profit_1)
            return await self._close_position_on_broker(
                pos, exit_px, "CLOSED_TP", f"Target Profit 1 (TP1 @ {pos.take_profit_1:,.2f}) Tercapai"
            )
        if pos.stop_loss > 0 and bar_high >= pos.stop_loss:
            exit_px = pos.stop_loss
            is_profit_stop = pos.stop_loss < pos.entry_price
            reason = (
                f"Trailing Stop Profit Terkunci (@ {pos.stop_loss:,.2f})"
                if is_profit_stop
                else f"Batas Rugi Stop-Loss (@ {pos.stop_loss:,.2f}) Tersentuh"
            )
            return await self._close_position_on_broker(
                pos, exit_px, "CLOSED_TP" if is_profit_stop else "CLOSED_SL", reason
            )

        if news_s_sent is not None and news_s_sent >= 0.30:
            return await self._execute_emergency_news_exit_and_reverse(
                pos=pos,
                new_side="BUY",
                curr_price=curr_price,
                est_atr=est_atr,
                news_s_sent=news_s_sent,
                unr_pnl=unr_pnl,
                catalyst_label="Berita Positif Kuat",
            )

        candidate_sl = pos.stop_loss
        if news_s_sent is not None and 0.18 <= news_s_sent < 0.30:
            if curr_price <= pos.entry_price - 0.30 * est_atr:
                candidate_sl = min(candidate_sl, round(pos.entry_price, 4))
            else:
                candidate_sl = min(candidate_sl, round(curr_price + 0.40 * est_atr, 4))
        if bar_low <= pos.entry_price - 0.60 * est_atr:
            be_plus = round(pos.entry_price - 0.20 * est_atr, 4)
            candidate_sl = min(candidate_sl, be_plus)
            trailed = self.engine.ensemble_agent.stop_loss_manager.update_trailing_stop(
                "SELL", bar_low, candidate_sl, trail_dist
            )
            candidate_sl = min(candidate_sl, trailed)

        if candidate_sl < pos.stop_loss - 1e-6:
            return await self._apply_trailing_stop_update(
                pos, candidate_sl, curr_price, unr_pnl, unr_pct
            )
        return PositionMonitorResult(
            action_taken="HOLDING",
            position=pos,
            current_price=curr_price,
            unrealized_pnl_usdt=round(unr_pnl, 2),
            unrealized_pnl_pct=round(unr_pct, 2),
        )

    async def _execute_emergency_news_exit_and_reverse(
        self,
        pos: UserPositionRecord,
        new_side: Literal["BUY", "SELL"],
        curr_price: float,
        est_atr: float,
        news_s_sent: float,
        unr_pnl: float,
        catalyst_label: str,
    ) -> PositionMonitorResult:
        """Close `pos` immediately on strong opposing news and reverse into `new_side` if broker close succeeded."""
        is_prof = (
            curr_price >= pos.entry_price
            if pos.side == "BUY"
            else curr_price <= pos.entry_price
        )
        close_res = await self._close_position_on_broker(
            pos,
            curr_price,
            "CLOSED_TP" if is_prof else "CLOSED_SL",
            f"Proteksi Darurat {catalyst_label} (Sentimen {news_s_sent:+.2f}) — Posisi {pos.side} Ditutup Dini",
            broker_pnl_override=unr_pnl,
        )
        if close_res.action_taken in ("CLOSED_TP", "CLOSED_SL"):
            rev_pos = await self._open_news_reversal_position(
                closed_pos=pos,
                new_side=new_side,
                curr_price=curr_price,
                est_atr=est_atr,
                news_s_sent=news_s_sent,
            )
            if rev_pos is not None:
                close_res.re_entry_position = rev_pos
                close_res.exit_reason = (
                    f"Proteksi Darurat {catalyst_label} (Sentimen {news_s_sent:+.2f}) — "
                    f"Posisi {pos.side} Ditutup Dini & Otomatis Balik Arah {new_side} @ ${rev_pos.entry_price:,.2f} "
                    f"(SL: ${rev_pos.stop_loss:,.2f} | TP1: ${rev_pos.take_profit_1:,.2f})"
                )
        return close_res

    async def _apply_trailing_stop_update(
        self,
        pos: UserPositionRecord,
        candidate_sl: float,
        curr_price: float,
        unr_pnl: float,
        unr_pct: float,
    ) -> PositionMonitorResult:
        """Persist ratcheted Trailing Stop in SQLite, sync to Exness MT5, and return TRAILING_UPDATED."""
        self.engine.vault.update_position_stop_loss(pos.position_id, candidate_sl)
        pos.stop_loss = candidate_sl
        await self._sync_trailing_sl_on_broker(pos, candidate_sl)
        return PositionMonitorResult(
            action_taken="TRAILING_UPDATED",
            position=pos,
            current_price=curr_price,
            unrealized_pnl_usdt=round(unr_pnl, 2),
            unrealized_pnl_pct=round(unr_pct, 2),
            new_stop_loss=candidate_sl,
        )

    def _evaluate_live_position_news_sentiment(
        self,
        symbol: str,
        headlines_override: Optional[list[str]] = None,
    ) -> Optional[float]:
        """
        Evaluate live breaking news sentiment while holding an open position.
        Returns `s_sent` in [-1.0, +1.0], or `None` if news evaluation is disabled and no override is given.
        """
        try:
            if headlines_override is None and not getattr(self.engine.settings, "use_live_news_feed", False):
                return None
            sent_res = self.engine.ensemble_agent.sentiment_analyzer.analyze_headlines(
                headlines=headlines_override,
                symbol=symbol,
            )
            if not sent_res.headline_scores:
                return None
            return float(sent_res.s_sent)
        except Exception as exc:
            logger.debug("In-position news sentiment check skipped", symbol=symbol, error=str(exc))
            return None

    async def _sync_trailing_sl_on_broker(
        self,
        pos: UserPositionRecord,
        new_stop_loss: float,
    ) -> None:
        """Push updated Trailing Stop-Loss to Exness MT5 in real time via `/OrderModifySafe`."""
        creds = self.engine.vault.resolve_credentials(pos.user_id)
        if creds is not None:
            await self.engine.mt5_client.modify_mt5_ticket(
                account_id_or_login=creds.api_key,
                token_or_password=creds.api_secret,
                ticket=pos.order_id,
                stop_loss=new_stop_loss,
                take_profit=pos.take_profit_1,
                server_name=creds.server_name,
            )

    async def _close_position_on_broker(
        self,
        pos: UserPositionRecord,
        exit_price: float,
        action_type: Literal["CLOSED_TP", "CLOSED_SL"],
        exit_reason: str,
        broker_pnl_override: Optional[float] = None,
    ) -> PositionMonitorResult:
        """Execute closing order on Exness MT5 (`/OrderCloseSafe`) and mark position CLOSED in UserVault."""
        net_pnl = (
            round(float(broker_pnl_override), 2)
            if broker_pnl_override is not None
            else estimate_mt5_pnl_usdt(
                pos.symbol, pos.side, pos.entry_price, exit_price, pos.quantity
            )
        )
        pnl_pct = round((net_pnl / max(pos.notional_usdt, 1e-6)) * 100.0, 2)

        creds = self.engine.vault.resolve_credentials(pos.user_id)
        close_ok = True
        if creds is not None:
            try:
                close_ok = await self.engine.mt5_client.close_mt5_ticket(
                    account_id_or_login=creds.api_key,
                    token_or_password=creds.api_secret,
                    ticket=pos.order_id,
                    lots=pos.quantity,
                    server_name=creds.server_name,
                )
            except Exception as exc:
                logger.warning("Closing order on broker encountered error", error=str(exc))
                close_ok = False

        if not close_ok:
            return PositionMonitorResult(
                action_taken="HOLDING",
                position=pos,
                current_price=exit_price,
                unrealized_pnl_usdt=net_pnl,
                unrealized_pnl_pct=pnl_pct,
                exit_reason="Gagal menutup posisi di server Exness MT5 (akan dicoba ulang)",
            )

        closed_rec = self.engine.vault.close_position(pos.position_id, exit_price, net_pnl)
        return PositionMonitorResult(
            action_taken=action_type,
            position=closed_rec or pos,
            current_price=exit_price,
            unrealized_pnl_usdt=0.0,
            unrealized_pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            realized_pnl_usdt=net_pnl,
        )

    async def _open_news_reversal_position(
        self,
        closed_pos: UserPositionRecord,
        new_side: Literal["BUY", "SELL"],
        curr_price: float,
        est_atr: float,
        news_s_sent: float,
    ) -> Optional[UserPositionRecord]:
        """
        Immediately open a counter-position (`new_side`) aligned with strong breaking news
        (`s_sent <= -0.30` -> `SELL`, `s_sent >= +0.30` -> `BUY`) right after an Emergency News Exit.
        """
        try:
            creds = self.engine.vault.resolve_credentials(closed_pos.user_id)
            if (
                creds is None
                or not creds.autotrade_enabled
                or creds.verified_usdt_balance < 10.0
            ):
                return None
            eff_atr = max(est_atr, curr_price * 0.002)
            sr_levels = self.engine.ensemble_agent.sr_detector.detect_levels(
                None, current_price=curr_price, atr_14=eff_atr
            )
            targets = self.engine.ensemble_agent.stop_loss_manager.calculate_realistic_targets(
                entry_price=curr_price,
                atr_14=eff_atr,
                action=new_side,
                support_1=sr_levels.support_1,
                support_2=sr_levels.support_2,
                resistance_1=sr_levels.resistance_1,
                resistance_2=sr_levels.resistance_2,
            )
            order_res = await self.engine.mt5_client.place_market_order(
                account_id_or_login=creds.api_key,
                token_or_password=creds.api_secret,
                symbol=closed_pos.symbol,
                side=new_side,
                quantity=closed_pos.quantity,
                reference_price=curr_price,
                stop_loss=targets.stop_loss,
                take_profit=targets.take_profit_1,
                server_name=creds.server_name,
                testnet=creds.testnet,
                custom_lot=closed_pos.quantity,
            )
            if not order_res.executed:
                return None
            anchored_sl, anchored_tp1, anchored_tp2 = anchor_targets_to_fill_price(
                side=new_side,
                fill_price=order_res.avg_price,
                ref_entry=curr_price,
                ref_sl=targets.stop_loss,
                ref_tp1=targets.take_profit_1,
                ref_tp2=targets.take_profit_2,
            )
            conf = round(min(0.95, max(0.75, abs(news_s_sent))), 4)
            return self.engine.vault.record_open_position(
                user_id=closed_pos.user_id,
                symbol=closed_pos.symbol,
                side=new_side,
                entry_price=order_res.avg_price,
                quantity=order_res.executed_qty,
                notional_usdt=order_res.notional_usdt,
                stop_loss=anchored_sl,
                take_profit_1=anchored_tp1,
                take_profit_2=anchored_tp2,
                confidence=conf,
                order_id=order_res.order_id,
                testnet=creds.testnet,
            )
        except Exception as exc:
            logger.warning("Failed to open news reversal position", symbol=closed_pos.symbol, error=str(exc))
            return None
