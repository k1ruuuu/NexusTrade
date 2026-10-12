from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Optional

from ai_engine.auto_trader.mt5_exness_client import resolve_mt5_symbol
from ai_engine.auto_trader.position_monitor import (
    PositionMonitorMixin,
    format_position_closed_notification,
)
from ai_engine.auto_trader.session_models import (
    AutoTradeSessionState,
    NotifyCallback,
    PositionMonitorResult,
    is_session_duration_token,
    parse_custom_lot_token,
    parse_session_duration,
)
from config.logging_config import get_logger
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe

logger = get_logger("nexustrade.auto_trader.session")

__all__ = [
    "AutoTradeSessionManager",
    "AutoTradeSessionState",
    "NotifyCallback",
    "PositionMonitorMixin",
    "PositionMonitorResult",
    "format_position_closed_notification",
    "is_session_duration_token",
    "parse_custom_lot_token",
    "parse_session_duration",
]


class AutoTradeSessionManager(PositionMonitorMixin):
    """
    Manages timed autonomous trading sessions (e.g., `/autotrade XAUUSD 5H`):
    - Periodically scans market & news while session timer is active.
    - Inherits open position management, Trailing Stop, and Emergency News Reversal from PositionMonitorMixin.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._sessions: dict[tuple[int, str], AutoTradeSessionState] = {}

    def get_user_sessions(self, user_id: int) -> list[AutoTradeSessionState]:
        uid = int(user_id)
        return [
            s
            for (u, _), s in self._sessions.items()
            if u == uid and s.status == "RUNNING" and s.remaining_seconds > 0
        ]

    def stop_user_sessions(
        self,
        user_id: int,
        symbol: Optional[str] = None,
    ) -> list[AutoTradeSessionState]:
        uid = int(user_id)
        stopped: list[AutoTradeSessionState] = []
        target_sym = normalize_symbol(symbol) if symbol else None
        for (u, sym), sess in list(self._sessions.items()):
            if u == uid and (target_sym is None or sym == target_sym):
                sess.status = "STOPPED"
                if sess.task is not None and not sess.task.done():
                    sess.task.cancel()
                stopped.append(sess)
                self._sessions.pop((u, sym), None)
        return stopped

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
    ) -> tuple[AutoTradeSessionState, Any]:
        """
        Start a 24/7 Non-Stop autonomous trading session for `(user_id, symbol)`
        analyzing `1D, 1H, 15M, 5M` trends and executing on `1m`:
        1. Enables `autotrade_enabled=True` for `user_id`.
        2. Runs the initial cycle immediately.
        3. Spawns a fast real-time background `asyncio.Task` that updates floating & realized profit
           live on Telegram without delay until `/autotrade off` is called.
        """
        uid = int(user_id)
        clean_sym = normalize_symbol(symbol)
        clean_tf = normalize_timeframe(timeframe) if timeframe != "1m" else "1m"
        dur_secs, dur_label = parse_session_duration(duration_code)

        # Cancel any previous session on the same (user_id, symbol)
        self.stop_user_sessions(uid, symbol=clean_sym)
        self.engine.vault.set_autotrade_enabled(uid, True)

        now_dt = datetime.now(timezone.utc)
        end_dt = now_dt + timedelta(seconds=dur_secs)
        sess = AutoTradeSessionState(
            session_id=f"SESS-{uid}-{clean_sym}-{int(now_dt.timestamp())}",
            user_id=uid,
            symbol=clean_sym,
            timeframe=clean_tf,
            duration_seconds=dur_secs,
            duration_label=dur_label,
            started_at_ts=now_dt.timestamp(),
            ends_at_ts=end_dt.timestamp(),
            started_at_iso=now_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            ends_at_iso=(
                "24/7 Non-Stop (Hingga /autotrade off)"
                if dur_secs >= 30_000_000
                else end_dt.strftime("%Y-%m-%d %H:%M UTC")
            ),
            last_updated_iso=now_dt.strftime("%H:%M:%S UTC"),
            custom_lot=custom_lot,
        )
        self._sessions[(uid, clean_sym)] = sess

        # Execute immediate first cycle
        first_report = await self._run_single_session_cycle(sess, notify_callback=None)

        # Spawn real-time 24/7 loop
        sess.task = asyncio.create_task(
            self._session_loop(sess, poll_interval_seconds, notify_callback, live_card_callback)
        )
        return sess, first_report

    async def _run_single_session_cycle(
        self,
        sess: AutoTradeSessionState,
        notify_callback: Optional[NotifyCallback] = None,
    ) -> Any:
        sess.cycles_completed += 1
        sess.last_updated_iso = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        open_positions = self.engine.vault.get_open_positions(sess.user_id, symbol=sess.symbol)

        if open_positions:
            mon = await self.monitor_and_manage_open_position(
                user_id=sess.user_id,
                symbol=sess.symbol,
                timeframe="1m",
            )
            if mon.action_taken in ("CLOSED_TP", "CLOSED_SL") and mon.position is not None:
                sess.positions_closed += 1
                if mon.realized_pnl_usdt >= 0:
                    sess.winning_trades += 1
                else:
                    sess.losing_trades += 1
                if mon.re_entry_position is not None:
                    sess.positions_opened += 1
                sess.realized_pnl_usdt = round(sess.realized_pnl_usdt + mon.realized_pnl_usdt, 2)
                sess.live_open_position = mon.re_entry_position
                sess.live_current_price = mon.current_price
                sess.live_floating_pnl_usdt = 0.0
                sess.live_floating_pnl_pct = 0.0
                sess.last_check_summary = (
                    f"Posisi {mon.position.side} ditutup @ ${mon.current_price:,.2f} "
                    f"({mon.exit_reason} | PnL: ${mon.realized_pnl_usdt:+.2f})"
                )
                if notify_callback is not None:
                    await notify_callback(
                        sess.user_id,
                        format_position_closed_notification(
                            mon,
                            header_title=f"AUTO-TRADE 24/7 MENUTUP POSISI — {sess.symbol}",
                            pnl_label="Hasil PnL Bersih",
                            remaining_human=sess.remaining_human,
                        ),
                    )
            elif mon.action_taken == "TRAILING_UPDATED" and mon.position is not None:
                sess.live_open_position = mon.position
                sess.live_current_price = mon.current_price
                sess.live_floating_pnl_usdt = mon.unrealized_pnl_usdt
                sess.live_floating_pnl_pct = mon.unrealized_pnl_pct
                sess.last_check_summary = (
                    f"Mengawal posisi {mon.position.side}: Trailing SL digeser ke ${mon.new_stop_loss:,.2f} "
                    f"(Floating Profit: ${mon.unrealized_pnl_usdt:+.2f})"
                )
            elif mon.position is not None:
                sess.live_open_position = mon.position
                sess.live_current_price = mon.current_price
                sess.live_floating_pnl_usdt = mon.unrealized_pnl_usdt
                sess.live_floating_pnl_pct = mon.unrealized_pnl_pct
                sess.last_check_summary = (
                    f"Menahan posisi {mon.position.side} @ ${mon.position.entry_price:,.2f} ➔ "
                    f"Sekarang ${mon.current_price:,.2f} (Floating: ${mon.unrealized_pnl_usdt:+.2f} / {mon.unrealized_pnl_pct:+.2f}%)"
                )
            return mon

        # No open position -> run 6-Gate Multi-Timeframe (1D/1H/15M/5M -> 1m) evaluation
        sess.live_open_position = None
        sess.live_floating_pnl_usdt = 0.0
        sess.live_floating_pnl_pct = 0.0
        report = await self.engine.evaluate_and_execute(
            user_id=sess.user_id,
            symbol=sess.symbol,
            timeframe="1m",
            require_autotrade_toggle=True,
            custom_lot=sess.custom_lot,
        )
        if report.decision is not None:
            sess.live_current_price = report.decision.entry_price
            sess.mtf_status_line = report.decision.mtf_summary

        if report.executed and report.order_result is not None and report.position_record is not None:
            sess.positions_opened += 1
            sess.live_open_position = report.position_record
            sess.live_current_price = report.order_result.avg_price
            sess.last_check_summary = (
                f"Membuka posisi {report.order_result.side} (1m) @ ${report.order_result.avg_price:,.2f} "
                f"(Ticket #{report.order_result.order_id})"
            )
            if notify_callback is not None:
                await notify_callback(
                    sess.user_id,
                    f"🚀 <b>AUTO-TRADE 24/7 MEMBUKA POSISI BARU (1M) — {sess.symbol}</b>\n"
                    f"• <b>Aksi &amp; Lot:</b> <b>{report.order_result.side} {report.order_result.symbol} ({report.order_result.executed_qty} Lot)</b>\n"
                    f"• <b>Harga Masuk (1m):</b> <code>${report.order_result.avg_price:,.2f}</code>\n"
                    f"• <b>Stop Loss:</b> <code>${report.position_record.stop_loss:,.2f}</code> | "
                    f"<b>TP1:</b> <code>${report.position_record.take_profit_1:,.2f}</code>\n"
                    f"• <b>Tren MTF:</b> <i>{sess.mtf_status_line}</i>",
                )
        else:
            sess.last_check_summary = report.rejection_summary or "Memantau sinyal 1D/1H/15M/5M ➔ 1M..."
        return report
    async def close_session_open_positions(
        self,
        sess: AutoTradeSessionState,
        reason: str = "Batas waktu sesi berakhir (Auto-Close @ Market)",
        notify_callback: Optional[NotifyCallback] = None,
    ) -> list[PositionMonitorResult]:
        """
        Automatically close all remaining OPEN positions for `(sess.user_id, sess.symbol)`
        at the current market price when the session timer expires before TP is reached.
        """
        creds = self.engine.vault.resolve_credentials(sess.user_id)
        open_positions = self.engine.vault.get_open_positions(sess.user_id, symbol=sess.symbol)
        tracked_tickets = {p.order_id for p in open_positions}
        live_px_map: dict[str, float] = {}
        live_pnl_map: dict[str, float] = {}

        if (
            creds is not None
            and creds.provider == "exness_mt5"
            and not self.engine.mt5_client._is_exness_trial_demo(creds.api_key, creds.testnet)
        ):
            live_orders = await self.engine.mt5_client.get_live_open_orders(
                creds.api_key, creds.api_secret, creds.server_name
            )
            target_syms = {
                sess.symbol.upper(),
                resolve_mt5_symbol(sess.symbol, "m").upper(),
                resolve_mt5_symbol(sess.symbol, "").upper(),
            }
            if live_orders:
                for ord_item in live_orders:
                    if not isinstance(ord_item, dict):
                        continue
                    ord_sym = str(ord_item.get("symbol", "")).upper()
                    ticket_str = str(ord_item.get("ticket") or ord_item.get("order") or "")
                    lots_val = float(ord_item.get("lots") or 0.01)
                    close_px = float(ord_item.get("closePrice") or ord_item.get("openPrice") or 0.0)
                    if close_px > 0:
                        live_px_map[ticket_str] = close_px
                    if ord_item.get("profit") is not None:
                        live_pnl_map[ticket_str] = float(ord_item.get("profit") or 0.0)
                    # Only close untracked orphan tickets directly here; tracked positions are closed once below
                    if ord_sym in target_syms and ticket_str and ticket_str not in tracked_tickets:
                        await self.engine.mt5_client.close_mt5_ticket(
                            account_id_or_login=creds.api_key,
                            token_or_password=creds.api_secret,
                            ticket=ticket_str,
                            lots=lots_val,
                            server_name=creds.server_name,
                        )

        results: list[PositionMonitorResult] = []
        if not open_positions:
            return results

        df = self.engine.ensemble_agent.load_market_series(
            sess.symbol, timeframe=sess.timeframe, limit=10
        )
        fallback_price = (
            float(df["close"].iloc[-1])
            if (df is not None and len(df) > 0)
            else open_positions[0].entry_price
        )

        for pos in open_positions:
            curr_price = live_px_map.get(pos.order_id, fallback_price)
            is_profit = (
                curr_price >= pos.entry_price
                if pos.side == "BUY"
                else curr_price <= pos.entry_price
            )
            mon = await self._close_position_on_broker(
                pos=pos,
                exit_price=curr_price,
                action_type="CLOSED_TP" if is_profit else "CLOSED_SL",
                exit_reason=reason,
                broker_pnl_override=live_pnl_map.get(pos.order_id),
            )
            if mon.action_taken not in ("CLOSED_TP", "CLOSED_SL"):
                continue
            sess.positions_closed += 1
            sess.realized_pnl_usdt = round(sess.realized_pnl_usdt + mon.realized_pnl_usdt, 2)
            sess.last_check_summary = (
                f"Waktu sesi habis: Posisi {pos.side} ditutup otomatis @ {curr_price:,.2f} "
                f"(PnL: ${mon.realized_pnl_usdt:+.2f})"
            )
            results.append(mon)
            if notify_callback is not None:
                await notify_callback(
                    sess.user_id,
                    f"⏰ <b>WAKTU SESI HABIS — POSISI DITUTUP OTOMATIS ({sess.symbol})</b>\n"
                    f"• <b>Alasan:</b> {reason}\n"
                    f"• <b>Aksi &amp; Harga Masuk:</b> {pos.side} @ <code>${pos.entry_price:,.2f}</code>\n"
                    f"• <b>Harga Penutupan (Market):</b> <code>${curr_price:,.2f}</code>\n"
                    f"• <b>Hasil PnL Bersih:</b> <code>${mon.realized_pnl_usdt:+.2f} USDT</code>",
                )
        return results

    async def _session_loop(
        self,
        sess: AutoTradeSessionState,
        poll_interval_seconds: float,
        notify_callback: Optional[NotifyCallback] = None,
        live_card_callback: Optional[NotifyCallback] = None,
    ) -> None:
        from telegram_bot.views.setup_formatters import format_timed_session_started_card

        last_rendered_card = ""
        try:
            while sess.status == "RUNNING" and time.time() < sess.ends_at_ts:
                sleep_s = min(poll_interval_seconds, max(1.0, sess.ends_at_ts - time.time()))
                await asyncio.sleep(sleep_s)
                if sess.status != "RUNNING" or time.time() >= sess.ends_at_ts:
                    break
                cycle_res = await self._run_single_session_cycle(sess, notify_callback=notify_callback)
                if live_card_callback is not None:
                    try:
                        new_card = format_timed_session_started_card(sess, cycle_res)
                        if new_card != last_rendered_card:
                            last_rendered_card = new_card
                            await live_card_callback(sess.user_id, new_card)
                    except Exception:
                        pass

            if sess.status == "RUNNING" and not sess.is_continuous_24_7:
                await self.close_session_open_positions(
                    sess,
                    reason=f"Batas waktu sesi {sess.duration_label} habis sebelum menyentuh TP (Auto-Close @ Market)",
                    notify_callback=notify_callback,
                )
                sess.status = "COMPLETED"
                self._sessions.pop((sess.user_id, sess.symbol), None)
                if notify_callback is not None:
                    await notify_callback(
                        sess.user_id,
                        f"🏁 <b>SESI AUTO-TRADE {sess.duration_label.upper()} SELESAI — {sess.symbol}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• <b>Siklus Analisis Dijalankan:</b> <code>{sess.cycles_completed}x</code>\n"
                        f"• <b>Posisi Dibuka / Ditutup:</b> <code>{sess.positions_opened} dibuka / {sess.positions_closed} ditutup</code>\n"
                        f"• <b>Total Realized PnL Sesi:</b> <code>${sess.realized_pnl_usdt:+.2f} USDT</code>",
                    )
        except asyncio.CancelledError:
            sess.status = "STOPPED"
            return
        except Exception as exc:
            logger.warning("Timed auto-trade session loop error", session_id=sess.session_id, error=str(exc))
