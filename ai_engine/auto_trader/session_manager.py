from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
import time
from typing import Any, Callable, Coroutine, Literal, Optional

from ai_engine.auto_trader.user_vault import UserPositionRecord
from config.logging_config import get_logger
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe

logger = get_logger("nexustrade.auto_trader.session")

DURATION_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(s|sec|detik|m|min|menit|h|hr|jam|hour|hours|d|day|days|hari)$",
    re.IGNORECASE,
)


def is_session_duration_token(token: str) -> bool:
    """
    Return True if `token` represents a timed auto-trade session duration
    (e.g., '5H', '2h', '30M', '5jam', '45menit', '1D', '2hari').
    Distinguishes session durations from lowercase candle intervals ('1m', '5m', '15m')
    unless uppercase ('30M', '5H') or explicit Indonesian unit ('jam', 'menit', 'hari') is used.
    """
    cleaned = token.strip()
    if not cleaned:
        return False
    m = DURATION_RE.match(cleaned)
    if not m:
        return False
    val = float(m.group(1))
    unit = m.group(2)
    # Standard lowercase candle intervals ('1m', '5m', '15m', '30m') when <= 30m:
    # treat '5H', '1H', '2H', '5jam', '30menit', '60m', '2h', '4h' in /autotrade <SYM> <DUR> as session duration
    if unit.lower() in ("h", "hr", "jam", "hour", "hours", "d", "day", "days", "hari", "s", "sec", "detik", "menit", "min"):
        return val > 0
    if unit == "M" or val >= 10:
        return val > 0
    return False


def parse_session_duration(raw: str, default_hours: float = 5.0) -> tuple[int, str]:
    """
    Parse a timed auto-trade duration string (e.g. '5H', '2jam', '30M', '1D')
    into `(duration_seconds, human_label)`.
    """
    cleaned = raw.strip()
    m = DURATION_RE.match(cleaned)
    if not m:
        if cleaned.isdigit():
            hrs = max(1, int(cleaned))
            return hrs * 3600, f"{hrs} Jam"
        secs = int(default_hours * 3600)
        return secs, f"{int(default_hours)} Jam"

    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("s", "sec", "detik"):
        secs = max(5, int(val))
        return secs, f"{secs} Detik"
    if unit in ("m", "min", "menit"):
        mins = max(1, int(val))
        return mins * 60, f"{mins} Menit"
    if unit in ("d", "day", "days", "hari"):
        days = max(1, int(val))
        return days * 86400, f"{days} Hari ({days * 24} Jam)"

    # Hours ('h', 'hr', 'jam', 'hour', 'hours')
    if val.is_integer():
        hrs = max(1, int(val))
        return hrs * 3600, f"{hrs} Jam"
    secs = max(60, int(val * 3600))
    return secs, f"{val:.1f} Jam"


@dataclass
class PositionMonitorResult:
    """Result of monitoring & holding an open position during an active Auto-Trade session."""

    action_taken: Literal["NONE", "HOLDING", "TRAILING_UPDATED", "CLOSED_TP", "CLOSED_SL"]
    position: Optional[UserPositionRecord] = None
    current_price: float = 0.0
    unrealized_pnl_usdt: float = 0.0
    unrealized_pnl_pct: float = 0.0
    new_stop_loss: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl_usdt: float = 0.0


@dataclass
class AutoTradeSessionState:
    """State of a timed autonomous trading session (e.g. `/autotrade XAUUSD 5H`)."""

    session_id: str
    user_id: int
    symbol: str
    timeframe: str
    duration_seconds: int
    duration_label: str
    started_at_ts: float
    ends_at_ts: float
    started_at_iso: str
    ends_at_iso: str
    status: Literal["RUNNING", "COMPLETED", "STOPPED"] = "RUNNING"
    cycles_completed: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    realized_pnl_usdt: float = 0.0
    last_check_summary: str = "Memulai sesi pemantauan..."
    task: Optional[asyncio.Task[Any]] = field(default=None, repr=False)

    @property
    def remaining_seconds(self) -> int:
        return max(0, int(self.ends_at_ts - time.time()))

    @property
    def remaining_human(self) -> str:
        rem = self.remaining_seconds
        if rem <= 0 or self.status != "RUNNING":
            return "Selesai (0 menit)"
        hrs = rem // 3600
        mins = (rem % 3600) // 60
        secs = rem % 60
        if hrs > 0:
            return f"{hrs} jam {mins} menit"
        if mins > 0:
            return f"{mins} menit {secs} detik"
        return f"{secs} detik"


NotifyCallback = Callable[[int, str], Coroutine[Any, Any, None]]


class AutoTradeSessionManager:
    """
    Manages timed autonomous trading sessions (e.g., `/autotrade XAUUSD 5H`):
    - Periodically scans market & news while session timer is active.
    - Holds open positions to maximize profit toward TP1 (1:2 RRR) / TP2 (1:3 RRR)
      while ratcheting dynamic Trailing Stop-Loss (never closing immediately for tiny profit).
    - Automatically closes position on broker & vault when TP or Trailing Stop is hit.
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

    async def monitor_and_manage_open_position(
        self,
        user_id: int,
        symbol: str,
        timeframe: str = "15m",
        price_override: Optional[float] = None,
        high_override: Optional[float] = None,
        low_override: Optional[float] = None,
    ) -> PositionMonitorResult:
        """
        Inspect active OPEN position for `(user_id, symbol)`:
        - Does NOT close prematurely on small fluctuations.
        - Ratchets Trailing Stop into profit once price moves >= +1.0 * ATR in favor.
        - Closes position on broker & vault ONLY when TP1/TP2 or Stop-Loss/Trailing-Stop is hit.
        """
        if hasattr(self.engine, "sync_live_mt5_positions"):
            await self.engine.sync_live_mt5_positions(user_id)
        clean_sym = normalize_symbol(symbol)
        open_positions = self.engine.vault.get_open_positions(user_id, symbol=clean_sym)
        if not open_positions:
            return PositionMonitorResult(action_taken="NONE")

        pos = open_positions[0]
        df = self.engine.ensemble_agent.load_market_series(clean_sym, timeframe=timeframe, limit=30)
        if df is not None and len(df) > 0:
            last_row = df.iloc[-1]
            curr_price = float(price_override if price_override is not None else last_row["close"])
            bar_high = float(high_override if high_override is not None else max(curr_price, float(last_row["high"])))
            bar_low = float(low_override if low_override is not None else min(curr_price, float(last_row["low"])))
        else:
            curr_price = float(price_override or pos.entry_price)
            bar_high = float(high_override or curr_price)
            bar_low = float(low_override or curr_price)

        initial_stop_dist = max(abs(pos.entry_price - pos.stop_loss), pos.entry_price * 0.005)
        est_atr = initial_stop_dist / max(self.engine.settings.atr_stop_loss_multiplier, 1.0)

        if pos.side == "BUY":
            unr_pnl = (curr_price - pos.entry_price) * pos.quantity
            unr_pct = ((curr_price - pos.entry_price) / max(pos.entry_price, 1e-9)) * 100.0

            # 1. Check Take-Profit 2 (1:3) or Take-Profit 1 (1:2)
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

            # 2. Check Stop-Loss / Ratcheted Trailing Stop
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

            # 3. Hold position & ratchet Trailing Stop once move >= +1.0 * ATR in favor
            candidate_sl = pos.stop_loss
            if bar_high >= pos.entry_price + 1.0 * est_atr:
                be_plus = round(pos.entry_price + 0.35 * est_atr, 4)
                candidate_sl = max(candidate_sl, be_plus)
                trailed = self.engine.ensemble_agent.stop_loss_manager.update_trailing_stop(
                    "BUY", bar_high, candidate_sl, initial_stop_dist
                )
                candidate_sl = max(candidate_sl, trailed)

            if candidate_sl > pos.stop_loss + 1e-6:
                self.engine.vault.update_position_stop_loss(pos.position_id, candidate_sl)
                pos.stop_loss = candidate_sl
                return PositionMonitorResult(
                    action_taken="TRAILING_UPDATED",
                    position=pos,
                    current_price=curr_price,
                    unrealized_pnl_usdt=round(unr_pnl, 2),
                    unrealized_pnl_pct=round(unr_pct, 2),
                    new_stop_loss=candidate_sl,
                )

            return PositionMonitorResult(
                action_taken="HOLDING",
                position=pos,
                current_price=curr_price,
                unrealized_pnl_usdt=round(unr_pnl, 2),
                unrealized_pnl_pct=round(unr_pct, 2),
            )

        # SELL (Short) position management
        unr_pnl = (pos.entry_price - curr_price) * pos.quantity
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

        candidate_sl = pos.stop_loss
        if bar_low <= pos.entry_price - 1.0 * est_atr:
            be_plus = round(pos.entry_price - 0.35 * est_atr, 4)
            candidate_sl = min(candidate_sl, be_plus)
            trailed = self.engine.ensemble_agent.stop_loss_manager.update_trailing_stop(
                "SELL", bar_low, candidate_sl, initial_stop_dist
            )
            candidate_sl = min(candidate_sl, trailed)

        if candidate_sl < pos.stop_loss - 1e-6:
            self.engine.vault.update_position_stop_loss(pos.position_id, candidate_sl)
            pos.stop_loss = candidate_sl
            return PositionMonitorResult(
                action_taken="TRAILING_UPDATED",
                position=pos,
                current_price=curr_price,
                unrealized_pnl_usdt=round(unr_pnl, 2),
                unrealized_pnl_pct=round(unr_pct, 2),
                new_stop_loss=candidate_sl,
            )

        return PositionMonitorResult(
            action_taken="HOLDING",
            position=pos,
            current_price=curr_price,
            unrealized_pnl_usdt=round(unr_pnl, 2),
            unrealized_pnl_pct=round(unr_pct, 2),
        )

    async def _close_position_on_broker(
        self,
        pos: UserPositionRecord,
        exit_price: float,
        action_type: Literal["CLOSED_TP", "CLOSED_SL"],
        exit_reason: str,
    ) -> PositionMonitorResult:
        """Execute closing counter-order on broker and mark position CLOSED in UserVault."""
        raw_pnl = (
            (exit_price - pos.entry_price) * pos.quantity
            if pos.side == "BUY"
            else (pos.entry_price - exit_price) * pos.quantity
        )
        fee = (pos.entry_price + exit_price) * pos.quantity * 0.0005
        net_pnl = round(raw_pnl - fee, 2)
        pnl_pct = round((net_pnl / max(pos.notional_usdt, 1e-6)) * 100.0, 2)

        creds = self.engine.vault.resolve_credentials(pos.user_id)
        close_side: Literal["BUY", "SELL"] = "SELL" if pos.side == "BUY" else "BUY"
        if creds is not None:
            try:
                if creds.provider == "exness_mt5":
                    await self.engine.mt5_client.close_mt5_ticket(
                        account_id_or_login=creds.api_key,
                        token_or_password=creds.api_secret,
                        ticket=pos.order_id,
                        lots=pos.quantity,
                        server_name=creds.server_name,
                    )
                else:
                    await self.engine.binance_client.place_market_order(
                        api_key=creds.api_key,
                        api_secret=creds.api_secret,
                        symbol=pos.symbol,
                        side=close_side,
                        quantity=pos.quantity,
                        reference_price=exit_price,
                        testnet=creds.testnet,
                    )
            except Exception as exc:
                logger.warning("Closing order on broker encountered error", error=str(exc))

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

    async def start_timed_session(
        self,
        user_id: int,
        symbol: str,
        duration_code: str = "5H",
        timeframe: str = "15m",
        poll_interval_seconds: float = 30.0,
        notify_callback: Optional[NotifyCallback] = None,
    ) -> tuple[AutoTradeSessionState, Any]:
        """
        Start a timed autonomous trading session for `(user_id, symbol)` running for `duration_code` (e.g. '5H'):
        1. Enables `autotrade_enabled=True` for `user_id`.
        2. Runs the initial cycle immediately (either managing an existing open position or running the 6-Gate check).
        3. Spawns a background `asyncio.Task` that continues monitoring & trading until `ends_at_ts`.
        """
        uid = int(user_id)
        clean_sym = normalize_symbol(symbol)
        clean_tf = normalize_timeframe(timeframe)
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
            started_at_iso=now_dt.strftime("%Y-%m-%d %H:%M UTC"),
            ends_at_iso=end_dt.strftime("%Y-%m-%d %H:%M UTC"),
        )
        self._sessions[(uid, clean_sym)] = sess

        # Execute immediate first cycle
        first_report = await self._run_single_session_cycle(sess, notify_callback=None)

        # Spawn background loop for the remainder of the session window
        sess.task = asyncio.create_task(
            self._session_loop(sess, poll_interval_seconds, notify_callback)
        )
        return sess, first_report

    async def _run_single_session_cycle(
        self,
        sess: AutoTradeSessionState,
        notify_callback: Optional[NotifyCallback] = None,
    ) -> Any:
        sess.cycles_completed += 1
        open_positions = self.engine.vault.get_open_positions(sess.user_id, symbol=sess.symbol)

        if open_positions:
            mon = await self.monitor_and_manage_open_position(
                user_id=sess.user_id,
                symbol=sess.symbol,
                timeframe=sess.timeframe,
            )
            if mon.action_taken in ("CLOSED_TP", "CLOSED_SL") and mon.position is not None:
                sess.positions_closed += 1
                sess.realized_pnl_usdt = round(sess.realized_pnl_usdt + mon.realized_pnl_usdt, 2)
                sess.last_check_summary = (
                    f"Posisi {mon.position.side} ditutup @ {mon.current_price:,.2f} "
                    f"({mon.exit_reason} | PnL: ${mon.realized_pnl_usdt:+.2f})"
                )
                if notify_callback is not None:
                    icon = "🎯" if mon.action_taken == "CLOSED_TP" else "🛡️"
                    await notify_callback(
                        sess.user_id,
                        f"{icon} <b>AUTO-TRADE MENUTUP POSISI — {sess.symbol}</b>\n"
                        f"• <b>Alasan:</b> {mon.exit_reason}\n"
                        f"• <b>Harga Keluar:</b> <code>${mon.current_price:,.2f}</code>\n"
                        f"• <b>Hasil PnL Bersih:</b> <code>${mon.realized_pnl_usdt:+.2f} USDT</code>\n"
                        f"• <b>Sisa Waktu Sesi:</b> <code>{sess.remaining_human}</code>",
                    )
            elif mon.action_taken == "TRAILING_UPDATED" and mon.position is not None:
                sess.last_check_summary = (
                    f"Mengawal posisi {mon.position.side}: Trailing Stop digeser ke ${mon.new_stop_loss:,.2f} "
                    f"(Floating PnL: ${mon.unrealized_pnl_usdt:+.2f})"
                )
            elif mon.position is not None:
                sess.last_check_summary = (
                    f"Menahan posisi {mon.position.side} menuju Target Profit "
                    f"(Floating PnL: ${mon.unrealized_pnl_usdt:+.2f} / {mon.unrealized_pnl_pct:+.2f}%)"
                )
            return mon

        # No open position -> run 6-Gate evaluation to see if an accurate entry exists now
        report = await self.engine.evaluate_and_execute(
            user_id=sess.user_id,
            symbol=sess.symbol,
            timeframe=sess.timeframe,
            require_autotrade_toggle=True,
        )
        if report.executed and report.order_result is not None:
            sess.positions_opened += 1
            sess.last_check_summary = (
                f"Membuka posisi {report.order_result.side} @ ${report.order_result.avg_price:,.2f} "
                f"(Order ID: {report.order_result.order_id})"
            )
            if notify_callback is not None:
                await notify_callback(
                    sess.user_id,
                    f"🚀 <b>SESI AUTO-TRADE MEMBUKA POSISI BARU — {sess.symbol}</b>\n"
                    f"• <b>Aksi:</b> <b>{report.order_result.side} {report.order_result.symbol}</b>\n"
                    f"• <b>Harga Masuk:</b> <code>${report.order_result.avg_price:,.2f}</code>\n"
                    f"• <b>Stop Loss:</b> <code>${report.position_record.stop_loss:,.2f}</code> | "
                    f"<b>TP1:</b> <code>${report.position_record.take_profit_1:,.2f}</code>\n"
                    f"• <b>Sisa Waktu Sesi:</b> <code>{sess.remaining_human}</code>",
                )
        else:
            sess.last_check_summary = report.rejection_summary or "Menunggu konfirmasi 6 Gerbang..."
        return report

    async def _session_loop(
        self,
        sess: AutoTradeSessionState,
        poll_interval_seconds: float,
        notify_callback: Optional[NotifyCallback] = None,
    ) -> None:
        try:
            while sess.status == "RUNNING" and time.time() < sess.ends_at_ts:
                sleep_s = min(poll_interval_seconds, max(1.0, sess.ends_at_ts - time.time()))
                await asyncio.sleep(sleep_s)
                if sess.status != "RUNNING" or time.time() >= sess.ends_at_ts:
                    break
                await self._run_single_session_cycle(sess, notify_callback=notify_callback)

            if sess.status == "RUNNING":
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
