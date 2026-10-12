from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
import time
from typing import Any, Callable, Coroutine, Literal, Optional

from ai_engine.auto_trader.user_vault import UserPositionRecord

DURATION_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(s|sec|detik|m|min|menit|h|hr|jam|hour|hours|d|day|days|hari)$",
    re.IGNORECASE,
)

LOT_TOKEN_RE = re.compile(
    r"^(?:lot[=:]?)?(\d+(?:\.\d+)?)(?:lots?|l)?$",
    re.IGNORECASE,
)


def is_session_duration_token(token: str) -> bool:
    """
    Return True if `token` represents a timed auto-trade session duration
    (e.g., '1m', '5m', '30M', '5H', '2h', '5jam', '30menit', '1D', '45s').
    """
    cleaned = token.strip()
    if not cleaned:
        return False
    m = DURATION_RE.match(cleaned)
    if not m:
        return False
    return float(m.group(1)) > 0


def parse_custom_lot_token(token: str) -> Optional[float]:
    """
    Parse a custom lot size token (e.g. '0.01', '0.05', '0.1', '1.5', '0.05lot', 'lot=0.05').
    Returns None if token is a duration or non-lot string.
    """
    cleaned = token.strip()
    if not cleaned or is_session_duration_token(cleaned):
        return None
    m = LOT_TOKEN_RE.match(cleaned)
    if not m:
        return None
    val = float(m.group(1))
    if 0.001 <= val <= 50.0:
        return round(max(0.01, val), 2)
    return None


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
    custom_lot: Optional[float] = None
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
