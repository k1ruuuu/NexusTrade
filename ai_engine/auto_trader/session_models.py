from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
import time
from typing import Any, Callable, Coroutine, Literal, Optional

from ai_engine.auto_trader.user_vault import UserPositionRecord

DURATION_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(s|sec|detik|m|min|menit|h|hr|j|jam|hour|hours|d|day|days|hari)$",
    re.IGNORECASE,
)

LOT_TOKEN_RE = re.compile(
    r"^(?:lot[=:]?)?(\d+(?:\.\d+)?)(?:lots?|l)?$",
    re.IGNORECASE,
)
MAX_POS_RE = re.compile(r"^(?:layer[=:]?|layers[=:]?|pos[=:]?|max[=:]?|x)?(\d+)$", re.IGNORECASE)


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
    Returns None if token is a duration, non-lot string, or an integer layer count (e.g. '4', '5').
    """
    cleaned = token.strip()
    if not cleaned or is_session_duration_token(cleaned):
        return None
    # If it's a bare integer <= 20 without decimal and without explicit 'lot' keyword,
    # reserve it for max_positions (layers count)
    if cleaned.isdigit() and "." not in cleaned and "lot" not in cleaned.lower() and int(cleaned) <= 20:
        return None
    m = LOT_TOKEN_RE.match(cleaned)
    if not m:
        return None
    val = float(m.group(1))
    if 0.001 <= val <= 50.0:
        return round(max(0.01, val), 2)
    return None


def parse_max_positions_token(token: str) -> Optional[int]:
    """
    Parse maximum concurrent positions / layer count token (e.g. '4', '5', 'x4', 'layer=4', 'pos:5').
    Returns int between 1 and 20 if valid, else None.
    """
    cleaned = token.strip()
    if not cleaned or is_session_duration_token(cleaned):
        return None
    if "." in cleaned or "lot" in cleaned.lower():
        return None
    m = MAX_POS_RE.match(cleaned)
    if not m:
        return None
    val = int(m.group(1))
    if 1 <= val <= 20:
        return val
    return None
SUPPORTED_MACRO_EVENTS = (
    "FOMC",      # Fed Interest Rate Decision, Powell Press Conference, Minutes, Dot Plot
    "NFP",       # Non-Farm Payrolls & US Unemployment
    "CPI",       # Consumer Price Index & Core CPI
    "PPI",       # Producer Price Index & Core PPI
    "PCE",       # Core PCE Inflation (Fed's Target Indicator)
    "GDP",       # Gross Domestic Product Growth (QoQ / YoY)
    "IMF",       # IMF World Economic Outlook & Global Growth/Debt Risk
    "RETAIL",    # US Retail Sales & Consumer Spending
    "PMI",       # ISM Manufacturing / Services PMI & Flash PMI
    "JOBLESS",   # US Initial Jobless Claims & Labor Softening
)

MACRO_EVENT_ALIASES: dict[str, str] = {
    "FED": "FOMC",
    "POWELL": "FOMC",
    "RATES": "FOMC",
    "RATE": "FOMC",
    "INTEREST_RATE": "FOMC",
    "INTEREST_RATES": "FOMC",
    "FOMC_MINUTES": "FOMC",
    "PAYROLL": "NFP",
    "PAYROLLS": "NFP",
    "JOBS": "NFP",
    "UNEMPLOYMENT": "NFP",
    "INFLATION": "CPI",
    "CORE_CPI": "CPI",
    "PRODUCER_PRICE": "PPI",
    "CORE_PCE": "PCE",
    "CLAIMS": "JOBLESS",
    "JOBLESS_CLAIMS": "JOBLESS",
    "INITIAL_CLAIMS": "JOBLESS",
    "ISM": "PMI",
    "SERVICES_PMI": "PMI",
    "MANUFACTURING_PMI": "PMI",
    "RETAIL_SALES": "RETAIL",
    "GROWTH": "GDP",
    "US_GDP": "GDP",
    "WEO": "IMF",
    "WORLD_BANK": "IMF",
}


def parse_macro_event_mode_token(token: str) -> Optional[str]:
    """
    Parse macro event sniper mode token (e.g. 'FOMC', 'IMF', 'NFP', 'CPI', 'PPI', 'PCE', 'GDP', 'RETAIL', 'PMI', 'JOBLESS').
    Supports aliases like 'FED', 'POWELL', 'JOBS', 'CLAIMS', 'ISM', 'WEO'.
    Returns normalized uppercase string if valid, else None.
    """
    cleaned = token.strip().upper()
    if cleaned.startswith("EVENT:") or cleaned.startswith("MODE:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if cleaned in SUPPORTED_MACRO_EVENTS:
        return cleaned
    if cleaned in MACRO_EVENT_ALIASES:
        return MACRO_EVENT_ALIASES[cleaned]
    return None

def parse_session_duration(raw: str = "24/7", default_hours: float = 87600.0) -> tuple[int, str]:
    """
    Parse a session mode/duration string into `(duration_seconds, human_label)`.
    Defaults to 24/7 Non-Stop continuous mode (until manually disabled via `/autotrade off`).
    """
    cleaned = (raw or "24/7").strip()
    if cleaned.lower() in ("24/7", "24x7", "nonstop", "non-stop", "continuous", "unlimited", "auto", ""):
        return 315_360_000, "24/7 Non-Stop"
    m = DURATION_RE.match(cleaned)
    if not m:
        if cleaned.isdigit():
            hrs = max(1, int(cleaned))
            return hrs * 3600, f"{hrs} Jam"
        return 315_360_000, "24/7 Non-Stop"
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
    re_entry_position: Optional[UserPositionRecord] = None

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
    winning_trades: int = 0
    losing_trades: int = 0
    realized_pnl_usdt: float = 0.0
    live_current_price: float = 0.0
    live_floating_pnl_usdt: float = 0.0
    live_floating_pnl_pct: float = 0.0
    live_open_position: Optional[UserPositionRecord] = None
    mtf_status_line: str = "1D • 1H • 15M • 5M ➔ Action 1M: SIAP"
    last_updated_iso: str = ""
    last_check_summary: str = "Memulai pemantauan 24/7 Non-Stop..."
    custom_lot: Optional[float] = None
    macro_event_mode: Optional[str] = None
    short_id: str = ""
    max_positions: int = 1
    is_trade_mode: bool = False
    ea_mode: Optional[
        Literal[
            "EA_SCALPING",
            "EA_EXPERT_ADVISOR",
            "EA_GRID_SCALPER",
            "EA_NX_BREAK",
            "EA_NX_PARSIAL",
            "EA_NX_REVERSE",
            "EA_NX_POC",
        ]
    ] = None
    last_broker_req_ts: float = 0.0
    min_step_points: float = 10.0
    stops_level_pips: float = 3.0
    max_spread_pips: float = 4.0
    last_exit_reason: Optional[str] = None
    task: Optional[asyncio.Task[Any]] = field(default=None, repr=False)

    @property
    def is_continuous_24_7(self) -> bool:
        return self.duration_seconds >= 30_000_000 or "24/7" in self.duration_label

    @property
    def remaining_seconds(self) -> int:
        return max(0, int(self.ends_at_ts - time.time()))

    @property
    def remaining_human(self) -> str:
        if self.status != "RUNNING":
            return "Dihentikan (Nonaktif)"
        if self.is_continuous_24_7:
            elapsed = max(0, int(time.time() - self.started_at_ts))
            hrs = elapsed // 3600
            mins = (elapsed % 3600) // 60
            secs = elapsed % 60
            return f"24/7 Non-Stop (Aktif {hrs}j {mins}m {secs}d)"
        rem = self.remaining_seconds
        if rem <= 0:
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
