from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

AccessStatus = Literal["ok", "expired", "unauthorized"]

DURATION_PRESETS: dict[str, tuple[int, str]] = {
    "1d": (1, "1 Hari"),
    "1hari": (1, "1 Hari"),
    "1_hari": (1, "1 Hari"),
    "1day": (1, "1 Hari"),
    "1": (1, "1 Hari"),
    "1w": (7, "1 Minggu"),
    "7d": (7, "1 Minggu"),
    "1minggu": (7, "1 Minggu"),
    "1_minggu": (7, "1 Minggu"),
    "1week": (7, "1 Minggu"),
    "7": (7, "1 Minggu"),
    "1m": (30, "1 Bulan"),
    "30d": (30, "1 Bulan"),
    "1bulan": (30, "1 Bulan"),
    "1_bulan": (30, "1 Bulan"),
    "1month": (30, "1 Bulan"),
    "30": (30, "1 Bulan"),
}


def parse_expiry_duration(code: str) -> tuple[int, str]:
    """Parse duration preset ('1d'/'1hari', '1w'/'1minggu', '1m'/'1bulan') into (days, label)."""
    cleaned = code.strip().lower().replace(" ", "")
    if cleaned in DURATION_PRESETS:
        return DURATION_PRESETS[cleaned]
    if cleaned.endswith("d") and cleaned[:-1].isdigit():
        days = max(1, int(cleaned[:-1]))
        return days, f"{days} Hari"
    if cleaned.isdigit():
        days = max(1, int(cleaned))
        return days, f"{days} Hari"
    raise ValueError(
        f"Durasi '{code}' tidak dikenali. Gunakan: 1d (1 Hari), 1w (1 Minggu), atau 1m (1 Bulan)."
    )


@dataclass
class ManagedUserRecord:
    """Admin-managed Telegram user subscription with expiration timestamp."""

    user_id: int
    label: str
    plan_label: str
    expires_at_ts: float
    expires_at_iso: str
    added_by: int
    created_at: str
    updated_at: str

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc).timestamp() > self.expires_at_ts

    @property
    def remaining_human(self) -> str:
        rem_sec = self.expires_at_ts - datetime.now(timezone.utc).timestamp()
        if rem_sec <= 0:
            return "EXPIRED (Habis)"
        days = int(rem_sec // 86400)
        hours = int((rem_sec % 86400) // 3600)
        if days > 0:
            return f"{days} hari {hours} jam"
        mins = max(1, int((rem_sec % 3600) // 60))
        return f"{hours} jam {mins} menit"


@dataclass
class AuditLogRecord:
    """Single user or admin activity audit log entry."""

    log_id: int
    user_id: int
    username: str
    role: str
    event_type: str
    action: str
    status: str
    created_at: str
