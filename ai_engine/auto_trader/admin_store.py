from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Literal, Optional

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.admin.store")

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


class AdminStore:
    """
    Persistent SQLite store for Admin Operations:
    1. User Management (Add, Edit Expiry: 1d/1w/1m, Delete/Revoke)
    2. Maintenance Mode Toggle (ON / OFF)
    3. Broadcast Recipient Aggregation
    4. Comprehensive User Activity Audit Trail
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db_path = db_path or (self.settings.processed_data_dir / "user_vault.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS managed_users (
                    user_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    plan_label TEXT NOT NULL,
                    expires_at_ts REAL NOT NULL,
                    expires_at_iso TEXT NOT NULL,
                    added_by INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_users (
                    user_id INTEGER PRIMARY KEY,
                    revoked_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_by INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_audit_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # 1. Management User (Add, Edit Expiry, Delete, Check Access)
    # ------------------------------------------------------------------
    def add_or_update_user(
        self,
        user_id: int,
        duration_code: str = "1m",
        label: str = "Telegram User",
        admin_id: int = 0,
        now_override: Optional[datetime] = None,
    ) -> ManagedUserRecord:
        """Add a new user or update an existing user's expiration duration (1d, 1w, 1m)."""
        uid = int(user_id)
        days, plan_label = parse_expiry_duration(duration_code)
        now_dt = now_override or datetime.now(timezone.utc)
        expires_dt = now_dt + timedelta(days=days)
        expires_ts = expires_dt.timestamp()
        expires_iso = expires_dt.strftime("%Y-%m-%d %H:%M UTC")
        now_iso = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        with self._connect() as conn:
            conn.execute("DELETE FROM revoked_users WHERE user_id = ?", (uid,))
            conn.execute(
                """
                INSERT INTO managed_users (
                    user_id, label, plan_label, expires_at_ts, expires_at_iso,
                    added_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    label = CASE WHEN excluded.label != 'Telegram User' THEN excluded.label ELSE managed_users.label END,
                    plan_label = excluded.plan_label,
                    expires_at_ts = excluded.expires_at_ts,
                    expires_at_iso = excluded.expires_at_iso,
                    added_by = excluded.added_by,
                    updated_at = excluded.updated_at
                """,
                (
                    uid,
                    label,
                    plan_label,
                    expires_ts,
                    expires_iso,
                    int(admin_id),
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()

        return self.get_managed_user(uid)  # type: ignore[return-value]

    def edit_user_expiry(
        self,
        user_id: int,
        duration_code: str,
        admin_id: int = 0,
        now_override: Optional[datetime] = None,
    ) -> Optional[ManagedUserRecord]:
        """Update the expiration duration of a user (creates managed record if in static list)."""
        existing = self.get_managed_user(user_id)
        label = existing.label if existing else f"User {user_id}"
        return self.add_or_update_user(
            user_id=user_id,
            duration_code=duration_code,
            label=label,
            admin_id=admin_id,
            now_override=now_override,
        )

    def delete_user(self, user_id: int) -> bool:
        """Delete a user from managed_users and mark as revoked."""
        uid = int(user_id)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM managed_users WHERE user_id = ?", (uid,))
            conn.execute(
                "INSERT OR REPLACE INTO revoked_users (user_id, revoked_at) VALUES (?, ?)",
                (uid, now_iso),
            )
            conn.commit()
            return cur.rowcount > 0 or uid in {int(x) for x in self.settings.telegram_allowed_user_ids}

    def get_managed_user(self, user_id: int) -> Optional[ManagedUserRecord]:
        uid = int(user_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM managed_users WHERE user_id = ?", (uid,)
            ).fetchone()
        if row is None:
            return None
        return ManagedUserRecord(
            user_id=int(row["user_id"]),
            label=str(row["label"]),
            plan_label=str(row["plan_label"]),
            expires_at_ts=float(row["expires_at_ts"]),
            expires_at_iso=str(row["expires_at_iso"]),
            added_by=int(row["added_by"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def list_managed_users(self) -> list[ManagedUserRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM managed_users ORDER BY expires_at_ts DESC"
            ).fetchall()
        return [
            ManagedUserRecord(
                user_id=int(r["user_id"]),
                label=str(r["label"]),
                plan_label=str(r["plan_label"]),
                expires_at_ts=float(r["expires_at_ts"]),
                expires_at_iso=str(r["expires_at_iso"]),
                added_by=int(r["added_by"]),
                created_at=str(r["created_at"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]

    def is_revoked(self, user_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM revoked_users WHERE user_id = ?", (int(user_id),)
            ).fetchone()
        return row is not None

    def check_user_access(
        self,
        user_id: Optional[int],
        static_allowed_ids: set[int],
    ) -> tuple[bool, AccessStatus, Optional[ManagedUserRecord]]:
        """
        Check whether `user_id` is authorized and not expired:
        - Admin users are always authorized ('ok').
        - Managed users in SQLite are checked against `expires_at_ts` ('ok' vs 'expired').
        - Explicitly deleted/revoked users are denied ('unauthorized').
        - Static `allowed_ids` or `telegram_open_access` are allowed if not revoked.
        """
        if user_id is None:
            return False, "unauthorized", None
        uid = int(user_id)
        if self.settings.is_admin_user(uid):
            return True, "ok", None

        managed = self.get_managed_user(uid)
        if managed is not None:
            if managed.is_expired:
                return False, "expired", managed
            return True, "ok", managed

        if self.is_revoked(uid):
            return False, "unauthorized", None

        if self.settings.telegram_open_access or uid in static_allowed_ids:
            return True, "ok", None

        return False, "unauthorized", None

    # ------------------------------------------------------------------
    # 2. Maintenance Mode (ON / OFF)
    # ------------------------------------------------------------------
    def is_maintenance_mode(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_settings WHERE key = 'maintenance_mode'"
            ).fetchone()
        return bool(row and str(row["value"]) == "1")

    def set_maintenance_mode(self, enabled: bool, admin_id: int = 0) -> bool:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        val = "1" if enabled else "0"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO system_settings (key, value, updated_by, updated_at)
                VALUES ('maintenance_mode', ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (val, int(admin_id), now_iso),
            )
            conn.commit()
        return enabled

    # ------------------------------------------------------------------
    # 3. Broadcast Recipients Aggregation
    # ------------------------------------------------------------------
    def get_broadcast_recipients(self) -> list[int]:
        """Collect all unique active Telegram user IDs for Admin broadcast."""
        recipients: set[int] = set()
        now_ts = datetime.now(timezone.utc).timestamp()

        for uid in self.settings.telegram_allowed_user_ids:
            if not self.is_revoked(int(uid)):
                recipients.add(int(uid))
        for uid in self.settings.telegram_admin_user_ids:
            recipients.add(int(uid))
        if self.settings.telegram_admin_chat_id:
            recipients.add(int(self.settings.telegram_admin_chat_id))

        with self._connect() as conn:
            for r in conn.execute(
                "SELECT user_id FROM managed_users WHERE expires_at_ts > ?", (now_ts,)
            ).fetchall():
                recipients.add(int(r["user_id"]))

            try:
                for r in conn.execute("SELECT user_id FROM user_accounts").fetchall():
                    uid = int(r["user_id"])
                    if not self.is_revoked(uid):
                        recipients.add(uid)
            except sqlite3.OperationalError:
                pass

            for r in conn.execute(
                "SELECT DISTINCT user_id FROM user_audit_logs WHERE status = 'ALLOWED'"
            ).fetchall():
                uid = int(r["user_id"])
                if not self.is_revoked(uid):
                    managed = self.get_managed_user(uid)
                    if managed is None or not managed.is_expired:
                        recipients.add(uid)

        return sorted(recipients)

    # ------------------------------------------------------------------
    # 4. Audit Trail Logging & Querying
    # ------------------------------------------------------------------
    def log_activity(
        self,
        user_id: Optional[int],
        username: Optional[str],
        role: str,
        event_type: str,
        action: str,
        status: str = "ALLOWED",
    ) -> None:
        uid = int(user_id or 0)
        uname = (username or f"user_{uid}").strip()
        clean_action = action.strip()
        # Redact raw secrets from /setup logs
        if clean_action.lower().startswith("/setup") and len(clean_action.split()) >= 3:
            parts = clean_action.split()
            clean_action = f"{parts[0]} {parts[1]} [REDACTED_API_KEY]"

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_audit_logs (
                    user_id, username, role, event_type, action, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (uid, uname, role, event_type, clean_action[:180], status, now_iso),
            )
            conn.commit()

    def get_audit_logs(
        self,
        limit: int = 20,
        user_id: Optional[int] = None,
    ) -> list[AuditLogRecord]:
        cap = max(1, min(int(limit), 100))
        with self._connect() as conn:
            if user_id is not None:
                rows = conn.execute(
                    "SELECT * FROM user_audit_logs WHERE user_id = ? ORDER BY log_id DESC LIMIT ?",
                    (int(user_id), cap),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM user_audit_logs ORDER BY log_id DESC LIMIT ?",
                    (cap,),
                ).fetchall()

        return [
            AuditLogRecord(
                log_id=int(r["log_id"]),
                user_id=int(r["user_id"]),
                username=str(r["username"]),
                role=str(r["role"]),
                event_type=str(r["event_type"]),
                action=str(r["action"]),
                status=str(r["status"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
