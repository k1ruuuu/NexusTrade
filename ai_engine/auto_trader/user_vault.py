from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3
from typing import Any, Literal, Optional

from ai_engine.auto_trader.vault_crypto import (
    decrypt_secret_for_user,
    encrypt_secret_for_user,
    mask_api_key,
)
from ai_engine.auto_trader.vault_positions import (
    BrokerProvider,
    CredentialSource,
    UserBinanceCredentials,
    UserPositionRecord,
    UserRole,
    close_position_in_db,
    compute_broker_account_id,
    insert_open_position,
    query_open_positions,
    refresh_live_exness_balance,
)
from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.auto_trader.vault")

__all__ = [
    "BrokerProvider",
    "CredentialSource",
    "UserBinanceCredentials",
    "UserPositionRecord",
    "UserRole",
    "UserVault",
    "mask_api_key",
    "refresh_live_exness_balance",
]
class UserVault:
    """
    Encrypted SQLite Multi-User Credential Vault for NexusTrade AI.
    Enforces strict role isolation:
    - Regular users ('user') MUST configure personal Binance credentials via /setup.
      They can NEVER access or fall back to .env credentials.
    - Admin users ('admin') may use their personal /setup record OR fall back to .env
      BINANCE_API_KEY / BINANCE_API_SECRET.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db_path = db_path or (self.settings.processed_data_dir / "user_vault.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._admin_env_autotrade: dict[int, bool] = {}
        self._linked_position_sizer: Any = None
        self._init_db()
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_accounts (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    encrypted_api_key TEXT NOT NULL,
                    encrypted_api_secret TEXT NOT NULL,
                    masked_api_key TEXT NOT NULL,
                    testnet INTEGER NOT NULL DEFAULT 1,
                    autotrade_enabled INTEGER NOT NULL DEFAULT 0,
                    risk_per_trade_pct REAL NOT NULL DEFAULT 0.01,
                    verified_usdt_balance REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'exness_mt5',
                    server_name TEXT NOT NULL DEFAULT 'Exness-MT5Trial7'
                )
                """
            )
            for col_sql in (
                "ALTER TABLE user_accounts ADD COLUMN provider TEXT NOT NULL DEFAULT 'exness_mt5'",
                "ALTER TABLE user_accounts ADD COLUMN server_name TEXT NOT NULL DEFAULT 'Exness-MT5Trial7'",
                "ALTER TABLE user_accounts ADD COLUMN account_id TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_positions (
                    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    notional_usdt REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    order_id TEXT NOT NULL,
                    testnet INTEGER NOT NULL DEFAULT 1,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    exit_price REAL,
                    pnl_usdt REAL NOT NULL DEFAULT 0.0,
                    account_id TEXT NOT NULL DEFAULT ''
                )
                """
            )
            try:
                conn.execute("ALTER TABLE user_positions ADD COLUMN account_id TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_daily_risk (
                    user_id INTEGER NOT NULL,
                    utc_date TEXT NOT NULL,
                    start_equity REAL NOT NULL,
                    current_equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    account_id TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (user_id, utc_date)
                )
                """
            )
            try:
                conn.execute("ALTER TABLE user_daily_risk ADD COLUMN account_id TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_daily_risk (
                    account_id TEXT NOT NULL,
                    utc_date TEXT NOT NULL,
                    start_equity REAL NOT NULL,
                    current_equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    breaker_latched INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (account_id, utc_date)
                )
                """
            )
            conn.commit()
        if self.db_path.exists():
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass

    def encrypt_for_user(self, user_id: int, plaintext: str) -> str:
        """Encrypt a secret string with per-user PBKDF2-HMAC-SHA256 + authenticated tag."""
        return encrypt_secret_for_user(self.settings.vault_encryption_secret, user_id, plaintext)

    def decrypt_for_user(self, user_id: int, token: str) -> str:
        """Verify HMAC tag and decrypt a user-bound secret string."""
        return decrypt_secret_for_user(self.settings.vault_encryption_secret, user_id, token)

    def get_user_role(self, user_id: int) -> UserRole:
        return "admin" if self.settings.is_admin_user(user_id) else "user"

    def save_user_credentials(
        self,
        user_id: int,
        api_key: str,
        api_secret: str,
        username: str = "Trader",
        testnet: bool = True,
        risk_per_trade_pct: Optional[float] = None,
        verified_usdt_balance: float = 0.0,
        autotrade_enabled: bool = False,
        provider: BrokerProvider = "exness_mt5",
        server_name: Optional[str] = None,
    ) -> UserBinanceCredentials:
        """Encrypt and persist a user's personal Exness Go / MT5 credentials."""
        clean_key = api_key.strip()
        clean_secret = api_secret.strip()
        min_key_len = 5
        min_sec_len = 6
        if len(clean_key) < min_key_len or len(clean_secret) < min_sec_len:
            raise ValueError(
                f"Kredensial tidak valid (Login MT5 min {min_key_len} karakter, Password min {min_sec_len} karakter)."
            )

        eff_risk = (
            float(risk_per_trade_pct)
            if risk_per_trade_pct is not None
            else min(0.01, self.settings.max_account_risk_per_trade)
        )
        eff_risk = max(0.001, min(eff_risk, self.settings.max_account_risk_per_trade))
        resolved_server = server_name or (
            "Exness-MT5Trial7" if testnet else "Exness-MT5Real"
        )
        acct_id = compute_broker_account_id(provider, resolved_server, clean_key, testnet)
        enc_key = self.encrypt_for_user(user_id, clean_key)
        enc_secret = self.encrypt_for_user(user_id, clean_secret)
        masked = mask_api_key(clean_key)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_accounts (
                    user_id, username, encrypted_api_key, encrypted_api_secret,
                    masked_api_key, testnet, autotrade_enabled, risk_per_trade_pct,
                    verified_usdt_balance, updated_at, provider, server_name, account_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    encrypted_api_key = excluded.encrypted_api_key,
                    encrypted_api_secret = excluded.encrypted_api_secret,
                    masked_api_key = excluded.masked_api_key,
                    testnet = excluded.testnet,
                    autotrade_enabled = excluded.autotrade_enabled,
                    risk_per_trade_pct = excluded.risk_per_trade_pct,
                    verified_usdt_balance = excluded.verified_usdt_balance,
                    updated_at = excluded.updated_at,
                    provider = excluded.provider,
                    server_name = excluded.server_name,
                    account_id = excluded.account_id
                """,
                (
                    int(user_id),
                    username or f"user_{user_id}",
                    enc_key,
                    enc_secret,
                    masked,
                    1 if testnet else 0,
                    1 if autotrade_enabled else 0,
                    eff_risk,
                    float(verified_usdt_balance),
                    now_iso,
                    provider,
                    resolved_server,
                    acct_id,
                ),
            )
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            init_eq = float(verified_usdt_balance) if verified_usdt_balance > 0 else self.settings.portfolio_initial_capital
            conn.execute(
                """
                INSERT INTO user_daily_risk (user_id, utc_date, start_equity, current_equity, realized_pnl, account_id)
                VALUES (?, ?, ?, ?, 0.0, ?)
                ON CONFLICT(user_id, utc_date) DO NOTHING
                """,
                (int(user_id), today_utc, init_eq, init_eq, acct_id),
            )
            conn.execute(
                """
                INSERT INTO account_daily_risk (account_id, utc_date, start_equity, current_equity, realized_pnl, breaker_latched)
                VALUES (?, ?, ?, ?, 0.0, 0)
                ON CONFLICT(account_id, utc_date) DO NOTHING
                """,
                (acct_id, today_utc, init_eq, init_eq),
            )
            conn.commit()
        logger.info(
            "Saved encrypted trading credentials in UserVault",
            user_id=user_id,
            role=self.get_user_role(user_id),
            provider=provider,
            server_name=resolved_server,
            masked_key=masked,
            testnet=testnet,
        )
        return UserBinanceCredentials(
            user_id=int(user_id),
            username=username or f"user_{user_id}",
            role=self.get_user_role(user_id),
            source="user_setup",
            api_key=clean_key,
            api_secret=clean_secret,
            masked_api_key=masked,
            testnet=testnet,
            autotrade_enabled=autotrade_enabled,
            risk_per_trade_pct=eff_risk,
            verified_usdt_balance=float(verified_usdt_balance),
            provider=provider,
            server_name=resolved_server,
            account_id=acct_id,
            updated_at=now_iso,
        )

    def resolve_credentials(self, user_id: int) -> Optional[UserBinanceCredentials]:
        """
        Resolve Binance API credentials for user_id with strict Admin vs User isolation:
        1. If user_id has a personal /setup record in SQLite, decrypt and return it.
        2. Else if user_id has Admin role AND .env has BINANCE_API_KEY + BINANCE_API_SECRET,
           return the Admin .env credentials.
        3. Else return None (Regular users NEVER fall back to .env credentials).
        """
        uid = int(user_id)
        role = self.get_user_role(uid)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_accounts WHERE user_id = ?", (uid,)
            ).fetchone()

        if row is not None:
            api_key = self.decrypt_for_user(uid, str(row["encrypted_api_key"]))
            api_secret = self.decrypt_for_user(uid, str(row["encrypted_api_secret"]))
            row_keys = row.keys()
            prov: BrokerProvider = "exness_mt5"
            srv_name = (
                str(row["server_name"])
                if ("server_name" in row_keys and str(row["server_name"]).strip())
                else ("Exness-MT5Trial7" if bool(row["testnet"]) else "Exness-MT5Real")
            )
            acct_id = (
                str(row["account_id"])
                if ("account_id" in row_keys and str(row["account_id"]).strip())
                else compute_broker_account_id(prov, srv_name, api_key, bool(row["testnet"]))
            )
            return UserBinanceCredentials(
                user_id=uid,
                username=str(row["username"]),
                role=role,
                source="user_setup",
                api_key=api_key,
                api_secret=api_secret,
                masked_api_key=str(row["masked_api_key"]),
                testnet=bool(row["testnet"]),
                autotrade_enabled=bool(row["autotrade_enabled"]),
                risk_per_trade_pct=float(row["risk_per_trade_pct"]),
                verified_usdt_balance=float(row["verified_usdt_balance"]),
                provider=prov,
                server_name=srv_name,
                account_id=acct_id,
                updated_at=str(row["updated_at"]),
            )
        # Strict Admin-only .env fallback
        if role == "admin":
            env_key = self.settings.binance_api_key.strip()
            env_secret = self.settings.binance_api_secret.strip()
            if len(env_key) >= 5 and len(env_secret) >= 6:
                srv_env = "Exness-MT5Trial7" if self.settings.binance_testnet else "Exness-MT5Real"
                acct_env = compute_broker_account_id("exness_mt5", srv_env, env_key, self.settings.binance_testnet)
                return UserBinanceCredentials(
                    user_id=uid,
                    username="Admin (.env)",
                    role="admin",
                    source="admin_env",
                    api_key=env_key,
                    api_secret=env_secret,
                    masked_api_key=mask_api_key(env_key),
                    testnet=self.settings.binance_testnet,
                    autotrade_enabled=self._admin_env_autotrade.get(uid, False),
                    risk_per_trade_pct=min(0.01, self.settings.max_account_risk_per_trade),
                    verified_usdt_balance=self.settings.portfolio_initial_capital,
                    provider="exness_mt5",
                    server_name=srv_env,
                    account_id=acct_env,
                    updated_at="env-configured",
                )

        return None

    def set_autotrade_enabled(self, user_id: int, enabled: bool) -> bool:
        """Toggle autotrade_enabled for a configured user (or Admin .env account)."""
        uid = int(user_id)
        creds = self.resolve_credentials(uid)
        if creds is None:
            return False

        if creds.source == "user_setup":
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with self._connect() as conn:
                conn.execute(
                    "UPDATE user_accounts SET autotrade_enabled = ?, updated_at = ? WHERE user_id = ?",
                    (1 if enabled else 0, now_iso, uid),
                )
                conn.commit()
            return True

        self._admin_env_autotrade[uid] = enabled
        return True

    def update_verified_balance(self, user_id: int, usdt_balance: float) -> None:
        uid = int(user_id)
        creds = self.resolve_credentials(uid)
        acct_id = creds.account_id if creds else f"user:{uid}"
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        today_utc = now_dt.strftime("%Y-%m-%d")
        bal = float(usdt_balance)
        with self._connect() as conn:
            conn.execute(
                "UPDATE user_accounts SET verified_usdt_balance = ?, updated_at = ? WHERE user_id = ?",
                (bal, now_iso, uid),
            )
            conn.execute(
                """
                INSERT INTO user_daily_risk (user_id, utc_date, start_equity, current_equity, realized_pnl, account_id)
                VALUES (?, ?, ?, ?, 0.0, ?)
                ON CONFLICT(user_id, utc_date) DO UPDATE SET
                    current_equity = excluded.current_equity
                """,
                (uid, today_utc, bal, bal, acct_id),
            )
            conn.execute(
                """
                INSERT INTO account_daily_risk (account_id, utc_date, start_equity, current_equity, realized_pnl, breaker_latched)
                VALUES (?, ?, ?, ?, 0.0, 0)
                ON CONFLICT(account_id, utc_date) DO UPDATE SET
                    current_equity = excluded.current_equity
                """,
                (acct_id, today_utc, bal, bal),
            )
            conn.commit()
        self._sync_linked_sizer(uid)

    def delete_user_credentials(self, user_id: int) -> bool:
        """Remove personal Broker credentials for user_id from the vault (/unlink)."""
        uid = int(user_id)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM user_accounts WHERE user_id = ?", (uid,))
            conn.commit()
            deleted = cur.rowcount > 0
        self._admin_env_autotrade.pop(uid, None)
        return deleted

    def record_open_position(
        self,
        user_id: int,
        symbol: str,
        side: Literal["BUY", "SELL"],
        entry_price: float,
        quantity: float,
        notional_usdt: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        confidence: float,
        order_id: str,
        testnet: bool = True,
        account_id: Optional[str] = None,
        status: Literal["OPEN", "CLOSED", "FILLED_UNPROTECTED"] = "OPEN",
    ) -> UserPositionRecord:
        creds = self.resolve_credentials(int(user_id))
        resolved_acct = account_id or (creds.account_id if creds else f"user:{int(user_id)}")
        with self._connect() as conn:
            return insert_open_position(
                conn=conn,
                user_id=user_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                notional_usdt=notional_usdt,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                confidence=confidence,
                order_id=order_id,
                testnet=testnet,
                account_id=resolved_acct,
                status=status,
            )

    def get_open_positions(
        self,
        user_id: int,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> list[UserPositionRecord]:
        creds = self.resolve_credentials(int(user_id))
        resolved_acct = account_id or (creds.account_id if creds else None)
        with self._connect() as conn:
            return query_open_positions(conn, user_id=user_id, symbol=symbol, account_id=resolved_acct)

    def update_position_stop_loss(self, position_id: int, new_stop_loss: float) -> None:
        """Ratchet trailing stop-loss on an active OPEN position."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE user_positions SET stop_loss = ? WHERE position_id = ? AND status IN ('OPEN', 'FILLED_UNPROTECTED')",
                (float(new_stop_loss), int(position_id)),
            )
            conn.commit()

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        pnl_usdt: float,
    ) -> Optional[UserPositionRecord]:
        """
        Idempotently mark an open position as CLOSED and update the account/user UTC daily drawdown ledger
        atomically in a single transaction (R04 / NT-03). Repeated calls on an already-closed position do NOT double-count PnL.
        """
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        newly_closed = False
        with self._connect() as conn:
            closed_rec, newly_closed = close_position_in_db(
                conn, position_id, exit_price, pnl_usdt, commit=False
            )
            if closed_rec is not None and newly_closed:
                uid = int(closed_rec.user_id)
                acct_id = closed_rec.account_id or f"user:{uid}"
                acc_row = conn.execute(
                    "SELECT verified_usdt_balance FROM user_accounts WHERE user_id = ?", (uid,)
                ).fetchone()
                base_eq = (
                    float(acc_row["verified_usdt_balance"])
                    if acc_row is not None and float(acc_row["verified_usdt_balance"]) > 0
                    else self.settings.portfolio_initial_capital
                )
                conn.execute(
                    """
                    INSERT INTO user_daily_risk (user_id, utc_date, start_equity, current_equity, realized_pnl, account_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, utc_date) DO UPDATE SET
                        realized_pnl = user_daily_risk.realized_pnl + excluded.realized_pnl,
                        current_equity = MAX(0.0, user_daily_risk.start_equity + user_daily_risk.realized_pnl + excluded.realized_pnl)
                    """,
                    (uid, today_utc, base_eq, max(0.0, base_eq + float(pnl_usdt)), float(pnl_usdt), acct_id),
                )
                conn.execute(
                    """
                    INSERT INTO account_daily_risk (account_id, utc_date, start_equity, current_equity, realized_pnl, breaker_latched)
                    VALUES (?, ?, ?, ?, ?, 0)
                    ON CONFLICT(account_id, utc_date) DO UPDATE SET
                        realized_pnl = account_daily_risk.realized_pnl + excluded.realized_pnl,
                        current_equity = MAX(0.0, account_daily_risk.start_equity + account_daily_risk.realized_pnl + excluded.realized_pnl)
                    """,
                    (acct_id, today_utc, base_eq, max(0.0, base_eq + float(pnl_usdt)), float(pnl_usdt)),
                )
            conn.commit()
        if closed_rec is not None and newly_closed:
            self._sync_linked_sizer(closed_rec.user_id)
        return closed_rec

    def get_user_daily_drawdown(
        self,
        user_id: int,
        account_id: Optional[str] = None,
    ) -> tuple[float, float, float]:
        """Return `(drawdown_fraction, start_equity, current_equity)` scoped by `account_id` (or `user_id`) on current UTC date."""
        uid = int(user_id)
        creds = self.resolve_credentials(uid)
        acct_id = account_id or (creds.account_id if creds else None)
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._connect() as conn:
            row = None
            if acct_id:
                row = conn.execute(
                    "SELECT start_equity, current_equity, realized_pnl FROM account_daily_risk WHERE account_id = ? AND utc_date = ?",
                    (acct_id, today_utc),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT start_equity, current_equity, realized_pnl FROM user_daily_risk WHERE user_id = ? AND utc_date = ?",
                    (uid, today_utc),
                ).fetchone()
        if row is None:
            eq = float(creds.verified_usdt_balance) if creds and creds.verified_usdt_balance > 0 else self.settings.portfolio_initial_capital
            return 0.0, eq, eq
        start_eq = max(float(row["start_equity"]), 1e-6)
        curr_eq = float(row["current_equity"])
        real_pnl = float(row["realized_pnl"])
        loss = max(0.0, start_eq - curr_eq, -real_pnl)
        dd = round(min(1.0, loss / start_eq), 6)
        return dd, start_eq, min(curr_eq, start_eq - loss)

    def _sync_linked_sizer(self, user_id: int) -> None:
        if self._linked_position_sizer is not None:
            _, start_eq, curr_eq = self.get_user_daily_drawdown(user_id)
            if start_eq > 0:
                self._linked_position_sizer.daily_start_equity = start_eq
                self._linked_position_sizer.current_equity = curr_eq
    def list_autotrade_enabled_user_ids(self) -> list[int]:
        """Return all user_ids (including Admin .env accounts) that currently have autotrade_enabled=True."""
        enabled_ids: set[int] = set()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_id FROM user_accounts WHERE autotrade_enabled = 1"
            ).fetchall()
            for r in rows:
                enabled_ids.add(int(r["user_id"]))

        for uid, is_on in self._admin_env_autotrade.items():
            if is_on and self.get_user_role(uid) == "admin":
                enabled_ids.add(int(uid))

        return sorted(enabled_ids)
