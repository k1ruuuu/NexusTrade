from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
from typing import Literal, Optional

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.auto_trader.vault")

UserRole = Literal["admin", "user"]
CredentialSource = Literal["user_setup", "admin_env"]


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe Telegram display (e.g., 'ABCD****WXYZ')."""
    cleaned = api_key.strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:4]}****{cleaned[-4:]}"


@dataclass
class UserBinanceCredentials:
    """Decrypted in-memory Binance credentials isolated to a single Telegram user_id."""

    user_id: int
    username: str
    role: UserRole
    source: CredentialSource
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    masked_api_key: str = "****"
    testnet: bool = True
    autotrade_enabled: bool = False
    risk_per_trade_pct: float = 0.01
    verified_usdt_balance: float = 0.0
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


@dataclass
class UserPositionRecord:
    """Active or closed auto-trade position belonging to a specific Telegram user_id."""

    position_id: int
    user_id: int
    symbol: str
    side: Literal["BUY", "SELL"]
    entry_price: float
    quantity: float
    notional_usdt: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float
    status: Literal["OPEN", "CLOSED"]
    order_id: str
    testnet: bool
    opened_at: str
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_usdt: float = 0.0


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
                    updated_at TEXT NOT NULL
                )
                """
            )
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
                    pnl_usdt REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            conn.commit()

    def _derive_keys(self, user_id: int, nonce: bytes) -> tuple[bytes, bytes]:
        master = self.settings.vault_encryption_secret.encode("utf-8")
        salt = f"nexustrade_uid_{int(user_id)}_".encode("utf-8") + nonce
        dk = hashlib.pbkdf2_hmac("sha256", master, salt, 60_000, dklen=64)
        return dk[:32], dk[32:]

    @staticmethod
    def _xor_keystream(data: bytes, enc_key: bytes, nonce: bytes) -> bytes:
        out = bytearray(len(data))
        block_count = (len(data) + 31) // 32
        for counter in range(block_count):
            ctr_bytes = counter.to_bytes(4, "big")
            keystream_block = hmac.new(enc_key, nonce + ctr_bytes, hashlib.sha256).digest()
            start = counter * 32
            end = min(start + 32, len(data))
            for i in range(start, end):
                out[i] = data[i] ^ keystream_block[i - start]
        return bytes(out)

    def encrypt_for_user(self, user_id: int, plaintext: str) -> str:
        """Encrypt a secret string with per-user PBKDF2-HMAC-SHA256 + authenticated tag."""
        nonce = secrets.token_bytes(16)
        enc_key, mac_key = self._derive_keys(user_id, nonce)
        raw_bytes = plaintext.encode("utf-8")
        ciphertext = self._xor_keystream(raw_bytes, enc_key, nonce)
        tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        payload = nonce + tag + ciphertext
        return base64.urlsafe_b64encode(payload).decode("ascii")

    def decrypt_for_user(self, user_id: int, token: str) -> str:
        """Verify HMAC tag and decrypt a user-bound secret string."""
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        if len(raw) < 48:
            raise ValueError("Invalid encrypted vault token length")
        nonce = raw[:16]
        tag = raw[16:48]
        ciphertext = raw[48:]
        enc_key, mac_key = self._derive_keys(user_id, nonce)
        expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Vault token integrity check failed (wrong user_id or tampered)")
        plaintext_bytes = self._xor_keystream(ciphertext, enc_key, nonce)
        return plaintext_bytes.decode("utf-8")

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
    ) -> UserBinanceCredentials:
        """Encrypt and persist a user's personal Binance API credentials."""
        clean_key = api_key.strip()
        clean_secret = api_secret.strip()
        if len(clean_key) < 8 or len(clean_secret) < 8:
            raise ValueError("API Key dan API Secret harus minimal 8 karakter.")

        eff_risk = (
            float(risk_per_trade_pct)
            if risk_per_trade_pct is not None
            else min(0.01, self.settings.max_account_risk_per_trade)
        )
        eff_risk = max(0.001, min(eff_risk, self.settings.max_account_risk_per_trade))

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
                    verified_usdt_balance, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    encrypted_api_key = excluded.encrypted_api_key,
                    encrypted_api_secret = excluded.encrypted_api_secret,
                    masked_api_key = excluded.masked_api_key,
                    testnet = excluded.testnet,
                    autotrade_enabled = excluded.autotrade_enabled,
                    risk_per_trade_pct = excluded.risk_per_trade_pct,
                    verified_usdt_balance = excluded.verified_usdt_balance,
                    updated_at = excluded.updated_at
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
                ),
            )
            conn.commit()

        logger.info(
            "Saved encrypted Binance credentials in UserVault",
            user_id=user_id,
            role=self.get_user_role(user_id),
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
                updated_at=str(row["updated_at"]),
            )

        # Strict Admin-only .env fallback
        if role == "admin":
            env_key = self.settings.binance_api_key.strip()
            env_secret = self.settings.binance_api_secret.strip()
            if len(env_key) >= 8 and len(env_secret) >= 8:
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
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._connect() as conn:
            conn.execute(
                "UPDATE user_accounts SET verified_usdt_balance = ?, updated_at = ? WHERE user_id = ?",
                (float(usdt_balance), now_iso, uid),
            )
            conn.commit()

    def delete_user_credentials(self, user_id: int) -> bool:
        """Remove personal Binance credentials for user_id from the vault (/unlink)."""
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
    ) -> UserPositionRecord:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO user_positions (
                    user_id, symbol, side, entry_price, quantity, notional_usdt,
                    stop_loss, take_profit_1, take_profit_2, confidence,
                    status, order_id, testnet, opened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
                """,
                (
                    int(user_id),
                    symbol.upper(),
                    side,
                    float(entry_price),
                    float(quantity),
                    float(notional_usdt),
                    float(stop_loss),
                    float(take_profit_1),
                    float(take_profit_2),
                    float(confidence),
                    str(order_id),
                    1 if testnet else 0,
                    now_iso,
                ),
            )
            conn.commit()
            pos_id = int(cur.lastrowid or 1)

        return UserPositionRecord(
            position_id=pos_id,
            user_id=int(user_id),
            symbol=symbol.upper(),
            side=side,
            entry_price=float(entry_price),
            quantity=float(quantity),
            notional_usdt=float(notional_usdt),
            stop_loss=float(stop_loss),
            take_profit_1=float(take_profit_1),
            take_profit_2=float(take_profit_2),
            confidence=float(confidence),
            status="OPEN",
            order_id=str(order_id),
            testnet=testnet,
            opened_at=now_iso,
        )

    def get_open_positions(
        self,
        user_id: int,
        symbol: Optional[str] = None,
    ) -> list[UserPositionRecord]:
        uid = int(user_id)
        with self._connect() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM user_positions WHERE user_id = ? AND status = 'OPEN' AND symbol = ? ORDER BY position_id DESC",
                    (uid, symbol.upper()),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM user_positions WHERE user_id = ? AND status = 'OPEN' ORDER BY position_id DESC",
                    (uid,),
                ).fetchall()

        return [
            UserPositionRecord(
                position_id=int(r["position_id"]),
                user_id=int(r["user_id"]),
                symbol=str(r["symbol"]),
                side="BUY" if str(r["side"]) == "BUY" else "SELL",
                entry_price=float(r["entry_price"]),
                quantity=float(r["quantity"]),
                notional_usdt=float(r["notional_usdt"]),
                stop_loss=float(r["stop_loss"]),
                take_profit_1=float(r["take_profit_1"]),
                take_profit_2=float(r["take_profit_2"]),
                confidence=float(r["confidence"]),
                status="OPEN" if str(r["status"]) == "OPEN" else "CLOSED",
                order_id=str(r["order_id"]),
                testnet=bool(r["testnet"]),
                opened_at=str(r["opened_at"]),
                closed_at=str(r["closed_at"]) if r["closed_at"] else None,
                exit_price=float(r["exit_price"]) if r["exit_price"] is not None else None,
                pnl_usdt=float(r["pnl_usdt"]),
            )
            for r in rows
        ]
