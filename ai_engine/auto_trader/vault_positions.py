from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import sqlite3
import time
from typing import Any, Literal, Optional

UserRole = Literal["admin", "user"]
CredentialSource = Literal["user_setup", "admin_env"]
BrokerProvider = Literal["exness_mt5"]


@dataclass
class UserBinanceCredentials:
    """Decrypted in-memory Broker/Exchange credentials isolated to a single Telegram user_id."""

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
    provider: BrokerProvider = "exness_mt5"
    server_name: str = "Exness-MT5Trial7"
    account_id: str = ""
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
    status: Literal["OPEN", "CLOSED", "FILLED_UNPROTECTED"]
    order_id: str
    testnet: bool
    opened_at: str
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_usdt: float = 0.0
    account_id: str = ""


def compute_broker_account_id(
    provider: str,
    server_name: str,
    api_key: str,
    testnet: bool,
) -> str:
    """Build canonical BrokerAccountIdentity key (`provider:server:env:login`)."""
    prov = (provider or "exness_mt5").strip().lower()
    srv = (server_name or "Exness-MT5Trial7").strip().upper()
    env_tag = "TEST" if testnet else "LIVE"
    login = (api_key or "unknown").strip()
    return f"{prov}:{srv}:{env_tag}:{login}"


def row_to_position_record(r: sqlite3.Row) -> UserPositionRecord:
    raw_status = str(r["status"])
    status_val: Literal["OPEN", "CLOSED", "FILLED_UNPROTECTED"] = (
        "CLOSED"
        if raw_status == "CLOSED"
        else ("FILLED_UNPROTECTED" if raw_status == "FILLED_UNPROTECTED" else "OPEN")
    )
    r_keys = r.keys()
    acct_id = str(r["account_id"]) if ("account_id" in r_keys and r["account_id"]) else ""
    return UserPositionRecord(
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
        status=status_val,
        order_id=str(r["order_id"]),
        testnet=bool(r["testnet"]),
        opened_at=str(r["opened_at"]),
        closed_at=str(r["closed_at"]) if r["closed_at"] else None,
        exit_price=float(r["exit_price"]) if r["exit_price"] is not None else None,
        pnl_usdt=float(r["pnl_usdt"]),
        account_id=acct_id,
    )

def insert_open_position(
    conn: sqlite3.Connection,
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
    account_id: str = "",
    status: Literal["OPEN", "CLOSED", "FILLED_UNPROTECTED"] = "OPEN",
) -> UserPositionRecord:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        """
        INSERT INTO user_positions (
            user_id, symbol, side, entry_price, quantity, notional_usdt,
            stop_loss, take_profit_1, take_profit_2, confidence,
            status, order_id, testnet, opened_at, account_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            status,
            str(order_id),
            1 if testnet else 0,
            now_iso,
            str(account_id),
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
        status=status,
        order_id=str(order_id),
        testnet=testnet,
        opened_at=now_iso,
        account_id=str(account_id),
    )


def query_open_positions(
    conn: sqlite3.Connection,
    user_id: int,
    symbol: Optional[str] = None,
    account_id: Optional[str] = None,
) -> list[UserPositionRecord]:
    uid = int(user_id)
    if account_id:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM user_positions WHERE (account_id = ? OR (account_id = '' AND user_id = ?)) AND status IN ('OPEN', 'FILLED_UNPROTECTED') AND symbol = ? ORDER BY position_id DESC",
                (account_id, uid, symbol.upper()),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM user_positions WHERE (account_id = ? OR (account_id = '' AND user_id = ?)) AND status IN ('OPEN', 'FILLED_UNPROTECTED') ORDER BY position_id DESC",
                (account_id, uid),
            ).fetchall()
    elif symbol:
        rows = conn.execute(
            "SELECT * FROM user_positions WHERE user_id = ? AND status IN ('OPEN', 'FILLED_UNPROTECTED') AND symbol = ? ORDER BY position_id DESC",
            (uid, symbol.upper()),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM user_positions WHERE user_id = ? AND status IN ('OPEN', 'FILLED_UNPROTECTED') ORDER BY position_id DESC",
            (uid,),
        ).fetchall()
    return [row_to_position_record(r) for r in rows]


def close_position_in_db(
    conn: sqlite3.Connection,
    position_id: int,
    exit_price: float,
    pnl_usdt: float,
    commit: bool = True,
) -> tuple[Optional[UserPositionRecord], bool]:
    """
    Idempotently transition a position to CLOSED only if it is not already CLOSED.
    Returns `(position_record, newly_closed)`.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pid = int(position_id)
    cur = conn.execute(
        """
        UPDATE user_positions
        SET status = 'CLOSED', closed_at = ?, exit_price = ?, pnl_usdt = ?
        WHERE position_id = ? AND status != 'CLOSED'
        """,
        (now_iso, float(exit_price), float(pnl_usdt), pid),
    )
    newly_closed = cur.rowcount > 0
    if commit:
        conn.commit()
    row = conn.execute(
        "SELECT * FROM user_positions WHERE position_id = ?", (pid,)
    ).fetchone()
    if row is None:
        return None, False
    return row_to_position_record(row), newly_closed

_LAST_BALANCE_REFRESH: dict[int, float] = {}

async def refresh_live_exness_balance(
    vault: Any,
    mt5_client: Any,
    user_id: int,
    creds: Optional[UserBinanceCredentials],
) -> None:
    """
    Shared helper to refresh `verified_usdt_balance` live from Exness MT5
    when the user has a non-sandbox `exness_mt5` account configured.
    Cached for 30 seconds to prevent hammering MT5 API on every user command.
    """
    if (
        creds is None
        or creds.source != "user_setup"
        or creds.provider != "exness_mt5"
        or mt5_client._is_exness_trial_demo(creds.api_key, creds.testnet)
    ):
        return

    now = time.time()
    if now - _LAST_BALANCE_REFRESH.get(user_id, 0.0) < 30.0 and creds.verified_usdt_balance > 0:
        return

    try:
        ver = await asyncio.wait_for(
            mt5_client.verify_credentials(
                creds.api_key,
                creds.api_secret,
                server_name=creds.server_name,
                testnet=creds.testnet,
            ),
            timeout=5.0,
        )
        _LAST_BALANCE_REFRESH[user_id] = now
        if ver.valid and ver.total_usdt >= 0:
            vault.update_verified_balance(user_id, ver.total_usdt)
            creds.verified_usdt_balance = ver.total_usdt
    except Exception:
        pass
