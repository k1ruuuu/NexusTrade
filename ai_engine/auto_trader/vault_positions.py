from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import sqlite3
from typing import Any, Literal, Optional

UserRole = Literal["admin", "user"]
CredentialSource = Literal["user_setup", "admin_env"]
BrokerProvider = Literal["binance_spot", "binance_futures", "exness_mt5"]


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
    provider: BrokerProvider = "binance_spot"
    server_name: str = "Binance-Spot"
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


def row_to_position_record(r: sqlite3.Row) -> UserPositionRecord:
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
        status="OPEN" if str(r["status"]) == "OPEN" else "CLOSED",
        order_id=str(r["order_id"]),
        testnet=bool(r["testnet"]),
        opened_at=str(r["opened_at"]),
        closed_at=str(r["closed_at"]) if r["closed_at"] else None,
        exit_price=float(r["exit_price"]) if r["exit_price"] is not None else None,
        pnl_usdt=float(r["pnl_usdt"]),
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
) -> UserPositionRecord:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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


def query_open_positions(
    conn: sqlite3.Connection,
    user_id: int,
    symbol: Optional[str] = None,
) -> list[UserPositionRecord]:
    uid = int(user_id)
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
    return [row_to_position_record(r) for r in rows]


def close_position_in_db(
    conn: sqlite3.Connection,
    position_id: int,
    exit_price: float,
    pnl_usdt: float,
) -> Optional[UserPositionRecord]:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pid = int(position_id)
    conn.execute(
        """
        UPDATE user_positions
        SET status = 'CLOSED', closed_at = ?, exit_price = ?, pnl_usdt = ?
        WHERE position_id = ?
        """,
        (now_iso, float(exit_price), float(pnl_usdt), pid),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM user_positions WHERE position_id = ?", (pid,)
    ).fetchone()
    if row is None:
        return None
    return row_to_position_record(row)


async def refresh_live_exness_balance(
    vault: Any,
    mt5_client: Any,
    user_id: int,
    creds: Optional[UserBinanceCredentials],
) -> None:
    """
    Shared helper to refresh `verified_usdt_balance` live from Exness MT5
    when the user has a non-sandbox `exness_mt5` account configured.
    """
    if (
        creds is None
        or creds.source != "user_setup"
        or creds.provider != "exness_mt5"
        or mt5_client._is_exness_trial_demo(creds.api_key, creds.testnet)
    ):
        return
    ver = await mt5_client.verify_credentials(
        creds.api_key,
        creds.api_secret,
        server_name=creds.server_name,
        testnet=creds.testnet,
    )
    if ver.valid and ver.total_usdt >= 0:
        vault.update_verified_balance(user_id, ver.total_usdt)
        creds.verified_usdt_balance = ver.total_usdt
