from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import time
from typing import Any, Literal, Optional
from urllib.parse import urlencode

import httpx

from config.logging_config import get_logger
from config.settings import get_settings
from data_pipeline.market_symbols import is_crypto_symbol, normalize_symbol

logger = get_logger("nexustrade.auto_trader.binance_client")

BINANCE_LIVE_BASE_URL = "https://api.binance.com"
BINANCE_TESTNET_BASE_URL = "https://testnet.binance.vision"
BINANCE_FUTURES_LIVE_URL = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"


@dataclass
class BinanceAccountVerification:
    """Result of verifying a user's Binance API Key & Secret against /api/v3/account."""

    valid: bool
    can_trade: bool
    usdt_free: float
    usdt_locked: float
    total_usdt: float
    asset_count: int
    account_type: str
    testnet: bool
    error_message: Optional[str] = None


@dataclass
class BinanceOrderResult:
    """Result of placing an automated order on a user's Binance account."""

    executed: bool
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    executed_qty: float
    avg_price: float
    notional_usdt: float
    testnet: bool
    status: str
    error_message: Optional[str] = None


class BinanceSpotClient:
    """
    Asynchronous HMAC-SHA256 authenticated Binance Spot & Testnet REST API client.
    Supports per-user credentials for account verification (/setup) and order execution (/autotrade).
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def resolve_base_url(testnet: bool, futures: bool = False) -> str:
        if futures:
            return BINANCE_FUTURES_TESTNET_URL if testnet else BINANCE_FUTURES_LIVE_URL
        return BINANCE_TESTNET_BASE_URL if testnet else BINANCE_LIVE_BASE_URL
    @staticmethod
    def sign_params(params: dict[str, Any], api_secret: str) -> str:
        query_string = urlencode(params)
        signature = hmac.new(
            api_secret.strip().encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query_string}&signature={signature}"

    @staticmethod
    def _is_sandbox_demo_key(api_key: str) -> bool:
        """Allow deterministic sandbox verification ONLY in non-production environments."""
        if get_settings().environment == "production":
            return False
        upper = api_key.strip().upper()
        return upper.startswith("TESTNET_") or upper.startswith("DEMO_")
    async def verify_credentials(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ) -> BinanceAccountVerification:
        """
        Verify a user's Binance API Key and Secret by signing GET /api/v3/account.
        Confirm trading permission and retrieve real-time USDT balance.
        """
        clean_key = api_key.strip()
        clean_secret = api_secret.strip()
        if len(clean_key) < 8 or len(clean_secret) < 8:
            return BinanceAccountVerification(
                valid=False,
                can_trade=False,
                usdt_free=0.0,
                usdt_locked=0.0,
                total_usdt=0.0,
                asset_count=0,
                account_type="INVALID",
                testnet=testnet,
                error_message="API Key dan API Secret harus minimal 8 karakter.",
            )

        if self._is_sandbox_demo_key(clean_key) and testnet:
            return BinanceAccountVerification(
                valid=True,
                can_trade=True,
                usdt_free=10000.0,
                usdt_locked=0.0,
                total_usdt=10000.0,
                asset_count=2,
                account_type="SPOT_TESTNET_SANDBOX",
                testnet=True,
            )

        base_url = self.resolve_base_url(testnet)
        params = {
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
        }
        signed_query = self.sign_params(params, clean_secret)
        url = f"{base_url}/api/v3/account?{signed_query}"
        headers = {"X-MBX-APIKEY": clean_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    msg = str(err_json.get("msg") or f"HTTP {resp.status_code}")[:160]
                except Exception:
                    msg = f"HTTP {resp.status_code}"
                return BinanceAccountVerification(
                    valid=False,
                    can_trade=False,
                    usdt_free=0.0,
                    usdt_locked=0.0,
                    total_usdt=0.0,
                    asset_count=0,
                    account_type="REJECTED",
                    testnet=testnet,
                    error_message=f"Binance menolak kredensial ({msg})",
                )

            payload = resp.json()
            can_trade = bool(payload.get("canTrade", False))
            account_type = str(payload.get("accountType", "SPOT"))
            balances = payload.get("balances", [])

            usdt_free = 0.0
            usdt_locked = 0.0
            non_zero_assets = 0
            for item in balances:
                free_v = float(item.get("free", 0.0))
                locked_v = float(item.get("locked", 0.0))
                if free_v + locked_v > 0:
                    non_zero_assets += 1
                if str(item.get("asset", "")).upper() == "USDT":
                    usdt_free = free_v
                    usdt_locked = locked_v

            if not can_trade:
                return BinanceAccountVerification(
                    valid=False,
                    can_trade=False,
                    usdt_free=usdt_free,
                    usdt_locked=usdt_locked,
                    total_usdt=usdt_free + usdt_locked,
                    asset_count=non_zero_assets,
                    account_type=account_type,
                    testnet=testnet,
                    error_message="API Key valid namun izin 'Enable Spot & Margin Trading' belum diaktifkan.",
                )

            return BinanceAccountVerification(
                valid=True,
                can_trade=True,
                usdt_free=round(usdt_free, 2),
                usdt_locked=round(usdt_locked, 2),
                total_usdt=round(usdt_free + usdt_locked, 2),
                asset_count=non_zero_assets,
                account_type=account_type,
                testnet=testnet,
            )
        except Exception as exc:
            logger.warning("Binance account verification failed", error=str(exc), testnet=testnet)
            return BinanceAccountVerification(
                valid=False,
                can_trade=False,
                usdt_free=0.0,
                usdt_locked=0.0,
                total_usdt=0.0,
                asset_count=0,
                account_type="UNREACHABLE",
                testnet=testnet,
                error_message=f"Gagal menghubungi server Binance ({type(exc).__name__})",
            )

    @staticmethod
    def format_order_quantity(quantity: float, price: float) -> float:
        """Round order quantity to exchange-compatible step precision based on unit price."""
        if price >= 10000:
            return round(max(0.0001, quantity), 4)
        if price >= 100:
            return round(max(0.001, quantity), 3)
        if price >= 1:
            return round(max(0.01, quantity), 2)
        return round(max(1.0, quantity), 0)

    async def place_market_order(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        reference_price: float,
        testnet: bool = True,
    ) -> BinanceOrderResult:
        """Place a signed MARKET order on the user's Binance Spot / Testnet account."""
        is_crypto, clean_sym = is_crypto_symbol(symbol)
        if not is_crypto:
            return BinanceOrderResult(
                executed=False,
                order_id="NONE",
                symbol=normalize_symbol(symbol, strip_hyphen=True),
                side=side,
                executed_qty=0.0,
                avg_price=reference_price,
                notional_usdt=0.0,
                testnet=testnet,
                status="UNSUPPORTED_SYMBOL",
                error_message=f"Simbol {symbol.upper()} bukan pasangan Crypto yang didukung di Binance Spot.",
            )
        formatted_qty = self.format_order_quantity(quantity, reference_price)
        est_notional = round(formatted_qty * reference_price, 2)

        if self._is_sandbox_demo_key(api_key) and testnet:
            sim_order_id = f"SIM-{int(time.time() * 1000)}"
            return BinanceOrderResult(
                executed=True,
                order_id=sim_order_id,
                symbol=clean_sym,
                side=side,
                executed_qty=formatted_qty,
                avg_price=reference_price,
                notional_usdt=est_notional,
                testnet=True,
                status="FILLED (TESTNET SANDBOX)",
            )

        base_url = self.resolve_base_url(testnet)
        params = {
            "symbol": clean_sym,
            "side": side,
            "type": "MARKET",
            "quantity": f"{formatted_qty:f}".rstrip("0").rstrip("."),
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
        }
        signed_query = self.sign_params(params, api_secret.strip())
        url = f"{base_url}/api/v3/order?{signed_query}"
        headers = {"X-MBX-APIKEY": api_key.strip()}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, headers=headers)
            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    msg = str(err_json.get("msg") or f"HTTP {resp.status_code}")[:160]
                except Exception:
                    msg = f"HTTP {resp.status_code}"
                return BinanceOrderResult(
                    executed=False,
                    order_id="NONE",
                    symbol=clean_sym,
                    side=side,
                    executed_qty=0.0,
                    avg_price=reference_price,
                    notional_usdt=0.0,
                    testnet=testnet,
                    status="REJECTED",
                    error_message=f"Order ditolak Binance ({msg})",
                )

            data = resp.json()
            order_status = str(data.get("status", "FILLED")).upper()
            exec_qty = float(data.get("executedQty", 0.0))
            cum_quote = float(data.get("cummulativeQuoteQty", 0.0))
            if order_status not in ("FILLED", "PARTIALLY_FILLED") or exec_qty <= 0.0:
                return BinanceOrderResult(
                    executed=False,
                    order_id=str(data.get("orderId", "NONE")),
                    symbol=clean_sym,
                    side=side,
                    executed_qty=0.0,
                    avg_price=reference_price,
                    notional_usdt=0.0,
                    testnet=testnet,
                    status=order_status,
                    error_message=f"Order Binance tidak terisi (status={order_status}, executedQty={exec_qty})",
                )
            avg_px = (cum_quote / exec_qty) if (exec_qty > 0 and cum_quote > 0) else reference_price
            return BinanceOrderResult(
                executed=True,
                order_id=str(data.get("orderId", f"ORD-{int(time.time())}")),
                symbol=clean_sym,
                side=side,
                executed_qty=round(exec_qty, 6),
                avg_price=round(avg_px, 4),
                notional_usdt=round(cum_quote if cum_quote > 0 else (exec_qty * reference_price), 2),
                testnet=testnet,
                status=order_status,
            )
        except Exception as exc:
            logger.warning("Binance order execution failed", symbol=clean_sym, error=str(exc))
            return BinanceOrderResult(
                executed=False,
                order_id="NONE",
                symbol=clean_sym,
                side=side,
                executed_qty=0.0,
                avg_price=reference_price,
                notional_usdt=0.0,
                testnet=testnet,
                status="ERROR",
                error_message=f"Gagal mengirim order ke Binance ({type(exc).__name__})",
            )
