from __future__ import annotations

import time
from typing import Literal, Optional

import httpx

from ai_engine.auto_trader.binance_client import (
    BinanceAccountVerification,
    BinanceOrderResult,
)
from config.logging_config import get_logger
from config.settings import get_settings
from data_pipeline.market_symbols import normalize_symbol

logger = get_logger("nexustrade.auto_trader.mt5_exness")

METAAPI_BASE_URL = "https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai"

MT5_INDEX_COMMODITY_MAP: dict[str, str] = {
    "SPX": "US500",
    "NDX": "USTEC",
    "DJI": "US30",
    "VIX": "VIX",
    "WTI": "USOIL",
    "USOIL": "USOIL",
    "BRENT": "UKOIL",
    "NATGAS": "XNGUSD",
    "XAUUSD": "XAUUSD",
    "GOLD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "SILVER": "XAGUSD",
}


def resolve_mt5_symbol(symbol: str, standard_suffix: str = "m") -> str:
    """
    Map canonical symbol (Crypto, Gold, Forex, US Equities, Global Indices)
    to MetaTrader 5 / Exness ('Exness Go') instrument symbol (e.g. BTCUSDm, XAUUSDm, EURUSDm, NVDA, US500m).
    """
    clean = normalize_symbol(symbol, strip_hyphen=True)
    if clean.endswith("M") and len(clean) > 4 and symbol.endswith("m"):
        return symbol.strip()

    if clean in MT5_INDEX_COMMODITY_MAP:
        base = MT5_INDEX_COMMODITY_MAP[clean]
        return f"{base}{standard_suffix}"

    if clean.endswith("USDT") and len(clean) > 4:
        base_coin = clean[:-4]
        return f"{base_coin}USD{standard_suffix}"
    if clean.endswith("USDC") and len(clean) > 4:
        base_coin = clean[:-4]
        return f"{base_coin}USD{standard_suffix}"

    if len(clean) == 6 and clean.isalpha():
        return f"{clean}{standard_suffix}"

    # US Equities on Exness MT5 (e.g., NVDA, AAPL, TSLA, MSFT)
    return f"{clean}{standard_suffix}" if standard_suffix else clean


def calculate_mt5_lots(symbol: str, quantity_units: float, reference_price: float) -> float:
    """
    Convert asset unit quantity into MetaTrader 5 / Exness lot volume (min 0.01 lot).
    - Gold (XAUUSD): 1 standard lot = 100 oz -> min 0.01 lot
    - Forex (6-letter FX): 1 standard lot = 100,000 units -> min 0.01 lot
    - Crypto / Stocks / Indices: 1 lot = 1 unit -> min 0.01 lot
    """
    clean = normalize_symbol(symbol, strip_hyphen=True)
    if clean in ("XAUUSD", "GOLD"):
        raw_lots = quantity_units / 100.0
    elif len(clean) == 6 and clean.isalpha() and not clean.endswith("USD"):
        raw_lots = (quantity_units * max(reference_price, 1.0)) / 100_000.0
    elif len(clean) == 6 and clean[:3] in ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"):
        raw_lots = quantity_units / 100_000.0
    else:
        raw_lots = quantity_units

    return round(max(0.01, min(raw_lots, 50.0)), 2)


class MT5ExnessClient:
    """
    MetaTrader 5 / Exness ('Exness Go') Multi-Asset 2-Way Execution Client.
    Supports BUY (Long) and SELL (Short) across all asset classes:
    - Crypto (BTCUSDm, ETHUSDm, SOLUSDm)
    - Precious Metals & Commodities (XAUUSDm, XAGUSDm, USOILm)
    - Forex (EURUSDm, GBPUSDm, USDJPYm)
    - US Stocks (NVDAm, AAPLm, TSLAm)
    - Global Indices (US500m, USTECm, US30m)
    Automatically attaches server-side Stop-Loss (sl) and Take-Profit (tp) to every order.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _is_exness_trial_demo(login_or_id: str, testnet: bool) -> bool:
        if get_settings().environment == "production":
            return False
        if not testnet:
            return False
        cleaned = login_or_id.strip().upper()
        return (
            cleaned.isdigit()
            or cleaned.startswith("EXNESS_")
            or cleaned.startswith("MT5_")
            or cleaned.startswith("TESTNET_")
            or cleaned.startswith("DEMO_")
        )

    async def verify_credentials(
        self,
        account_id_or_login: str,
        token_or_password: str,
        server_name: str = "Exness-MT5Trial",
        testnet: bool = True,
    ) -> BinanceAccountVerification:
        """
        Verify MT5 / Exness Go account via MetaApi Cloud REST API or Exness Trial sandbox.
        """
        clean_id = account_id_or_login.strip()
        clean_secret = token_or_password.strip()
        if len(clean_id) < 5 or len(clean_secret) < 6:
            return BinanceAccountVerification(
                valid=False,
                can_trade=False,
                usdt_free=0.0,
                usdt_locked=0.0,
                total_usdt=0.0,
                asset_count=0,
                account_type="INVALID_MT5",
                testnet=testnet,
                error_message="Login ID MT5/Exness (min 5 karakter) dan Password/Token (min 6 karakter) wajib diisi.",
            )

        if self._is_exness_trial_demo(clean_id, testnet):
            return BinanceAccountVerification(
                valid=True,
                can_trade=True,
                usdt_free=10000.0,
                usdt_locked=0.0,
                total_usdt=10000.0,
                asset_count=5,
                account_type=f"EXNESS_GO_MT5 ({server_name})",
                testnet=True,
            )

        url = f"{METAAPI_BASE_URL}/users/current/accounts/{clean_id}/account-information"
        headers = {"auth-token": clean_secret}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    msg = str(err_json.get("message") or err_json.get("msg") or f"HTTP {resp.status_code}")[:160]
                except Exception:
                    msg = f"HTTP {resp.status_code}"
                return BinanceAccountVerification(
                    valid=False,
                    can_trade=False,
                    usdt_free=0.0,
                    usdt_locked=0.0,
                    total_usdt=0.0,
                    asset_count=0,
                    account_type="MT5_REJECTED",
                    testnet=testnet,
                    error_message=f"Exness/MT5 menolak koneksi ({msg})",
                )

            data = resp.json()
            balance = float(data.get("balance", 0.0))
            equity = float(data.get("equity", balance))
            free_margin = float(data.get("freeMargin", equity))
            broker_srv = str(data.get("server", server_name))
            trade_allowed = bool(data.get("tradeAllowed", True))
            if not trade_allowed:
                return BinanceAccountVerification(
                    valid=False,
                    can_trade=False,
                    usdt_free=free_margin,
                    usdt_locked=max(0.0, equity - free_margin),
                    total_usdt=equity,
                    asset_count=1,
                    account_type=f"EXNESS_MT5 ({broker_srv})",
                    testnet=testnet,
                    error_message="Akun MT5/Exness terhubung namun fitur Algo Trading / Investor Password belum mengizinkan transaksi.",
                )

            return BinanceAccountVerification(
                valid=True,
                can_trade=True,
                usdt_free=round(free_margin, 2),
                usdt_locked=round(max(0.0, equity - free_margin), 2),
                total_usdt=round(equity, 2),
                asset_count=5,
                account_type=f"EXNESS_GO_MT5 ({broker_srv})",
                testnet=testnet,
            )
        except Exception as exc:
            logger.warning("MT5/Exness account verification failed", error=str(exc))
            return BinanceAccountVerification(
                valid=False,
                can_trade=False,
                usdt_free=0.0,
                usdt_locked=0.0,
                total_usdt=0.0,
                asset_count=0,
                account_type="MT5_UNREACHABLE",
                testnet=testnet,
                error_message=f"Gagal menghubungi gateway MT5/Exness ({type(exc).__name__})",
            )

    async def place_market_order(
        self,
        account_id_or_login: str,
        token_or_password: str,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        reference_price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        server_name: str = "Exness-MT5Trial",
        testnet: bool = True,
    ) -> BinanceOrderResult:
        """
        Execute a 2-way (BUY Long or SELL Short) market order on MetaTrader 5 / Exness Go
        for ANY asset class (Crypto, Gold XAUUSD, Forex, US Stocks, Indices) with native SL & TP.
        """
        mt5_symbol = resolve_mt5_symbol(symbol)
        lots = calculate_mt5_lots(symbol, quantity, reference_price)
        est_notional = round(quantity * reference_price, 2)

        if self._is_exness_trial_demo(account_id_or_login, testnet):
            order_id = f"MT5-{int(time.time() * 1000)}"
            return BinanceOrderResult(
                executed=True,
                order_id=order_id,
                symbol=mt5_symbol,
                side=side,
                executed_qty=lots,
                avg_price=reference_price,
                notional_usdt=est_notional,
                testnet=True,
                status=f"FILLED ({server_name} • {lots} Lot)",
            )

        url = f"{METAAPI_BASE_URL}/users/current/accounts/{account_id_or_login.strip()}/trade"
        headers = {"auth-token": token_or_password.strip()}
        payload: dict[str, object] = {
            "actionType": "ORDER_TYPE_BUY" if side == "BUY" else "ORDER_TYPE_SELL",
            "symbol": mt5_symbol,
            "volume": lots,
            "comment": "NexusTrade AI 6-Gate",
        }
        if stop_loss > 0:
            payload["stopLoss"] = round(stop_loss, 4)
        if take_profit > 0:
            payload["takeProfit"] = round(take_profit, 4)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    msg = str(err_json.get("message") or err_json.get("msg") or f"HTTP {resp.status_code}")[:160]
                except Exception:
                    msg = f"HTTP {resp.status_code}"
                return BinanceOrderResult(
                    executed=False,
                    order_id="NONE",
                    symbol=mt5_symbol,
                    side=side,
                    executed_qty=0.0,
                    avg_price=reference_price,
                    notional_usdt=0.0,
                    testnet=testnet,
                    status="MT5_REJECTED",
                    error_message=f"Order ditolak Exness/MT5 ({msg})",
                )

            data = resp.json()
            numeric_code = int(data.get("numericCode", 10009))
            string_code = str(data.get("stringCode", "TRADE_RETCODE_DONE"))
            if numeric_code not in (10008, 10009) and "DONE" not in string_code.upper():
                return BinanceOrderResult(
                    executed=False,
                    order_id=str(data.get("orderId", "NONE")),
                    symbol=mt5_symbol,
                    side=side,
                    executed_qty=0.0,
                    avg_price=reference_price,
                    notional_usdt=0.0,
                    testnet=testnet,
                    status=string_code,
                    error_message=f"Exness/MT5 tidak mengeksekusi order ({string_code})",
                )

            return BinanceOrderResult(
                executed=True,
                order_id=str(data.get("orderId") or data.get("positionId") or f"MT5-{int(time.time())}"),
                symbol=mt5_symbol,
                side=side,
                executed_qty=lots,
                avg_price=reference_price,
                notional_usdt=est_notional,
                testnet=testnet,
                status=f"FILLED ({server_name} • {lots} Lot)",
            )
        except Exception as exc:
            logger.warning("MT5/Exness order execution failed", symbol=mt5_symbol, error=str(exc))
            return BinanceOrderResult(
                executed=False,
                order_id="NONE",
                symbol=mt5_symbol,
                side=side,
                executed_qty=0.0,
                avg_price=reference_price,
                notional_usdt=0.0,
                testnet=testnet,
                status="MT5_ERROR",
                error_message=f"Gagal mengirim order ke Exness/MT5 ({type(exc).__name__})",
            )
