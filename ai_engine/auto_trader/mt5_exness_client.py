from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Optional
import uuid

import httpx

from ai_engine.auto_trader.binance_client import (
    BinanceAccountVerification,
    BinanceOrderResult,
)
from config.logging_config import get_logger
from config.settings import get_settings
from data_pipeline.market_symbols import normalize_symbol

logger = get_logger("nexustrade.auto_trader.mt5_exness")

METAAPI_PROVISIONING_URL = "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai"
METAAPI_DEFAULT_CLIENT_URL = "https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai"
METAAPI_REGIONS: tuple[str, ...] = ("new-york", "london", "singapore")

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
    to MetaTrader 5 / Exness ('Exness Go') instrument symbol (e.g. BTCUSDm, XAUUSDm, EURUSDm, NVDAm, US500m).
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
    Real-Time MetaTrader 5 / Exness ('Exness Go') Multi-Asset 2-Way Execution Client via MetaApi Cloud REST Bridge.
    Supports two connection modes in `/setup exness`:
      1. Direct MetaApi `(METAAPI_ACCOUNT_ID, METAAPI_TOKEN)`
      2. Auto-Provisioning with Exness `(MT5_LOGIN_ID, MT5_PASSWORD, Exness-MT5Trial...)` + `METAAPI_TOKEN`
    Reads 100% real-time balance/equity and executes live BUY (Long) & SELL (Short) orders with SL & TP.
    """

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _is_exness_trial_demo(login_or_id: str, testnet: bool) -> bool:
        """
        Return True ONLY for explicit sandbox test prefixes (EXNESS_DEMO_, MT5_DEMO_, TESTNET_DEMO_).
        Real numeric Exness MT5 Login IDs (e.g. '198273645') are NEVER intercepted here!
        """
        if get_settings().environment == "production":
            return False
        if not testnet:
            return False
        cleaned = login_or_id.strip().upper()
        return (
            cleaned.startswith("EXNESS_DEMO_")
            or cleaned.startswith("MT5_DEMO_")
            or cleaned.startswith("TESTNET_DEMO_")
        )

    @staticmethod
    def _looks_like_jwt_token(value: str) -> bool:
        stripped = value.strip()
        return stripped.startswith("eyJ") and len(stripped) > 40

    @staticmethod
    def _candidate_client_urls(region: Optional[str] = None) -> list[str]:
        cfg_region = (region or get_settings().metaapi_region or "new-york").strip().lower()
        urls = [f"https://mt-client-api-v1.{cfg_region}.agiliumtrade.ai"]
        for r in METAAPI_REGIONS:
            candidate = f"https://mt-client-api-v1.{r}.agiliumtrade.ai"
            if candidate not in urls:
                urls.append(candidate)
        urls.append(METAAPI_DEFAULT_CLIENT_URL)
        return urls

    async def provision_or_resolve_account(
        self,
        mt5_login: str,
        mt5_password: str,
        server_name: str,
        metaapi_token: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Automatically find or provision an Exness MT5 account on MetaApi Cloud using
        `(mt5_login, mt5_password, server_name)` and deploy it for real-time trading.
        Returns `(metaapi_account_id, region, error_message)`.
        """
        clean_login = mt5_login.strip()
        clean_pass = mt5_password.strip()
        clean_srv = server_name.strip() or "Exness-MT5Trial"
        token = metaapi_token.strip()
        headers = {"auth-token": token}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # 1. Check if this MT5 login is already provisioned in the user's MetaApi account
                list_resp = await client.get(
                    f"{METAAPI_PROVISIONING_URL}/users/current/accounts",
                    headers=headers,
                )
                account_id: Optional[str] = None
                region: str = get_settings().metaapi_region or "new-york"
                state: str = "DEPLOYED"

                if list_resp.status_code == 200:
                    items = list_resp.json()
                    if isinstance(items, list):
                        for item in items:
                            if str(item.get("login", "")).strip() == clean_login:
                                account_id = str(item.get("_id") or item.get("id"))
                                region = str(item.get("region") or region)
                                state = str(item.get("state", "DEPLOYED"))
                                break
                elif list_resp.status_code in (401, 403):
                    return None, None, "METAAPI_TOKEN tidak valid atau kadaluarsa (HTTP 401/403 dari MetaApi)."

                # 2. If not yet provisioned, create the MT5 account on MetaApi Cloud
                if not account_id:
                    create_headers = {
                        "auth-token": token,
                        "transaction-id": uuid.uuid4().hex,
                    }
                    create_payload = {
                        "name": f"NexusTrade-{clean_login}",
                        "type": "cloud-g2",
                        "login": clean_login,
                        "password": clean_pass,
                        "server": clean_srv,
                        "platform": "mt5",
                        "magic": 202610,
                    }
                    create_resp = await client.post(
                        f"{METAAPI_PROVISIONING_URL}/users/current/accounts",
                        headers=create_headers,
                        json=create_payload,
                    )
                    if create_resp.status_code not in (200, 201, 202):
                        try:
                            err_j = create_resp.json()
                            msg = str(err_j.get("message") or err_j.get("msg") or f"HTTP {create_resp.status_code}")[:160]
                        except Exception:
                            msg = f"HTTP {create_resp.status_code}"
                        return None, None, f"Gagal mendaftarkan akun MT5 ke MetaApi ({msg})"

                    created_data = create_resp.json()
                    account_id = str(created_data.get("id") or created_data.get("_id") or "")
                    state = str(created_data.get("state", "CREATED"))

                if not account_id:
                    return None, None, "MetaApi tidak mengembalikan Account ID."

                # 3. Deploy account if not yet DEPLOYED
                if state != "DEPLOYED":
                    await client.post(
                        f"{METAAPI_PROVISIONING_URL}/users/current/accounts/{account_id}/deploy",
                        headers=headers,
                    )

                # 4. Poll briefly for broker connection readiness
                for _ in range(8):
                    st_resp = await client.get(
                        f"{METAAPI_PROVISIONING_URL}/users/current/accounts/{account_id}",
                        headers=headers,
                    )
                    if st_resp.status_code == 200:
                        st_data = st_resp.json()
                        region = str(st_data.get("region") or region)
                        conn_st = str(st_data.get("connectionStatus", "")).upper()
                        if conn_st == "CONNECTED":
                            break
                    await asyncio.sleep(1.5)

                return account_id, region, None
        except Exception as exc:
            logger.warning("MetaApi auto-provisioning failed", login=clean_login, error=str(exc))
            return None, None, f"Gagal menghubungi server provisioning MetaApi ({type(exc).__name__})"

    async def verify_credentials(
        self,
        account_id_or_login: str,
        token_or_password: str,
        server_name: str = "Exness-MT5Trial",
        testnet: bool = True,
        metaapi_token_override: Optional[str] = None,
    ) -> BinanceAccountVerification:
        """
        Verify MT5 / Exness Go account in real-time via MetaApi Cloud:
        - If `account_id_or_login` is a numeric MT5 Login ID (e.g. '198273645') and `token_or_password`
          is the MT5 password, automatically provisions/resolves the MetaApi Account ID using `METAAPI_TOKEN`.
        - If `account_id_or_login` is already a MetaApi Account ID and `token_or_password` is a MetaApi JWT token,
          queries `/account-information` directly.
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
                account_type=f"EXNESS_GO_MT5_SANDBOX ({server_name})",
                testnet=True,
            )

        settings = get_settings()
        eff_metaapi_token = (
            metaapi_token_override
            or (clean_secret if self._looks_like_jwt_token(clean_secret) else settings.metaapi_token)
        ).strip()

        target_account_id = clean_id
        target_region: Optional[str] = settings.metaapi_region

        # If user passed numeric Exness MT5 Login + MT5 Password, auto-provision via MetaApi
        if clean_id.isdigit() and not self._looks_like_jwt_token(clean_secret):
            if not eff_metaapi_token:
                return BinanceAccountVerification(
                    valid=False,
                    can_trade=False,
                    usdt_free=0.0,
                    usdt_locked=0.0,
                    total_usdt=0.0,
                    asset_count=0,
                    account_type="METAAPI_TOKEN_REQUIRED",
                    testnet=testnet,
                    error_message=(
                        f"Agar akun Exness ({clean_id} @ {server_name}) terhubung REAL-TIME dari Linux, "
                        f"isi METAAPI_TOKEN (gratis di https://app.metaapi.cloud) pada file .env atau gunakan format: "
                        f"/setup exness <METAAPI_ACCOUNT_ID> <METAAPI_TOKEN> {server_name}"
                    ),
                )
            prov_id, prov_region, prov_err = await self.provision_or_resolve_account(
                mt5_login=clean_id,
                mt5_password=clean_secret,
                server_name=server_name,
                metaapi_token=eff_metaapi_token,
            )
            if prov_err or not prov_id:
                return BinanceAccountVerification(
                    valid=False,
                    can_trade=False,
                    usdt_free=0.0,
                    usdt_locked=0.0,
                    total_usdt=0.0,
                    asset_count=0,
                    account_type="MT5_PROVISION_FAILED",
                    testnet=testnet,
                    error_message=prov_err or "Gagal menghubungkan Login MT5 ke MetaApi.",
                )
            target_account_id = prov_id
            target_region = prov_region

        auth_token = eff_metaapi_token or clean_secret
        headers = {"auth-token": auth_token}
        last_err = "Server MetaApi tidak merespons"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                for base_url in self._candidate_client_urls(target_region):
                    url = f"{base_url}/users/current/accounts/{target_account_id}/account-information"
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
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
                                error_message="Akun Exness/MT5 terhubung namun menggunakan Investor Password (Read-Only). Gunakan Master Trading Password.",
                            )
                        return BinanceAccountVerification(
                            valid=True,
                            can_trade=True,
                            usdt_free=round(free_margin, 2),
                            usdt_locked=round(max(0.0, equity - free_margin), 2),
                            total_usdt=round(equity, 2),
                            asset_count=5,
                            account_type=f"EXNESS_GO_MT5 ({broker_srv} • ID:{target_account_id[:8]})",
                            testnet=testnet,
                        )
                    try:
                        err_json = resp.json()
                        last_err = str(err_json.get("message") or err_json.get("msg") or f"HTTP {resp.status_code}")[:160]
                    except Exception:
                        last_err = f"HTTP {resp.status_code}"
                    if resp.status_code in (401, 403):
                        break

            return BinanceAccountVerification(
                valid=False,
                can_trade=False,
                usdt_free=0.0,
                usdt_locked=0.0,
                total_usdt=0.0,
                asset_count=0,
                account_type="MT5_REJECTED",
                testnet=testnet,
                error_message=f"Exness/MT5 menolak koneksi ({last_err})",
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
        Execute a real-time 2-way (BUY Long or SELL Short) market order on MetaTrader 5 / Exness Go
        for ANY asset class (Crypto, Gold XAUUSD, Forex, US Stocks, Indices) with native SL & TP.
        Automatically tries standard Exness 'm' suffix (e.g. XAUUSDm) and raw symbol (XAUUSD).
        """
        mt5_symbol_m = resolve_mt5_symbol(symbol, standard_suffix="m")
        mt5_symbol_raw = resolve_mt5_symbol(symbol, standard_suffix="")
        lots = calculate_mt5_lots(symbol, quantity, reference_price)
        est_notional = round(quantity * reference_price, 2)

        if self._is_exness_trial_demo(account_id_or_login, testnet):
            order_id = f"MT5-{int(time.time() * 1000)}"
            return BinanceOrderResult(
                executed=True,
                order_id=order_id,
                symbol=mt5_symbol_m,
                side=side,
                executed_qty=lots,
                avg_price=reference_price,
                notional_usdt=est_notional,
                testnet=True,
                status=f"FILLED ({server_name} • {lots} Lot)",
            )

        settings = get_settings()
        clean_id = account_id_or_login.strip()
        clean_secret = token_or_password.strip()
        auth_token = (
            clean_secret if self._looks_like_jwt_token(clean_secret) else settings.metaapi_token
        ).strip()

        target_account_id = clean_id
        target_region: Optional[str] = settings.metaapi_region
        if clean_id.isdigit() and not self._looks_like_jwt_token(clean_secret) and auth_token:
            prov_id, prov_reg, _ = await self.provision_or_resolve_account(
                clean_id, clean_secret, server_name, auth_token
            )
            if prov_id:
                target_account_id = prov_id
                target_region = prov_reg

        headers = {"auth-token": auth_token or clean_secret}
        last_err = "Order ditolak"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                for base_url in self._candidate_client_urls(target_region):
                    url = f"{base_url}/users/current/accounts/{target_account_id}/trade"
                    for candidate_sym in (mt5_symbol_m, mt5_symbol_raw):
                        payload: dict[str, Any] = {
                            "actionType": "ORDER_TYPE_BUY" if side == "BUY" else "ORDER_TYPE_SELL",
                            "symbol": candidate_sym,
                            "volume": lots,
                            "comment": "NexusTrade AI 6-Gate",
                        }
                        if stop_loss > 0:
                            payload["stopLoss"] = round(stop_loss, 4)
                        if take_profit > 0:
                            payload["takeProfit"] = round(take_profit, 4)

                        resp = await client.post(url, headers=headers, json=payload)
                        if resp.status_code != 200:
                            try:
                                err_json = resp.json()
                                last_err = str(
                                    err_json.get("message") or err_json.get("msg") or f"HTTP {resp.status_code}"
                                )[:160]
                            except Exception:
                                last_err = f"HTTP {resp.status_code}"
                            if "symbol" in last_err.lower() and candidate_sym == mt5_symbol_m:
                                continue
                            break

                        data = resp.json()
                        numeric_code = int(data.get("numericCode", 10009))
                        string_code = str(data.get("stringCode", "TRADE_RETCODE_DONE"))
                        if numeric_code in (10008, 10009) or "DONE" in string_code.upper():
                            return BinanceOrderResult(
                                executed=True,
                                order_id=str(
                                    data.get("orderId") or data.get("positionId") or f"MT5-{int(time.time())}"
                                ),
                                symbol=candidate_sym,
                                side=side,
                                executed_qty=lots,
                                avg_price=reference_price,
                                notional_usdt=est_notional,
                                testnet=testnet,
                                status=f"FILLED ({server_name} • {lots} Lot)",
                            )
                        last_err = f"{string_code} (code={numeric_code})"
                        if "SYMBOL" in string_code.upper() and candidate_sym == mt5_symbol_m:
                            continue
                        return BinanceOrderResult(
                            executed=False,
                            order_id=str(data.get("orderId", "NONE")),
                            symbol=candidate_sym,
                            side=side,
                            executed_qty=0.0,
                            avg_price=reference_price,
                            notional_usdt=0.0,
                            testnet=testnet,
                            status=string_code,
                            error_message=f"Exness/MT5 tidak mengeksekusi order ({last_err})",
                        )

            return BinanceOrderResult(
                executed=False,
                order_id="NONE",
                symbol=mt5_symbol_m,
                side=side,
                executed_qty=0.0,
                avg_price=reference_price,
                notional_usdt=0.0,
                testnet=testnet,
                status="MT5_REJECTED",
                error_message=f"Order ditolak Exness/MT5 ({last_err})",
            )
        except Exception as exc:
            logger.warning("MT5/Exness order execution failed", symbol=mt5_symbol_m, error=str(exc))
            return BinanceOrderResult(
                executed=False,
                order_id="NONE",
                symbol=mt5_symbol_m,
                side=side,
                executed_qty=0.0,
                avg_price=reference_price,
                notional_usdt=0.0,
                testnet=testnet,
                status="MT5_ERROR",
                error_message=f"Gagal mengirim order ke Exness/MT5 ({type(exc).__name__})",
            )
