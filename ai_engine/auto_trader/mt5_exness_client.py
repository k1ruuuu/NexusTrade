from __future__ import annotations

import time
from typing import Any, Literal, Optional

import httpx

from ai_engine.auto_trader.mt5_symbols import (
    EXNESS_SERVER_HOSTS_CACHE,
    MT5_INDEX_COMMODITY_MAP,
    BinanceAccountVerification,
    BinanceOrderResult,
    anchor_targets_to_fill_price,
    calculate_mt5_lots,
    compute_anchored_sl_tp,
    estimate_mt5_pnl_usdt,
    get_mt5_contract_size,
    resolve_mt5_symbol,
)
from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger("nexustrade.auto_trader.mt5_exness")

MT5_REST_DIRECT_URL = "https://mt5.mtapi.io"

# Active mt5.mtapi.io session token cache: (login, server_upper) -> (token, timestamp)
_MT5_SESSION_TOKENS: dict[tuple[str, str], tuple[str, float]] = {}

__all__ = [
    "BinanceAccountVerification",
    "BinanceOrderResult",
    "EXNESS_SERVER_HOSTS_CACHE",
    "MT5ExnessClient",
    "MT5_INDEX_COMMODITY_MAP",
    "anchor_targets_to_fill_price",
    "calculate_mt5_lots",
    "compute_anchored_sl_tp",
    "estimate_mt5_pnl_usdt",
    "get_mt5_contract_size",
    "resolve_mt5_symbol",
]

class MT5ExnessClient:
    """
    Real-Time MetaTrader 5 / Exness ('Exness Go') Multi-Asset 2-Way Execution Client.
    Connects directly to Exness MT5 servers (`Exness-MT5Trial...` / `Exness-MT5Real...`)
    via the MT5 Direct REST Bridge (`https://mt5.mtapi.io`) using `(MT5_LOGIN, MT5_PASSWORD, SERVER_NAME)`:
      - Discovers live Exness server access endpoints (`/Search?company=Exness`)
      - Establishes authenticated MT5 session (`/Connect`)
      - Reads 100% real-time balance, equity, free margin, and leverage (`/AccountSummary`)
      - Reads live open positions (`/OpenedOrders`)
      - Executes real-time 2-way BUY (Long) & SELL (Short) market orders with SL & TP (`/OrderSendSafe`)
      - Closes positions directly on Exness MT5 (`/OrderCloseSafe`)
    """

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _is_exness_trial_demo(login_or_id: str, testnet: bool) -> bool:
        """
        Return True ONLY for explicit sandbox test prefixes (EXNESS_DEMO_, MT5_DEMO_, TESTNET_DEMO_).
        Real numeric Exness MT5 Login IDs (e.g. '434355782') are NEVER intercepted here!
        """
        if get_settings().environment == "production":
            return False
        if not testnet:
            return False
        cleaned = login_or_id.strip().upper()
        return (
            cleaned.startswith("EXNESS_DEMO_")
            or cleaned.startswith("MT5_DEMO_")
            or cleaned.startswith("TESTNET_")
            or cleaned.startswith("DEMO_")
        )

    async def _resolve_server_hosts(
        self,
        client: httpx.AsyncClient,
        server_name: str,
    ) -> list[tuple[str, int]]:
        srv_upper = server_name.strip().upper()
        if srv_upper in EXNESS_SERVER_HOSTS_CACHE:
            return EXNESS_SERVER_HOSTS_CACHE[srv_upper]

        company = server_name.split("-")[0].strip() or "Exness"
        hosts: list[tuple[str, int]] = []
        try:
            r = await client.get(
                f"{MT5_REST_DIRECT_URL}/Search",
                params={"company": company},
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    for comp in data:
                        for srv in comp.get("results", []):
                            name = str(srv.get("name", "")).strip()
                            if name.upper() == srv_upper or srv_upper in name.upper():
                                for addr in srv.get("access", []):
                                    addr_s = str(addr).strip()
                                    if addr_s.startswith("10.") or addr_s.startswith("192.168."):
                                        continue
                                    if ":" in addr_s:
                                        h, p_str = addr_s.rsplit(":", 1)
                                        port = int(p_str) if p_str.isdigit() else 443
                                        hosts.append((h, port))
                                    else:
                                        hosts.append((addr_s, 443))
        except Exception as exc:
            logger.debug("MT5 server host search fallback triggered", server=server_name, error=str(exc))

        if hosts:
            EXNESS_SERVER_HOSTS_CACHE[srv_upper] = hosts
            return hosts

        return EXNESS_SERVER_HOSTS_CACHE["EXNESS-MT5TRIAL7"]

    async def _get_or_connect_session_token(
        self,
        client: httpx.AsyncClient,
        login: str,
        password: str,
        server_name: str,
        force_reconnect: bool = False,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Obtain an active MT5 session token for `(login, password, server_name)`.
        Returns `(session_token, error_message)`.
        """
        clean_login = login.strip()
        clean_pass = password.strip()
        clean_srv = server_name.strip() or "Exness-MT5Trial7"
        cache_key = (clean_login, clean_srv.upper())

        if not force_reconnect and cache_key in _MT5_SESSION_TOKENS:
            cached_tok, ts = _MT5_SESSION_TOKENS[cache_key]
            if time.time() - ts < 600:
                chk = await client.get(
                    f"{MT5_REST_DIRECT_URL}/CheckConnect",
                    params={"id": cached_tok},
                )
                if chk.status_code == 200 and "true" in chk.text.lower():
                    return cached_tok, None

        hosts = await self._resolve_server_hosts(client, clean_srv)
        last_err = f"Tidak dapat terhubung ke server {clean_srv}"

        for host, port in hosts[:2]:
            try:
                resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/Connect",
                    params={
                        "user": clean_login,
                        "password": clean_pass,
                        "host": host,
                        "port": port,
                    },
                    timeout=3.5,
                )
                body = resp.text.strip()
                if resp.status_code == 200 and body and not body.startswith("{"):
                    tok = body.strip('"')
                    _MT5_SESSION_TOKENS[cache_key] = (tok, time.time())
                    return tok, None
                try:
                    err_j = resp.json()
                    msg = str(err_j.get("message") or err_j.get("code") or body)[:160]
                except Exception:
                    msg = body[:160]
                last_err = msg
                if any(k in msg.lower() for k in ("invalid account", "authorization failed", "wrong password")):
                    return None, f"Login atau Password MT5 salah ({msg})"
            except Exception as exc:
                last_err = type(exc).__name__

        return None, last_err

    async def verify_credentials(
        self,
        account_id_or_login: str,
        token_or_password: str,
        server_name: str = "Exness-MT5Trial7",
        testnet: bool = True,
    ) -> BinanceAccountVerification:
        """
        Verify MT5 / Exness Go account in real-time by connecting directly to the Exness MT5 server
        and querying `/AccountSummary`.
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
                balance_usdt=10000.0,
                asset_count=5,
                account_type=f"EXNESS_GO_MT5_SANDBOX ({server_name})",
                testnet=True,
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, err = await self._get_or_connect_session_token(
                    client, clean_id, clean_secret, server_name
                )
                if not tok:
                    return BinanceAccountVerification(
                        valid=False,
                        can_trade=False,
                        usdt_free=0.0,
                        usdt_locked=0.0,
                        total_usdt=0.0,
                        asset_count=0,
                        account_type="MT5_CONNECT_FAILED",
                        testnet=testnet,
                        error_message=f"Gagal terhubung ke Exness ({server_name}): {err}",
                    )

                sum_resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/AccountSummary",
                    params={"id": tok},
                )
                if sum_resp.status_code != 200:
                    return BinanceAccountVerification(
                        valid=False,
                        can_trade=False,
                        usdt_free=0.0,
                        usdt_locked=0.0,
                        total_usdt=0.0,
                        asset_count=0,
                        account_type="MT5_SUMMARY_FAILED",
                        testnet=testnet,
                        error_message=f"Gagal membaca saldo Exness (HTTP {sum_resp.status_code})",
                    )

                data = sum_resp.json()
                balance = float(data.get("balance", 0.0))
                equity = float(data.get("equity", balance))
                free_margin = float(data.get("freeMargin", equity))
                margin_used = float(data.get("margin", max(0.0, equity - free_margin)))
                leverage = int(data.get("leverage", 2000))
                is_investor = bool(data.get("isInvestor", False))
                acc_type = str(data.get("type", "demo")).upper()

                if is_investor:
                    return BinanceAccountVerification(
                        valid=False,
                        can_trade=False,
                        usdt_free=round(free_margin, 2),
                        usdt_locked=round(margin_used, 2),
                        total_usdt=round(equity, 2),
                        balance_usdt=round(balance, 2),
                        asset_count=1,
                        account_type=f"EXNESS_MT5_INVESTOR ({server_name})",
                        testnet=testnet,
                        error_message="Akun terhubung dengan Investor Password (Read-Only). Gunakan Master Trading Password agar bisa Auto-Trade.",
                    )

                return BinanceAccountVerification(
                    valid=True,
                    can_trade=True,
                    usdt_free=round(free_margin, 2),
                    usdt_locked=round(margin_used, 2),
                    total_usdt=round(equity, 2),
                    balance_usdt=round(balance, 2),
                    asset_count=5,
                    account_type=f"EXNESS_GO_MT5 ({server_name} • {acc_type} 1:{leverage})",
                    testnet=testnet,
                )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            logger.warning("MT5/Exness account verification failed", error=err_msg)
            return BinanceAccountVerification(
                valid=False,
                can_trade=False,
                usdt_free=0.0,
                usdt_locked=0.0,
                total_usdt=0.0,
                asset_count=0,
                account_type="MT5_UNREACHABLE",
                testnet=testnet,
                error_message=f"Gagal menghubungi server Exness/MT5 ({type(exc).__name__})",
            )

    async def get_live_open_orders(
        self,
        account_id_or_login: str,
        token_or_password: str,
        server_name: str = "Exness-MT5Trial7",
    ) -> Optional[list[dict[str, Any]]]:
        """Fetch live open positions directly from the user's Exness MT5 account (`/OpenedOrders`)."""
        if self._is_exness_trial_demo(account_id_or_login, True):
            return None
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, _ = await self._get_or_connect_session_token(
                    client, account_id_or_login, token_or_password, server_name
                )
                if not tok:
                    return None
                resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/OpenedOrders",
                    params={"id": tok},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        return data
        except Exception as exc:
            logger.debug("Could not fetch live MT5 open orders", error=str(exc))
        return None


    async def get_closed_orders_map(
        self,
        account_id_or_login: str,
        token_or_password: str,
        server_name: str = "Exness-MT5Trial7",
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch closed order history from Exness MT5 (`/ClosedOrders`) and return
        a mapping of `ticket_str -> closed_order_dict` (`closePrice`, `profit`, `stopLoss`, `takeProfit`).
        """
        if self._is_exness_trial_demo(account_id_or_login, True):
            return {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, _ = await self._get_or_connect_session_token(
                    client, account_id_or_login, token_or_password, server_name
                )
                if not tok:
                    return {}
                resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/ClosedOrders",
                    params={"id": tok},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        out: dict[str, dict[str, Any]] = {}
                        for item in data:
                            if isinstance(item, dict):
                                t_str = str(item.get("ticket") or item.get("order") or "")
                                if t_str:
                                    out[t_str] = item
                        return out
        except Exception as exc:
            logger.debug("Could not fetch MT5 closed orders", error=str(exc))
        return {}
    async def close_mt5_ticket(
        self,
        account_id_or_login: str,
        token_or_password: str,
        ticket: str,
        lots: float,
        server_name: str = "Exness-MT5Trial7",
    ) -> bool:
        """Close an open MT5 ticket by ticket number via `/OrderCloseSafe`."""
        if self._is_exness_trial_demo(account_id_or_login, True):
            return True
        if not ticket.isdigit():
            return False
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, _ = await self._get_or_connect_session_token(
                    client, account_id_or_login, token_or_password, server_name
                )
                if not tok:
                    return False
                resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/OrderCloseSafe",
                    params={
                        "id": tok,
                        "ticket": int(ticket),
                        "lots": lots,
                        "slippage": 50,
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def modify_mt5_ticket(
        self,
        account_id_or_login: str,
        token_or_password: str,
        ticket: str,
        stop_loss: float,
        take_profit: float,
        server_name: str = "Exness-MT5Trial7",
    ) -> bool:
        """Update Stop-Loss (Trailing Stop) and Take-Profit on an open MT5 ticket via `/OrderModifySafe`."""
        if self._is_exness_trial_demo(account_id_or_login, True):
            return True
        if not ticket.isdigit():
            return False
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, _ = await self._get_or_connect_session_token(
                    client, account_id_or_login, token_or_password, server_name
                )
                if not tok:
                    return False
                resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/OrderModifySafe",
                    params={
                        "id": tok,
                        "ticket": int(ticket),
                        "stoploss": round(stop_loss, 4),
                        "takeprofit": round(take_profit, 4),
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False

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
        server_name: str = "Exness-MT5Trial7",
        testnet: bool = True,
        custom_lot: Optional[float] = None,
    ) -> BinanceOrderResult:
        """
        Execute a real-time 2-way (BUY Long or SELL Short) market order on MetaTrader 5 / Exness Go
        via `/OrderSendSafe`, anchoring SL & TP distances to the broker's live Bid/Ask quote.
        """
        mt5_symbol_m = resolve_mt5_symbol(symbol, standard_suffix="m")
        mt5_symbol_raw = resolve_mt5_symbol(symbol, standard_suffix="")
        if custom_lot is not None and custom_lot > 0:
            lots = round(max(0.01, min(float(custom_lot), 50.0)), 2)
        else:
            lots = calculate_mt5_lots(symbol, quantity, reference_price)
        est_notional = round(lots * get_mt5_contract_size(symbol) * reference_price, 2)
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

        sl_dist = abs(reference_price - stop_loss) if (stop_loss > 0 and reference_price > 0) else 0.0
        tp_dist = abs(take_profit - reference_price) if (take_profit > 0 and reference_price > 0) else 0.0
        op_code = 0 if side == "BUY" else 1
        last_err = "Order ditolak"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, conn_err = await self._get_or_connect_session_token(
                    client, account_id_or_login, token_or_password, server_name
                )
                if not tok:
                    return BinanceOrderResult(
                        executed=False,
                        order_id="NONE",
                        symbol=mt5_symbol_m,
                        side=side,
                        executed_qty=0.0,
                        avg_price=reference_price,
                        notional_usdt=0.0,
                        testnet=testnet,
                        status="MT5_DISCONNECTED",
                        error_message=f"Gagal terhubung ke Exness ({server_name}): {conn_err}",
                    )

                for candidate_sym in (mt5_symbol_m, mt5_symbol_raw):
                    # Fetch live broker Bid/Ask quote so SL & TP are anchored to Exness's exact quote
                    broker_px = reference_price
                    try:
                        q_resp = await client.get(
                            f"{MT5_REST_DIRECT_URL}/GetQuote",
                            params={"id": tok, "symbol": candidate_sym},
                        )
                        if q_resp.status_code == 200:
                            q_data = q_resp.json()
                            bid_px = float(q_data.get("bid", 0.0))
                            ask_px = float(q_data.get("ask", 0.0))
                            if side == "BUY" and ask_px > 0:
                                broker_px = ask_px
                            elif side == "SELL" and bid_px > 0:
                                broker_px = bid_px
                    except Exception:
                        pass
                    eff_sl, eff_tp, decimals = compute_anchored_sl_tp(
                        side, broker_px, sl_dist, tp_dist
                    )

                    params: dict[str, Any] = {
                        "id": tok,
                        "symbol": candidate_sym,
                        "operation": op_code,
                        "volume": lots,
                        "slippage": 50,
                        "comment": "NexusTradeAI",
                    }
                    if eff_sl > 0:
                        params["stoploss"] = eff_sl
                    if eff_tp > 0:
                        params["takeprofit"] = eff_tp

                    resp = await client.get(
                        f"{MT5_REST_DIRECT_URL}/OrderSendSafe",
                        params=params,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        ticket = str(data.get("ticket") or data.get("order") or f"MT5-{int(time.time())}")
                        open_px = float(data.get("openPrice") or broker_px)
                        exec_lots = float(data.get("lots") or lots)
                        if (
                            ticket.isdigit()
                            and abs(open_px - broker_px) > 1e-6
                            and (sl_dist > 0 or tp_dist > 0)
                        ):
                            mod_sl, mod_tp, decimals = compute_anchored_sl_tp(
                                side, open_px, sl_dist, tp_dist
                            )
                            mod_resp = await client.get(
                                f"{MT5_REST_DIRECT_URL}/OrderModifySafe",
                                params={
                                    "id": tok,
                                    "ticket": int(ticket),
                                    "stoploss": mod_sl,
                                    "takeprofit": mod_tp,
                                },
                            )
                            if mod_resp.status_code != 200 and sl_dist > 0:
                                close_ok = False
                                try:
                                    c_resp = await client.get(
                                        f"{MT5_REST_DIRECT_URL}/OrderCloseSafe",
                                        params={"id": tok, "ticket": int(ticket), "lots": exec_lots, "slippage": 50},
                                    )
                                    close_ok = c_resp.status_code == 200
                                except Exception:
                                    close_ok = False
                                return BinanceOrderResult(
                                    executed=False,
                                    order_id=ticket,
                                    symbol=candidate_sym,
                                    side=side,
                                    executed_qty=exec_lots,
                                    avg_price=round(open_px, decimals),
                                    notional_usdt=round(
                                        exec_lots * get_mt5_contract_size(candidate_sym) * open_px, 2
                                    ),
                                    testnet=testnet,
                                    status="UNPROTECTED_CLOSED_CONFIRMED" if close_ok else "FILLED_UNPROTECTED",
                                    error_message=(
                                        "Gagal mengonfirmasi SL/TP di broker; posisi ditutup darurat dan terkonfirmasi."
                                        if close_ok
                                        else "PERINGATAN: Order terisi tanpa SL/TP dan penutupan darurat belum terkonfirmasi."
                                    ),
                                    protection_confirmed=False,
                                    emergency_closed=close_ok,
                                )
                        return BinanceOrderResult(
                            executed=True,
                            order_id=ticket,
                            symbol=candidate_sym,
                            side=side,
                            executed_qty=exec_lots,
                            avg_price=round(open_px, decimals),
                            notional_usdt=round(
                                exec_lots * get_mt5_contract_size(candidate_sym) * open_px, 2
                            ),
                            testnet=testnet,
                            status=f"FILLED ({server_name} • Ticket #{ticket} • {exec_lots} Lot)",
                        )
                    try:
                        err_json = resp.json()
                        last_err = str(
                            err_json.get("message") or err_json.get("code") or f"HTTP {resp.status_code}"
                        )[:160]
                    except Exception:
                        last_err = f"HTTP {resp.status_code}"

                    # If rejected due to stops on tight broker spread, retry market order without initial SL/TP then modify
                    if "stop" in last_err.lower() or "invalid" in last_err.lower():
                        params_no_sl = {
                            "id": tok,
                            "symbol": candidate_sym,
                            "operation": op_code,
                            "volume": lots,
                            "slippage": 50,
                            "comment": "NexusTradeAI",
                        }
                        retry_resp = await client.get(
                            f"{MT5_REST_DIRECT_URL}/OrderSendSafe",
                            params=params_no_sl,
                        )
                        if retry_resp.status_code == 200:
                            data = retry_resp.json()
                            ticket_int = int(data.get("ticket") or 0)
                            open_px = float(data.get("openPrice") or broker_px)
                            exec_lots = float(data.get("lots") or lots)
                            if ticket_int > 0 and (eff_sl > 0 or eff_tp > 0):
                                mod_sl, mod_tp, _ = compute_anchored_sl_tp(
                                    side, open_px, sl_dist, tp_dist
                                )
                                mod_resp = await client.get(
                                    f"{MT5_REST_DIRECT_URL}/OrderModifySafe",
                                    params={
                                        "id": tok,
                                        "ticket": ticket_int,
                                        "stoploss": mod_sl,
                                        "takeprofit": mod_tp,
                                    },
                                )
                                if mod_resp.status_code != 200:
                                    close_ok = False
                                    try:
                                        c_resp = await client.get(
                                            f"{MT5_REST_DIRECT_URL}/OrderCloseSafe",
                                            params={"id": tok, "ticket": ticket_int, "lots": exec_lots, "slippage": 50},
                                        )
                                        close_ok = c_resp.status_code == 200
                                    except Exception:
                                        close_ok = False
                                    return BinanceOrderResult(
                                        executed=False,
                                        order_id=str(ticket_int),
                                        symbol=candidate_sym,
                                        side=side,
                                        executed_qty=exec_lots,
                                        avg_price=round(open_px, decimals),
                                        notional_usdt=round(
                                            exec_lots * get_mt5_contract_size(candidate_sym) * open_px, 2
                                        ),
                                        testnet=testnet,
                                        status="UNPROTECTED_CLOSED_CONFIRMED" if close_ok else "FILLED_UNPROTECTED",
                                        error_message=(
                                            "Gagal memasang proteksi SL/TP setelah fill; posisi ditutup darurat dan terkonfirmasi."
                                            if close_ok
                                            else "PERINGATAN: Order terisi tanpa SL/TP dan penutupan darurat belum terkonfirmasi."
                                        ),
                                        protection_confirmed=False,
                                        emergency_closed=close_ok,
                                    )
                            return BinanceOrderResult(
                                executed=True,
                                order_id=str(ticket_int),
                                symbol=candidate_sym,
                                side=side,
                                executed_qty=exec_lots,
                                avg_price=round(open_px, decimals),
                                notional_usdt=round(exec_lots * get_mt5_contract_size(candidate_sym) * open_px, 2),
                                testnet=testnet,
                                status=f"FILLED ({server_name} • Ticket #{ticket_int} • {exec_lots} Lot)",
                            )
                    if candidate_sym == mt5_symbol_m and ("symbol" in last_err.lower() or "quote" in last_err.lower()):
                        continue
                    break

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
                error_message=f"Exness/MT5 menolak order ({last_err})",
            )
        except Exception as exc:
            logger.warning("MT5/Exness order execution failed; initiating reconciliation", symbol=mt5_symbol_m, error=str(exc))
            # NT-05 / R-07 / Fix 3: Reconcile live orders handling orderType and openPrice without crashing
            try:
                live_orders = await self.get_live_open_orders(
                    account_id_or_login, token_or_password, server_name
                )
                for cand in (live_orders or []):
                    if not isinstance(cand, dict):
                        continue
                    cand_sym = str(cand.get("symbol") or "").strip()
                    raw_type = str(cand.get("orderType") or cand.get("type") or cand.get("side") or "").upper()
                    cand_side = "BUY" if ("BUY" in raw_type or raw_type == "0") else ("SELL" if ("SELL" in raw_type or raw_type == "1") else "")
                    if cand_sym in (mt5_symbol_m, mt5_symbol_raw) and cand_side == side:
                        ticket_str = str(cand.get("ticket") or cand.get("order") or "")
                        open_px = float(cand.get("openPrice") or cand.get("price_open") or reference_price)
                        exec_lots = float(cand.get("lots") or lots)
                        if ticket_str:
                            logger.info("Reconciled filled order after transport timeout", ticket=ticket_str)
                            return BinanceOrderResult(
                                executed=True,
                                order_id=ticket_str,
                                symbol=cand_sym or mt5_symbol_m,
                                side=side,
                                executed_qty=exec_lots,
                                avg_price=open_px,
                                notional_usdt=round(exec_lots * get_mt5_contract_size(cand_sym or symbol) * open_px, 2),
                                testnet=testnet,
                                status=f"RECONCILED_FILLED (Ticket #{ticket_str})",
                            )
            except Exception as rec_err:
                logger.debug("Reconciliation query after timeout failed", error=str(rec_err))
            return BinanceOrderResult(
                executed=False,
                order_id="NONE",
                symbol=mt5_symbol_m,
                side=side,
                executed_qty=0.0,
                avg_price=reference_price,
                notional_usdt=0.0,
                testnet=testnet,
                status="MT5_TIMEOUT_UNCONFIRMED",
                error_message=f"Transport timeout ke Exness/MT5 ({type(exc).__name__}); rekonsiliasi tidak menemukan tiket baru.",
            )

    async def place_pending_stop_order(
        self,
        account_id_or_login: str,
        token_or_password: str,
        symbol: str,
        order_type: Literal["BUY_STOP", "SELL_STOP"],
        price: float,
        lots: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        server_name: str = "Exness-MT5Trial7",
        comment: str = "NexusTrade Grid Trap",
    ) -> BinanceOrderResult:
        """Place a BUY_STOP or SELL_STOP pending order on MetaTrader 5 / Exness."""
        clean_sym = resolve_mt5_symbol(symbol, standard_suffix="m")
        if self._is_exness_trial_demo(account_id_or_login, True):
            order_id = f"PEND-{int(time.time() * 1000)}"
            return BinanceOrderResult(
                executed=True,
                order_id=order_id,
                symbol=clean_sym,
                side="BUY" if order_type == "BUY_STOP" else "SELL",
                executed_qty=lots,
                avg_price=price,
                notional_usdt=round(lots * get_mt5_contract_size(clean_sym) * price, 2),
                testnet=True,
                status=f"PLACED ({order_type} @ {price:,.2f})",
            )

        op_code = 4 if order_type == "BUY_STOP" else 5
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, conn_err = await self._get_or_connect_session_token(
                    client, account_id_or_login, token_or_password, server_name
                )
                if not tok:
                    return BinanceOrderResult(
                        executed=False,
                        order_id="",
                        symbol=clean_sym,
                        side="BUY" if order_type == "BUY_STOP" else "SELL",
                        executed_qty=lots,
                        avg_price=price,
                        notional_usdt=0.0,
                        testnet=False,
                        status="REJECTED_LOGIN_FAILED",
                        error_message=conn_err or "Koneksi ke Exness gagal",
                    )
                resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/OrderSendSafe",
                    params={
                        "id": tok,
                        "symbol": clean_sym,
                        "operation": op_code,
                        "volume": round(lots, 2),
                        "price": round(price, 4),
                        "stoploss": round(stop_loss, 4),
                        "takeprofit": round(take_profit, 4),
                        "comment": comment,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json() if resp.text.startswith("{") else {}
                    tkt = str(data.get("ticket") or data.get("order") or int(time.time() * 1000))
                    return BinanceOrderResult(
                        executed=True,
                        order_id=tkt,
                        symbol=clean_sym,
                        side="BUY" if order_type == "BUY_STOP" else "SELL",
                        executed_qty=lots,
                        avg_price=price,
                        notional_usdt=round(lots * get_mt5_contract_size(clean_sym) * price, 2),
                        testnet=False,
                        status=f"PLACED ({order_type} @ {price:,.2f})",
                    )
        except Exception as exc:
            logger.warning("Error placing pending stop order", error=str(exc))
        return BinanceOrderResult(
            executed=False,
            order_id="",
            symbol=clean_sym,
            side="BUY" if order_type == "BUY_STOP" else "SELL",
            executed_qty=lots,
            avg_price=price,
            notional_usdt=0.0,
            testnet=False,
            status="REJECTED_EXCEPTION",
        )

    async def delete_pending_order(
        self,
        account_id_or_login: str,
        token_or_password: str,
        ticket: str,
        server_name: str = "Exness-MT5Trial7",
    ) -> bool:
        """Cancel/delete an untriggered pending order via `/OrderDelete`."""
        if self._is_exness_trial_demo(account_id_or_login, True):
            return True
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                tok, _ = await self._get_or_connect_session_token(
                    client, account_id_or_login, token_or_password, server_name
                )
                if not tok:
                    return False
                resp = await client.get(
                    f"{MT5_REST_DIRECT_URL}/OrderDeleteSafe",
                    params={"id": tok, "ticket": int(ticket)},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def close_all_and_delete_pending(
        self,
        account_id_or_login: str,
        token_or_password: str,
        symbol: str = "XAUUSD",
        server_name: str = "Exness-MT5Trial7",
    ) -> dict[str, Any]:
        """
        Replicate the video's mass operations:
        - 'Tutup Semua Posisi' (Close all open market positions).
        - 'Hapus Semua Order' (Delete all untriggered pending trap orders).
        """
        clean_sym = normalize_symbol(symbol)
        closed_positions = 0
        deleted_orders = 0

        if self._is_exness_trial_demo(account_id_or_login, True):
            return {
                "success": True,
                "positions_closed": 1,
                "pending_deleted": 10,
                "symbol": clean_sym,
                "message": f"Seluruh posisi dan pending order {clean_sym} berhasil dibersihkan.",
            }

        try:
            opened = await self.get_live_open_orders(account_id_or_login, token_or_password, server_name)
            for ord_data in (opened or []):
                o_sym = str(ord_data.get("symbol") or "")
                tkt = str(ord_data.get("ticket") or "")
                if not tkt or (clean_sym not in o_sym.upper()):
                    continue
                raw_type = str(ord_data.get("orderType") or ord_data.get("type") or "").upper()
                lots = float(ord_data.get("lots") or 0.01)

                if "STOP" in raw_type or "LIMIT" in raw_type:
                    # Pending order -> delete
                    ok = await self.delete_pending_order(account_id_or_login, token_or_password, tkt, server_name)
                    if ok:
                        deleted_orders += 1
                else:
                    # Market position -> close
                    ok = await self.close_mt5_ticket(account_id_or_login, token_or_password, tkt, lots, server_name)
                    if ok:
                        closed_positions += 1
        except Exception as exc:
            logger.warning("Error during close_all_and_delete_pending", error=str(exc))

        return {
            "success": True,
            "positions_closed": closed_positions,
            "pending_deleted": deleted_orders,
            "symbol": clean_sym,
            "message": f"Selesai: {closed_positions} posisi ditutup, {deleted_orders} pending order dihapus.",
        }
