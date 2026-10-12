from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from data_pipeline.market_symbols import normalize_symbol


@dataclass
class BinanceAccountVerification:
    """Result of verifying a user's Exness Go / MT5 account credentials."""

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
    """Result of placing an automated market order on a user's Exness Go / MT5 account."""

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

EXNESS_SERVER_HOSTS_CACHE: dict[str, list[tuple[str, int]]] = {
    "EXNESS-MT5TRIAL7": [
        ("52.221.81.217", 443),
        ("13.114.223.90", 443),
        ("47.245.95.97", 443),
        ("13.232.4.164", 443),
        ("47.81.14.118", 443),
        ("8.213.236.229", 443),
    ],
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


def compute_anchored_sl_tp(
    side: Literal["BUY", "SELL"],
    anchor_price: float,
    sl_dist: float,
    tp_dist: float,
) -> tuple[float, float, int]:
    """
    Compute Stop-Loss and Take-Profit anchored to the broker's live Bid/Ask quote price.
    Returns `(stop_loss, take_profit, decimals)`.
    """
    decimals = 2 if anchor_price >= 10.0 else (4 if anchor_price >= 0.1 else 5)
    if side == "BUY":
        eff_sl = round(anchor_price - sl_dist, decimals) if sl_dist > 0 else 0.0
        eff_tp = round(anchor_price + tp_dist, decimals) if tp_dist > 0 else 0.0
    else:
        eff_sl = round(anchor_price + sl_dist, decimals) if sl_dist > 0 else 0.0
        eff_tp = round(max(anchor_price * 0.05, anchor_price - tp_dist), decimals) if tp_dist > 0 else 0.0
    return eff_sl, eff_tp, decimals


def anchor_targets_to_fill_price(
    side: Literal["BUY", "SELL"],
    fill_price: float,
    ref_entry: float,
    ref_sl: float,
    ref_tp1: float,
    ref_tp2: float,
) -> tuple[float, float, float]:
    """
    Anchor (stop_loss, take_profit_1, take_profit_2) distances to the actual broker fill price (`fill_price`)
    so that Spot vs Futures basis differences never invert or distort SL/TP levels.
    """
    if fill_price <= 0 or ref_entry <= 0:
        return ref_sl, ref_tp1, ref_tp2
    sl_dist = abs(ref_entry - ref_sl) if ref_sl > 0 else 0.0
    tp1_dist = abs(ref_tp1 - ref_entry) if ref_tp1 > 0 else 0.0
    tp2_dist = abs(ref_tp2 - ref_entry) if ref_tp2 > 0 else 0.0
    eff_sl, eff_tp1, decimals = compute_anchored_sl_tp(side, fill_price, sl_dist, tp1_dist)
    _, eff_tp2, _ = compute_anchored_sl_tp(side, fill_price, sl_dist, tp2_dist)
    return (
        eff_sl if ref_sl > 0 else 0.0,
        eff_tp1 if ref_tp1 > 0 else 0.0,
        round(eff_tp2, decimals) if ref_tp2 > 0 else 0.0,
    )


def get_mt5_contract_size(symbol: str) -> float:
    """
    Return standard MetaTrader 5 / Exness contract size (units per 1.00 standard lot):
    - Gold (XAUUSD): 100 Troy Ounces
    - Silver (XAGUSD): 5,000 Troy Ounces
    - Crude Oil (WTI / USOIL / BRENT / UKOIL): 1,000 Barrels
    - 6-letter Forex pairs (EURUSD, GBPUSD, USDJPY, etc.): 100,000 currency units
    - Crypto (BTCUSD, ETHUSD), US Stocks, Global Indices: 1 unit per lot
    """
    clean = normalize_symbol(symbol, strip_hyphen=True).upper()
    if clean.endswith("M") and len(clean) > 4 and symbol.endswith("m"):
        clean = clean[:-1]
    if clean in ("XAUUSD", "GOLD"):
        return 100.0
    if clean in ("XAGUSD", "SILVER"):
        return 5000.0
    if clean in ("WTI", "USOIL", "BRENT", "UKOIL"):
        return 1000.0
    fx_ccys = {"EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"}
    if len(clean) == 6 and clean[:3] in fx_ccys and clean[3:] in fx_ccys:
        return 100_000.0
    return 1.0


def estimate_mt5_pnl_usdt(
    symbol: str,
    side: Literal["BUY", "SELL"],
    entry_price: float,
    exit_price: float,
    lots: float,
) -> float:
    """
    Calculate MT5 position PnL in USD accounting for instrument contract size per lot.
    """
    clean = normalize_symbol(symbol, strip_hyphen=True).upper()
    if clean.endswith("M") and len(clean) > 4 and symbol.endswith("m"):
        clean = clean[:-1]
    contract_size = get_mt5_contract_size(clean)
    price_diff = (exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)
    raw_pnl = price_diff * float(lots) * contract_size
    # Convert quote-currency PnL to USD when USD is the base currency (e.g. USDJPY, USDCAD, USDCHF)
    if len(clean) == 6 and clean.startswith("USD") and not clean.endswith("USD"):
        raw_pnl = raw_pnl / max(exit_price, 1e-6)
    return round(raw_pnl, 2)
