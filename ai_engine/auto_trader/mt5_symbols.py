from __future__ import annotations

from typing import Literal

from data_pipeline.market_symbols import normalize_symbol

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
