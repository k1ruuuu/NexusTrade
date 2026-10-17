from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Optional
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
    balance_usdt: float = 0.0
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
    protection_confirmed: bool = True
    emergency_closed: bool = False
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
    Convert asset unit quantity into MetaTrader 5 / Exness lot volume using instrument contract size:
    - Gold (XAUUSD): 100 oz / lot
    - Silver (XAGUSD): 5,000 oz / lot
    - Crude Oil (WTI / USOIL / BRENT / UKOIL): 1,000 bbl / lot
    - 6-letter Forex pairs: 100,000 units / lot
    - Crypto / Stocks / Indices: 1 unit / lot
    Floors to 0.01 lot step so unit-to-lot conversion never rounds up beyond risk budget.
    """
    import math

    clean = normalize_symbol(symbol, strip_hyphen=True).upper()
    if clean.endswith("M") and len(clean) > 4 and symbol.endswith("m"):
        clean = clean[:-1]
    contract_size = get_mt5_contract_size(clean)
    if len(clean) == 6 and clean.isalpha() and not clean.endswith("USD") and clean.startswith("USD"):
        raw_lots = (quantity_units * max(reference_price, 1e-6)) / contract_size
    else:
        raw_lots = quantity_units / max(contract_size, 1.0)

    floored = math.floor(max(0.0, raw_lots) * 100.0 + 1e-9) / 100.0
    return round(max(0.01, min(floored, 50.0)), 2)

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


def get_mt5_pip_size(symbol: str, price: float = 0.0) -> float:
    """
    Return instrument-specific 1.0 pip size in price currency units:
    - Gold (XAUUSD / GOLD): 1 pip = 0.10 ($5.00 move is 50 pips)
    - Silver (XAGUSD / SILVER): 1 pip = 0.01
    - 6-letter FX ending in JPY (USDJPY, EURJPY, GBPJPY): 1 pip = 0.01
    - Standard 6-letter FX (EURUSD, GBPUSD, AUDUSD): 1 pip = 0.0001
    - Crypto (BTCUSD, BTCUSDT, ETHUSD): 1 pip = 1.0 ($50.00 move is 50 pips)
    - Global Indices (US30, US500, USTEC): 1 pip = 1.0 index point
    - Oil (USOIL, UKOIL, WTI, BRENT): 1 pip = 0.01
    """
    sym = normalize_symbol(symbol, strip_hyphen=True).upper()
    if sym.endswith("M") and len(sym) > 4 and symbol.endswith("m"):
        sym = sym[:-1]
    if any(k in sym for k in ("XAU", "GOLD")):
        return 0.10
    if any(k in sym for k in ("XAG", "SILVER")):
        return 0.01
    if any(k in sym for k in ("BTC", "ETH", "SOL", "CRYPTO")):
        return 1.0
    if any(k in sym for k in ("US30", "US500", "USTEC", "SPX", "NDX", "DJI")):
        return 1.0
    if any(k in sym for k in ("USOIL", "UKOIL", "WTI", "BRENT", "OIL")):
        return 0.01
    if len(sym) == 6 and sym.endswith("JPY"):
        return 0.01
    if len(sym) == 6 and any(c in sym for c in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF")):
        return 0.0001
    if price >= 1000.0:
        return 1.0
    if price >= 10.0:
        return 0.10
    return 0.0001

def get_mt5_digits_and_point(symbol: str, price: float = 0.0) -> tuple[int, float]:
    """
    Return standard MetaTrader 5 (digits, point) tuple for any supported instrument:
    - Gold (XAUUSD), Indices, Crypto, Oil: digits = 2, point = 0.01
    - Silver (XAGUSD): digits = 3, point = 0.001
    - 6-letter FX ending in JPY (USDJPY, EURJPY): digits = 3, point = 0.001
    - Standard 6-letter FX (EURUSD, GBPUSD): digits = 5, point = 0.00001
    """
    sym = normalize_symbol(symbol, strip_hyphen=True).upper()
    if sym.endswith("M") and len(sym) > 4 and symbol.endswith("m"):
        sym = sym[:-1]
    if any(k in sym for k in ("XAU", "GOLD", "BTC", "ETH", "SOL", "US30", "US500", "USTEC", "SPX", "NDX", "DJI", "USOIL", "UKOIL", "WTI", "BRENT")):
        return 2, 0.01
    if any(k in sym for k in ("XAG", "SILVER")):
        return 3, 0.001
    if len(sym) == 6 and sym.endswith("JPY"):
        return 3, 0.001
    if len(sym) == 6 and any(c in sym for c in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF")):
        return 5, 0.00001
    if price >= 1000.0:
        return 2, 0.01
    if price >= 20.0:
        return 3, 0.001
    if price >= 0.1:
        return 4, 0.0001
    return 5, 0.00001


def get_mt5_point_size(symbol: str, price: float = 0.0) -> float:
    """Return 1.0 point size."""
    return get_mt5_digits_and_point(symbol, price)[1]

def calculate_ea_sl_tp(
    symbol: str,
    side: Literal["BUY", "SELL"],
    entry_price: float,
    sl_pips: float = 50.0,
    tp_pips: float = 150.0,
    stops_level_pips: float = 3.0,
) -> tuple[float, float, int]:
    """
    Calculate EA Stop-Loss and Take-Profit enforcing Stops Level & Spread Protector:
    - Guarantees minimum distance from market price >= stops_level_pips (Prevents MT5 Error 130).
    - Returns (stop_loss, take_profit, decimals).
    """
    pip_sz = get_mt5_pip_size(symbol, entry_price)
    decimals = 2 if entry_price >= 10.0 else (4 if entry_price >= 0.1 else 5)
    eff_sl_pips = max(float(sl_pips), float(stops_level_pips))
    eff_tp_pips = max(float(tp_pips), float(stops_level_pips))

    sl_dist = eff_sl_pips * pip_sz
    tp_dist = eff_tp_pips * pip_sz

    if side == "BUY":
        sl = round(entry_price - sl_dist, decimals)
        tp = round(entry_price + tp_dist, decimals)
    else:
        sl = round(entry_price + sl_dist, decimals)
        tp = round(max(entry_price * 0.01, entry_price - tp_dist), decimals)
    return sl, tp, decimals


@dataclass
class GridScalperConfig:
    """Configuration for High-Frequency M1 Pending Stop Grid EA (Video Mode)."""
    layers_per_side: int = 5              # 5 Buy Stops + 5 Sell Stops per cycle
    first_offset_pips: float = 10.0       # Distance from current Bid/Ask to Layer 1 ($1.00 on Gold)
    grid_step_pips: float = 10.0          # Distance between consecutive stop layers ($1.00 on Gold)
    take_profit_pips: float = 15.0        # Tight scalping TP per pending order ($1.50 on Gold)
    stop_loss_pips: float = 25.0          # Protective SL per pending order ($2.50 on Gold)
    base_lot: float = 0.01
    max_lot: float = 2.00                 # Video caps at 2.00 lots per order
    compounding_enabled: bool = True
    equity_per_001_lot: float = 15.0      # Dynamic compounding: $10->$0.01, $45->0.03, $120->0.08, $1500->1.00, $3000+->2.00
    target_equity_usd: float = 5000.0     # Auto-stop & close all when equity target is reached
    max_equity_drawdown_pct: float = 0.35 # Emergency basket equity stop


def calculate_compounded_grid_lot(
    equity_usd: float,
    cfg: Optional[GridScalperConfig] = None,
    volume_min: float = 0.01,
    volume_step: float = 0.01,
) -> float:
    """
    Replicate the video's aggressive equity-based lot compounding progression:
    - Equity $10 - $29     -> 0.01 lot
    - Equity $30 - $60     -> 0.03 lot
    - Equity $120 - $250   -> 0.08 lot
    - Equity $1,000+       -> 1.00 lot
    - Equity $3,000+       -> 2.00 lot
    """
    config = cfg or GridScalperConfig()
    if not config.compounding_enabled or equity_usd <= 20.0:
        return config.base_lot
    raw_lot = (equity_usd / max(config.equity_per_001_lot, 1.0)) * 0.01
    stepped = math.floor((raw_lot + 1e-9) / volume_step) * volume_step
    return round(max(volume_min, min(stepped, config.max_lot)), 2)


def build_pending_stop_grid_levels(
    symbol: str,
    bid_price: float,
    ask_price: float,
    cfg: Optional[GridScalperConfig] = None,
) -> list[dict[str, Any]]:
    """
    Generate symmetric BUY_STOP (above Ask) and SELL_STOP (below Bid) trap levels
    normalized for any MT5 instrument (XAUUSD, BTCUSD, US30, EURUSD, etc.).
    """
    config = cfg or GridScalperConfig()
    pip_sz = get_mt5_pip_size(symbol, ask_price)
    _, point_sz = get_mt5_digits_and_point(symbol, ask_price)
    decimals = 2 if point_sz >= 0.01 else (3 if point_sz >= 0.001 else 5)

    first_dist = max(config.first_offset_pips * pip_sz, 3.0 * pip_sz)
    step_dist = max(config.grid_step_pips * pip_sz, 2.0 * pip_sz)
    tp_dist = config.take_profit_pips * pip_sz
    sl_dist = config.stop_loss_pips * pip_sz

    orders: list[dict[str, Any]] = []
    for i in range(config.layers_per_side):
        # BUY_STOP above current Ask
        buy_entry = round(ask_price + first_dist + (i * step_dist), decimals)
        buy_tp = round(buy_entry + tp_dist, decimals) if tp_dist > 0 else 0.0
        buy_sl = round(buy_entry - sl_dist, decimals) if sl_dist > 0 else 0.0
        orders.append({
            "side": "BUY_STOP",
            "op_code": 4,  # MT5 OP_BUYSTOP / ORDER_TYPE_BUY_STOP
            "layer": i + 1,
            "price": buy_entry,
            "stop_loss": buy_sl,
            "take_profit": buy_tp,
        })

        # SELL_STOP below current Bid
        sell_entry = round(bid_price - first_dist - (i * step_dist), decimals)
        sell_tp = round(sell_entry - tp_dist, decimals) if tp_dist > 0 else 0.0
        sell_sl = round(sell_entry + sl_dist, decimals) if sl_dist > 0 else 0.0
        orders.append({
            "side": "SELL_STOP",
            "op_code": 5,  # MT5 OP_SELLSTOP / ORDER_TYPE_SELL_STOP
            "layer": i + 1,
            "price": sell_entry,
            "stop_loss": sell_sl,
            "take_profit": sell_tp,
        })
    return orders
