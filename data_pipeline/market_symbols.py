from __future__ import annotations
import re

SAFE_SYMBOL_RE = re.compile(r"[^A-Z0-9._^-]")
ALLOWED_TIMEFRAMES: set[str] = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1W"}

BINANCE_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1W": "1w",
}

YAHOO_INTERVAL_RANGE_MAP: dict[str, tuple[str, str]] = {
    "1m": ("1m", "5d"),
    "5m": ("5m", "1mo"),
    "15m": ("15m", "1mo"),
    "30m": ("30m", "1mo"),
    "1h": ("60m", "3mo"),
    "4h": ("60m", "3mo"),
    "1d": ("1d", "5y"),
    "1W": ("1wk", "5y"),
}

YAHOO_SYMBOL_MAP: dict[str, str] = {
    # Commodities (Gold, Silver, Platinum, Palladium, Energy, Base Metals)
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",
    "XAGUSD": "SI=F",
    "SILVER": "SI=F",
    "XPTUSD": "PL=F",
    "PLATINUM": "PL=F",
    "XPDUSD": "PA=F",
    "PALLADIUM": "PA=F",
    "WTI": "CL=F",
    "USOIL": "CL=F",
    "BRENT": "BZ=F",
    "UKOIL": "BZ=F",
    "NATGAS": "NG=F",
    "COPPER": "HG=F",
    # Global Indices (US, Europe, Asia-Pacific)
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "US500": "^GSPC",
    "NDX": "^NDX",
    "NAS100": "^NDX",
    "USTEC": "^NDX",
    "DJI": "^DJI",
    "US30": "^DJI",
    "RUT": "^RUT",
    "US2000": "^RUT",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "DAX": "^GDAXI",
    "GER40": "^GDAXI",
    "FTSE": "^FTSE",
    "UK100": "^FTSE",
    "CAC40": "^FCHI",
    "FRA40": "^FCHI",
    "NIKKEI": "^N225",
    "JPN225": "^N225",
    "HSI": "^HSI",
    "HK50": "^HSI",
    "ASX200": "^AXJO",
    "AUS200": "^AXJO",
    "KOSPI": "^KS11",
    "IHSG": "^JKSE",
    "JKSE": "^JKSE",
    "NIFTY": "^NSEI",
    "STI": "^STI",
    # Explicit Major Forex Pairs
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
}

FOREX_CURRENCIES: set[str] = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "AUD",
    "NZD",
    "CAD",
    "CHF",
    "SGD",
    "HKD",
    "CNY",
    "IDR",
    "MYR",
    "THB",
    "INR",
    "KRW",
    "SEK",
    "NOK",
    "DKK",
    "ZAR",
    "MXN",
    "TRY",
    "BRL",
    "PLN",
}

KNOWN_CRYPTO_BASES: set[str] = {
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "DOT",
    "LINK",
    "SHIB",
    "PEPE",
    "UNI",
    "LTC",
    "BCH",
    "AAVE",
    "SUSHI",
    "CRV",
    "GRT",
    "BAT",
    "XTZ",
    "YFI",
    "SUI",
    "TON",
    "ARB",
    "OP",
    "APT",
    "NEAR",
    "INJ",
    "PAXG",
    "TRUMP",
    "FET",
    "RENDER",
    "WIF",
    "BONK",
    "FLOKI",
    "SEI",
    "TIA",
    "JUP",
    "ENA",
    "ONDO",
    "PENDLE",
    "WLD",
    "FIL",
    "ATOM",
    "ETC",
    "XLM",
    "ALGO",
    "VET",
    "ICP",
    "HBAR",
    "KAS",
    "TAO",
    "ORDI",
    "STX",
    "IMX",
    "RUNE",
    "SAND",
    "MANA",
    "AXS",
    "GALA",
    "LDO",
    "MKR",
    "COMP",
    "SNX",
    "1INCH",
    "ENS",
    "ZEC",
    "DASH",
    "XMR",
    "EOS",
    "TRX",
    "SKY",
}


def normalize_symbol(symbol: str, strip_hyphen: bool = False) -> str:
    """Normalize exchange-prefixed or slash-delimited ticker to canonical uppercase symbol."""
    cleaned = symbol.upper().split(":")[-1].replace("/", "").strip()
    cleaned = SAFE_SYMBOL_RE.sub("", cleaned).replace("..", "")
    if strip_hyphen:
        cleaned = cleaned.replace("-", "")
    return cleaned[:24] or "BTCUSDT"


def normalize_timeframe(timeframe: str, default: str = "15m") -> str:
    """Validate timeframe against allowed intervals."""
    tf = timeframe.strip()
    return tf if tf in ALLOWED_TIMEFRAMES else default


def resolve_yahoo_ticker(symbol: str) -> str:
    """Map canonical symbol, Crypto pair, or 6-letter ISO Forex pair to Yahoo Finance ticker."""
    clean_sym = normalize_symbol(symbol)
    if clean_sym in YAHOO_SYMBOL_MAP:
        return YAHOO_SYMBOL_MAP[clean_sym]
    if (
        len(clean_sym) == 6
        and clean_sym[:3] in FOREX_CURRENCIES
        and clean_sym[3:] in FOREX_CURRENCIES
    ):
        return f"{clean_sym}=X"
    for suffix in ("USDT", "USDC", "USD"):
        if clean_sym.endswith(suffix) and len(clean_sym) > len(suffix):
            base = clean_sym[: -len(suffix)]
            if base in KNOWN_CRYPTO_BASES:
                return f"{base}-USD"
    if clean_sym in KNOWN_CRYPTO_BASES:
        return f"{clean_sym}-USD"
    return clean_sym


def is_crypto_symbol(symbol: str) -> tuple[bool, str]:
    """Return (is_crypto, normalized_binance_pair)."""
    clean = normalize_symbol(symbol, strip_hyphen=True)
    if clean in YAHOO_SYMBOL_MAP:
        return False, clean
    if (
        len(clean) == 6
        and clean[:3] in FOREX_CURRENCIES
        and clean[3:] in FOREX_CURRENCIES
    ):
        return False, clean
    if clean.endswith("USDT") or clean.endswith("USDC"):
        return True, clean
    if clean.endswith("USD"):
        base = clean[:-3]
        if base in KNOWN_CRYPTO_BASES:
            return True, f"{base}USDT"
    if clean in KNOWN_CRYPTO_BASES:
        return True, f"{clean}USDT"
    return False, clean
