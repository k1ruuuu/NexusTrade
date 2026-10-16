from __future__ import annotations

from typing import Literal, Tuple

SentimentLabel = Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]
MacroStance = Literal["HAWKISH", "DOVISH", "NEUTRAL"]

BULLISH_LEXICON: dict[str, float] = {
    "surge": 0.85,
    "surged": 0.85,
    "surges": 0.85,
    "soar": 0.90,
    "soared": 0.90,
    "rally": 0.80,
    "rallies": 0.80,
    "rebound": 0.72,
    "rebounds": 0.72,
    "reclaim": 0.70,
    "reclaims": 0.70,
    "revive": 0.65,
    "revives": 0.65,
    "jump": 0.68,
    "jumps": 0.68,
    "rise": 0.58,
    "rises": 0.58,
    "demand": 0.52,
    "defend": 0.55,
    "support": 0.45,
    "breakout": 0.78,
    "bullish": 0.85,
    "bulls": 0.72,
    "outperform": 0.75,
    "upgrade": 0.75,
    "upgraded": 0.75,
    "beat": 0.70,
    "beats": 0.70,
    "record": 0.65,
    "profit": 0.65,
    "profits": 0.65,
    "growth": 0.60,
    "gain": 0.60,
    "gains": 0.60,
    "spike": 0.68,
    "spikes": 0.68,
    "accumulation": 0.72,
    "inflow": 0.75,
    "inflows": 0.75,
    "institutional": 0.45,
    "adoption": 0.65,
    "approval": 0.80,
    "approved": 0.80,
    "strong": 0.55,
    "expansion": 0.60,
    "upside": 0.70,
    "buy": 0.55,
    "positive": 0.65,
    "recovery": 0.60,
    "dividend": 0.50,
    "high": 0.45,
    "highs": 0.50,
}

BEARISH_LEXICON: dict[str, float] = {
    "plunge": -0.90,
    "plunged": -0.90,
    "plunges": -0.90,
    "crash": -0.95,
    "crashed": -0.95,
    "selloff": -0.85,
    "bearish": -0.85,
    "bears": -0.72,
    "warning": -0.58,
    "downgrade": -0.78,
    "downgraded": -0.78,
    "miss": -0.72,
    "missed": -0.72,
    "misses": -0.72,
    "loss": -0.68,
    "losses": -0.68,
    "deficit": -0.65,
    "decline": -0.65,
    "declined": -0.65,
    "drop": -0.60,
    "dropped": -0.60,
    "outflow": -0.75,
    "outflows": -0.75,
    "liquidation": -0.82,
    "liquidations": -0.82,
    "hack": -0.92,
    "exploit": -0.90,
    "lawsuit": -0.78,
    "investigation": -0.70,
    "ban": -0.85,
    "default": -0.90,
    "bankruptcy": -0.95,
    "recession": -0.80,
    "weak": -0.55,
    "downside": -0.70,
    "negative": -0.65,
    "slump": -0.75,
    "fear": -0.65,
    "slip": -0.60,
    "slips": -0.60,
    "slipped": -0.60,
    "dip": -0.52,
    "dips": -0.52,
    "pullback": -0.55,
    "cautious": -0.48,
    "sell": -0.55,
    "selling": -0.58,
    "pressure": -0.52,
    "correction": -0.62,
    "stall": -0.45,
    "stalls": -0.45,
    "lower": -0.48,
    "fall": -0.58,
    "falls": -0.58,
    "fell": -0.58,
    "tumble": -0.75,
    "tumbles": -0.75,
    "retreat": -0.55,
    "retreats": -0.55,
    "reject": -0.58,
    "rejected": -0.58,
    "rejection": -0.58,
}

BOOTSTRAP_TRAINING_CORPUS: list[tuple[str, int]] = [
    ("Institutional spot Bitcoin ETF inflows surge to record high on strong accumulation", 1),
    ("Company reports strong Q3 profit growth and beats revenue expectations with upside guidance", 1),
    ("Bullish breakout as bulls defend critical support and volume rallies", 1),
    ("Validator upgrades and institutional adoption drive record network expansion", 1),
    ("Central bank gold accumulation supports strong XAUUSD rebound and inflows", 1),
    ("Exchange hack triggers massive liquidation selloff and regulatory lawsuit", -1),
    ("Company misses earnings guidance and reports severe quarterly loss amid slump", -1),
    ("Bearish breakdown as bears target lower support following downgrade warning", -1),
    ("Institutional outflows accelerate amid recession fear and liquidity deficit", -1),
    ("Scheduled maintenance window completed during overnight session with flat volume", 0),
    ("Market participants await central bank minutes as trading range remains unchanged", 0),
    ("Options settlement concludes near equilibrium with balanced order book depth", 0),
]

DEFAULT_ASSET_HEADLINES: dict[str, list[str]] = {
    "BTC": [
        "Institutional spot Bitcoin ETF inflows surge to record high on strong accumulation",
        "Bullish breakout in BTC derivatives as institutional adoption and profit growth expand",
        "Spot Bitcoin demand rallies with strong institutional inflows and positive market structure",
    ],
    "ETH": [
        "Ethereum Layer-2 transaction volume reaches record expansion alongside steady staking inflows",
        "Institutional custody demand for ETH strengthens as DeFi protocol yields stabilize",
        "Gas fee volatility remains neutral during off-peak settlement windows",
    ],
    "SOL": [
        "Solana decentralized exchange volume surges past major competitors amid strong retail activity",
        "Validator client upgrades improve network throughput and institutional staking confidence",
    ],
    "XAUUSD": [
        "Central bank gold accumulation and sovereign safe-haven demand support strong XAUUSD breakout",
        "Real Treasury yield compression drives institutional capital inflows into bullion markets",
        "Spot gold consolidates near record highs as macro inflation expectations remain elevated",
    ],
}


def classify_sentiment_polarity(
    polarity: float,
    threshold: float = 0.15,
) -> SentimentLabel:
    """Classify scalar sentiment polarity in [-1.0, 1.0] into discrete SentimentLabel."""
    if polarity >= threshold:
        return "POSITIVE"
    if polarity <= -threshold:
        return "NEGATIVE"
    return "NEUTRAL"

# -----------------------------------------------------------------------------
# Institutional Macroeconomic "Hawkish vs Dovish" Monetary Policy Lexicon
# -----------------------------------------------------------------------------
# Hawkish = Strong US Inflation/Jobs -> Fed Hikes/Holds -> DXY Up -> Gold (XAUUSD) Down (SELL)
# Dovish = Weak US Inflation/Jobs -> Fed Cuts/Eases -> DXY Down -> Gold (XAUUSD) Up (BUY)

HAWKISH_LEXICON: dict[str, float] = {
    "hot": 0.70,
    "hotter": 0.85,
    "hottest": 0.90,
    "accelerate": 0.75,
    "accelerates": 0.75,
    "accelerated": 0.75,
    "accelerating": 0.75,
    "beat": 0.80,
    "beats": 0.80,
    "surge": 0.85,
    "surges": 0.85,
    "surging": 0.85,
    "sticky": 0.80,
    "robust": 0.75,
    "resilient": 0.70,
    "overheating": 0.90,
    "hike": 0.85,
    "hikes": 0.85,
    "hiking": 0.85,
    "hawkish": 0.90,
    "higher for longer": 0.90,
    "inflation persists": 0.85,
    "tight labor": 0.80,
    "wage growth": 0.75,
    "strong jobs": 0.85,
    "payroll surge": 0.85,
    "cpi beat": 0.90,
    "ppi beat": 0.90,
    "retail sales beat": 0.80,
    "retail sales surge": 0.85,
    "gdp beats": 0.85,
    "gdp surges": 0.85,
    "pmi expands": 0.80,
    "ism beats": 0.80,
    "fomc hawkish": 0.95,
    "powell hawkish": 0.95,
    "rate hike on table": 0.90,
    "imf upgrades us growth": 0.85,
    "imf raises us forecast": 0.85,
    "jobless claims drop": 0.80,
    "quantitative tightening": 0.80,
}

DOVISH_LEXICON: dict[str, float] = {
    "cool": -0.70,
    "cools": -0.75,
    "cooled": -0.75,
    "cooling": -0.80,
    "slow": -0.65,
    "slows": -0.70,
    "slowed": -0.70,
    "slowing": -0.75,
    "slowdown": -0.80,
    "miss": -0.80,
    "misses": -0.80,
    "missed": -0.80,
    "disappoint": -0.75,
    "disappoints": -0.80,
    "disappointing": -0.80,
    "soft": -0.70,
    "soften": -0.75,
    "softens": -0.75,
    "softening": -0.80,
    "weak": -0.75,
    "weaker": -0.80,
    "weakness": -0.80,
    "cut": -0.85,
    "cuts": -0.85,
    "cutting": -0.85,
    "rate cut": -0.90,
    "dovish": -0.90,
    "dovish pivot": -0.95,
    "easing": -0.85,
    "disinflation": -0.85,
    "deflation": -0.80,
    "jobless claims jump": -0.80,
    "jobless claims surge": -0.85,
    "unemployment rises": -0.85,
    "cooling inflation": -0.90,
    "cpi miss": -0.90,
    "ppi miss": -0.90,
    "nfp miss": -0.90,
    "fomc dovish": -0.95,
    "powell dovish": -0.95,
    "dovish pause": -0.90,
    "rate cuts on horizon": -0.85,
    "gdp misses": -0.85,
    "gdp contracts": -0.90,
    "gdp slows": -0.80,
    "retail sales slump": -0.85,
    "retail sales miss": -0.80,
    "pmi contracts": -0.80,
    "pmi misses": -0.80,
    "imf warns global slowdown": -0.85,
    "imf cuts global growth": -0.85,
    "debt distress": -0.80,
    "recession fear": -0.85,
    "recession warning": -0.90,
}

def compute_macro_monetary_bias(text: str) -> Tuple[float, MacroStance]:
    """
    Quantify macroeconomic stance from text:
      - Score > 0: HAWKISH (Strong US inflation / jobs -> USD bullish)
      - Score < 0: DOVISH (Cooling US inflation / jobs -> USD bearish)
      - Score ~ 0: NEUTRAL
    """
    lower_text = text.lower()
    import re

    score = 0.0
    matched_count = 0

    for phrase, wt in HAWKISH_LEXICON.items():
        if " " in phrase:
            if phrase in lower_text:
                score += wt * 1.3
                matched_count += 1
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", lower_text):
                score += wt
                matched_count += 1

    for phrase, wt in DOVISH_LEXICON.items():
        if " " in phrase:
            if phrase in lower_text:
                score += wt * 1.3
                matched_count += 1
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", lower_text):
                score += wt
                matched_count += 1

    if matched_count == 0:
        return 0.0, "NEUTRAL"

    normalized = float(max(-1.0, min(1.0, score / max(1.0, matched_count * 0.75))))
    if normalized >= 0.15:
        return round(normalized, 4), "HAWKISH"
    elif normalized <= -0.15:
        return round(normalized, 4), "DOVISH"
    return round(normalized, 4), "NEUTRAL"


def macro_monetary_to_gold_signal(
    macro_score: float,
    asset_symbol: str = "XAUUSD",
) -> Tuple[float, str]:
    """
    Macro Monetary Policy Transmission to Gold (XAUUSD) & Commodities:
      - Strong US Inflation / Jobs (Hawkish, macro_score > 0)
        -> Fed hikes / holds higher -> DXY strengthens
        -> Gold (XAUUSD) drops -> SELL signal (-1.0)
      - Weak US Inflation / Jobs (Dovish, macro_score < 0)
        -> Fed cuts -> DXY weakens
        -> Gold (XAUUSD) rallies -> BUY signal (+1.0)

    Returns: (gold_sentiment_score [-1.0, +1.0], action_recommendation ['BUY', 'SELL', 'HOLD'])
    """
    clean_sym = asset_symbol.upper().replace("/", "").replace("-", "")
    # Inverse transmission for XAUUSD / Gold and Crypto
    if any(k in clean_sym for k in ("XAU", "GOLD", "BTC", "ETH")):
        # Invert: Hawkish USD (+score) = Bearish Gold (-score)
        gold_score = -float(macro_score)
    else:
        # Direct for USD / DXY
        gold_score = float(macro_score)

    gold_score = float(max(-1.0, min(1.0, gold_score)))
    if gold_score >= 0.18:
        return round(gold_score, 4), "BUY"
    elif gold_score <= -0.18:
        return round(gold_score, 4), "SELL"
    return round(gold_score, 4), "HOLD"
