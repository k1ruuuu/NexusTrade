from __future__ import annotations

import html

from ai_engine.models.ensemble_agent import SignalDecision


def render_progress_bar(value: float, length: int = 10) -> str:
    """Render an ASCII block progress bar for value in [0.0, 1.0]."""
    clamped = min(max(float(value), 0.0), 1.0)
    filled = int(round(clamped * length))
    return "█" * filled + "░" * (length - filled)


def format_price(value: float) -> str:
    """Format price with human-friendly decimal precision based on magnitude."""
    abs_val = abs(float(value))
    if abs_val >= 100.0:
        return f"${value:,.2f}"
    if abs_val >= 1.0:
        return f"${value:,.4f}".rstrip("0").rstrip(".")
    if abs_val > 0:
        return f"${value:,.6f}".rstrip("0").rstrip(".")
    return "$0.00"


def format_pct_diff(target: float, entry: float) -> str:
    """Return signed percentage difference from entry price."""
    if entry <= 0:
        return "0.00%"
    diff_pct = ((target - entry) / entry) * 100.0
    return f"{diff_pct:+.2f}%"


def format_badge(action: str, confidence: float) -> str:
    """Return institutional status badge with human-readable action label."""
    act = action.upper()
    if act == "BUY":
        return "🟢 <b>STRONG BUY (Beli Kuat)</b>" if confidence >= 0.72 else "🟢 <b>BUY (Beli)</b>"
    if act == "SELL":
        return "🔴 <b>STRONG SELL (Jual Kuat)</b>" if confidence >= 0.72 else "🔴 <b>SELL (Jual)</b>"
    return "🟡 <b>NEUTRAL / HOLD (Tunggu Dulu)</b>"


def describe_tech_score(s_tech: float, rsi: float) -> str:
    if s_tech >= 0.50:
        trend = "Tren Naik Kuat"
    elif s_tech >= 0.15:
        trend = "Tren Cenderung Naik"
    elif s_tech <= -0.50:
        trend = "Tren Turun Kuat"
    elif s_tech <= -0.15:
        trend = "Tren Cenderung Turun"
    else:
        trend = "Arah Mendatar"

    if rsi >= 70.0:
        rsi_note = "Jenuh Beli"
    elif rsi <= 30.0:
        rsi_note = "Jenuh Jual"
    else:
        rsi_note = "Normal"
    return f"{trend}, RSI {rsi:.0f} ({rsi_note})"


def describe_sentiment_score(s_sent: float) -> str:
    if s_sent >= 0.35:
        return "Berita Sangat Positif"
    if s_sent >= 0.15:
        return "Berita Cenderung Positif"
    if s_sent <= -0.35:
        return "Berita Sangat Negatif"
    if s_sent <= -0.15:
        return "Berita Cenderung Negatif"
    return "Berita Netral / Seimbang"


def describe_volatility_score(s_vol: float, regime: str) -> str:
    reg = regime.upper()
    if reg == "LOW" or s_vol < 0.25:
        return "Tenang / Risiko Rendah"
    if reg == "NORMAL" or s_vol < 0.60:
        return "Stabil / Normal"
    if reg == "ELEVATED":
        return "Gejolak Cukup Tinggi"
    return "Ekstrem / Bahaya"


def build_human_signal_summary(decision: SignalDecision) -> str:
    if not decision.approved:
        return (
            f"⚠️ <b>Sinyal ditahan otomatis</b> demi keamanan modal Anda "
            f"({html.escape(decision.rejection_reason or 'Batas risiko terlampaui')})."
        )
    act = decision.action.upper()
    conf_pct = decision.confidence * 100.0
    if act == "BUY":
        strength = "sangat kuat" if conf_pct >= 72 else "cukup positif"
        return (
            f"💡 <b>Kesimpulan:</b> Peluang <b>BELI</b> {strength} ({conf_pct:.0f}%). "
            f"Patuhi <i>Stop Loss</i> &amp; target di bawah."
        )
    if act == "SELL":
        strength = "sangat kuat" if conf_pct >= 72 else "cukup kuat"
        return (
            f"💡 <b>Kesimpulan:</b> Tekanan <b>JUAL</b> {strength} ({conf_pct:.0f}%). "
            f"Waspadai koreksi &amp; patuhi <i>Stop Loss</i>."
        )
    return (
        f"💡 <b>Kesimpulan:</b> Arah belum pasti ({conf_pct:.0f}%). "
        f"Sebaiknya <b>TUNGGU (HOLD)</b> sampai tren jelas."
    )
