"""
Universal Candlestick & Price Action Reader for NexusTrade AI:
Digunakan secara terpadu oleh fitur /autotrade, /trade, dan seluruh varian EA MT5.

Mengimplementasikan 3 Pilar Filosofi Candlestick:
1. Siapa yang Menang? -> Lihat BODY (Dominasi Buyers vs Sellers).
2. Di Mana Harga Ditolak? -> Lihat WICK / Sumbu (Penolakan Bawah vs Penolakan Atas).
3. Ke Mana Arah Trennya? -> Lihat RANGKAIAN Lilin (Swing High, Swing Low, Higher Highs/Lows -> Uptrend/Downtrend).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SingleCandleAnatomy:
    """Anatomi dan karakter 1 buah lilin candlestick."""
    open: float
    high: float
    low: float
    close: float
    color: Literal["PUTIH", "MERAH", "DOJI"]  # Putih (Harga Naik), Merah (Harga Turun), Doji (Netral)
    price_change: float
    range_total: float
    body_size: float
    body_ratio: float          # Persentase bodi terhadap total rentang (0.0 - 1.0)
    upper_wick: float
    upper_wick_ratio: float    # Persentase sumbu atas terhadap total rentang (0.0 - 1.0)
    lower_wick: float
    lower_wick_ratio: float    # Persentase sumbu bawah terhadap total rentang (0.0 - 1.0)
    
    # 3 Pilar Perilaku (Behaviour)
    winner: str                # Siapa yang menang? (BUYERS MENANG MUTLAK / SELLERS MENANG / SEIMBANG)
    rejection_site: str        # Di mana harga ditolak? (DITOLAK DI LEVEL BAWAH / DITOLAK DI LEVEL ATAS / NORMAL)
    pattern_label: str         # Pola lilin teridentifikasi (Pinbar, Engulfing, Hammer, Marubozu, dll)


@dataclass
class CandlestickSeriesAnalysis:
    """Analisa rangkaian candlestick, Swing High, Swing Low, dan arah tren."""
    trend: Literal["UPTREND", "DOWNTREND", "SIDEWAYS"]
    trend_description: str
    latest_candle: SingleCandleAnatomy
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)
    latest_swing_high: float = 0.0
    latest_swing_low: float = 0.0
    market_structure: str = ""
    three_pillars_summary: str = ""


class CandlestickReader:
    """Mesin pembaca candlestick dan aksi harga (Price Action Engine)."""

    @classmethod
    def read_single_candle(
        cls,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        prev_open: Optional[float] = None,
        prev_close: Optional[float] = None,
    ) -> SingleCandleAnatomy:
        """
        Membaca anatomi 1 buah lilin candlestick:
        - OHLC
        - Warna (Putih = Naik, Merah = Turun, Doji = Ragu-ragu)
        - Siapa yang menang? (Lihat Body)
        - Di mana harga ditolak? (Lihat Wick)
        """
        c_range = max(high_p - low_p, 1e-8)
        body = abs(close_p - open_p)
        upper_wick = max(0.0, high_p - max(open_p, close_p))
        lower_wick = max(0.0, min(open_p, close_p) - low_p)

        body_ratio = body / c_range
        upper_wick_ratio = upper_wick / c_range
        lower_wick_ratio = lower_wick / c_range

        # 1. Warna Candlestick (Sesuai Konvensi: Putih = Naik, Merah = Turun)
        if close_p > open_p:
            color: Literal["PUTIH", "MERAH", "DOJI"] = "PUTIH"
        elif close_p < open_p:
            color = "MERAH"
        else:
            color = "DOJI"

        # 2. Pilar 1: Siapa yang Menang? (Lihat Body)
        if color == "PUTIH":
            if body_ratio >= 0.70:
                winner = "BUYERS MENANG MUTLAK (Marubozu Putih/Hijau Solid)"
            elif body_ratio >= 0.45:
                winner = "BUYERS MENDOMINASI (Bodi Putih/Hijau Kuat)"
            else:
                winner = "BUYERS UNGGUL TIPIS (Tekanan Imbang)"
        elif color == "MERAH":
            if body_ratio >= 0.70:
                winner = "SELLERS MENANG MUTLAK (Marubozu Merah Solid)"
            elif body_ratio >= 0.45:
                winner = "SELLERS MENDOMINASI (Bodi Merah Kuat)"
            else:
                winner = "SELLERS UNGGUL TIPIS (Tekanan Imbang)"
        else:
            winner = "SEIMBANG / PERANG RAGU-RAGU (Doji Tanpa Pemenang)"

        # 3. Pilar 2: Di Mana Harga Ditolak? (Lihat Wick / Sumbu)
        if lower_wick_ratio >= 0.45 and upper_wick_ratio <= 0.20:
            rejection = f"DITOLAK DI LEVEL BAWAH (Buyers Mendorong Naik, Sumbu Bawah {lower_wick_ratio*100:.0f}%)"
            pattern = "PINBAR_REJECTION_BAWAH (HAMMER)" if color == "PUTIH" else "HANGING_MAN_REJECTION"
        elif upper_wick_ratio >= 0.45 and lower_wick_ratio <= 0.20:
            rejection = f"DITOLAK DI LEVEL ATAS (Sellers Menekan Turun, Sumbu Atas {upper_wick_ratio*100:.0f}%)"
            pattern = "PINBAR_REJECTION_ATAS (SHOOTING_STAR)" if color == "MERAH" else "INVERTED_HAMMER_REJECTION"
        elif lower_wick_ratio >= 0.35 and upper_wick_ratio >= 0.35:
            rejection = "DITOLAK DI KEDUA SISI (Volatilitas Tinggi / Spinning Top)"
            pattern = "SPINNING_TOP"
        elif body_ratio >= 0.65:
            rejection = "TIDAK ADA PENOLAKAN BERARTI (Ekspansi Bodi Penuh)"
            pattern = "MOMENTUM_SOLID_EXPANSION"
        else:
            rejection = "PENOLAKAN NORMAL DI BATAS SPREAD"
            pattern = "LILIN_NORMAL"

        # Cek Engulfing jika data bar sebelumnya tersedia
        if prev_open is not None and prev_close is not None:
            if color == "PUTIH" and prev_close < prev_open and close_p >= prev_open and open_p <= prev_close:
                pattern = "BULLISH_ENGULFING (Lilin Putih Menelan Lilin Merah Sebelumnya)"
                winner = "BUYERS MENANG BESAR (Reversal Bullish Engulfing)"
            elif color == "MERAH" and prev_close > prev_open and close_p <= prev_open and open_p >= prev_close:
                pattern = "BEARISH_ENGULFING (Lilin Merah Menelan Lilin Putih Sebelumnya)"
                winner = "SELLERS MENANG BESAR (Reversal Bearish Engulfing)"

        return SingleCandleAnatomy(
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            color=color,
            price_change=close_p - open_p,
            range_total=c_range,
            body_size=body,
            body_ratio=round(body_ratio, 4),
            upper_wick=round(upper_wick, 4),
            upper_wick_ratio=round(upper_wick_ratio, 4),
            lower_wick=round(lower_wick, 4),
            lower_wick_ratio=round(lower_wick_ratio, 4),
            winner=winner,
            rejection_site=rejection,
            pattern_label=pattern,
        )

    @classmethod
    def analyze_series_and_swings(
        cls,
        df: pd.DataFrame,
        swing_window: int = 3,
    ) -> CandlestickSeriesAnalysis:
        """
        Menganalisa rangkaian lilin untuk mengidentifikasi:
        - Swing High (Puncak)
        - Swing Low (Lembah)
        - Higher Highs (HH) / Higher Lows (HL) ➔ UPTREND
        - Lower Lows (LL) / Lower Highs (LH) ➔ DOWNTREND
        - Pilar 3: Ke mana arah tren dari rangkaian strukturnya?
        """
        if len(df) < 5:
            c0 = float(df["close"].iloc[-1]) if len(df) > 0 else 2650.0
            dummy = cls.read_single_candle(c0, c0 + 1.0, c0 - 1.0, c0)
            return CandlestickSeriesAnalysis(
                trend="SIDEWAYS",
                trend_description="Data lilin kurang dari 5 bar",
                latest_candle=dummy,
            )

        highs = pd.to_numeric(df["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(df["low"], errors="coerce").astype(np.float64).values
        closes = pd.to_numeric(df["close"], errors="coerce").astype(np.float64).values
        opens = pd.to_numeric(df["open"], errors="coerce").astype(np.float64).values

        # 1. Hitung Swing Highs dan Swing Lows lokal
        n = len(df)
        swing_highs: List[float] = []
        swing_lows: List[float] = []

        w = max(1, swing_window)
        for i in range(w, n - w):
            # Swing High jika lebih tinggi dari w bar sebelum dan sesudah
            if all(highs[i] >= highs[i - k] for k in range(1, w + 1)) and all(highs[i] >= highs[i + k] for k in range(1, w + 1)):
                swing_highs.append(float(highs[i]))
            # Swing Low jika lebih rendah dari w bar sebelum dan sesudah
            if all(lows[i] <= lows[i - k] for k in range(1, w + 1)) and all(lows[i] <= lows[i + k] for k in range(1, w + 1)):
                swing_lows.append(float(lows[i]))

        latest_sh = swing_highs[-1] if swing_highs else float(np.max(highs[-10:]))
        latest_sl = swing_lows[-1] if swing_lows else float(np.min(lows[-10:]))

        # 2. Pilar 3: Ke Mana Tren dari Rangkaian Strukturnya?
        # Analisis Higher High / Higher Low vs Lower Low / Lower High
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1] > swing_highs[-2]
            hl = swing_lows[-1] > swing_lows[-2]
            lh = swing_highs[-1] < swing_highs[-2]
            ll = swing_lows[-1] < swing_lows[-2]

            if hh and hl:
                trend: Literal["UPTREND", "DOWNTREND", "SIDEWAYS"] = "UPTREND"
                structure = "HIGHER HIGHS (HH) & HIGHER LOWS (HL) BERURUTAN"
                trend_desc = "📈 UPTREND KUAT: Struktur pasar mencetak puncak dan lembah yang semakin meninggi."
            elif lh and ll:
                trend = "DOWNTREND"
                structure = "LOWER HIGHS (LH) & LOWER LOWS (LL) BERURUTAN"
                trend_desc = "📉 DOWNTREND KUAT: Struktur pasar mencetak puncak dan lembah yang semakin menurun."
            elif hh and ll:
                trend = "SIDEWAYS"
                structure = "EXPANDING MARKET / MEGAFON"
                trend_desc = "↔️ SIDEWAYS / MEGAPOHON: Volatilitas melebar tanpa arah tren konsisten."
            else:
                trend = "SIDEWAYS"
                structure = "KONSOLIDASI MENYEMPIT (TRIANGLE/RANGE)"
                trend_desc = "↔️ SIDEWAYS: Pasar berkonsolidasi di dalam rentang S/R."
        else:
            # Fallback jika data swing belum mencukupi: baca kemiringan EMA dan momentum close
            c_series = pd.Series(closes)
            ema9 = float(c_series.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(c_series.ewm(span=21, adjust=False).mean().iloc[-1])
            slope = closes[-1] - closes[0]

            if ema9 > ema21 and slope > 0:
                trend = "UPTREND"
                structure = "MOMENTUM NAIK KONSISTEN"
                trend_desc = "📈 UPTREND: Rentang lilin bergerak konsisten naik di atas EMA 21."
            elif ema9 < ema21 and slope < 0:
                trend = "DOWNTREND"
                structure = "MOMENTUM TURUN KONSISTEN"
                trend_desc = "📉 DOWNTREND: Rentang lilin bergerak konsisten turun di bawah EMA 21."
            else:
                trend = "SIDEWAYS"
                structure = "KONSOLIDASI"
                trend_desc = "↔️ SIDEWAYS: Lilin berosilasi di sekitar rata-rata harga."

        # Analisis lilin terakhir (bar terakhir selesai atau berjalan)
        last_o, last_h, last_l, last_c = opens[-1], highs[-1], lows[-1], closes[-1]
        prev_o = opens[-2] if n >= 2 else None
        prev_c = closes[-2] if n >= 2 else None
        candle_anatomy = cls.read_single_candle(last_o, last_h, last_l, last_c, prev_o, prev_c)

        # 3. Rangkuman 3 Pilar Perilaku (Behaviour Summary)
        summary = (
            f"1. PEMENANG: {candle_anatomy.winner} (Bodi {candle_anatomy.body_ratio*100:.0f}%, Warna {candle_anatomy.color})\n"
            f"2. PENOLAKAN: {candle_anatomy.rejection_site}\n"
            f"3. ARAH TREN: {trend_desc} (Struktur: {structure})"
        )

        return CandlestickSeriesAnalysis(
            trend=trend,
            trend_description=trend_desc,
            latest_candle=candle_anatomy,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            latest_swing_high=latest_sh,
            latest_swing_low=latest_sl,
            market_structure=structure,
            three_pillars_summary=summary,
        )
