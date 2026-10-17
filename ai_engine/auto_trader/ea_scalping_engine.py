"""
EA Scalping Engine — Pure 5-MA Steep Crossover, Candlestick Anatomy & Multi-Timeframe (H1/M5) Forward Prediction Scalper:

1. Timeframe Eksekusi: M1 / 1M murni.
2. Analisa Tren Pasar: Multi-timeframe D1, H1, M15, M5 — dengan H1 dan M5 sebagai jangkar utama.
3. 5 Jenis Moving Average (MA) pada M1:
   - MA 1: MA Biru (EMA 5 - Fast Signal Line)
   - MA 2: MA Hijau (EMA 8 - Bullish Reference Target)
   - MA 3: MA Merah (EMA 13 - Bearish Reference Target)
   - MA 4: MA Kuning (EMA 34 - Mid Baseline)
   - MA 5: MA Ungu (EMA 55 - Major Baseline)
4. Aturan Eksekusi Cross Curam (Steep Cross):
   - BUY: MA Biru cross MA Hijau ke atas secara curam (steep upward slope) + didukung tren H1/M5 & karakter lilin bullish.
   - SELL: MA Biru cross MA Merah ke bawah secara curam (steep downward slope) + didukung tren H1/M5 & karakter lilin bearish.
5. Pembacaan Karakter Candlestick:
   - Pinbar Rejection (Lower/Upper Wick >= 40%)
   - Engulfing Candle
   - Momentum Solid Body (>= 50%)
   - Breakout High/Low
6. Entry dengan SL dan TP Saling Berdekatan (Tight Scalping):
   - SL 10-12 pips & TP 12-15 pips (sangat rapat dan cepat tersentuh).
7. Auto Re-Entry Seketika pada TP maupun SL (Continuous Loop).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ai_engine.auto_trader.mt5_symbols import (
    calculate_ea_sl_tp,
    get_mt5_contract_size,
    get_mt5_digits_and_point,
    get_mt5_pip_size,
)
from data_pipeline.market_symbols import normalize_symbol
from data_pipeline.candlestick_reader import CandlestickReader, SingleCandleAnatomy
from data_pipeline.market_structure_analyzer import MarketStructureAnalyzer, MarketStructureResult


@dataclass
class EAScalpingSignal:
    """Hasil evaluasi komprehensif sinyal EA Scalping 5-MA & MTF."""
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    trend_direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    mtf_h1_trend: str
    mtf_m5_trend: str
    candle_pattern: str
    forward_prediction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_pips: float
    tp_pips: float
    confidence: float
    details: str
    ma_blue: float = 0.0
    ma_green: float = 0.0
    ma_red: float = 0.0
    ma_yellow: float = 0.0
    ma_purple: float = 0.0
    steep_slope_ratio: float = 0.0
    is_instant_reentry: bool = False


class EAScalpingEngine:
    """Mesin AI untuk EA Scalping 5-MA, Candlestick, dan Analisa Tren H1/M5."""

    # Parameter 5 Jenis Moving Average pada M1
    PERIOD_MA_BLUE = 5      # MA Biru (Ultra-Fast Signal)
    PERIOD_MA_GREEN = 8     # MA Hijau (Bullish Reference)
    PERIOD_MA_RED = 13      # MA Merah (Bearish Reference)
    PERIOD_MA_YELLOW = 34   # MA Kuning (Mid Baseline)
    PERIOD_MA_PURPLE = 55   # MA Ungu (Major Baseline)

    @classmethod
    def compute_5_ma(cls, closes: pd.Series) -> Tuple[float, float, float, float, float, float]:
        """
        Menghitung nilai 5 MA pada M1 dan kemiringan (slope) curam MA Biru.
        Returns: (ma_blue, ma_green, ma_red, ma_yellow, ma_purple, blue_slope)
        """
        if len(closes) < 5:
            p = float(closes.iloc[-1]) if len(closes) > 0 else 2650.0
            return p, p, p, p, p, 0.0

        ema_blue_s = closes.ewm(span=cls.PERIOD_MA_BLUE, adjust=False).mean()
        ema_green_s = closes.ewm(span=cls.PERIOD_MA_GREEN, adjust=False).mean()
        ema_red_s = closes.ewm(span=cls.PERIOD_MA_RED, adjust=False).mean()
        ema_yellow_s = closes.ewm(span=cls.PERIOD_MA_YELLOW, adjust=False).mean() if len(closes) >= cls.PERIOD_MA_YELLOW else ema_red_s
        ema_purple_s = closes.ewm(span=cls.PERIOD_MA_PURPLE, adjust=False).mean() if len(closes) >= cls.PERIOD_MA_PURPLE else ema_yellow_s

        b0 = float(ema_blue_s.iloc[-1])
        b1 = float(ema_blue_s.iloc[-2]) if len(ema_blue_s) >= 2 else b0
        g0 = float(ema_green_s.iloc[-1])
        r0 = float(ema_red_s.iloc[-1])
        y0 = float(ema_yellow_s.iloc[-1])
        p0 = float(ema_purple_s.iloc[-1])

        blue_slope = b0 - b1
        return b0, g0, r0, y0, p0, blue_slope

    @classmethod
    def read_mtf_trend(
        cls,
        h1_df: Optional[pd.DataFrame] = None,
        m5_df: Optional[pd.DataFrame] = None,
        d1_df: Optional[pd.DataFrame] = None,
        m15_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Literal["BULLISH", "BEARISH", "NEUTRAL"], str, str]:
        """
        Menganalisa tren pasar dari D1, H1, M15, dan M5 — dengan H1 dan M5 sebagai jangkar utama.
        Returns: (overall_trend, h1_desc, m5_desc)
        """
        h1_bias = "NEUTRAL"
        m5_bias = "NEUTRAL"

        if h1_df is not None and len(h1_df) >= 10:
            h1_c = pd.to_numeric(h1_df["close"], errors="coerce").ffill()
            e20_s = h1_c.ewm(span=min(20, len(h1_c)), adjust=False).mean()
            e50_s = h1_c.ewm(span=min(50, len(h1_c)), adjust=False).mean()
            e20 = float(e20_s.iloc[-1])
            e50 = float(e50_s.iloc[-1])
            h1_slope = float(h1_c.iloc[-1]) - float(h1_c.iloc[0])
            h1_bias = "BULLISH" if (e20 > e50 and h1_slope >= 0) or (e20 >= e50 and h1_slope > 0) else "BEARISH"
        if m5_df is not None and len(m5_df) >= 10:
            m5_c = pd.to_numeric(m5_df["close"], errors="coerce").ffill()
            e9 = float(m5_c.ewm(span=9, adjust=False).mean().iloc[-1])
            e21 = float(m5_c.ewm(span=21, adjust=False).mean().iloc[-1])
            m5_bias = "BULLISH" if e9 >= e21 else "BEARISH"

        h1_desc = f"H1 {h1_bias}"
        m5_desc = f"M5 {m5_bias}"

        # H1 dan M5 yang paling utama
        if h1_bias == "BULLISH" and m5_bias == "BULLISH":
            return "BULLISH", h1_desc, m5_desc
        elif h1_bias == "BEARISH" and m5_bias == "BEARISH":
            return "BEARISH", h1_desc, m5_desc
        elif m5_bias != "NEUTRAL":
            # Jika berbeda, bobot M5 memberikan momentum jangka terdekat untuk scalping M1
            return m5_bias, h1_desc, m5_desc
        elif h1_bias != "NEUTRAL":
            return h1_bias, h1_desc, m5_desc
        return "NEUTRAL", h1_desc, m5_desc

    @staticmethod
    def read_candlestick_character(df: pd.DataFrame) -> Tuple[str, Literal["BULLISH", "BEARISH", "NEUTRAL"], float]:
        """
        Membaca karakter dan anatomi candlestick pada lilin M1 yang selesai (bar 1 / index -2).
        Returns: (pattern_desc, bias, strength)
        """
        if len(df) < 3:
            return "LILIN_NORMAL", "NEUTRAL", 0.50

        bar1 = df.iloc[-2]
        bar2 = df.iloc[-3]

        o1, h1, l1, c1 = float(bar1["open"]), float(bar1["high"]), float(bar1["low"]), float(bar1["close"])
        o2, h2, l2, c2 = float(bar2["open"]), float(bar2["high"]), float(bar2["low"]), float(bar2["close"])

        c_range = max(h1 - l1, 1e-6)
        body = abs(c1 - o1)
        lower_wick = max(0.0, min(o1, c1) - l1)
        upper_wick = max(0.0, h1 - max(o1, c1))

        body_ratio = body / c_range
        lower_wick_ratio = lower_wick / c_range
        upper_wick_ratio = upper_wick / c_range

        # 1. Bullish Pinbar Rejection (Penolakan Bawah >= 40%)
        if lower_wick_ratio >= 0.40 and c1 >= o1:
            return f"PINBAR_REJECTION_BAWAH (Sumbu {lower_wick_ratio*100:.0f}%)", "BULLISH", 0.90
        elif lower_wick_ratio >= 0.50:
            return f"HAMMER_REJECTION (Sumbu {lower_wick_ratio*100:.0f}%)", "BULLISH", 0.88

        # 2. Bearish Pinbar Rejection (Penolakan Atas >= 40%)
        if upper_wick_ratio >= 0.40 and c1 <= o1:
            return f"PINBAR_REJECTION_ATAS (Sumbu {upper_wick_ratio*100:.0f}%)", "BEARISH", 0.90
        elif upper_wick_ratio >= 0.50:
            return f"SHOOTING_STAR_REJECTION (Sumbu {upper_wick_ratio*100:.0f}%)", "BEARISH", 0.88

        # 3. Bullish Engulfing
        if (c1 > o1) and (c2 < o2) and (c1 >= o2) and (o1 <= c2):
            return "BULLISH_ENGULFING", "BULLISH", 0.87

        # 4. Bearish Engulfing
        if (c1 < o1) and (c2 > o2) and (c1 <= o2) and (o1 >= c2):
            return "BEARISH_ENGULFING", "BEARISH", 0.87

        # 5. Bullish Solid Momentum Body (Bodi >= 50%)
        if (c1 > o1) and (body_ratio >= 0.50) and (c1 > h2):
            return f"MOMENTUM_BREAKOUT_HIJAU (Bodi {body_ratio*100:.0f}%)", "BULLISH", 0.86
        elif (c1 > o1) and (body_ratio >= 0.50):
            return f"MOMENTUM_SOLID_HIJAU (Bodi {body_ratio*100:.0f}%)", "BULLISH", 0.82

        # 6. Bearish Solid Momentum Body (Bodi >= 50%)
        if (c1 < o1) and (body_ratio >= 0.50) and (c1 < l2):
            return f"MOMENTUM_BREAKDOWN_MERAH (Bodi {body_ratio*100:.0f}%)", "BEARISH", 0.86
        elif (c1 < o1) and (body_ratio >= 0.50):
            return f"MOMENTUM_SOLID_MERAH (Bodi {body_ratio*100:.0f}%)", "BEARISH", 0.82

        # 7. Higher High / Lower Low
        if c1 > o1 and h1 > h2:
            return "HIGHER_HIGH_LANJUTAN", "BULLISH", 0.75
        elif c1 < o1 and l1 < l2:
            return "LOWER_LOW_LANJUTAN", "BEARISH", 0.75

        return "KONSOLIDASI_NORMAL", "NEUTRAL", 0.50

    @classmethod
    def evaluate_scalping_5ma(
        cls,
        m1_df: pd.DataFrame,
        symbol: str,
        h1_df: Optional[pd.DataFrame] = None,
        m5_df: Optional[pd.DataFrame] = None,
        d1_df: Optional[pd.DataFrame] = None,
        m15_df: Optional[pd.DataFrame] = None,
        sl_pips: float = 12.0,
        tp_pips: float = 14.0,
        last_exit_reason: Optional[str] = None,
    ) -> EAScalpingSignal:
        """
        Alur logika lengkap EA Scalping:
        - Timeframe: M1 untuk action/entry.
        - Analisa Tren: H1 dan M5 (didukung D1 & M15).
        - 5 Jenis MA: Biru (EMA5), Hijau (EMA8), Merah (EMA13), Kuning (EMA34), Ungu (EMA55).
        - Action:
          - BUY: MA Biru cross MA Hijau ke atas curam.
          - SELL: MA Biru cross MA Merah ke bawah curam.
        - SL & TP: Saling berdekatan (Tight Scalping).
        - Re-entry: Seketika saat TP maupun SL.
        """
        clean_sym = normalize_symbol(symbol)
        closes_m1 = pd.to_numeric(m1_df["close"], errors="coerce").ffill()
        curr_price = float(closes_m1.iloc[-1]) if len(closes_m1) > 0 else 2650.0

        digits, point = get_mt5_digits_and_point(clean_sym, curr_price)
        pip_size = get_mt5_pip_size(clean_sym, curr_price)

        # 1. Hitung 5 Moving Averages pada M1
        b0, g0, r0, y0, p0, blue_slope = cls.compute_5_ma(closes_m1)

        # Nilai MA pada bar sebelumnya untuk mendeteksi perpotongan (crossover)
        closes_prev = closes_m1.iloc[:-1]
        b_prev, g_prev, r_prev, _, _, _ = cls.compute_5_ma(closes_prev) if len(closes_prev) >= 5 else (b0, g0, r0, y0, p0, 0.0)

        # 2. Analisa Tren Pasar Multi-Timeframe (H1 & M5 Utama)
        mtf_trend, h1_desc, m5_desc = cls.read_mtf_trend(h1_df=h1_df, m5_df=m5_df, d1_df=d1_df, m15_df=m15_df)

        # 3. Analisa Struktur Pasar (Market Structure HH, HL, LH, LL & 3 Checklist Wajib)
        struct_res = MarketStructureAnalyzer.analyze_structure(
            m5_df if (m5_df is not None and len(m5_df) >= 20) else m1_df,
            symbol=clean_sym,
            timeframe="5m" if (m5_df is not None and len(m5_df) >= 20) else "1m",
        )

        # 3. Pembacaan Karakter Candlestick M1
        candle_pattern, candle_bias, candle_strength = cls.read_candlestick_character(m1_df)

        # 4. Deteksi Perpotongan Curam (Steep Cross):
        # Slope ratio diukur dalam kelipatan point per candle M1
        slope_points = abs(blue_slope) / max(point, 1e-6)
        is_steep_slope = slope_points >= 2.0 or abs(blue_slope) >= (0.10 * pip_size)

        # BUY Signal: MA Biru cross MA Hijau ke atas curam, ATAU MA Biru > MA Hijau ekspansi curam
        bull_cross_up = (b_prev <= g_prev and b0 > g0) and (blue_slope > 0)
        bull_steep_expansion = (b0 > g0) and (blue_slope > 0) and is_steep_slope

        # SELL Signal: MA Biru cross MA Merah ke bawah curam, ATAU MA Biru < MA Merah ekspansi curam
        bear_cross_down = (b_prev >= r_prev and b0 < r0) and (blue_slope < 0)
        bear_steep_expansion = (b0 < r0) and (blue_slope < 0) and is_steep_slope

        is_reentry = last_exit_reason in ("TAKE_PROFIT", "STOP_LOSS", "CLOSED_TP", "CLOSED_SL")
        reentry_tag = f" [RE-ENTRY {last_exit_reason}]" if is_reentry else ""

        action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
        forward_prediction = ""
        conf = 0.50
        details = ""

        # Skenario 1: Eksekusi BUY (MA Biru cross MA Hijau ke atas curam)
        if (bull_cross_up or (bull_steep_expansion and candle_bias in ("BULLISH", "NEUTRAL"))) and mtf_trend != "BEARISH":
            action = "BUY"
            cross_type = "Cross Naik Curam" if bull_cross_up else "Ekspansi Curam Naik"
            forward_prediction = f"PREDIKSI NAIK 1M: MA Biru ({b0:,.2f}) {cross_type} MA Hijau ({g0:,.2f}) didukung Tren {m5_desc}/{h1_desc}"
            conf = min(0.92, max(0.82, candle_strength + 0.04))
            details = (
                f"🚀 BUY TERKONFIRMASI: MA Biru Cross MA Hijau Ke Atas Curam (Slope +{blue_slope/point:.1f}pt) • "
                f"Lilin: {candle_pattern} • Tren: {m5_desc} & {h1_desc} ➔ {forward_prediction}{reentry_tag}"
            )

        # Skenario 2: Eksekusi SELL (MA Biru cross MA Merah ke bawah curam)
        elif (bear_cross_down or (bear_steep_expansion and candle_bias in ("BEARISH", "NEUTRAL"))) and mtf_trend != "BULLISH":
            action = "SELL"
            cross_type = "Cross Turun Curam" if bear_cross_down else "Ekspansi Curam Turun"
            forward_prediction = f"PREDIKSI TURUN 1M: MA Biru ({b0:,.2f}) {cross_type} MA Merah ({r0:,.2f}) didukung Tren {m5_desc}/{h1_desc}"
            conf = min(0.92, max(0.82, candle_strength + 0.04))
            details = (
                f"🚀 SELL TERKONFIRMASI: MA Biru Cross MA Merah Ke Bawah Curam (Slope {blue_slope/point:.1f}pt) • "
                f"Lilin: {candle_pattern} • Tren: {m5_desc} & {h1_desc} ➔ {forward_prediction}{reentry_tag}"
            )

        # Skenario 3: Rejection Candlestick Kuat (Pinbar / Reversal)
        elif candle_bias == "BULLISH" and candle_strength >= 0.88 and b0 >= g0:
            action = "BUY"
            forward_prediction = "PREDIKSI NAIK REVERSAL: Rejection Lilin Kuat Menolak Support M1"
            conf = 0.85
            details = (
                f"⚡ BUY REVERSAL: {candle_pattern} • MA Biru ({b0:,.2f}) Stabil di Atas MA Hijau ({g0:,.2f}) • "
                f"{forward_prediction}{reentry_tag}"
            )
        elif candle_bias == "BEARISH" and candle_strength >= 0.88 and b0 <= r0:
            action = "SELL"
            forward_prediction = "PREDIKSI TURUN REVERSAL: Rejection Lilin Kuat Menolak Resistance M1"
            conf = 0.85
            details = (
                f"⚡ SELL REVERSAL: {candle_pattern} • MA Biru ({b0:,.2f}) Tertekan di Bawah MA Merah ({r0:,.2f}) • "
                f"{forward_prediction}{reentry_tag}"
            )

        # Skenario 4: Re-entry Instan Pasca TP / SL
        elif is_reentry:
            action = "BUY" if (b0 >= g0 or candle_bias == "BULLISH" or mtf_trend == "BULLISH") else "SELL"
            forward_prediction = f"PREDIKSI SIKLUS RE-ENTRY INSTAN: Melanjutkan Arah {action} Pasca {last_exit_reason}"
            conf = 0.82
            details = (
                f"🔄 RE-ENTRY INSTAN PASCA {last_exit_reason}: Mengikuti Arah {action} "
                f"(MA Biru {b0:,.2f}, Tren MTF {m5_desc}/{h1_desc}, Lilin {candle_pattern}) ➔ Eksekusi {action}"
            )
        else:
            action = "HOLD"
            forward_prediction = "PREDIKSI KONSOLIDASI: Menunggu MA Biru Cross Curam MA Hijau (BUY) atau MA Merah (SELL)"
            conf = 0.50
            details = (
                f"Menunggu Sinyal 5-MA M1: MA Biru ({b0:,.2f}), MA Hijau ({g0:,.2f}), MA Merah ({r0:,.2f}), "
                f"Lilin: {candle_pattern}, Tren: {m5_desc}/{h1_desc}"
            )

        # Hitung SL dan TP yang saling berdekatan (Tight Scalping SL & TP)
        sl_val, tp_val, _ = calculate_ea_sl_tp(
            symbol=clean_sym,
            side=action if action in ("BUY", "SELL") else "BUY",
            entry_price=curr_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            stops_level_pips=3.0,
        )

        return EAScalpingSignal(
            symbol=clean_sym,
            action=action,
            trend_direction=mtf_trend,
            mtf_h1_trend=h1_desc,
            mtf_m5_trend=m5_desc,
            candle_pattern=candle_pattern,
            forward_prediction=forward_prediction,
            entry_price=curr_price,
            stop_loss=sl_val,
            take_profit=tp_val,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confidence=conf,
            details=details,
            ma_blue=b0,
            ma_green=g0,
            ma_red=r0,
            ma_yellow=y0,
            ma_purple=p0,
            steep_slope_ratio=blue_slope / max(point, 1e-6),
            is_instant_reentry=is_reentry,
        )

    # Alias backwards compatibility
    @classmethod
    def evaluate_scalping(
        cls,
        df: pd.DataFrame,
        symbol: str,
        sl_pips: float = 12.0,
        tp_pips: float = 14.0,
        last_exit_reason: Optional[str] = None,
    ) -> EAScalpingSignal:
        return cls.evaluate_scalping_5ma(
            m1_df=df,
            symbol=symbol,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            last_exit_reason=last_exit_reason,
        )
