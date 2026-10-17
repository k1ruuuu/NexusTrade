"""
EA Scalping Engine — Professional High-Frequency Trend & Candlestick Forward Prediction Scalper:
1. Membaca Arah Trend: Mengombinasikan EMA Cepat (9), EMA Sedang (21), dan EMA Tren (50) pada timeframe M1/M5.
2. Membaca Candlestick: Analisis anatomi lilin real-time (Pinbar Rejection wick >= 40%, Engulfing, Momentum Body >= 50%, Breakout High/Low).
3. Membaca Pergerakan Market Kedepannya (Forward Prediction): Menggabungkan arah tren dan konfirmasi candlestick untuk memprediksi ekspansi harga berikutnya.
4. Entry dengan SL dan TP Saling Berdekatan: Tight scalping risk-reward (10-15 pips SL dan 12-18 pips TP) untuk eksekusi kilat.
5. Auto Re-entry Seketika pada TP maupun SL: Langsung membaca ulang market dan membuka posisi baru begitu posisi sebelumnya terealisasi.
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


@dataclass
class EAScalpingSignal:
    """Hasil evaluasi komprehensif sinyal EA Scalping."""
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    trend_direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    candle_pattern: str
    forward_prediction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_pips: float
    tp_pips: float
    confidence: float
    details: str
    is_instant_reentry: bool = False


class EAScalpingEngine:
    """Mesin AI untuk EA Scalping berbasis Trend, Candlestick, dan Forward Prediction."""

    @staticmethod
    def read_trend(df: pd.DataFrame) -> Tuple[Literal["BULLISH", "BEARISH", "NEUTRAL"], float, float, float]:
        """
        Membaca arah tren market menggunakan EMA 9, EMA 21, dan EMA 50.
        Returns: (trend_direction, ema9, ema21, ema50)
        """
        if len(df) < 10:
            p = float(df["close"].iloc[-1]) if len(df) > 0 else 2650.0
            return "NEUTRAL", p, p, p

        closes = pd.to_numeric(df["close"], errors="coerce").ffill()
        ema9_s = closes.ewm(span=9, adjust=False).mean()
        ema21_s = closes.ewm(span=21, adjust=False).mean()
        ema50_s = closes.ewm(span=50, adjust=False).mean() if len(closes) >= 50 else ema21_s

        e9 = float(ema9_s.iloc[-1])
        e21 = float(ema21_s.iloc[-1])
        e50 = float(ema50_s.iloc[-1])
        e9_prev = float(ema9_s.iloc[-2]) if len(ema9_s) >= 2 else e9
        e21_prev = float(ema21_s.iloc[-2]) if len(ema21_s) >= 2 else e21

        slope_9 = e9 - e9_prev
        slope_21 = e21 - e21_prev

        if (e9 >= e21 and slope_9 >= -1e-6) or (e9 > e50 and slope_21 >= 0):
            return "BULLISH", e9, e21, e50
        elif (e9 <= e21 and slope_9 <= 1e-6) or (e9 < e50 and slope_21 <= 0):
            return "BEARISH", e9, e21, e50
        else:
            return "NEUTRAL", e9, e21, e50

    @staticmethod
    def read_candlestick(df: pd.DataFrame) -> Tuple[str, Literal["BULLISH", "BEARISH", "NEUTRAL"], float]:
        """
        Membaca anatomi candlestick pada candle terakhir selesai (bar 1 / index -2).
        Returns: (pattern_name, bias, strength)
        """
        if len(df) < 3:
            return "NORMAL_CANDLE", "NEUTRAL", 0.50

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

        # 1. Bullish Pinbar / Rejection Wick Bawah (penolakan harga bawah)
        if lower_wick_ratio >= 0.40 and c1 >= o1:
            return f"BULLISH_PINBAR_REJECTION ({lower_wick_ratio*100:.0f}% Wick)", "BULLISH", 0.90
        elif lower_wick_ratio >= 0.50:
            return f"BULLISH_HAMMER_REJECTION ({lower_wick_ratio*100:.0f}% Wick)", "BULLISH", 0.88

        # 2. Bearish Pinbar / Rejection Wick Atas (penolakan harga atas)
        if upper_wick_ratio >= 0.40 and c1 <= o1:
            return f"BEARISH_PINBAR_REJECTION ({upper_wick_ratio*100:.0f}% Wick)", "BEARISH", 0.90
        elif upper_wick_ratio >= 0.50:
            return f"BEARISH_SHOOTING_STAR ({upper_wick_ratio*100:.0f}% Wick)", "BEARISH", 0.88

        # 3. Bullish Engulfing (candle hijau menelan candle merah sebelumnya)
        if (c1 > o1) and (c2 < o2) and (c1 >= o2) and (o1 <= c2):
            return "BULLISH_ENGULFING", "BULLISH", 0.87

        # 4. Bearish Engulfing (candle merah menelan candle hijau sebelumnya)
        if (c1 < o1) and (c2 > o2) and (c1 <= o2) and (o1 >= c2):
            return "BEARISH_ENGULFING", "BEARISH", 0.87

        # 5. Bullish Momentum Body (bodi hijau solid)
        if (c1 > o1) and (body_ratio >= 0.50) and (c1 > h2):
            return f"BULLISH_MOMENTUM_BREAKOUT ({body_ratio*100:.0f}% Bodi)", "BULLISH", 0.85
        elif (c1 > o1) and (body_ratio >= 0.50):
            return f"BULLISH_MOMENTUM_BODY ({body_ratio*100:.0f}% Bodi)", "BULLISH", 0.82

        # 6. Bearish Momentum Body (bodi merah solid)
        if (c1 < o1) and (body_ratio >= 0.50) and (c1 < l2):
            return f"BEARISH_MOMENTUM_BREAKDOWN ({body_ratio*100:.0f}% Bodi)", "BEARISH", 0.85
        elif (c1 < o1) and (body_ratio >= 0.50):
            return f"BEARISH_MOMENTUM_BODY ({body_ratio*100:.0f}% Bodi)", "BEARISH", 0.82

        # 7. Higher High / Lower Low continuation
        if c1 > o1 and h1 > h2:
            return "BULLISH_HIGHER_HIGH", "BULLISH", 0.75
        elif c1 < o1 and l1 < l2:
            return "BEARISH_LOWER_LOW", "BEARISH", 0.75

        return "NORMAL_CONSOLIDATION_CANDLE", "NEUTRAL", 0.50

    @classmethod
    def evaluate_scalping(
        cls,
        df: pd.DataFrame,
        symbol: str,
        sl_pips: float = 12.0,
        tp_pips: float = 14.0,
        last_exit_reason: Optional[str] = None,
    ) -> EAScalpingSignal:
        """
        Mengevaluasi kondisi scalping komprehensif:
        - Membaca Arah Tren (Trend Reading)
        - Membaca Anatomi Candlestick (Candle Reading)
        - Memprediksi Pergerakan Market Kedepannya (Forward Prediction)
        - Menghitung SL dan TP yang Saling Berdekatan (Tight Scalping SL & TP)
        """
        clean_sym = normalize_symbol(symbol)
        curr_price = float(df["close"].iloc[-1]) if len(df) > 0 else 2650.0

        trend_dir, e9, e21, e50 = cls.read_trend(df)
        candle_pattern, candle_bias, candle_strength = cls.read_candlestick(df)

        is_reentry = last_exit_reason in ("TAKE_PROFIT", "STOP_LOSS", "CLOSED_TP", "CLOSED_SL")
        reentry_tag = f" [RE-ENTRY {last_exit_reason}]" if is_reentry else ""

        # Prediksi pergerakan market kedepannya:
        # Konfluensi 1: Tren Bullish + Candlestick Bullish ➔ Prediksi ekspansi NAIK (BUY)
        if trend_dir == "BULLISH" and candle_bias in ("BULLISH", "NEUTRAL"):
            action = "BUY"
            forward_pred = "PREDIKSI NAIK: Momentum Tren Bullish & Candlestick Konfirmasi Ekspansi ke Atas"
            conf = min(0.92, max(0.80, candle_strength + 0.05))
            details = (
                f"📈 Tren: BULLISH (EMA9 {e9:,.2f} ≥ EMA21 {e21:,.2f}) • "
                f"Lilin: {candle_pattern} • "
                f"{forward_pred} ➔ Eksekusi BUY (SL {sl_pips:.0f}p / TP {tp_pips:.0f}p Berdekatan){reentry_tag}"
            )

        # Konfluensi 2: Tren Bearish + Candlestick Bearish ➔ Prediksi ekspansi TURUN (SELL)
        elif trend_dir == "BEARISH" and candle_bias in ("BEARISH", "NEUTRAL"):
            action = "SELL"
            forward_pred = "PREDIKSI TURUN: Momentum Tren Bearish & Candlestick Konfirmasi Penekanan ke Bawah"
            conf = min(0.92, max(0.80, candle_strength + 0.05))
            details = (
                f"📉 Tren: BEARISH (EMA9 {e9:,.2f} ≤ EMA21 {e21:,.2f}) • "
                f"Lilin: {candle_pattern} • "
                f"{forward_pred} ➔ Eksekusi SELL (SL {sl_pips:.0f}p / TP {tp_pips:.0f}p Berdekatan){reentry_tag}"
            )

        # Konfluensi 3: Rejection Kuat / Reversal Candle (misal Pinbar Rejection kuat melawan EMA)
        elif candle_bias == "BULLISH" and candle_strength >= 0.88:
            action = "BUY"
            forward_pred = "PREDIKSI NAIK (REVERSAL): Rejection Bawah Kuat Menolak Support"
            conf = 0.84
            details = (
                f"⚡ Rejection Bullish Terkonfirmasi: {candle_pattern} • "
                f"{forward_pred} ➔ Eksekusi Scalping Reversal BUY (SL {sl_pips:.0f}p / TP {tp_pips:.0f}p){reentry_tag}"
            )
        elif candle_bias == "BEARISH" and candle_strength >= 0.88:
            action = "SELL"
            forward_pred = "PREDIKSI TURUN (REVERSAL): Rejection Atas Kuat Menolak Resistance"
            conf = 0.84
            details = (
                f"⚡ Rejection Bearish Terkonfirmasi: {candle_pattern} • "
                f"{forward_pred} ➔ Eksekusi Scalping Reversal SELL (SL {sl_pips:.0f}p / TP {tp_pips:.0f}p){reentry_tag}"
            )

        # Jika sedang dalam siklus re-entry instan pasca SL/TP, gunakan bias candle atau arah tren terkuat
        elif is_reentry:
            action = "BUY" if (trend_dir == "BULLISH" or candle_bias == "BULLISH") else "SELL"
            forward_pred = f"PREDIKSI LANJUTAN: Siklus Re-Entry Instan Pasca {last_exit_reason}"
            conf = 0.82
            details = (
                f"🔄 Re-Entry Instan Pasca {last_exit_reason}: Mengikuti Arah {action} "
                f"(Tren: {trend_dir}, Lilin: {candle_pattern}) ➔ Eksekusi {action} SL & TP Berdekatan"
            )
        else:
            action = "HOLD"
            forward_pred = "PREDIKSI KONSOLIDASI: Menunggu Konfirmasi Lilin Searah Tren"
            conf = 0.50
            details = (
                f"Menunggu Konfirmasi Candlestick Searah Tren (Tren: {trend_dir}, "
                f"EMA9: {e9:,.2f}, EMA21: {e21:,.2f}, Lilin: {candle_pattern})"
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
            trend_direction=trend_dir,
            candle_pattern=candle_pattern,
            forward_prediction=forward_pred,
            entry_price=curr_price,
            stop_loss=sl_val,
            take_profit=tp_val,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confidence=conf,
            details=details,
            is_instant_reentry=is_reentry,
        )
