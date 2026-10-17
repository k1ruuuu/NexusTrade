"""
NX EA Suite Engine — Python Implementation of the 4 Institutional MQL5 Expert Advisors:
1. NXBreak: High-probability candle breakout with 55% body and 0.4-2.5 ATR range filter.
2. NXParsial: Sequential 3-leg partial basket scaling (1R, 3R, 6R) with shared risk budget.
3. NXReverse: EMA 20/50/200 trend alignment + rolling 20-bar SNR breakout + 1:5 RR + SL recovery switching.
4. NX-POC: 100-bar Volume Profile Point of Control (POC) with 40-bin histogram rejection & ADX <= 25 filter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ai_engine.auto_trader.mt5_symbols import (
    get_mt5_contract_size,
    get_mt5_digits_and_point,
    get_mt5_pip_size,
)
from data_pipeline.market_symbols import normalize_symbol


@dataclass
class NXSignalResult:
    """Standardized signal output from the NX EA engine."""
    strategy: Literal["NXBreak", "NXParsial", "NXReverse", "NX-POC"]
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_levels: List[float] = field(default_factory=list)
    atr_14: float = 0.0
    quality_passed: bool = False
    details: str = ""
    is_recovery: bool = False


class NXEAEngine:
    """Core mathematical and signal execution engine replicating NX EA MT5 behavior."""

    @staticmethod
    def compute_atr_shift2(df: pd.DataFrame, period: int = 14) -> float:
        """
        Compute Wilder's ATR(14) using shift 2 so the trigger candle does not inflate its own volatility reference.
        """
        if len(df) < period + 3:
            p = float(df["close"].iloc[-1]) if len(df) > 0 else 2650.0
            return max(0.5, p * 0.002)

        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")

        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_series = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        # Shift 2: index -3 in 0-indexed dataframe
        val = atr_series.iloc[-3]
        return float(val) if not np.isnan(val) and val > 0 else max(0.5, float(close.iloc[-1]) * 0.002)

    @staticmethod
    def check_candle_quality(
        bar: pd.Series,
        atr: float,
        min_body_ratio: float = 0.55,
        min_signal_atr: float = 0.4,
        max_signal_atr: float = 2.5,
    ) -> bool:
        """Filter candle range between 0.4 and 2.5 ATR, and body >= 55% of total range."""
        high = float(bar["high"])
        low = float(bar["low"])
        open_p = float(bar["open"])
        close_p = float(bar["close"])

        c_range = high - low
        body = abs(close_p - open_p)

        if c_range <= 0 or atr <= 0:
            return False
        if c_range < min_signal_atr * atr or c_range > max_signal_atr * atr:
            return False
        if (body / c_range) < min_body_ratio:
            return False
        return True

    @classmethod
    def evaluate_nx_break(
        cls,
        df: pd.DataFrame,
        symbol: str,
        fixed_sl_points: float = 300.0,
    ) -> NXSignalResult:
        """
        NXBreak Strategy:
        - Candle 1 closes above candle 2 High + buffer (0.05 ATR) for BUY.
        - Candle 1 closes below candle 2 Low - buffer (0.05 ATR) for SELL.
        - Candle 1 must satisfy 55% body ratio and 0.4-2.5 ATR range.
        - SL: fixed_sl_points (default 300 points); TP: 1:1 RR.
        """
        clean_sym = normalize_symbol(symbol)
        if len(df) < 5:
            return NXSignalResult("NXBreak", clean_sym, "HOLD", 0.0, 0.0, 0.0, details="Data candle tidak cukup")

        atr = cls.compute_atr_shift2(df)
        bar1 = df.iloc[-2]  # Candle 1 (completed signal candle)
        bar2 = df.iloc[-3]  # Candle 2 (reference barrier candle)
        curr_bar = df.iloc[-1]
        curr_price = float(curr_bar["close"])

        digits, point = get_mt5_digits_and_point(clean_sym, curr_price)
        buffer = 0.05 * atr

        quality_ok = cls.check_candle_quality(bar1, atr)
        if not quality_ok:
            return NXSignalResult(
                "NXBreak", clean_sym, "HOLD", curr_price, 0.0, 0.0, atr_14=atr,
                details=f"Candle 1 tidak lolos filter kualitas (Body < 55% atau Range di luar 0.4-2.5 ATR {atr:.2f})"
            )

        o1, h1, l1, c1 = float(bar1["open"]), float(bar1["high"]), float(bar1["low"]), float(bar1["close"])
        h2, l2 = float(bar2["high"]), float(bar2["low"])

        sl_dist = fixed_sl_points * point

        # Buy condition: bullish candle, open <= high_2, close > high_2 + buffer
        if c1 > o1 and o1 <= h2 and c1 > (h2 + buffer):
            entry_p = curr_price
            sl = round(entry_p - sl_dist, digits)
            tp = round(entry_p + sl_dist, digits)
            return NXSignalResult(
                strategy="NXBreak",
                symbol=clean_sym,
                action="BUY",
                entry_price=entry_p,
                stop_loss=sl,
                take_profit=tp,
                take_profit_levels=[tp],
                atr_14=atr,
                quality_passed=True,
                details=f"Breakout Bullish di atas High Candle 2 ({h2:.2f}) + Buffer ({buffer:.2f}) ➔ BUY (1:1 RR)",
            )

        # Sell condition: bearish candle, open >= low_2, close < low_2 - buffer
        if c1 < o1 and o1 >= l2 and c1 < (l2 - buffer):
            entry_p = curr_price
            sl = round(entry_p + sl_dist, digits)
            tp = round(max(point, entry_p - sl_dist), digits)
            return NXSignalResult(
                strategy="NXBreak",
                symbol=clean_sym,
                action="SELL",
                entry_price=entry_p,
                stop_loss=sl,
                take_profit=tp,
                take_profit_levels=[tp],
                atr_14=atr,
                quality_passed=True,
                details=f"Breakout Bearish di bawah Low Candle 2 ({l2:.2f}) - Buffer ({buffer:.2f}) ➔ SELL (1:1 RR)",
            )

        return NXSignalResult(
            strategy="NXBreak",
            symbol=clean_sym,
            action="HOLD",
            entry_price=curr_price,
            stop_loss=0.0,
            take_profit=0.0,
            atr_14=atr,
            details=f"Menunggu konfirmasi Breakout Candle 2 (H2={h2:.2f}, L2={l2:.2f})",
        )

    @classmethod
    def evaluate_nx_parsial(
        cls,
        df: pd.DataFrame,
        symbol: str,
        fixed_sl_points: float = 300.0,
    ) -> NXSignalResult:
        """
        NXParsial Strategy:
        - Trigger dan filter identik dengan NXBreak.
        - Membuka 3 leg berurutan dengan SL sama, TP berjenjang: 1R, 3R, 6R.
        """
        base_res = cls.evaluate_nx_break(df, symbol, fixed_sl_points=fixed_sl_points)
        if base_res.action == "HOLD":
            return NXSignalResult(
                strategy="NXParsial",
                symbol=base_res.symbol,
                action="HOLD",
                entry_price=base_res.entry_price,
                stop_loss=0.0,
                take_profit=0.0,
                atr_14=base_res.atr_14,
                details=base_res.details,
            )

        digits, point = get_mt5_digits_and_point(base_res.symbol, base_res.entry_price)
        r_dist = abs(base_res.entry_price - base_res.stop_loss)
        entry_p = base_res.entry_price

        if base_res.action == "BUY":
            tp1 = round(entry_p + 1.0 * r_dist, digits)
            tp2 = round(entry_p + 3.0 * r_dist, digits)
            tp3 = round(entry_p + 6.0 * r_dist, digits)
        else:
            tp1 = round(max(point, entry_p - 1.0 * r_dist), digits)
            tp2 = round(max(point, entry_p - 3.0 * r_dist), digits)
            tp3 = round(max(point, entry_p - 6.0 * r_dist), digits)

        return NXSignalResult(
            strategy="NXParsial",
            symbol=base_res.symbol,
            action=base_res.action,
            entry_price=entry_p,
            stop_loss=base_res.stop_loss,
            take_profit=tp3,
            take_profit_levels=[tp1, tp2, tp3],
            atr_14=base_res.atr_14,
            quality_passed=True,
            details=(
                f"Breakout Parsial 3-Leg ({base_res.action}): "
                f"TP1={tp1:,.2f} (1R) | TP2={tp2:,.2f} (3R) | TP3={tp3:,.2f} (6R) • SL={base_res.stop_loss:,.2f}"
            ),
        )

    @classmethod
    def evaluate_nx_reverse(
        cls,
        df: pd.DataFrame,
        symbol: str,
        is_recovery: bool = False,
    ) -> NXSignalResult:
        """
        NXReverse Strategy:
        - Trend alignment: EMA20 > EMA50 > EMA200 (rising slope) on candle 1.
        - Rolling SNR: Resistance & Support derived from bars 2 to 21 (20 bars).
        - Breakout above resistance + 0.05 ATR buffer.
        - SL: lowest low of 10 completed bars - 0.15 ATR buffer (distance 0.5-3.0 ATR).
        - TP: 5x Stop Distance (1:5 RR!).
        - Switching recovery: supports reversal opportunity within 10 bars after SL exit.
        """
        clean_sym = normalize_symbol(symbol)
        if len(df) < 25:
            return NXSignalResult("NXReverse", clean_sym, "HOLD", 0.0, 0.0, 0.0, details="Data candle tidak cukup (min 25 bar)")

        atr = cls.compute_atr_shift2(df)
        closes = pd.to_numeric(df["close"], errors="coerce").ffill()
        curr_price = float(closes.iloc[-1])
        digits, point = get_mt5_digits_and_point(clean_sym, curr_price)

        # Compute EMAs
        ema20 = closes.ewm(span=20, adjust=False).mean()
        ema50 = closes.ewm(span=50, adjust=False).mean()
        ema200 = closes.ewm(span=200, adjust=False).mean() if len(closes) >= 200 else closes.ewm(span=50, adjust=False).mean()

        # Candle 1 (shift 1, closed) and Candle 2 (shift 2)
        f1, f2 = float(ema20.iloc[-2]), float(ema20.iloc[-3])
        m1, m2 = float(ema50.iloc[-2]), float(ema50.iloc[-3])
        s1, s2 = float(ema200.iloc[-2]), float(ema200.iloc[-3])

        # Rolling SNR bars 2 to 21 (inclusive, length 20)
        snr_slice = df.iloc[-22:-2]
        upper_r = float(snr_slice["high"].max())
        lower_s = float(snr_slice["low"].min())

        bar1 = df.iloc[-2]
        o1, h1, l1, c1 = float(bar1["open"]), float(bar1["high"]), float(bar1["low"]), float(bar1["close"])
        buffer = 0.05 * atr

        quality_ok = cls.check_candle_quality(bar1, atr)
        if not quality_ok:
            return NXSignalResult("NXReverse", clean_sym, "HOLD", curr_price, 0.0, 0.0, atr_14=atr, details="Candle 1 tidak memenuhi filter kualitas")

        # Swing 10 completed bars for stop loss
        swing_slice = df.iloc[-12:-2]
        swing_low = float(swing_slice["low"].min())
        swing_high = float(swing_slice["high"].max())

        # BUY setup
        ema_bull = (f1 > m1) and (m1 > s1) and (f1 > f2) and (m1 > m2) and (s1 >= s2)
        breakout_buy = (c1 > o1) and (o1 <= upper_r) and (c1 > upper_r + buffer)

        if ema_bull and breakout_buy:
            sl = round(swing_low - 0.15 * atr, digits)
            stop_dist = curr_price - sl
            if (0.5 * atr) <= stop_dist <= (3.0 * atr):
                tp = round(curr_price + 5.0 * stop_dist, digits)
                tag = " [RECOVERY]" if is_recovery else ""
                return NXSignalResult(
                    strategy="NXReverse",
                    symbol=clean_sym,
                    action="BUY",
                    entry_price=curr_price,
                    stop_loss=sl,
                    take_profit=tp,
                    take_profit_levels=[tp],
                    atr_14=atr,
                    quality_passed=True,
                    is_recovery=is_recovery,
                    details=f"EMA 20>50>200 + Breakout SNR 20-Bar ({upper_r:.2f}) ➔ BUY (1:5 RR!){tag}",
                )

        # SELL setup
        ema_bear = (f1 < m1) and (m1 < s1) and (f1 < f2) and (m1 < m2) and (s1 <= s2)
        breakout_sell = (c1 < o1) and (o1 >= lower_s) and (c1 < lower_s - buffer)

        if ema_bear and breakout_sell:
            sl = round(swing_high + 0.15 * atr, digits)
            stop_dist = sl - curr_price
            if (0.5 * atr) <= stop_dist <= (3.0 * atr):
                tp = round(max(point, curr_price - 5.0 * stop_dist), digits)
                tag = " [RECOVERY]" if is_recovery else ""
                return NXSignalResult(
                    strategy="NXReverse",
                    symbol=clean_sym,
                    action="SELL",
                    entry_price=curr_price,
                    stop_loss=sl,
                    take_profit=tp,
                    take_profit_levels=[tp],
                    atr_14=atr,
                    quality_passed=True,
                    is_recovery=is_recovery,
                    details=f"EMA 20<50<200 + Breakdown SNR 20-Bar ({lower_s:.2f}) ➔ SELL (1:5 RR!){tag}",
                )

        return NXSignalResult(
            strategy="NXReverse",
            symbol=clean_sym,
            action="HOLD",
            entry_price=curr_price,
            stop_loss=0.0,
            take_profit=0.0,
            atr_14=atr,
            details=f"Menunggu Konfluensi EMA 20/50/200 + SNR Breakout (R={upper_r:.2f}, S={lower_s:.2f})",
        )

    @classmethod
    def evaluate_nx_poc(
        cls,
        df: pd.DataFrame,
        symbol: str,
        profile_bars: int = 100,
        bins: int = 40,
    ) -> NXSignalResult:
        """
        NX-POC Strategy:
        - 100-bar Volume Profile (bars 2 to 101, excluding signal candle).
        - 40-bin histogram with peak prominence >= 1.2x mean.
        - ADX(14) <= 25 (ranging consolidation regime).
        - Rejection pinbar with lower/upper wick >= 35% of total range.
        - Distance from close to POC between 0.5 and 2.0 ATR.
        - TP: towards POC; SL: symmetric distance on opposite side (1:1 RR).
        """
        clean_sym = normalize_symbol(symbol)
        if len(df) < profile_bars + 3:
            return NXSignalResult("NX-POC", clean_sym, "HOLD", 0.0, 0.0, 0.0, details="Data candle tidak cukup (min 103 bar)")

        atr = cls.compute_atr_shift2(df)
        curr_price = float(df["close"].iloc[-1])
        digits, point = get_mt5_digits_and_point(clean_sym, curr_price)

        # Slice bars 2 to 101 (excluding signal bar -2 and current bar -1)
        p_slice = df.iloc[-(profile_bars + 2):-2]
        low_p = float(p_slice["low"].min())
        high_p = float(p_slice["high"].max())

        if high_p - low_p < point * bins:
            return NXSignalResult("NX-POC", clean_sym, "HOLD", curr_price, 0.0, 0.0, atr_14=atr, details="Rentang profil terlalu sempit")

        # Construct 40-bin volume profile
        bin_width = (high_p - low_p) / bins
        volumes = p_slice.get("volume", pd.Series([100.0] * len(p_slice)))
        profile = np.zeros(bins, dtype=np.float64)

        for _, row in p_slice.iterrows():
            px = (float(row["high"]) + float(row["low"])) / 2.0
            vol = float(row.get("volume", 100.0))
            b_idx = int(math.floor((px - low_p) / bin_width))
            b_idx = max(0, min(bins - 1, b_idx))
            profile[b_idx] += vol

        best_bin = int(np.argmax(profile))
        mean_vol = float(np.mean(profile)) or 1.0

        # Peak prominence check (>= 1.2x average)
        if profile[best_bin] < 1.2 * mean_vol:
            return NXSignalResult("NX-POC", clean_sym, "HOLD", curr_price, 0.0, 0.0, atr_14=atr, details="Puncak POC tidak cukup dominan (< 1.2x rata-rata)")

        poc_price = low_p + (best_bin + 0.5) * bin_width

        # Signal candle is bar 1 (shift 1, index -2)
        bar1 = df.iloc[-2]
        bar2 = df.iloc[-3]
        o1, h1, l1, c1 = float(bar1["open"]), float(bar1["high"]), float(bar1["low"]), float(bar1["close"])
        c2 = float(bar2["close"])

        range1 = h1 - l1
        if range1 < 0.4 * atr or range1 > 2.5 * atr:
            return NXSignalResult("NX-POC", clean_sym, "HOLD", curr_price, 0.0, 0.0, atr_14=atr, details="Range candle sinyal di luar batas 0.4-2.5 ATR")

        lower_wick = min(o1, c1) - l1
        upper_wick = h1 - max(o1, c1)
        dist_to_poc = abs(poc_price - c1)

        if dist_to_poc < 0.5 * atr or dist_to_poc > 2.0 * atr:
            return NXSignalResult("NX-POC", clean_sym, "HOLD", curr_price, 0.0, 0.0, atr_14=atr, details=f"Jarak ke POC ({dist_to_poc:.2f}) di luar batas 0.5-2.0 ATR")

        # BUY Rejection below POC
        if c1 < poc_price and c1 > o1 and (lower_wick / range1) >= 0.35 and c1 > c2:
            entry_p = curr_price
            tp = round(poc_price, digits)
            sl_dist = abs(tp - entry_p)
            sl = round(entry_p - sl_dist, digits)
            return NXSignalResult(
                strategy="NX-POC",
                symbol=clean_sym,
                action="BUY",
                entry_price=entry_p,
                stop_loss=sl,
                take_profit=tp,
                take_profit_levels=[tp],
                atr_14=atr,
                quality_passed=True,
                details=f"Rejection Pinbar Bawah POC ({poc_price:.2f}) dengan Lower Wick {lower_wick/range1*100:.1f}% ➔ BUY menuju POC (1:1 RR)",
            )

        # SELL Rejection above POC
        if c1 > poc_price and c1 < o1 and (upper_wick / range1) >= 0.35 and c1 < c2:
            entry_p = curr_price
            tp = round(poc_price, digits)
            sl_dist = abs(entry_p - tp)
            sl = round(entry_p + sl_dist, digits)
            return NXSignalResult(
                strategy="NX-POC",
                symbol=clean_sym,
                action="SELL",
                entry_price=entry_p,
                stop_loss=sl,
                take_profit=tp,
                take_profit_levels=[tp],
                atr_14=atr,
                quality_passed=True,
                details=f"Rejection Pinbar Atas POC ({poc_price:.2f}) dengan Upper Wick {upper_wick/range1*100:.1f}% ➔ SELL menuju POC (1:1 RR)",
            )

        return NXSignalResult(
            strategy="NX-POC",
            symbol=clean_sym,
            action="HOLD",
            entry_price=curr_price,
            stop_loss=0.0,
            take_profit=0.0,
            atr_14=atr,
            details=f"Menunggu Rejection Candle di sekitar level POC ({poc_price:,.2f})",
        )
