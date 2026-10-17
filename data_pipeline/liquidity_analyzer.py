"""
Liquidity & Smart Money Concepts (SMC) Engine for NexusTrade AI:
Mengimplementasikan pembacaan likuiditas pasar, perangkap stop-loss ritel, dan perburuan likuiditas oleh pemain besar.

3 Pilar Filosofi Likuiditas:
1. Tandai Equal Highs (EQH) dan Equal Lows (EQL):
   - EQH: Puncak-puncak sejajar yang dianggap resisten kuat oleh ritel -> Sarang stop-loss seller.
   - EQL: Lembah-lembah sejajar yang dianggap support kuat oleh ritel -> Sarang stop-loss buyer.
2. Atas High = Buy Side Liquidity (BSL), Bawah Low = Sell Side Liquidity (SSL):
   - BSL: Kumpulan Buy Stop (SL seller). Magnet harga di atas.
   - SSL: Kumpulan Sell Stop (SL buyer). Magnet harga di bawah.
3. Jangan Taruh SL di Tempat yang Terlalu Jelas:
   - Jangan pasang SL tepat di level EQH/EQL yang menjadi sasaran empuk sapuan likuiditas (Liquidity Sweep).
   - Proteksi SL dengan Anti-Stop-Hunt Buffer di luar zona sapuan.
   - Entry terbaik adalah SETELAH sapuan likuiditas (Purge & Rejection) terjadi!
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from data_pipeline.market_symbols import normalize_symbol


@dataclass
class LiquidityPool:
    """Zona tumpukan likuiditas pasar (Stop-Loss Pool)."""
    kind: Literal["BSL", "SSL"]                        # BSL di atas High, SSL di bawah Low
    price_level: float                                  # Level harga acuan likuiditas
    pool_type: Literal["EQUAL_HIGHS", "EQUAL_LOWS", "SWING_EXTREME"]
    touch_count: int                                    # Berapa kali harga menyentuh level sejajar ini
    neatness_score: float                               # Kerapian level (0.0 - 1.0, makin rapi makin banyak SL menumpuk)
    is_swept: bool = False                              # Apakah sudah disapu/dijebol oleh sumbu lilin (Liquidity Sweep)
    sweep_candle_index: Optional[int] = None
    sweep_rejection_wick: float = 0.0                   # Persentase sumbu penolakan pasca sapuan


@dataclass
class LiquiditySweepEvent:
    """Peristiwa sapuan likuiditas (Stop Hunt / Purge & Rejection)."""
    sweep_type: Literal["BULLISH_SSL_SWEEP", "BEARISH_BSL_SWEEP", "NONE"]
    pool_level: float
    sweep_extreme_price: float
    rejection_wick_ratio: float
    details: str
    is_fakeout_reversal: bool = False                   # True jika terjadi penolakan (bukan tembus lurus/displacement)


@dataclass
class SweepVsBreakoutAssessment:
    """Hasil diagnosis perbandingan mendalam antara Sweep (Jebakan) vs Breakout Asli."""
    level: float
    level_type: Literal["RESISTANCE_EQH", "SUPPORT_EQL", "KEY_HIGH", "KEY_LOW"]
    verdict: Literal["SWEEP_JEBAKAN", "BREAKOUT_ASLI", "BREAKDOWN_ASLI", "MENUNGGU_RETEST", "BELUM_TEMBUS"]
    pierce_type: Literal["WICK", "BODY", "BELUM_TEMBUS"]
    close_position: Literal["DI_DALAM_LEVEL", "DI_LUAR_LEVEL", "BELUM_TEMBUS"]
    retest_status: Literal["BERTAHAN_FLIP_SR", "GAGAL_BALIK_ARAH", "SEDANG_RETEST", "BELUM_RETEST"]
    retest_held: bool
    details: str
    three_checks_display: str

@dataclass
class LiquidityAnalysisResult:
    """Hasil diagnosis komprehensif peta likuiditas sebelum entry."""
    symbol: str
    timeframe: str
    current_price: float
    nearest_bsl: Optional[float] = None                # Buy Side Liquidity terdekat di atas harga
    nearest_ssl: Optional[float] = None                # Sell Side Liquidity terdekat di bawah harga
    equal_highs: List[LiquidityPool] = field(default_factory=list)
    equal_lows: List[LiquidityPool] = field(default_factory=list)
    active_pools: List[LiquidityPool] = field(default_factory=list)
    recent_sweep: Optional[LiquiditySweepEvent] = None
    
    # 3 Aturan Likuiditas
    rule_1_eq_summary: str = ""                        # Ringkasan Equal Highs / Lows
    rule_2_bsl_ssl_summary: str = ""                   # Posisi BSL & SSL
    rule_3_sl_placement_guide: str = ""                # Rekomendasi penempatan SL aman dari perburuan
    smart_entry_bias: Literal["BUY", "SELL", "WAIT"] = "WAIT"
    three_rules_display: str = ""
    sweep_vs_breakout: Optional[SweepVsBreakoutAssessment] = None


class LiquidityAnalyzer:
    """Mesin analisis likuiditas institusional (Liquidity & Stop Hunt Engine)."""

    @classmethod
    def detect_equal_highs_and_lows(
        cls,
        df: pd.DataFrame,
        tolerance_pct: float = 0.0012,  # Toleransi 0.12% untuk menganggap level sejajar
        window: int = 3,
    ) -> Tuple[List[LiquidityPool], List[LiquidityPool]]:
        """
        Aturan 1: Tandai Equal Highs (EQH) dan Equal Lows (EQL).
        Makin rapih levelnya, makin banyak SL yang menumpuk.
        """
        if len(df) < (window * 2 + 1):
            return [], []

        highs = pd.to_numeric(df["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(df["low"], errors="coerce").astype(np.float64).values
        n = len(df)

        swing_highs: List[Tuple[int, float]] = []
        swing_lows: List[Tuple[int, float]] = []

        for i in range(window, n - window):
            h_curr = highs[i]
            l_curr = lows[i]

            if all(h_curr >= highs[i - k] for k in range(1, window + 1)) and \
               all(h_curr >= highs[i + k] for k in range(1, window + 1)):
                swing_highs.append((i, float(h_curr)))

            if all(l_curr <= lows[i - k] for k in range(1, window + 1)) and \
               all(l_curr <= lows[i + k] for k in range(1, window + 1)):
                swing_lows.append((i, float(l_curr)))

        # Cari Equal Highs (Puncak-puncak sejajar dalam batas toleransi)
        eq_high_pools: List[LiquidityPool] = []
        visited_h = set()
        for i in range(len(swing_highs)):
            if i in visited_h:
                continue
            idx_i, p_i = swing_highs[i]
            cluster = [(idx_i, p_i)]
            for j in range(i + 1, len(swing_highs)):
                if j in visited_h:
                    continue
                idx_j, p_j = swing_highs[j]
                if abs(p_i - p_j) / max(p_i, 1e-6) <= tolerance_pct:
                    cluster.append((idx_j, p_j))
                    visited_h.add(j)

            if len(cluster) >= 2:
                visited_h.add(i)
                avg_level = float(np.mean([p for _, p in cluster]))
                diff_spread = float(np.max([p for _, p in cluster]) - np.min([p for _, p in cluster]))
                neatness = max(0.50, min(1.0, 1.0 - (diff_spread / max(avg_level * tolerance_pct, 1e-6))))
                eq_high_pools.append(
                    LiquidityPool(
                        kind="BSL",
                        price_level=round(avg_level, 4),
                        pool_type="EQUAL_HIGHS",
                        touch_count=len(cluster),
                        neatness_score=round(neatness, 4),
                    )
                )

        # Cari Equal Lows (Lembah-lembah sejajar dalam batas toleransi)
        eq_low_pools: List[LiquidityPool] = []
        visited_l = set()
        for i in range(len(swing_lows)):
            if i in visited_l:
                continue
            idx_i, p_i = swing_lows[i]
            cluster = [(idx_i, p_i)]
            for j in range(i + 1, len(swing_lows)):
                if j in visited_l:
                    continue
                idx_j, p_j = swing_lows[j]
                if abs(p_i - p_j) / max(p_i, 1e-6) <= tolerance_pct:
                    cluster.append((idx_j, p_j))
                    visited_l.add(j)

            if len(cluster) >= 2:
                visited_l.add(i)
                avg_level = float(np.mean([p for _, p in cluster]))
                diff_spread = float(np.max([p for _, p in cluster]) - np.min([p for _, p in cluster]))
                neatness = max(0.50, min(1.0, 1.0 - (diff_spread / max(avg_level * tolerance_pct, 1e-6))))
                eq_low_pools.append(
                    LiquidityPool(
                        kind="SSL",
                        price_level=round(avg_level, 4),
                        pool_type="EQUAL_LOWS",
                        touch_count=len(cluster),
                        neatness_score=round(neatness, 4),
                    )
                )

        return eq_high_pools, eq_low_pools

    @classmethod
    def detect_liquidity_sweep(
        cls,
        df: pd.DataFrame,
        bsl_levels: List[float],
        ssl_levels: List[float],
        lookback_bars: int = 5,
    ) -> Optional[LiquiditySweepEvent]:
        """
        Deteksi Sapuan Likuiditas (Liquidity Sweep / Stop Hunt / Purge & Rejection):
        - Bullish Sweep (SSL Sweep): Harga sempat menembus SSL/Equal Lows ke bawah, mengambil SL para buyer,
          namun lilin menolak kembali dan ditutup DI ATAS level tersebut (sumbu bawah panjang).
        - Bearish Sweep (BSL Sweep): Harga sempat menembus BSL/Equal Highs ke atas, mengambil SL para seller,
          namun lilin menolak kembali dan ditutup DI BAWAH level tersebut (sumbu atas panjang).
        - "Tidak setiap tembus itu jebakan": Jika lilin tembus dan ditutup solid (displacement), bukan sweep.
        """
        if len(df) < lookback_bars:
            return None

        subset = df.tail(lookback_bars)
        opens = pd.to_numeric(subset["open"], errors="coerce").astype(np.float64).values
        highs = pd.to_numeric(subset["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(subset["low"], errors="coerce").astype(np.float64).values
        closes = pd.to_numeric(subset["close"], errors="coerce").astype(np.float64).values

        n_bars = len(subset)
        # Evaluasi dari lilin terbaru mundur ke 3 lilin terakhir
        for i in range(n_bars - 1, max(-1, n_bars - 4), -1):
            o_bar, h_bar, l_bar, c_bar = opens[i], highs[i], lows[i], closes[i]
            bar_range = max(h_bar - l_bar, 1e-6)
            lower_wick = max(0.0, min(o_bar, c_bar) - l_bar)
            upper_wick = max(0.0, h_bar - max(o_bar, c_bar))

            lower_wick_ratio = lower_wick / bar_range
            upper_wick_ratio = upper_wick / bar_range

            # 1. Cek Bullish SSL Sweep (Menembus level SSL ke bawah lalu rejeksi naik)
            for ssl in ssl_levels:
                # Harga terendah sempat tembus di bawah SSL, tapi penutupan tetap di atas SSL
                if l_bar < ssl and c_bar >= ssl and lower_wick_ratio >= 0.35:
                    return LiquiditySweepEvent(
                        sweep_type="BULLISH_SSL_SWEEP",
                        pool_level=ssl,
                        sweep_extreme_price=l_bar,
                        rejection_wick_ratio=round(lower_wick_ratio, 4),
                        is_fakeout_reversal=True,
                        details=(
                            f"🎯 BULLISH LIQUIDITY SWEEP: Harga sempat menembus level SSL (${ssl:,.2f}) "
                            f"hingga ${l_bar:,.2f} untuk memancing SL para Buyer, lalu ditutup memantul kembali "
                            f"(Sumbu Bawah {lower_wick_ratio*100:.0f}%) ➔ Pemain besar menampung likuiditas, SIAP NAIK!"
                        ),
                    )

            # 2. Cek Bearish BSL Sweep (Menembus level BSL ke atas lalu rejeksi turun)
            for bsl in bsl_levels:
                # Harga tertinggi sempat tembus di atas BSL, tapi penutupan tetap di bawah BSL
                if h_bar > bsl and c_bar <= bsl and upper_wick_ratio >= 0.35:
                    return LiquiditySweepEvent(
                        sweep_type="BEARISH_BSL_SWEEP",
                        pool_level=bsl,
                        sweep_extreme_price=h_bar,
                        rejection_wick_ratio=round(upper_wick_ratio, 4),
                        is_fakeout_reversal=True,
                        details=(
                            f"🎯 BEARISH LIQUIDITY SWEEP: Harga sempat menembus level BSL (${bsl:,.2f}) "
                            f"hingga ${h_bar:,.2f} untuk memancing SL para Seller, lalu ditutup menekan kembali "
                            f"(Sumbu Atas {upper_wick_ratio*100:.0f}%) ➔ Pemain besar menampung likuiditas, SIAP TURUN!"
                        ),
                    )

        return None
    @classmethod
    def evaluate_sweep_vs_breakout(
        cls,
        df: pd.DataFrame,
        resistance_level: Optional[float] = None,
        support_level: Optional[float] = None,
        lookback_bars: int = 8,
    ) -> Optional[SweepVsBreakoutAssessment]:
        """
        Membedakan Sweep vs Breakout Asli berdasarkan 3 Checklist:
        1. Tembus pakai wick atau body?
        2. Candle nya close di mana? (Di luar atau di dalam level?)
        3. Saat retest, bertahan atau balik?

        Aturan Emas:
        - Jangan bereaksi saat harga baru menembus.
        - Tunggu candle nya close dulu, lalu lihat retest-nya!
        """
        if len(df) < lookback_bars:
            return None

        subset = df.tail(lookback_bars)
        opens = pd.to_numeric(subset["open"], errors="coerce").astype(np.float64).values
        highs = pd.to_numeric(subset["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(subset["low"], errors="coerce").astype(np.float64).values
        closes = pd.to_numeric(subset["close"], errors="coerce").astype(np.float64).values
        n = len(subset)

        # 1. Evaluasi Penembusan Resistance (Equal Highs / BSL)
        if resistance_level is not None and resistance_level > 0:
            r = resistance_level
            breach_idx = -1
            for i in range(n):
                if highs[i] > r:
                    breach_idx = i
                    break

            if breach_idx != -1:
                b_open, b_high, b_low, b_close = opens[breach_idx], highs[breach_idx], lows[breach_idx], closes[breach_idx]
                b_range = max(b_high - b_low, 1e-6)
                b_body = abs(b_close - b_open)
                upper_wick = max(0.0, b_high - max(b_open, b_close))

                # Skenario 1: SWEEP pada Resistance
                # Tembus hanya dengan sumbu (wick), candle close kembali di bawah level
                if b_close <= r:
                    pierce_type = "WICK"
                    close_pos = "DI_DALAM_LEVEL"
                    retest_st = "GAGAL_BALIK_ARAH"
                    verdict = "SWEEP_JEBAKAN"
                    detail = (
                        f"SWEEP JEBAKAN: Harga menembus Resistance (${r:,.2f}) hanya dengan sumbu (High ${b_high:,.2f}), "
                        f"tetapi candle CLOSE DI BAWAH (${b_close:,.2f}). Stop-Loss seller sudah diambil, breakout trader "
                        f"terjebak ➔ Waspada Berbalik Turun Cepat!"
                    )
                    held = False
                # Skenario 2: BREAKOUT pada Resistance
                # Body besar, close jelas di atas level
                else:
                    pierce_type = "BODY"
                    close_pos = "DI_LUAR_LEVEL"
                    # Cek retest pada candle setelah penembusan
                    post_bars_lows = lows[breach_idx + 1:] if breach_idx < n - 1 else []
                    post_bars_closes = closes[breach_idx + 1:] if breach_idx < n - 1 else []

                    if len(post_bars_lows) > 0:
                        # Apakah ada candle yang menguji kembali level resistance (retest)
                        retest_occurred = any(l <= r * 1.0015 for l in post_bars_lows)
                        retest_failed = any(c < r * 0.9995 for c in post_bars_closes)

                        if retest_occurred and not retest_failed:
                            verdict = "BREAKOUT_ASLI"
                            retest_st = "BERTAHAN_FLIP_SR"
                            detail = (
                                f"BREAKOUT ASLI: Candle tembus pakai BODY (Close ${b_close:,.2f} di atas Resistance ${r:,.2f}), "
                                f"kemudian retest berhasil BERTAHAN (Resistance berubah fungsi menjadi Support) ➔ Siap Lanjut Naik!"
                            )
                            held = True
                        elif retest_failed:
                            verdict = "SWEEP_JEBAKAN"
                            retest_st = "GAGAL_BALIK_ARAH"
                            detail = f"BREAKOUT GAGAL (FAKE BREAKOUT): Sempat close di atas, namun saat retest gagal bertahan dan jatuh kembali di bawah ${r:,.2f}."
                            held = False
                        else:
                            verdict = "MENUNGGU_RETEST"
                            retest_st = "BELUM_RETEST"
                            detail = f"CANDLE CLOSE DI ATAS RESISTANCE: Tanda buyer serius. JANGAN LANGSUNG FOMO! Tunggu retest ke level ${r:,.2f} dan pastikan bertahan."
                            held = False
                    else:
                        verdict = "MENUNGGU_RETEST"
                        retest_st = "BELUM_RETEST"
                        detail = f"Candle baru saja close di atas ${r:,.2f}. Tunggu candle berikutnya melakukan retest."
                        held = False

                checks = (
                    f"1. 🔍 TEMBUS PAKAI: {pierce_type}\n"
                    f"2. 🚪 CANDLE CLOSE: {close_pos} (Level ${r:,.2f})\n"
                    f"3. 🔄 STATUS RETEST: {retest_st} (Bertahan: {held})"
                )
                return SweepVsBreakoutAssessment(
                    level=r,
                    level_type="RESISTANCE_EQH",
                    verdict=verdict,
                    pierce_type=pierce_type,
                    close_position=close_pos,
                    retest_status=retest_st,
                    retest_held=held,
                    details=detail,
                    three_checks_display=checks,
                )

        # 2. Evaluasi Penembusan Support (Equal Lows / SSL)
        if support_level is not None and support_level > 0:
            s = support_level
            breach_idx = -1
            for i in range(n):
                if lows[i] < s:
                    breach_idx = i
                    break

            if breach_idx != -1:
                b_open, b_high, b_low, b_close = opens[breach_idx], highs[breach_idx], lows[breach_idx], closes[breach_idx]

                # Skenario 1: SWEEP pada Support
                # Tembus hanya dengan sumbu, candle close kembali di atas level
                if b_close >= s:
                    pierce_type = "WICK"
                    close_pos = "DI_DALAM_LEVEL"
                    retest_st = "GAGAL_BALIK_ARAH"
                    verdict = "SWEEP_JEBAKAN"
                    detail = (
                        f"SWEEP JEBAKAN: Harga menembus Support (${s:,.2f}) hanya dengan sumbu (Low ${b_low:,.2f}), "
                        f"tetapi candle CLOSE DI ATAS (${b_close:,.2f}). Stop-Loss buyer sudah diambil, breakdown trader "
                        f"terjebak ➔ Waspada Berbalik Naik Cepat!"
                    )
                    held = False
                # Skenario 2: BREAKDOWN pada Support
                # Body besar, close jelas di bawah level
                else:
                    pierce_type = "BODY"
                    close_pos = "DI_LUAR_LEVEL"
                    post_bars_highs = highs[breach_idx + 1:] if breach_idx < n - 1 else []
                    post_bars_closes = closes[breach_idx + 1:] if breach_idx < n - 1 else []

                    if len(post_bars_highs) > 0:
                        retest_occurred = any(h >= s * 0.9985 for h in post_bars_highs)
                        retest_failed = any(c > s * 1.0005 for c in post_bars_closes)

                        if retest_occurred and not retest_failed:
                            verdict = "BREAKDOWN_ASLI"
                            retest_st = "BERTAHAN_FLIP_SR"
                            detail = (
                                f"BREAKDOWN ASLI: Candle tembus pakai BODY (Close ${b_close:,.2f} di bawah Support ${s:,.2f}), "
                                f"kemudian retest berhasil BERTAHAN di bawah level (Support berubah fungsi jadi Resistance) ➔ Siap Lanjut Turun!"
                            )
                            held = True
                        elif retest_failed:
                            verdict = "SWEEP_JEBAKAN"
                            retest_st = "GAGAL_BALIK_ARAH"
                            detail = f"BREAKDOWN GAGAL (FAKE BREAKDOWN): Sempat close di bawah, namun saat retest melonjak kembali di atas ${s:,.2f}."
                            held = False
                        else:
                            verdict = "MENUNGGU_RETEST"
                            retest_st = "BELUM_RETEST"
                            detail = f"CANDLE CLOSE DI BAWAH SUPPORT: Tanda seller serius. JANGAN LANGSUNG FOMO! Tunggu retest ke level ${s:,.2f} dan pastikan bertahan."
                            held = False
                    else:
                        verdict = "MENUNGGU_RETEST"
                        retest_st = "BELUM_RETEST"
                        detail = f"Candle baru saja close di bawah ${s:,.2f}. Tunggu candle berikutnya melakukan retest."
                        held = False

                checks = (
                    f"1. 🔍 TEMBUS PAKAI: {pierce_type}\n"
                    f"2. 🚪 CANDLE CLOSE: {close_pos} (Level ${s:,.2f})\n"
                    f"3. 🔄 STATUS RETEST: {retest_st} (Bertahan: {held})"
                )
                return SweepVsBreakoutAssessment(
                    level=s,
                    level_type="SUPPORT_EQL",
                    verdict=verdict,
                    pierce_type=pierce_type,
                    close_position=close_pos,
                    retest_status=retest_st,
                    retest_held=held,
                    details=detail,
                    three_checks_display=checks,
                )

        return None


    @classmethod
    def analyze_liquidity(
        cls,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: str = "5m",
        tolerance_pct: float = 0.0012,
    ) -> LiquidityAnalysisResult:
        """
        Diagnosis Komprehensif 3 Aturan Likuiditas:
        1. Tandai Equal Highs (EQH) dan Equal Lows (EQL).
        2. Atas High = BSL, Bawah Low = SSL (Magnet Harga).
        3. Jangan taruh SL di tempat yang paling jelas (Panduan Penempatan SL Aman).
        """
        clean_sym = symbol.upper()
        curr_price = float(df["close"].iloc[-1]) if len(df) > 0 else 2650.0

        # Aturan 1: Deteksi Equal Highs & Equal Lows
        eq_highs, eq_lows = cls.detect_equal_highs_and_lows(df, tolerance_pct=tolerance_pct)

        # Cari Swing Puncak & Lembah Ekstrem Terakhir
        highs = pd.to_numeric(df["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(df["low"], errors="coerce").astype(np.float64).values
        recent_sh = float(np.max(highs[-20:])) if len(highs) >= 20 else float(np.max(highs))
        recent_sl = float(np.min(lows[-20:])) if len(lows) >= 20 else float(np.min(lows))

        # Aturan 2: Identifikasi Level BSL (Atas High) dan SSL (Bawah Low)
        all_bsl_levels = [p.price_level for p in eq_highs] + [recent_sh]
        all_ssl_levels = [p.price_level for p in eq_lows] + [recent_sl]

        # Ambil BSL dan SSL yang paling dekat dengan harga saat ini
        above_levels = [lvl for lvl in all_bsl_levels if lvl > curr_price]
        below_levels = [lvl for lvl in all_ssl_levels if lvl < curr_price]

        nearest_bsl = min(above_levels) if above_levels else recent_sh
        nearest_ssl = max(below_levels) if below_levels else recent_sl

        # Deteksi apakah baru saja terjadi sapuan likuiditas (Liquidity Sweep)
        sweep_event = cls.detect_liquidity_sweep(
            df=df,
            bsl_levels=all_bsl_levels,
            ssl_levels=all_ssl_levels,
            lookback_bars=6,
        )

        # Evaluasi Mendalam: Sweep (Jebakan) vs Breakout Asli
        svb_assessment = cls.evaluate_sweep_vs_breakout(
            df=df,
            resistance_level=nearest_bsl if nearest_bsl > 0 else None,
            support_level=nearest_ssl if nearest_ssl > 0 else None,
            lookback_bars=8,
        )

        # Susun Ringkasan Aturan 1
        if eq_highs and eq_lows:
            r1_text = f"Terdeteksi {len(eq_highs)} Equal Highs (${eq_highs[0].price_level:,.2f}) & {len(eq_lows)} Equal Lows (${eq_lows[0].price_level:,.2f})"
        elif eq_highs:
            r1_text = f"Terdeteksi Equal Highs rapi di ${eq_highs[0].price_level:,.2f} ({eq_highs[0].touch_count}x sentuhan, Kerapian {eq_highs[0].neatness_score*100:.0f}%)"
        elif eq_lows:
            r1_text = f"Terdeteksi Equal Lows rapi di ${eq_lows[0].price_level:,.2f} ({eq_lows[0].touch_count}x sentuhan, Kerapian {eq_lows[0].neatness_score*100:.0f}%)"
        else:
            r1_text = f"Puncak (${recent_sh:,.2f}) & Lembah (${recent_sl:,.2f}) Tersebar Normal (Tanpa Equal High/Low mencolok)"

        # Susun Ringkasan Aturan 2
        r2_text = (
            f"MAGNET LIKUIDITAS: Buy Side Liquidity (BSL) di atas ${nearest_bsl:,.2f} (SL Seller) | "
            f"Sell Side Liquidity (SSL) di bawah ${nearest_ssl:,.2f} (SL Buyer)"
        )

        # Susun Ringkasan Aturan 3: Panduan Penempatan SL Aman
        if sweep_event is not None and sweep_event.sweep_type == "BULLISH_SSL_SWEEP":
            smart_bias = "BUY"
            r3_text = (
                f"🛡️ SL AMAN BUY: Sapuan SSL telah selesai di ${sweep_event.sweep_extreme_price:,.2f}. "
                f"Taruh SL di bawah jarum sweep (${sweep_event.sweep_extreme_price - 1.0:,.2f}), BUKAN di level rata Equal Lows."
            )
        elif sweep_event is not None and sweep_event.sweep_type == "BEARISH_BSL_SWEEP":
            smart_bias = "SELL"
            r3_text = (
                f"🛡️ SL AMAN SELL: Sapuan BSL telah selesai di ${sweep_event.sweep_extreme_price:,.2f}. "
                f"Taruh SL di atas jarum sweep (${sweep_event.sweep_extreme_price + 1.0:,.2f}), BUKAN di level rata Equal Highs."
            )
        else:
            smart_bias = "WAIT"
            r3_text = (
                f"🛡️ JANGAN TARUH SL DI TEMPAT JELAS: Hindari pasang SL mepet tepat di ${nearest_ssl:,.2f} atau ${nearest_bsl:,.2f}. "
                f"Beri ruang penyangga (Anti-Sweep Buffer) agar tidak menjadi mangsa perburuan likuiditas."
            )

        three_rules_card = (
            f"1. 🎯 EQUAL HIGHS & LOWS: {r1_text}\n"
            f"2. 🧲 PETA LIKUIDITAS: {r2_text}\n"
            f"3. 🛡️ PENEMPATAN SL CERDAS: {r3_text}\n"
            f"4. ⚡ STATUS SAPUAN (SWEEP): {sweep_event.details if sweep_event else (svb_assessment.details if svb_assessment else 'Tidak ada sapuan instan dalam 3 bar terakhir')}"
        )
        if svb_assessment is not None:
            three_rules_card += f"\n5. ⚖️ EVALUASI TEMBUSAN: {svb_assessment.verdict}\n{svb_assessment.three_checks_display}"

        all_pools = eq_highs + eq_lows

        return LiquidityAnalysisResult(
            symbol=clean_sym,
            timeframe=timeframe,
            current_price=curr_price,
            nearest_bsl=nearest_bsl,
            nearest_ssl=nearest_ssl,
            equal_highs=eq_highs,
            equal_lows=eq_lows,
            active_pools=all_pools,
            recent_sweep=sweep_event,
            rule_1_eq_summary=r1_text,
            rule_2_bsl_ssl_summary=r2_text,
            rule_3_sl_placement_guide=r3_text,
            smart_entry_bias=smart_bias,
            sweep_vs_breakout=svb_assessment,
            three_rules_display=three_rules_card,
        )
