"""
Market Structure Analyzer (Analisis Struktur Pasar) for NexusTrade AI:
Mengimplementasikan pembacaan struktur pasar murni berdasarkan 4 Label Universal:
  - HH (Higher High)
  - HL (Higher Low)
  - LH (Lower High)
  - LL (Lower Low)

3 Pertanyaan Wajib Sebelum Entry:
1. Mana Swing High dan Swing Low yang jelas? (Hanya puncak & lembah menonjol, abaikan goyangan kecil).
2. HH-HL (Uptrend), LH-LL (Downtrend), atau Range (Sideways)?
3. Swing mana yang menjaga tren?
   - Uptrend: Dijaga oleh HL Terakhir (Jika dijebol ke bawah -> Uptrend Patah / Abnormal).
   - Downtrend: Dijaga oleh LH Terakhir (Jika dijebol ke atas -> Downtrend Patah / Abnormal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    """Titik Puncak (Swing High) atau Lembah (Swing Low) yang terverifikasi."""
    index: int
    price: float
    kind: Literal["HIGH", "LOW"]
    label: Literal["HH", "HL", "LH", "LL", "EQUAL_HIGH", "EQUAL_LOW", "FIRST"]
    bar_timestamp: str = ""
    prominence_atr: float = 0.0


@dataclass
class PullbackWaveAnalysis:
    """Hasil diagnosis karakter gelombang Impuls vs Koreksi/Pullback dan Kesehatannya."""
    current_phase: Literal["IMPULSE", "PULLBACK", "PULLBACK_COMPLETED", "CONSOLIDATION"]
    is_healthy: bool
    retest_depth_pct: float
    overlap_ratio: float
    guardian_level: float
    guardian_kind: str
    phase_explanation: str
    three_golden_rules: str

@dataclass
class StructureBreakEvent:
    """Peristiwa penembusan struktur pasar: Break of Structure (BOS) atau Market Structure Shift (MSS)."""
    event_type: Literal["BULLISH_BOS", "BEARISH_BOS", "BEARISH_MSS", "BULLISH_MSS", "NONE"]
    level_broken: float
    broken_swing_label: str
    broken_swing_kind: Literal["HIGH", "LOW", "NONE"]
    break_close_price: float
    is_confirmed_by_close: bool
    direction_meaning: Literal["CONTINUATION", "REVERSAL", "NONE"]
    details: str

@dataclass
class MarketStructureResult:
    """Hasil diagnosis struktur pasar komprehensif sebelum entry."""
    symbol: str
    timeframe: str
    current_price: float
    regime: Literal["UPTREND", "DOWNTREND", "RANGE_SIDEWAYS"]
    pattern_label: Literal["HH-HL", "LH-LL", "RANGE"]
    
    # 3 Pertanyaan Wajib Sebelum Entry
    # Pertanyaan 1: Di mana Swing High & Swing Low yang jelas?
    latest_swing_high: float
    latest_swing_low: float
    all_swings: List[SwingPoint] = field(default_factory=list)
    recent_swing_highs: List[float] = field(default_factory=list)
    recent_swing_lows: List[float] = field(default_factory=list)
    
    # Pertanyaan 2: HH-HL, LH-LL, atau Range?
    structure_summary: str = ""
    
    # Pertanyaan 3: Swing mana yang menjaga tren?
    guardian_swing_kind: Literal["HL", "LH", "RANGE_BOUNDS"] = "HL"
    guardian_price_level: float = 0.0
    is_trend_normal: bool = True     # True jika HL/LH penjaga belum jebol
    trend_status_detail: str = ""    # "NORMAL (HL Terjaga)" vs "ABNORMAL / PATAH (HL Jebol)"

    # Rekomendasi Arah Entry Berdasarkan Struktur
    structural_bias: Literal["BUY", "SELL", "WAIT"] = "WAIT"
    entry_allowed: bool = False
    pullback_analysis: Optional[PullbackWaveAnalysis] = None
    latest_structure_event: Optional[StructureBreakEvent] = None
    bos_count: int = 0
    mss_occurred: bool = False
    structure_peta_display: str = ""
    three_checks_display: str = ""

@dataclass
class TopDownStructureAnalysis:
    """Hasil integrasi analisa HTF (Arah/Tren Utama) vs LTF (Eksekusi)."""
    symbol: str
    htf_name: str
    ltf_name: str
    htf_regime: Literal["UPTREND", "DOWNTREND", "RANGE_SIDEWAYS"]
    htf_pattern: Literal["HH-HL", "LH-LL", "RANGE"]
    macro_bias: Literal["BUY_ONLY", "SELL_ONLY", "NEUTRAL"]
    macro_instruction: str
    htf_structure: MarketStructureResult
    ltf_structure: MarketStructureResult
    is_aligned: bool
    execution_verdict: str


class MarketStructureAnalyzer:
    """Mesin analisis struktur pasar (Market Structure Engine)."""

    @classmethod
    def identify_prominent_swings(
        cls,
        df: pd.DataFrame,
        window: int = 4,
        min_prominence_pct: float = 0.0015,
    ) -> List[SwingPoint]:
        """
        Langkah Pertama: Kenali Swing yang JELAS.
        Abaikan goyangan kecil di tengah jalan; hanya tandai puncak & lembah yang menonjol.
        - Swing High: High lebih tinggi dari 'window' bar di kiri dan kanannya.
        - Swing Low: Low lebih rendah dari 'window' bar di kiri dan kanannya.
        """
        if len(df) < (window * 2 + 1):
            return []

        highs = pd.to_numeric(df["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(df["low"], errors="coerce").astype(np.float64).values
        n = len(df)

        raw_swings: List[Tuple[int, float, Literal["HIGH", "LOW"]]] = []

        for i in range(window, n - window):
            h_curr = highs[i]
            l_curr = lows[i]

            # Swing High: puncak gelombang di mana kiri & kanannya lebih rendah
            is_sh = all(h_curr >= highs[i - k] for k in range(1, window + 1)) and \
                    all(h_curr >= highs[i + k] for k in range(1, window + 1))

            # Swing Low: lembah gelombang di mana kiri & kanannya lebih tinggi
            is_sl = all(l_curr <= lows[i - k] for k in range(1, window + 1)) and \
                    all(l_curr <= lows[i + k] for k in range(1, window + 1))

            if is_sh and not is_sl:
                raw_swings.append((i, float(h_curr), "HIGH"))
            elif is_sl and not is_sh:
                raw_swings.append((i, float(l_curr), "LOW"))

        # Filter goyangan kecil: Pastikan swing bergantian (High -> Low -> High)
        # dan memiliki jarak harga yang cukup (noise filtering)
        filtered: List[Tuple[int, float, Literal["HIGH", "LOW"]]] = []
        for s in raw_swings:
            if not filtered:
                filtered.append(s)
                continue
            prev = filtered[-1]
            if s[2] == prev[2]:
                # Jenis sama berturut-turut: ambil puncak yang lebih tinggi atau lembah yang lebih rendah
                if s[2] == "HIGH" and s[1] > prev[1]:
                    filtered[-1] = s
                elif s[2] == "LOW" and s[1] < prev[1]:
                    filtered[-1] = s
            else:
                # Memeriksa selisih harga minimum agar bukan sekadar goyangan semu
                diff_pct = abs(s[1] - prev[1]) / max(prev[1], 1e-6)
                if diff_pct >= min_prominence_pct:
                    filtered.append(s)

        # Beri 4 Label Standar: HH, HL, LH, LL
        labeled_swings: List[SwingPoint] = []
        last_high: Optional[float] = None
        last_low: Optional[float] = None

        for idx, px, kind in filtered:
            if kind == "HIGH":
                if last_high is None:
                    lbl = "FIRST"
                elif px > last_high * 1.0001:
                    lbl = "HH"  # Higher High: Puncak yang lebih tinggi
                elif px < last_high * 0.9999:
                    lbl = "LH"  # Lower High: Puncak yang lebih rendah
                else:
                    lbl = "EQUAL_HIGH"
                last_high = px
            else:
                if last_low is None:
                    lbl = "FIRST"
                elif px > last_low * 1.0001:
                    lbl = "HL"  # Higher Low: Lembah yang lebih tinggi
                elif px < last_low * 0.9999:
                    lbl = "LL"  # Lower Low: Lembah yang lebih rendah
                else:
                    lbl = "EQUAL_LOW"
                last_low = px

            labeled_swings.append(SwingPoint(index=idx, price=px, kind=kind, label=lbl))

        return labeled_swings

    @classmethod
    def analyze_structure(
        cls,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: str = "1m",
        window: int = 4,
    ) -> MarketStructureResult:
        """
        Mengevaluasi struktur pasar dan menjalankan 3 Checklist Wajib sebelum entry:
        1. Mana Swing High & Swing Low yang jelas?
        2. HH-HL, LH-LL, atau Range?
        3. Swing mana yang menjaga tren? (Apakah tren normal atau patah?)
        """
        clean_sym = symbol.upper()
        curr_price = float(df["close"].iloc[-1]) if len(df) > 0 else 2650.0

        swings = cls.identify_prominent_swings(df, window=window)

        all_highs = [s.price for s in swings if s.kind == "HIGH"]
        all_lows = [s.price for s in swings if s.kind == "LOW"]

        lat_sh = all_highs[-1] if all_highs else float(df["high"].max())
        lat_sl = all_lows[-1] if all_lows else float(df["low"].min())

        # Ambil 4 label swing terakhir
        high_labels = [s.label for s in swings if s.kind == "HIGH"]
        low_labels = [s.label for s in swings if s.kind == "LOW"]

        lat_h_label = high_labels[-1] if high_labels else "HIGH"
        lat_l_label = low_labels[-1] if low_labels else "LOW"

        # Kumpulkan swing HL terakhir dan LH terakhir
        hl_points = [s.price for s in swings if s.kind == "LOW" and s.label == "HL"]
        latest_hl = hl_points[-1] if hl_points else lat_sl

        lh_points = [s.price for s in swings if s.kind == "HIGH" and s.label == "LH"]
        latest_lh = lh_points[-1] if lh_points else lat_sh

        # Klasifikasi 2: HH-HL (Uptrend), LH-LL (Downtrend), atau Range?
        is_uptrend = (lat_h_label == "HH" or "HH" in high_labels[-2:]) and (lat_l_label == "HL" or "HL" in low_labels[-2:])
        is_downtrend = (lat_h_label == "LH" or "LH" in high_labels[-2:]) and (lat_l_label == "LL" or "LL" in low_labels[-2:])

        if is_uptrend:
            regime = "UPTREND"
            pattern_label = "HH-HL"
            guardian_kind = "HL"
            guardian_price = latest_hl
            # Kapan uptrend masih normal? Selama HL terakhir tidak ditembus ke bawah
            is_normal = curr_price >= latest_hl
            if is_normal:
                status_detail = f"NORMAL: HL Terakhir (${latest_hl:,.2f}) Terjaga Kuat ➔ Uptrend Utuh"
                bias = "BUY"
                allowed = True
            else:
                status_detail = f"ABNORMAL / PATAH: HL Terakhir (${latest_hl:,.2f}) Ditembus ke Bawah ➔ Struktur Uptrend Rusak"
                bias = "WAIT"
                allowed = False
            summary = "📈 UPTREND: Pola HH dan HL Terkonfirmasi Berulang"

        elif is_downtrend:
            regime = "DOWNTREND"
            pattern_label = "LH-LL"
            guardian_kind = "LH"
            guardian_price = latest_lh
            # Kapan downtrend masih normal? Selama LH terakhir belum ditembus ke atas
            is_normal = curr_price <= latest_lh
            if is_normal:
                status_detail = f"NORMAL: LH Terakhir (${latest_lh:,.2f}) Terjaga Kuat ➔ Downtrend Utuh"
                bias = "SELL"
                allowed = True
            else:
                status_detail = f"ABNORMAL / PATAH: LH Terakhir (${latest_lh:,.2f}) Ditembus ke Atas ➔ Struktur Downtrend Rusak"
                bias = "WAIT"
                allowed = False
            summary = "📉 DOWNTREND: Pola LH dan LL Terkonfirmasi Berulang"

        else:
            regime = "RANGE_SIDEWAYS"
            pattern_label = "RANGE"
            guardian_kind = "RANGE_BOUNDS"
            guardian_price = lat_sh
            is_normal = True
            status_detail = f"RANGE / SIDEWAYS: Puncak (${lat_sh:,.2f}) & Lembah (${lat_sl:,.2f}) Sejajar Tanpa Arah Jelas"
            bias = "WAIT"
            allowed = False
            summary = "↔️ RANGE / SIDEWAYS: Puncak & Lembah Sejajar, Struktur Belum Memberi Arah Pasti"

        # Analisa Karakter Gelombang: Impuls vs Koreksi/Pullback & Kesehatannya
        pb_res = cls.analyze_impulse_and_pullback(
            df=df,
            regime=regime,
            latest_sh=lat_sh,
            latest_sl=lat_sl,
            latest_hl=latest_hl,
            latest_lh=latest_lh,
        )

        # Deteksi Break of Structure (BOS) & Market Structure Shift (MSS)
        struct_event, bos_cnt, mss_flag = cls.detect_bos_and_mss(
            df=df,
            swings=swings,
            regime=regime,
            latest_hl=latest_hl,
            latest_lh=latest_lh,
            latest_sh=lat_sh,
            latest_sl=lat_sl,
        )

        # Jika terjadi MSS, struktur berbalik arah secara seketika!
        if struct_event is not None and struct_event.event_type == "BEARISH_MSS":
            regime = "DOWNTREND"
            pattern_label = "LH-LL"
            bias = "SELL"
            guardian_kind = "LH"
            guardian_price = latest_lh
            status_detail = struct_event.details
            summary = "📉 MSS BEARISH: Buyer Kehilangan Kendali, Struktur Berubah Jadi Downtrend"
        elif struct_event is not None and struct_event.event_type == "BULLISH_MSS":
            regime = "UPTREND"
            pattern_label = "HH-HL"
            bias = "BUY"
            guardian_kind = "HL"
            guardian_price = latest_hl
            status_detail = struct_event.details
            summary = "📈 MSS BULLISH: Seller Kehilangan Kendali, Struktur Berubah Jadi Uptrend"

        event_str = f"{struct_event.event_type} ({struct_event.direction_meaning})" if struct_event else "TIDAK_ADA_EVENT_BARU"

        # Tampilan 3 Checklist Wajib + 3 Aturan Trend & Pullback + 3 Aturan BOS/MSS
        checks_text = (
            f"1. 🎯 SWING HIGH & LOW MENONJOL: Swing High ${lat_sh:,.2f} | Swing Low ${lat_sl:,.2f}\n"
            f"2. 📊 POLA STRUKTUR: {pattern_label} ({summary})\n"
            f"3. 🛡️ SWING PENJAGA TREN: {guardian_kind} @ ${guardian_price:,.2f} ➔ {status_detail}\n"
            f"4. 🌊 FASE GELOMBANG: {pb_res.current_phase} ➔ {pb_res.phase_explanation}\n"
            f"5. ☕ KESEHATAN PULLBACK: {'🟢 SEHAT (HL/LH Terjaga)' if pb_res.is_healthy else '🔴 RUSAK / ABNORMAL'}\n"
            f"6. 🗺️ PERISTIWA STRUKTUR: {event_str}\n"
            f"   ↳ {struct_event.details if struct_event else 'Belum ada BOS atau MSS baru dalam 8 bar terakhir.'}"
        )
        return MarketStructureResult(
            symbol=clean_sym,
            timeframe=timeframe,
            current_price=curr_price,
            regime=regime,
            pattern_label=pattern_label,
            latest_swing_high=lat_sh,
            latest_swing_low=lat_sl,
            all_swings=swings,
            recent_swing_highs=all_highs[-3:],
            recent_swing_lows=all_lows[-3:],
            structure_summary=summary,
            guardian_swing_kind=guardian_kind,
            guardian_price_level=guardian_price,
            is_trend_normal=is_normal,
            trend_status_detail=status_detail,
            structural_bias=bias,
            entry_allowed=allowed,
            three_checks_display=checks_text,
            latest_structure_event=struct_event,
            bos_count=bos_cnt,
            mss_occurred=mss_flag,
            structure_peta_display=event_str,
            pullback_analysis=pb_res,
        )

    @classmethod
    def evaluate_top_down_htf_ltf(
        cls,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
        symbol: str = "XAUUSD",
        htf_name: str = "H1",
        ltf_name: str = "M5",
        proposed_action: Optional[Literal["BUY", "SELL"]] = None,
    ) -> TopDownStructureAnalysis:
        """
        Filosofi Timeframe Besar (HTF) vs Timeframe Kecil (LTF):
        - Timeframe Besar (H1/D1) = HTF. Dipakai untuk menentukan arah/tren utama market.
          Di timeframe besar, jika puncak dan lembahnya makin tinggi (HH-HL) -> Struktur Uptrend -> Arah utama: CARI NAIK.
          Sebaliknya, jika puncak dan lembahnya makin rendah (LH-LL) -> Struktur Downtrend -> Arah utama: CARI TURUN.
        - Timeframe Kecil (M5/M15/M1) = LTF. Dipakai khusus untuk timing eksekusi.
        - Syarat Eksekusi: LTF dilarang keras melawan arah arus tren HTF!
        """
        htf_struct = cls.analyze_structure(htf_df, symbol=symbol, timeframe=htf_name, window=3)
        ltf_struct = cls.analyze_structure(ltf_df, symbol=symbol, timeframe=ltf_name, window=2)

        if htf_struct.regime == "UPTREND":
            macro_bias: Literal["BUY_ONLY", "SELL_ONLY", "NEUTRAL"] = "BUY_ONLY"
            macro_inst = "📈 ARAH UTAMA HTF: UPTREND (Puncak & Lembah Meninggi) ➔ CARI NAIK (BUY ONLY)"
        elif htf_struct.regime == "DOWNTREND":
            macro_bias = "SELL_ONLY"
            macro_inst = "📉 ARAH UTAMA HTF: DOWNTREND (Puncak & Lembah Menurun) ➔ CARI TURUN (SELL ONLY)"
        else:
            macro_bias = "NEUTRAL"
            macro_inst = "↔️ ARAH UTAMA HTF: RANGE / SIDEWAYS ➔ Menunggu Breakout Valid"

        # Verifikasi Keselarasan Aksi Eksekusi
        if proposed_action == "BUY":
            if macro_bias == "BUY_ONLY":
                is_aligned = True
                verdict = f"✅ EKSEKUSI BUY DIIZINKAN: Selaras dengan Arah Utama {htf_name} (Cari Naik)."
            elif macro_bias == "SELL_ONLY":
                is_aligned = False
                verdict = f"🚫 EKSEKUSI BUY DITOLAK: Melawan Arah Utama {htf_name} yang sedang Downtrend (Cari Turun)!"
            else:
                is_aligned = True
                verdict = f"⚠️ EKSEKUSI BUY DIIZINKAN DENGAN HATI-HATI ({htf_name} sedang Sideways/Range)."
        elif proposed_action == "SELL":
            if macro_bias == "SELL_ONLY":
                is_aligned = True
                verdict = f"✅ EKSEKUSI SELL DIIZINKAN: Selaras dengan Arah Utama {htf_name} (Cari Turun)."
            elif macro_bias == "BUY_ONLY":
                is_aligned = False
                verdict = f"🚫 EKSEKUSI SELL DITOLAK: Melawan Arah Utama {htf_name} yang sedang Uptrend (Cari Naik)!"
            else:
                is_aligned = True
                verdict = f"⚠️ EKSEKUSI SELL DIIZINKAN DENGAN HATI-HATI ({htf_name} sedang Sideways/Range)."
        else:
            is_aligned = True
            verdict = f"💡 Menunggu Setup Eksekusi di {ltf_name} yang Selaras dengan {macro_bias}."

        return TopDownStructureAnalysis(
            symbol=symbol,
            htf_name=htf_name,
            ltf_name=ltf_name,
            htf_regime=htf_struct.regime,
            htf_pattern=htf_struct.pattern_label,
            macro_bias=macro_bias,
            macro_instruction=macro_inst,
            htf_structure=htf_struct,
            ltf_structure=ltf_struct,
            is_aligned=is_aligned,
            execution_verdict=verdict,
        )

    @classmethod
    def analyze_impulse_and_pullback(
        cls,
        df: pd.DataFrame,
        regime: Literal["UPTREND", "DOWNTREND", "RANGE_SIDEWAYS"],
        latest_sh: float,
        latest_sl: float,
        latest_hl: float,
        latest_lh: float,
        lookback_bars: int = 10,
    ) -> PullbackWaveAnalysis:
        """
        Filosofi Trend & Pullback:
        1. Bedakan Impuls dan Koreksi:
           - Impuls: Searah tren, candle besar, cepat, jaraknya jauh.
           - Koreksi/Pullback: Berlawanan arah tren, candle lebih kecil, pelan, dan saling tumpang tindih.
           - Uptrend sehat: Impuls lebih panjang dari koreksi (naik cepat, turun pelan).
        2. Tunggu Pullback:
           - Jangan entry di puncak ekstensi impuls. Tunggu pasar istirahat dan menyelesaikan pullback.
        3. Pastikan Pullback Sehat (HL/LH Terjaga):
           - Uptrend: Selama harga bertahan di atas HL terakhir -> Pullback sehat!
           - Downtrend: Selama harga bertahan di bawah LH terakhir -> Pullback sehat!
        """
        curr_price = float(df["close"].iloc[-1]) if len(df) > 0 else 2650.0
        subset = df.tail(lookback_bars)
        opens = pd.to_numeric(subset["open"], errors="coerce").astype(np.float64).values
        highs = pd.to_numeric(subset["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(subset["low"], errors="coerce").astype(np.float64).values
        closes = pd.to_numeric(subset["close"], errors="coerce").astype(np.float64).values

        n_bars = len(subset)
        if n_bars < 3:
            return PullbackWaveAnalysis(
                current_phase="CONSOLIDATION",
                is_healthy=True,
                retest_depth_pct=0.0,
                overlap_ratio=0.0,
                guardian_level=latest_hl if regime == "UPTREND" else latest_lh,
                guardian_kind="HL" if regime == "UPTREND" else "LH",
                phase_explanation="Data lilin belum mencukupi untuk analisis gelombang impuls/pullback.",
                three_golden_rules="Menunggu data lilin terbentuk.",
            )

        # Hitung overlap ratio antar lilin berurutan
        overlap_sum = 0.0
        for i in range(1, n_bars):
            rng = max(highs[i] - lows[i], 1e-6)
            overlap = max(0.0, min(highs[i], highs[i - 1]) - max(lows[i], lows[i - 1]))
            overlap_sum += (overlap / rng)
        avg_overlap = overlap_sum / max(n_bars - 1, 1)

        # Lilin terakhir
        last_body = abs(closes[-1] - opens[-1])
        last_range = max(highs[-1] - lows[-1], 1e-6)
        last_body_ratio = last_body / last_range
        last_is_white = closes[-1] > opens[-1]
        last_is_red = closes[-1] < opens[-1]

        if regime == "UPTREND":
            guardian_kind = "HL"
            guardian_level = latest_hl
            is_healthy = curr_price >= latest_hl
            total_impulse_range = max(latest_sh - latest_hl, 1e-6)
            retest_depth = max(0.0, (latest_sh - curr_price) / total_impulse_range * 100.0)

            if curr_price >= latest_sh * 0.999 and last_is_white and last_body_ratio >= 0.50:
                current_phase: Literal["IMPULSE", "PULLBACK", "PULLBACK_COMPLETED", "CONSOLIDATION"] = "IMPULSE"
                phase_expl = "⚡ GELOMBANG IMPULS NAIK: Lilin besar & cepat melaju ke atas (Jangan kejar harga di pucuk, tunggu pullback!)."
            elif curr_price < latest_sh and last_is_red and is_healthy:
                current_phase = "PULLBACK"
                phase_expl = f"☕ SEDANG PULLBACK ISTIRAHAT: Harga turun pelan (Retest {retest_depth:.1f}% dari puncak, HL ${latest_hl:,.2f} terjaga)."
            elif curr_price < latest_sh and last_is_white and is_healthy:
                current_phase = "PULLBACK_COMPLETED"
                phase_expl = f"🎯 PULLBACK SELESAI & SEHAT: Muncul lilin pembalikan di atas HL ${latest_hl:,.2f} ➔ SIAP ENTRY BUY!"
            elif not is_healthy:
                current_phase = "CONSOLIDATION"
                phase_expl = f"⚠️ PULLBACK RUSAK / ABNORMAL: HL Terakhir (${latest_hl:,.2f}) Ditembus ke Bawah (Struktur naik patah)!"
            else:
                current_phase = "PULLBACK"
                phase_expl = f"Koreksi normal di rentang ${latest_hl:,.2f} - ${latest_sh:,.2f}."

        elif regime == "DOWNTREND":
            guardian_kind = "LH"
            guardian_level = latest_lh
            is_healthy = curr_price <= latest_lh
            total_impulse_range = max(latest_lh - latest_sl, 1e-6)
            retest_depth = max(0.0, (curr_price - latest_sl) / total_impulse_range * 100.0)

            if curr_price <= latest_sl * 1.001 and last_is_red and last_body_ratio >= 0.50:
                current_phase = "IMPULSE"
                phase_expl = "⚡ GELOMBANG IMPULS TURUN: Lilin merah besar meluncur ke bawah (Jangan kejar harga di dasar, tunggu pullback!)."
            elif curr_price > latest_sl and last_is_white and is_healthy:
                current_phase = "PULLBACK"
                phase_expl = f"☕ SEDANG PULLBACK ISTIRAHAT: Harga naik pelan (Retest {retest_depth:.1f}% dari dasar, LH ${latest_lh:,.2f} terjaga)."
            elif curr_price > latest_sl and last_is_red and is_healthy:
                current_phase = "PULLBACK_COMPLETED"
                phase_expl = f"🎯 PULLBACK SELESAI & SEHAT: Muncul lilin merah penekanan di bawah LH ${latest_lh:,.2f} ➔ SIAP ENTRY SELL!"
            elif not is_healthy:
                current_phase = "CONSOLIDATION"
                phase_expl = f"⚠️ PULLBACK RUSAK / ABNORMAL: LH Terakhir (${latest_lh:,.2f}) Ditembus ke Atas (Struktur turun patah)!"
            else:
                current_phase = "PULLBACK"
                phase_expl = f"Koreksi normal di rentang ${latest_sl:,.2f} - ${latest_lh:,.2f}."

        else:
            guardian_kind = "RANGE_BOUNDS"
            guardian_level = latest_sh
            is_healthy = True
            retest_depth = 50.0
            current_phase = "CONSOLIDATION"
            phase_expl = f"↔️ PASAR RANGE / KONSOLIDASI: Lilin tumpang tindih ({avg_overlap*100:.0f}% Overlap) tanpa gelombang impuls dominan."

        golden_rules_text = (
            f"1. KARAKTER GELOMBANG: {current_phase} ({phase_expl})\n"
            f"2. TUNGGU PULLBACK: {'✅ SIAP ENTRY (Pullback Selesai)' if current_phase == 'PULLBACK_COMPLETED' else '⏳ TUNGGU PULLBACK SELESAI'}\n"
            f"3. KESEHATAN PULLBACK: {'🟢 SEHAT (' + guardian_kind + f' @ ${guardian_level:,.2f} Terjaga)' if is_healthy else '🔴 RUSAK / ABNORMAL'}"
        )

        return PullbackWaveAnalysis(
            current_phase=current_phase,
            is_healthy=is_healthy,
            retest_depth_pct=round(retest_depth, 2),
            overlap_ratio=round(avg_overlap, 4),
            guardian_level=guardian_level,
            guardian_kind=guardian_kind,
            phase_explanation=phase_expl,
            three_golden_rules=golden_rules_text,
        )

    @classmethod
    def detect_bos_and_mss(
        cls,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        regime: Literal["UPTREND", "DOWNTREND", "RANGE_SIDEWAYS"],
        latest_hl: float,
        latest_lh: float,
        latest_sh: float,
        latest_sl: float,
        lookback_bars: int = 8,
    ) -> Tuple[Optional[StructureBreakEvent], int, bool]:
        """
        Deteksi Penembusan Struktur Pasar (BOS vs MSS):
        - Searah Tren = BOS (Break of Structure ➔ Lanjut).
          - Uptrend: Candle close di atas Swing High sebelumnya (HH).
          - Downtrend: Candle close di bawah Swing Low sebelumnya (LL).
        - HL/LH Terakhir Tembus = MSS (Market Structure Shift ➔ Berubah Arah).
          - Dari Uptrend jadi Downtrend: Candle close di bawah Higher Low (HL) terakhir!
          - Dari Downtrend jadi Uptrend: Candle close di atas Lower High (LH) terakhir!
        - WAJIB CLOSE, bukan sekadar wick!
        """
        if len(df) < lookback_bars:
            return None, 0, False

        subset = df.tail(lookback_bars)
        highs = pd.to_numeric(subset["high"], errors="coerce").astype(np.float64).values
        lows = pd.to_numeric(subset["low"], errors="coerce").astype(np.float64).values
        closes = pd.to_numeric(subset["close"], errors="coerce").astype(np.float64).values

        n = len(subset)

        # 1. Cek Market Structure Shift (MSS) Terlebih Dahulu (Berubah Arah!)
        # Uptrend patah jika close di bawah HL Terakhir
        if regime == "UPTREND" and latest_hl > 0:
            for i in range(n - 1, max(-1, n - 4), -1):
                if closes[i] < latest_hl:
                    event = StructureBreakEvent(
                        event_type="BEARISH_MSS",
                        level_broken=latest_hl,
                        broken_swing_label="HL",
                        broken_swing_kind="LOW",
                        break_close_price=closes[i],
                        is_confirmed_by_close=True,
                        direction_meaning="REVERSAL",
                        details=(
                            f"MSS (MARKET STRUCTURE SHIFT) BEARISH: Candle berhasil CLOSE di bawah HL Terakhir (${latest_hl:,.2f})! "
                            f"Buyer kehilangan kendali, struktur berubah dari NAIK jadi TURUN ➔ Cari Lower High (LH) untuk Sell."
                        ),
                    )
                    return event, 0, True

        # Downtrend patah jika close di atas LH Terakhir
        if regime == "DOWNTREND" and latest_lh > 0:
            for i in range(n - 1, max(-1, n - 4), -1):
                if closes[i] > latest_lh:
                    event = StructureBreakEvent(
                        event_type="BULLISH_MSS",
                        level_broken=latest_lh,
                        broken_swing_label="LH",
                        broken_swing_kind="HIGH",
                        break_close_price=closes[i],
                        is_confirmed_by_close=True,
                        direction_meaning="REVERSAL",
                        details=(
                            f"MSS (MARKET STRUCTURE SHIFT) BULLISH: Candle berhasil CLOSE di atas LH Terakhir (${latest_lh:,.2f})! "
                            f"Seller kehilangan kendali, struktur berubah dari TURUN jadi NAIK ➔ Cari Higher Low (HL) untuk Buy."
                        ),
                    )
                    return event, 0, True

        # 2. Cek Break of Structure (BOS) (Searah Tren ➔ Lanjut!)
        # Bullish BOS di Uptrend: Close di atas Swing High sebelumnya
        if (regime == "UPTREND" or regime == "RANGE_SIDEWAYS") and latest_sh > 0:
            for i in range(n - 1, max(-1, n - 4), -1):
                if closes[i] > latest_sh:
                    event = StructureBreakEvent(
                        event_type="BULLISH_BOS",
                        level_broken=latest_sh,
                        broken_swing_label="HH",
                        broken_swing_kind="HIGH",
                        break_close_price=closes[i],
                        is_confirmed_by_close=True,
                        direction_meaning="CONTINUATION",
                        details=(
                            f"BOS (BREAK OF STRUCTURE) BULLISH: Candle berhasil CLOSE di atas Swing High (${latest_sh:,.2f})! "
                            f"Struktur uptrend berlanjut. Tunggu pullback untuk membentuk Higher Low (HL) baru."
                        ),
                    )
                    return event, 1, False

        # Bearish BOS di Downtrend: Close di bawah Swing Low sebelumnya
        if (regime == "DOWNTREND" or regime == "RANGE_SIDEWAYS") and latest_sl > 0:
            for i in range(n - 1, max(-1, n - 4), -1):
                if closes[i] < latest_sl:
                    event = StructureBreakEvent(
                        event_type="BEARISH_BOS",
                        level_broken=latest_sl,
                        broken_swing_label="LL",
                        broken_swing_kind="LOW",
                        break_close_price=closes[i],
                        is_confirmed_by_close=True,
                        direction_meaning="CONTINUATION",
                        details=(
                            f"BOS (BREAK OF STRUCTURE) BEARISH: Candle berhasil CLOSE di bawah Swing Low (${latest_sl:,.2f})! "
                            f"Struktur downtrend berlanjut. Tunggu pullback untuk membentuk Lower High (LH) baru."
                        ),
                    )
                    return event, 1, False

        return None, 0, False
