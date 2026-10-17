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
    three_checks_display: str = ""


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

        # Tampilan 3 Checklist Wajib
        checks_text = (
            f"1. 🎯 SWING HIGH & LOW MENONJOL: Swing High ${lat_sh:,.2f} | Swing Low ${lat_sl:,.2f}\n"
            f"2. 📊 POLA STRUKTUR: {pattern_label} ({summary})\n"
            f"3. 🛡️ SWING PENJAGA TREN: {guardian_kind} @ ${guardian_price:,.2f} ➔ {status_detail}"
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
        )
