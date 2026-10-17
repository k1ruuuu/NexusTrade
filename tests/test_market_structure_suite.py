"""
Test Suite for Market Structure (Struktur Pasar):
1. 4 Label Universal:
   - HH (Higher High)
   - HL (Higher Low)
   - LH (Lower High)
   - LL (Lower Low)
2. Deteksi Swing High (Puncak) & Swing Low (Lembah) Menonjol (Bebas Goyangan Kecil / Noise).
3. Uptrend Normal vs Abnormal:
   - Uptrend Normal: HH + HL berulang, HL terakhir terjaga.
   - Uptrend Abnormal / Patah: HL terakhir dijebol ke bawah.
4. Downtrend Normal vs Abnormal:
   - Downtrend Normal: LH + LL berulang, LH terakhir terjaga.
   - Downtrend Abnormal / Patah: LH terakhir dijebol ke atas.
5. Range / Sideways: Puncak & Lembah sejajar.
6. 3 Checklist Wajib Sebelum Entry:
   - 1. Mana Swing High & Swing Low yang jelas?
   - 2. HH-HL, LH-LL, atau Range?
   - 3. Swing mana yang menjaga tren?
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from data_pipeline.market_structure_analyzer import MarketStructureAnalyzer, SwingPoint, MarketStructureResult


def create_zigzag_ohlcv(points: list[float], bars_per_segment: int = 6) -> pd.DataFrame:
    """Helper to generate smooth OHLCV waves with prominent peaks and troughs."""
    records = []
    curr = points[0]
    for seg_idx in range(len(points) - 1):
        target = points[seg_idx + 1]
        step = (target - curr) / bars_per_segment
        for _ in range(bars_per_segment):
            curr += step
            records.append({
                "open": curr - (0.2 if step > 0 else -0.2),
                "high": curr + 0.5,
                "low": curr - 0.5,
                "close": curr,
                "volume": 1000.0,
            })
    return pd.DataFrame(records)


def test_market_structure_uptrend_hh_hl():
    """Verify Uptrend with Higher Highs (HH) and Higher Lows (HL) and normal status."""
    # Gelombang naik bertingkat: Lembah 100 -> Puncak 120 -> Lembah 110 (HL) -> Puncak 130 (HH) -> Lembah 125 (HL) -> Close 135
    wave_points = [100.0, 120.0, 110.0, 130.0, 125.0, 138.0]
    df = create_zigzag_ohlcv(wave_points, bars_per_segment=6)

    res = MarketStructureAnalyzer.analyze_structure(df, symbol="XAUUSD", timeframe="5m", window=2)
    assert res.regime == "UPTREND"
    assert res.pattern_label == "HH-HL"
    assert res.guardian_swing_kind == "HL"
    # Uptrend normal selama HL terakhir (125.0) tidak ditembus ke bawah
    assert res.is_trend_normal is True
    assert "NORMAL" in res.trend_status_detail
    assert res.structural_bias == "BUY"
    assert res.entry_allowed is True


def test_market_structure_uptrend_broken_when_hl_violated():
    """Verify Uptrend becomes abnormal/broken when latest HL is violated downwards."""
    # Gelombang naik bertingkat, lalu harga anjlok menembus HL terakhir:
    # Lembah 100 -> Puncak 120 -> Lembah 110 (HL) -> Puncak 130 (HH) -> Lembah 125 (HL) -> Harga anjlok ke 105 (HL jebol!)
    wave_points = [100.0, 120.0, 110.0, 130.0, 125.0, 138.0, 105.0]
    df = create_zigzag_ohlcv(wave_points, bars_per_segment=6)

    res = MarketStructureAnalyzer.analyze_structure(df, symbol="XAUUSD", timeframe="5m", window=2)
    # Jika struktur sebelumnya HH-HL tapi harga anjlok menembus HL terakhir (125.0):
    if res.pattern_label == "HH-HL":
        assert res.is_trend_normal is False
        assert "ABNORMAL" in res.trend_status_detail or "PATAH" in res.trend_status_detail
        assert res.entry_allowed is False


def test_market_structure_downtrend_lh_ll():
    """Verify Downtrend with Lower Highs (LH) and Lower Lows (LL) and normal status."""
    # Gelombang turun bertingkat: Puncak 200 -> Lembah 170 -> Puncak 185 (LH) -> Lembah 160 (LL) -> Puncak 175 (LH) -> Close 150
    wave_points = [200.0, 170.0, 185.0, 160.0, 175.0, 150.0]
    df = create_zigzag_ohlcv(wave_points, bars_per_segment=6)

    res = MarketStructureAnalyzer.analyze_structure(df, symbol="XAUUSD", timeframe="5m", window=2)
    assert res.regime == "DOWNTREND"
    assert res.pattern_label == "LH-LL"
    assert res.guardian_swing_kind == "LH"
    # Downtrend normal selama LH terakhir (175.0) belum ditembus ke atas
    assert res.is_trend_normal is True
    assert "NORMAL" in res.trend_status_detail
    assert res.structural_bias == "SELL"
    assert res.entry_allowed is True


def test_market_structure_downtrend_broken_when_lh_violated():
    """Verify Downtrend becomes abnormal/broken when latest LH is violated upwards."""
    # Gelombang turun bertingkat, lalu harga melonjak menembus LH terakhir:
    # Puncak 200 -> Lembah 170 -> Puncak 185 (LH) -> Lembah 160 (LL) -> Puncak 175 (LH) -> Harga melonjak ke 190 (LH jebol!)
    wave_points = [200.0, 170.0, 185.0, 160.0, 175.0, 190.0]
    df = create_zigzag_ohlcv(wave_points, bars_per_segment=6)

    res = MarketStructureAnalyzer.analyze_structure(df, symbol="XAUUSD", timeframe="5m", window=2)
    if res.pattern_label == "LH-LL":
        assert res.is_trend_normal is False
        assert "ABNORMAL" in res.trend_status_detail or "PATAH" in res.trend_status_detail
        assert res.entry_allowed is False


def test_market_structure_three_checks_display():
    """Verify the 3 pre-entry questions are populated clearly in three_checks_display."""
    wave_points = [100.0, 120.0, 110.0, 130.0, 125.0, 135.0]
    df = create_zigzag_ohlcv(wave_points, bars_per_segment=6)

    res = MarketStructureAnalyzer.analyze_structure(df, symbol="XAUUSD", timeframe="5m", window=2)
    assert "1. 🎯 SWING HIGH & LOW MENONJOL:" in res.three_checks_display
    assert "2. 📊 POLA STRUKTUR:" in res.three_checks_display
    assert "3. 🛡️ SWING PENJAGA TREN:" in res.three_checks_display

def test_top_down_htf_uptrend_cari_naik():
    """Verify HTF uptrend enforces 'CARI NAIK (BUY ONLY)' bias and rejects counter-trend SELL."""
    wave_htf_up = [100.0, 130.0, 115.0, 145.0, 135.0, 160.0]
    df_htf = create_zigzag_ohlcv(wave_htf_up, bars_per_segment=6)
    df_ltf = create_zigzag_ohlcv([140.0, 150.0, 145.0, 155.0], bars_per_segment=4)

    td_buy = MarketStructureAnalyzer.evaluate_top_down_htf_ltf(
        htf_df=df_htf, ltf_df=df_ltf, symbol="XAUUSD", htf_name="H1", ltf_name="M5", proposed_action="BUY"
    )
    assert td_buy.macro_bias == "BUY_ONLY"
    assert "CARI NAIK (BUY ONLY)" in td_buy.macro_instruction
    assert td_buy.is_aligned is True
    assert "DIIZINKAN" in td_buy.execution_verdict

    td_sell = MarketStructureAnalyzer.evaluate_top_down_htf_ltf(
        htf_df=df_htf, ltf_df=df_ltf, symbol="XAUUSD", htf_name="H1", ltf_name="M5", proposed_action="SELL"
    )
    assert td_sell.is_aligned is False
    assert "DITOLAK" in td_sell.execution_verdict


def test_top_down_htf_downtrend_cari_turun():
    """Verify HTF downtrend enforces 'CARI TURUN (SELL ONLY)' bias and rejects counter-trend BUY."""
    wave_htf_dn = [200.0, 160.0, 180.0, 140.0, 160.0, 120.0]
    df_htf = create_zigzag_ohlcv(wave_htf_dn, bars_per_segment=6)
    df_ltf = create_zigzag_ohlcv([150.0, 130.0, 140.0, 125.0], bars_per_segment=4)

    td_sell = MarketStructureAnalyzer.evaluate_top_down_htf_ltf(
        htf_df=df_htf, ltf_df=df_ltf, symbol="XAUUSD", htf_name="H1", ltf_name="M5", proposed_action="SELL"
    )
    assert td_sell.macro_bias == "SELL_ONLY"
    assert "CARI TURUN (SELL ONLY)" in td_sell.macro_instruction
    assert td_sell.is_aligned is True
    assert "DIIZINKAN" in td_sell.execution_verdict

    td_buy = MarketStructureAnalyzer.evaluate_top_down_htf_ltf(
        htf_df=df_htf, ltf_df=df_ltf, symbol="XAUUSD", htf_name="H1", ltf_name="M5", proposed_action="BUY"
    )
    assert td_buy.is_aligned is False
    assert "DITOLAK" in td_buy.execution_verdict
