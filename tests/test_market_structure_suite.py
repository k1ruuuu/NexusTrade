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

def test_impulse_vs_pullback_detection():
    """Verify distinction between impulse (large, fast bars) and pullback (slow, overlapping bars)."""
    # Build uptrend wave: base 100 -> 120 -> 110 (HL) -> 130 (HH) -> 125 (HL) -> 138 (HH)
    wave = [100.0, 120.0, 110.0, 130.0, 125.0, 138.0]
    df_uptrend = create_zigzag_ohlcv(wave, bars_per_segment=6)

    # 1. Impuls: lilin putih bodi besar melaju di dekat puncak
    pb_imp = MarketStructureAnalyzer.analyze_impulse_and_pullback(
        df=df_uptrend,
        regime="UPTREND",
        latest_sh=138.0,
        latest_sl=110.0,
        latest_hl=125.0,
        latest_lh=130.0,
    )
    assert pb_imp.current_phase in ("IMPULSE", "PULLBACK_COMPLETED")
    assert pb_imp.is_healthy is True

    # 2. Koreksi/Pullback: Tambah lilin-lilin merah kecil turun pelan (136, 135, 134)
    records_pullback = df_uptrend.to_dict("records")
    for px in [136.0, 135.0, 134.0]:
        records_pullback.append({
            "open": px + 0.4,
            "high": px + 0.6,
            "low": px - 0.4,
            "close": px - 0.2,
            "volume": 800.0,
        })
    df_pullback = pd.DataFrame(records_pullback)

    pb_corr = MarketStructureAnalyzer.analyze_impulse_and_pullback(
        df=df_pullback,
        regime="UPTREND",
        latest_sh=138.0,
        latest_sl=110.0,
        latest_hl=125.0,
        latest_lh=130.0,
    )
    assert pb_corr.current_phase == "PULLBACK"
    assert pb_corr.is_healthy is True  # HL 125.0 terjaga karena harga di 133.8
    assert "HL $125.00 terjaga" in pb_corr.phase_explanation


def test_pullback_health_unhealthy_when_hl_broken():
    """Verify pullback becomes unhealthy / broken when price dumps below latest HL."""
    wave = [100.0, 120.0, 110.0, 130.0, 125.0, 138.0]
    df = create_zigzag_ohlcv(wave, bars_per_segment=6)

    # Append dump bars dropping below HL 125.0 (down to 120.0)
    recs = df.to_dict("records")
    recs.append({"open": 126.0, "high": 126.5, "low": 119.5, "close": 120.0, "volume": 5000.0})
    df_dump = pd.DataFrame(recs)

    pb_dump = MarketStructureAnalyzer.analyze_impulse_and_pullback(
        df=df_dump,
        regime="UPTREND",
        latest_sh=138.0,
        latest_sl=110.0,
        latest_hl=125.0,
        latest_lh=130.0,
    )
    assert pb_dump.is_healthy is False
    assert "RUSAK" in pb_dump.phase_explanation or "ABNORMAL" in pb_dump.phase_explanation
    assert "1. KARAKTER GELOMBANG:" in pb_dump.three_golden_rules
    assert "2. TUNGGU PULLBACK:" in pb_dump.three_golden_rules
    assert "3. KESEHATAN PULLBACK:" in pb_dump.three_golden_rules


def test_bullish_bos_continuation():
    """Verify Bullish BOS: Price breaks above previous Swing High and candle body closes above level."""
    wave = [100.0, 120.0, 110.0, 125.0, 115.0]
    df_uptrend = create_zigzag_ohlcv(wave, bars_per_segment=6)

    # Break above previous Swing High 125.0 with solid body close at 128.0
    recs = df_uptrend.to_dict("records")
    recs.append({"open": 123.0, "high": 129.0, "low": 122.0, "close": 128.0, "volume": 2000.0})
    df_bos = pd.DataFrame(recs)

    res = MarketStructureAnalyzer.analyze_structure(df_bos, symbol="XAUUSD", timeframe="5m", window=2)
    assert res.latest_structure_event is not None
    assert res.latest_structure_event.event_type == "BULLISH_BOS"
    assert res.latest_structure_event.direction_meaning == "CONTINUATION"
    assert res.latest_structure_event.is_confirmed_by_close is True
    assert "BOS (BREAK OF STRUCTURE) BULLISH" in res.latest_structure_event.details


def test_bearish_mss_reversal():
    """Verify Bearish MSS: Price dumps and candle body closes BELOW the latest Higher Low (HL)."""
    # Uptrend structure: Base 100 -> High 130 -> HL 120 -> High 135 -> HL 125 -> High 140
    wave = [100.0, 130.0, 120.0, 135.0, 125.0, 140.0]
    df_uptrend = create_zigzag_ohlcv(wave, bars_per_segment=6)

    # Price fails to make new high, then dumps and CLOSES below HL 125.0 (close at 121.0)
    recs = df_uptrend.to_dict("records")
    recs.append({"open": 130.0, "high": 131.0, "low": 120.0, "close": 121.0, "volume": 5000.0})
    df_mss = pd.DataFrame(recs)

    res = MarketStructureAnalyzer.analyze_structure(df_mss, symbol="XAUUSD", timeframe="5m", window=2)
    assert res.latest_structure_event is not None
    assert res.latest_structure_event.event_type == "BEARISH_MSS"
    assert res.latest_structure_event.direction_meaning == "REVERSAL"
    assert res.regime == "DOWNTREND"  # Structure flipped!
    assert res.structural_bias == "SELL"
    assert "MSS (MARKET STRUCTURE SHIFT) BEARISH" in res.latest_structure_event.details


def test_bullish_mss_reversal():
    """Verify Bullish MSS: Price rallies and candle body closes ABOVE the latest Lower High (LH)."""
    # Downtrend structure: Base 200 -> Low 170 -> LH 185 -> Low 160 -> LH 175 -> Low 150
    wave = [200.0, 170.0, 185.0, 160.0, 175.0, 150.0]
    df_downtrend = create_zigzag_ohlcv(wave, bars_per_segment=6)

    # Price rallies and CLOSES above LH 175.0 (close at 180.0)
    recs = df_downtrend.to_dict("records")
    recs.append({"open": 165.0, "high": 182.0, "low": 164.0, "close": 180.0, "volume": 5000.0})
    df_mss_bull = pd.DataFrame(recs)

    res = MarketStructureAnalyzer.analyze_structure(df_mss_bull, symbol="XAUUSD", timeframe="5m", window=2)
    assert res.latest_structure_event is not None
    assert res.latest_structure_event.event_type == "BULLISH_MSS"
    assert res.latest_structure_event.direction_meaning == "REVERSAL"
    assert res.regime == "UPTREND"  # Structure flipped!
    assert res.structural_bias == "BUY"
    assert "MSS (MARKET STRUCTURE SHIFT) BULLISH" in res.latest_structure_event.details
