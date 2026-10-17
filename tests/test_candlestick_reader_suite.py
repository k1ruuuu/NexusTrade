"""
Test Suite for Universal Candlestick & Price Action Reader:
1. Membaca OHLC, Body, dan Wick (Sumbu).
2. Membedakan Warna Candlestick:
   - Putih (Close > Open ➔ Harga Naik).
   - Merah (Close < Open ➔ Harga Turun).
   - Doji (Close == Open ➔ Keseimbangan).
3. Mengidentifikasi Swing High, Swing Low, dan Uptrend / Downtrend.
4. Validasi 3 Perilaku (Behaviour):
   - Siapa yang menang? (Lihat Body).
   - Di mana harga ditolak? (Lihat Wick).
   - Ke mana trennya? (Lihat Rangkaian lilin).
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from data_pipeline.candlestick_reader import CandlestickReader, SingleCandleAnatomy, CandlestickSeriesAnalysis
from ai_engine.models.ensemble_agent import EnsembleDecisionAgent
from config.settings import Settings


def test_candlestick_ohlc_body_and_wick_calculations():
    """Verify exact OHLC, body size, upper wick, and lower wick calculations."""
    # Bullish candle: Open=100.0, High=110.0, Low=95.0, Close=108.0
    c_white = CandlestickReader.read_single_candle(
        open_p=100.0,
        high_p=110.0,
        low_p=95.0,
        close_p=108.0,
    )
    assert c_white.color == "PUTIH"
    assert c_white.body_size == 8.0          # 108 - 100
    assert c_white.range_total == 15.0       # 110 - 95
    assert c_white.upper_wick == 2.0         # 110 - 108
    assert c_white.lower_wick == 5.0         # 100 - 95
    assert abs(c_white.body_ratio - (8.0 / 15.0)) <= 1e-4

    # Bearish candle: Open=108.0, High=110.0, Low=95.0, Close=98.0
    c_red = CandlestickReader.read_single_candle(
        open_p=108.0,
        high_p=110.0,
        low_p=95.0,
        close_p=98.0,
    )
    assert c_red.color == "MERAH"
    assert c_red.body_size == 10.0          # 108 - 98
    assert c_red.upper_wick == 2.0          # 110 - 108
    assert c_red.lower_wick == 3.0          # 98 - 95


def test_candlestick_color_distinction():
    """Verify candle color: Putih when Close > Open, Merah when Close < Open."""
    white = CandlestickReader.read_single_candle(2600.0, 2610.0, 2595.0, 2608.0)
    assert white.color == "PUTIH"
    assert "BUYERS" in white.winner

    red = CandlestickReader.read_single_candle(2610.0, 2612.0, 2590.0, 2595.0)
    assert red.color == "MERAH"
    assert "SELLERS" in red.winner

    doji = CandlestickReader.read_single_candle(2600.0, 2605.0, 2595.0, 2600.0)
    assert doji.color == "DOJI"
    assert "SEIMBANG" in doji.winner


def test_behaviour_pillar_1_body_winner():
    """Pilar 1: Siapa yang menang? -> Lihat Body."""
    # Buyers Menang Mutlak: Bodi besar >= 70% dari range
    marubozu_white = CandlestickReader.read_single_candle(100.0, 110.0, 100.0, 109.0)
    assert "BUYERS MENANG MUTLAK" in marubozu_white.winner

    # Sellers Menang Mutlak: Bodi merah besar >= 70% dari range
    marubozu_red = CandlestickReader.read_single_candle(110.0, 110.0, 100.0, 101.0)
    assert "SELLERS MENANG MUTLAK" in marubozu_red.winner


def test_behaviour_pillar_2_wick_rejection():
    """Pilar 2: Di mana harga ditolak? -> Lihat Wick."""
    # Sumbu bawah panjang (Rejection di support/bawah): Open 105, Close 106, High 107, Low 95 (Lower wick = 105-95=10 out of 12 = 83%)
    rejection_down = CandlestickReader.read_single_candle(105.0, 107.0, 95.0, 106.0)
    assert "DITOLAK DI LEVEL BAWAH" in rejection_down.rejection_site
    assert "PINBAR" in rejection_down.pattern_label or "HAMMER" in rejection_down.pattern_label

    # Sumbu atas panjang (Rejection di resistance/atas): Open 97, Close 96, Low 95, High 110 (Upper wick = 110-97=13 out of 15 = 86%)
    rejection_up = CandlestickReader.read_single_candle(97.0, 110.0, 95.0, 96.0)
    assert "DITOLAK DI LEVEL ATAS" in rejection_up.rejection_site
    assert "PINBAR" in rejection_up.pattern_label or "SHOOTING_STAR" in rejection_up.pattern_label


def test_behaviour_pillar_3_series_swings_and_uptrend():
    """Pilar 3: Ke mana trennya? -> Lihat Rangkaian lilin, Swing High, Swing Low, dan Uptrend/Downtrend."""
    # Rangkaian Higher Highs & Higher Lows (Uptrend)
    records = []
    p = 100.0
    for i in range(30):
        # Wave pattern rising
        wave = 2.0 * np.sin(i * 0.5)
        p += 1.0 + wave
        records.append({
            "open": p - 0.5,
            "high": p + 2.0,
            "low": p - 2.0,
            "close": p + 1.0,
        })
    df_uptrend = pd.DataFrame(records)

    analysis = CandlestickReader.analyze_series_and_swings(df_uptrend, swing_window=2)
    assert analysis.trend == "UPTREND"
    assert "UPTREND" in analysis.trend_description
    assert analysis.latest_swing_high > 0.0
    assert analysis.latest_swing_low > 0.0
    assert "1. PEMENANG:" in analysis.three_pillars_summary
    assert "2. PENOLAKAN:" in analysis.three_pillars_summary
    assert "3. ARAH TREN:" in analysis.three_pillars_summary


def test_integration_in_ensemble_agent():
    """Verify EnsembleDecisionAgent indicators include candlestick anatomy and 3 pillars."""
    settings = Settings(env_file=None, use_live_market_data=False)
    agent = EnsembleDecisionAgent(settings=settings)

    dec = agent.analyze_multi_timeframe_autotrade("XAUUSD")
    assert dec is not None
    assert "candle_color" in dec.indicators
    assert dec.indicators["candle_color"] in ("PUTIH", "MERAH", "DOJI")
    assert "candle_winner" in dec.indicators
    assert "candle_rejection" in dec.indicators
    assert "candle_trend_series" in dec.indicators
    assert "candle_three_pillars" in dec.indicators
