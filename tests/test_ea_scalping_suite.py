"""
Test Suite for EA Scalping:
1. 5 Jenis Moving Average (MA) pada M1: Biru (EMA5), Hijau (EMA8), Merah (EMA13), Kuning (EMA34), Ungu (EMA55).
2. Cross Curam:
   - MA Biru cross MA Hijau ke atas curam ➔ BUY.
   - MA Biru cross MA Merah ke bawah curam ➔ SELL.
3. Analisa Tren Pasar Multi-Timeframe (D1, H1, M15, M5 — H1 dan M5 Utama).
4. Karakter & Anatomi Candlestick: Pinbar Rejection (Wick >= 40%), Engulfing, Momentum Body.
5. Entry dengan SL dan TP Saling Berdekatan (Tight Scalping).
6. Auto Re-Entry Seketika pada TP maupun SL.
7. Bypass Filter 6-Gate Ketat untuk Semua Mode EA (Tidak Terhambat Berita & Ruang S/R).
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from ai_engine.auto_trader.ea_scalping_engine import EAScalpingEngine, EAScalpingSignal
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.user_vault import UserVault
from config.settings import Settings


def generate_test_ohlcv(n_bars: int = 60, base_price: float = 2650.0, trend_step: float = 0.5) -> pd.DataFrame:
    """Helper to generate realistic candles for testing."""
    records = []
    p = base_price
    for i in range(n_bars):
        p += trend_step
        records.append({
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
            "open": p - 0.2,
            "high": p + 1.0,
            "low": p - 1.0,
            "close": p + 0.5,
            "volume": 1000.0,
        })
    return pd.DataFrame(records)


def test_ea_scalping_5_ma_computation_and_steep_slope():
    """Verify computation of all 5 MAs on M1 and steep slope calculation."""
    df = generate_test_ohlcv(60, base_price=2600.0, trend_step=1.0)
    closes = pd.to_numeric(df["close"], errors="coerce").ffill()
    b, g, r, y, p, slope = EAScalpingEngine.compute_5_ma(closes)

    # In a rising series, faster MAs must be greater than slower MAs
    assert b > g > r > y > p
    assert slope > 0.0


def test_ea_scalping_steep_crossover_buy_and_sell():
    """Verify MA Biru cross MA Hijau ke atas curam -> BUY, and MA Biru cross MA Merah ke bawah curam -> SELL."""
    # Skenario BUY: Strong rising trend
    df_buy = generate_test_ohlcv(60, base_price=2650.0, trend_step=1.5)
    sig_buy = EAScalpingEngine.evaluate_scalping_5ma(
        m1_df=df_buy,
        symbol="XAUUSD",
        sl_pips=12.0,
        tp_pips=14.0,
    )
    assert sig_buy.action == "BUY"
    assert sig_buy.ma_blue > sig_buy.ma_green
    assert "BUY TERKONFIRMASI" in sig_buy.details or "PREDIKSI NAIK" in sig_buy.forward_prediction

    # Skenario SELL: Strong falling trend with red falling candles
    df_sell = generate_test_ohlcv(60, base_price=2650.0, trend_step=-1.5)
    for idx in df_sell.index[-5:]:
        c_val = float(df_sell.loc[idx, "close"])
        df_sell.loc[idx, "open"] = c_val + 0.8
        df_sell.loc[idx, "high"] = c_val + 1.2
        df_sell.loc[idx, "low"] = c_val - 0.2
    sig_sell = EAScalpingEngine.evaluate_scalping_5ma(
        m1_df=df_sell,
        symbol="XAUUSD",
        sl_pips=12.0,
        tp_pips=14.0,
    )
    assert sig_sell.action == "SELL"
    assert sig_sell.ma_blue < sig_sell.ma_red
    assert "SELL TERKONFIRMASI" in sig_sell.details or "PREDIKSI TURUN" in sig_sell.forward_prediction


def test_ea_scalping_mtf_h1_and_m5_anchoring():
    """Verify MTF trend analysis where H1 and M5 serve as primary trend anchors."""
    h1_bull = generate_test_ohlcv(40, base_price=2600.0, trend_step=2.0)
    m5_bull = generate_test_ohlcv(40, base_price=2650.0, trend_step=1.0)

    trend, h1_desc, m5_desc = EAScalpingEngine.read_mtf_trend(h1_df=h1_bull, m5_df=m5_bull)
    assert trend == "BULLISH"
    assert "H1 BULLISH" in h1_desc
    assert "M5 BULLISH" in m5_desc

    h1_bear = generate_test_ohlcv(40, base_price=2700.0, trend_step=-2.0)
    m5_bear = generate_test_ohlcv(40, base_price=2650.0, trend_step=-1.0)
    trend_b, h1_b, m5_b = EAScalpingEngine.read_mtf_trend(h1_df=h1_bear, m5_df=m5_bear)
    assert trend_b == "BEARISH"
    assert "H1 BEARISH" in h1_b
    assert "M5 BEARISH" in m5_b


def test_ea_scalping_read_candlestick_character():
    """Verify candlestick anatomy reading (Pinbar Rejection, Engulfing, Momentum Body)."""
    df = generate_test_ohlcv(30, base_price=2650.0, trend_step=0.0)

    # Bullish Pinbar Rejection (Lower wick = 60% of range)
    df.loc[df.index[-2], "open"] = 2653.0
    df.loc[df.index[-2], "close"] = 2654.0
    df.loc[df.index[-2], "low"] = 2650.0
    df.loc[df.index[-2], "high"] = 2655.0
    pat, bias, strength = EAScalpingEngine.read_candlestick_character(df)
    assert "PINBAR" in pat or "REJECTION" in pat
    assert bias == "BULLISH"
    assert strength >= 0.85

    # Bearish Pinbar Rejection (Upper wick = 60% of range)
    df.loc[df.index[-2], "open"] = 2652.0
    df.loc[df.index[-2], "close"] = 2651.0
    df.loc[df.index[-2], "high"] = 2655.0
    df.loc[df.index[-2], "low"] = 2650.0
    pat_b, bias_b, strength_b = EAScalpingEngine.read_candlestick_character(df)
    assert "PINBAR" in pat_b or "REJECTION" in pat_b
    assert bias_b == "BEARISH"
    assert strength_b >= 0.85


def test_ea_scalping_tight_sl_and_tp():
    """Verify entry calculation enforces tight SL and TP that are close together."""
    df = generate_test_ohlcv(50, base_price=2650.0, trend_step=1.0)
    sig = EAScalpingEngine.evaluate_scalping_5ma(df, "XAUUSD", sl_pips=12.0, tp_pips=14.0)

    assert sig.action in ("BUY", "SELL")
    assert sig.entry_price > 0.0
    assert sig.stop_loss > 0.0
    assert sig.take_profit > 0.0

    sl_dist = abs(sig.entry_price - sig.stop_loss)
    tp_dist = abs(sig.take_profit - sig.entry_price)

    # Pada Gold, 12 pips = $1.20 dan 14 pips = $1.40
    assert abs(sl_dist - 1.20) <= 0.05
    assert abs(tp_dist - 1.40) <= 0.05
    # SL dan TP saling berdekatan (selisih jarak hanya $0.20)
    assert abs(tp_dist - sl_dist) <= 0.30


def test_ea_scalping_instant_reentry_on_tp_and_sl():
    """Verify instant re-entry trigger evaluates immediately when previous trade exits via TP or SL."""
    df = generate_test_ohlcv(50, base_price=2650.0, trend_step=0.8)

    # Re-entry pasca TP
    reentry_tp = EAScalpingEngine.evaluate_scalping_5ma(
        df, "XAUUSD", sl_pips=12.0, tp_pips=14.0, last_exit_reason="TAKE_PROFIT"
    )
    assert reentry_tp.is_instant_reentry is True
    assert reentry_tp.action in ("BUY", "SELL")
    assert "RE-ENTRY TAKE_PROFIT" in reentry_tp.details

    # Re-entry pasca SL
    reentry_sl = EAScalpingEngine.evaluate_scalping_5ma(
        df, "XAUUSD", sl_pips=12.0, tp_pips=14.0, last_exit_reason="STOP_LOSS"
    )
    assert reentry_sl.is_instant_reentry is True
    assert reentry_sl.action in ("BUY", "SELL")
    assert "RE-ENTRY STOP_LOSS" in reentry_sl.details


@pytest.mark.asyncio
async def test_ea_scalping_and_all_ea_modes_bypass_6_gates():
    """Verify all EA modes bypass Gate 2 (News) and Gate 4 (S/R room) without technical rejections."""
    settings = Settings(env_file=None, use_live_market_data=False)
    vault = UserVault(settings=settings)
    vault.save_user_credentials(
        user_id=106,
        api_key="EXNESS_DEMO_106",
        api_secret="secret",
        server_name="Exness-MT5Trial",
        testnet=True,
        verified_usdt_balance=1000.0,
        autotrade_enabled=True,
    )
    engine = SmartAutoTraderEngine(settings=settings, user_vault=vault)

    for ea_mode in ("EA_SCALPING", "EA_NX_BREAK", "EA_NX_PARSIAL", "EA_NX_REVERSE", "EA_NX_POC"):
        report = await engine.evaluate_and_execute(
            user_id=106,
            symbol="XAUUSD",
            ea_mode=ea_mode,
        )
        assert report is not None
        assert report.decision is not None

        # Verify Gate 2 (News) was bypassed and passed
        gate2 = next((g for g in report.gates if g.gate_number == 2), None)
        assert gate2 is not None
        assert gate2.passed is True
        assert "Gate 2 Dilewati" in gate2.detail

        # Verify Gate 4 (S/R Room) was bypassed and passed
        gate4 = next((g for g in report.gates if g.gate_number == 4), None)
        assert gate4 is not None
        assert gate4.passed is True
        assert "Gate 4 Dilewati" in gate4.detail
