"""
Test Suite for EA Scalping:
1. Membaca Arah Tren: Bullish / Bearish via EMA 9, EMA 21, EMA 50.
2. Membaca Candlestick: Pinbar Rejection (Lower/Upper Wick >= 40%), Engulfing, Momentum Body.
3. Membaca Pergerakan Market Kedepannya: Prediksi arah ekspansi harga berikutnya.
4. Entry dengan SL dan TP Saling Berdekatan: Tight scalping 12 pips SL / 14 pips TP.
5. Auto Re-Entry Seketika pada TP maupun SL.
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


def test_ea_scalping_read_trend():
    """Verify EAScalpingEngine accurately reads Bullish and Bearish trend directions."""
    # Bullish trend
    bull_df = generate_test_ohlcv(50, base_price=2600.0, trend_step=1.0)
    t_bull, e9, e21, e50 = EAScalpingEngine.read_trend(bull_df)
    assert t_bull == "BULLISH"
    assert e9 >= e21

    # Bearish trend
    bear_df = generate_test_ohlcv(50, base_price=2600.0, trend_step=-1.0)
    t_bear, e9_b, e21_b, e50_b = EAScalpingEngine.read_trend(bear_df)
    assert t_bear == "BEARISH"
    assert e9_b <= e21_b


def test_ea_scalping_read_candlestick():
    """Verify candlestick anatomy reading (Pinbar Rejection, Engulfing, Momentum Body)."""
    df = generate_test_ohlcv(30, base_price=2650.0, trend_step=0.0)

    # 1. Bullish Pinbar Rejection (Lower wick = 60% of range)
    # Range = 5.0 (2650 to 2655), Open=2653, Close=2654, Low=2650 -> lower wick = 2653 - 2650 = 3.0 (60%)
    df.loc[df.index[-2], "open"] = 2653.0
    df.loc[df.index[-2], "close"] = 2654.0
    df.loc[df.index[-2], "low"] = 2650.0
    df.loc[df.index[-2], "high"] = 2655.0
    pat, bias, strength = EAScalpingEngine.read_candlestick(df)
    assert "PINBAR" in pat or "REJECTION" in pat
    assert bias == "BULLISH"
    assert strength >= 0.85

    # 2. Bearish Pinbar Rejection (Upper wick = 60% of range)
    # Range = 5.0 (2650 to 2655), Open=2652, Close=2651, High=2655 -> upper wick = 2655 - 2652 = 3.0 (60%)
    df.loc[df.index[-2], "open"] = 2652.0
    df.loc[df.index[-2], "close"] = 2651.0
    df.loc[df.index[-2], "high"] = 2655.0
    df.loc[df.index[-2], "low"] = 2650.0
    pat_b, bias_b, strength_b = EAScalpingEngine.read_candlestick(df)
    assert "PINBAR" in pat_b or "REJECTION" in pat_b
    assert bias_b == "BEARISH"
    assert strength_b >= 0.85

    # 3. Bullish Momentum Body
    df.loc[df.index[-2], "open"] = 2650.5
    df.loc[df.index[-2], "close"] = 2654.5
    df.loc[df.index[-2], "low"] = 2650.0
    df.loc[df.index[-2], "high"] = 2655.0
    pat_m, bias_m, _ = EAScalpingEngine.read_candlestick(df)
    assert bias_m == "BULLISH"
    assert "MOMENTUM" in pat_m


def test_ea_scalping_tight_sl_and_tp():
    """Verify entry calculation enforces tight SL and TP that are close together."""
    df = generate_test_ohlcv(50, base_price=2650.0, trend_step=1.0)
    sig = EAScalpingEngine.evaluate_scalping(df, "XAUUSD", sl_pips=12.0, tp_pips=14.0)

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
    reentry_tp = EAScalpingEngine.evaluate_scalping(
        df, "XAUUSD", sl_pips=12.0, tp_pips=14.0, last_exit_reason="TAKE_PROFIT"
    )
    assert reentry_tp.is_instant_reentry is True
    assert reentry_tp.action in ("BUY", "SELL")
    assert "RE-ENTRY TAKE_PROFIT" in reentry_tp.details

    # Re-entry pasca SL
    reentry_sl = EAScalpingEngine.evaluate_scalping(
        df, "XAUUSD", sl_pips=12.0, tp_pips=14.0, last_exit_reason="STOP_LOSS"
    )
    assert reentry_sl.is_instant_reentry is True
    assert reentry_sl.action in ("BUY", "SELL")
    assert "RE-ENTRY STOP_LOSS" in reentry_sl.details


@pytest.mark.asyncio
async def test_ea_scalping_engine_integration():
    """Verify SmartAutoTraderEngine executes EA_SCALPING with full checklist."""
    settings = Settings(env_file=None, use_live_market_data=False)
    vault = UserVault(settings=settings)
    vault.save_user_credentials(
        user_id=105,
        api_key="EXNESS_DEMO_105",
        api_secret="secret",
        server_name="Exness-MT5Trial",
        testnet=True,
        verified_usdt_balance=1000.0,
        autotrade_enabled=True,
    )
    engine = SmartAutoTraderEngine(settings=settings, user_vault=vault)

    report = await engine.evaluate_and_execute(
        user_id=105,
        symbol="XAUUSD",
        ea_mode="EA_SCALPING",
    )
    assert report is not None
    assert report.decision is not None
    assert report.decision.indicators.get("ea_mode") == "EA_SCALPING"
    assert "trend" in report.decision.indicators
    assert "candle" in report.decision.indicators
    assert "prediction" in report.decision.indicators
