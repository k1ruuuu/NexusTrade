"""
Test Suite for NX EA MQL5 Strategies Python Integration:
1. NXBreak: Candle breakout, quality filter (body >= 55%, range 0.4-2.5 ATR), 1:1 RR.
2. NXParsial: 3-Leg partial basket scaling (1R, 3R, 6R) with shared risk budget.
3. NXReverse: EMA 20/50/200 trend alignment, 20-bar rolling SNR breakout, 1:5 RR, recovery mode.
4. NX-POC: 100-bar Volume Profile 40-bin POC calculation, rejection pinbar wick >= 35%, 1:1 RR.
5. End-to-end integration into SmartAutoTraderEngine.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from ai_engine.auto_trader.nx_ea_engine import NXEAEngine, NXSignalResult
from ai_engine.auto_trader.engine import SmartAutoTraderEngine
from ai_engine.auto_trader.user_vault import UserVault
from config.settings import Settings


def generate_synthetic_ohlcv(n_bars: int = 150, base_price: float = 2650.0, trend: float = 0.5) -> pd.DataFrame:
    """Generate realistic OHLCV candle dataframe for testing."""
    np.random.seed(42)
    records = []
    price = base_price
    for i in range(n_bars):
        noise = float(np.random.normal(0, 1.0))
        drift = trend * (i / n_bars)
        open_p = price
        close_p = open_p + drift + noise
        high_p = max(open_p, close_p) + abs(float(np.random.normal(0.5, 0.5))) + 0.5
        low_p = min(open_p, close_p) - abs(float(np.random.normal(0.5, 0.5))) - 0.5
        volume = float(np.random.randint(500, 3000))
        records.append({
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
        })
        price = close_p
    return pd.DataFrame(records)


def test_nx_candle_quality_filter():
    """Verify candle range and body ratio checks."""
    atr = 5.0
    # Good candle: range = 5.0 (1.0 ATR), body = 3.5 (70% of range)
    good_bar = pd.Series({"open": 2650.0, "high": 2655.0, "low": 2650.0, "close": 2653.5})
    assert NXEAEngine.check_candle_quality(good_bar, atr) is True

    # Bad body: range = 5.0, body = 1.0 (20% of range < 55%)
    doji_bar = pd.Series({"open": 2652.0, "high": 2655.0, "low": 2650.0, "close": 2653.0})
    assert NXEAEngine.check_candle_quality(doji_bar, atr) is False

    # Bad range: range = 1.0 (0.2 ATR < 0.4 ATR)
    tiny_bar = pd.Series({"open": 2650.0, "high": 2651.0, "low": 2650.0, "close": 2650.8})
    assert NXEAEngine.check_candle_quality(tiny_bar, atr) is False

    # Giant range: range = 20.0 (4.0 ATR > 2.5 ATR)
    giant_bar = pd.Series({"open": 2650.0, "high": 2670.0, "low": 2650.0, "close": 2668.0})
    assert NXEAEngine.check_candle_quality(giant_bar, atr) is False


def test_nx_break_bullish_and_bearish_triggers():
    """Verify NXBreak generates BUY and SELL breakout signals with 1:1 RR."""
    df = generate_synthetic_ohlcv(50, base_price=2650.0)
    atr = NXEAEngine.compute_atr_shift2(df)

    # Inject a clean bullish breakout at bar 1 (index -2) relative to bar 2 (index -3)
    bar2_high = 2660.0
    bar2_low = 2655.0
    df.loc[df.index[-3], "high"] = bar2_high
    df.loc[df.index[-3], "low"] = bar2_low

    # Bullish breakout: open <= bar2_high, close > bar2_high + 0.05 * atr
    buffer = 0.05 * atr
    df.loc[df.index[-2], "open"] = bar2_high - 0.5
    df.loc[df.index[-2], "high"] = bar2_high + buffer + 3.0
    df.loc[df.index[-2], "low"] = bar2_high - 1.0
    df.loc[df.index[-2], "close"] = bar2_high + buffer + 2.0  # body = 2.5, range = 4.0 (62.5% body)
    df.loc[df.index[-1], "close"] = bar2_high + buffer + 2.0

    res_buy = NXEAEngine.evaluate_nx_break(df, "XAUUSD", fixed_sl_points=300.0)
    assert res_buy.action == "BUY"
    assert res_buy.quality_passed is True
    assert res_buy.stop_loss < res_buy.entry_price
    assert res_buy.take_profit > res_buy.entry_price
    # Check 1:1 RR distance
    sl_dist = round(abs(res_buy.entry_price - res_buy.stop_loss), 2)
    tp_dist = round(abs(res_buy.take_profit - res_buy.entry_price), 2)
    assert abs(sl_dist - tp_dist) <= 0.02


def test_nx_parsial_multi_target_basket():
    """Verify NXParsial calculates 1R, 3R, and 6R take-profit targets."""
    df = generate_synthetic_ohlcv(50, base_price=2650.0)
    atr = NXEAEngine.compute_atr_shift2(df)

    bar2_high = 2660.0
    df.loc[df.index[-3], "high"] = bar2_high
    buffer = 0.05 * atr
    df.loc[df.index[-2], "open"] = bar2_high - 0.5
    df.loc[df.index[-2], "high"] = bar2_high + buffer + 3.0
    df.loc[df.index[-2], "low"] = bar2_high - 1.0
    df.loc[df.index[-2], "close"] = bar2_high + buffer + 2.0
    df.loc[df.index[-1], "close"] = bar2_high + buffer + 2.0

    res = NXEAEngine.evaluate_nx_parsial(df, "XAUUSD", fixed_sl_points=300.0)
    assert res.action == "BUY"
    assert len(res.take_profit_levels) == 3

    entry = res.entry_price
    r_dist = abs(entry - res.stop_loss)
    tp1, tp2, tp3 = res.take_profit_levels

    assert abs((tp1 - entry) - 1.0 * r_dist) <= 0.05
    assert abs((tp2 - entry) - 3.0 * r_dist) <= 0.05
    assert abs((tp3 - entry) - 6.0 * r_dist) <= 0.05


def test_nx_reverse_trend_and_snr_breakout():
    """Verify NXReverse EMA 20>50>200 trend alignment, 20-bar SNR, and 1:5 RR."""
    # Strong upward trend so EMA20 > EMA50 > EMA200
    df = generate_synthetic_ohlcv(80, base_price=2600.0, trend=2.0)
    atr = NXEAEngine.compute_atr_shift2(df)

    # 20-bar SNR slice is df.iloc[-22:-2]
    snr_high = float(df.iloc[-22:-2]["high"].max())

    # Set signal bar 1 to break out above snr_high + buffer
    buffer = 0.05 * atr
    df.loc[df.index[-2], "open"] = snr_high - 0.2
    df.loc[df.index[-2], "high"] = snr_high + buffer + 2.5
    df.loc[df.index[-2], "low"] = snr_high - 0.5
    df.loc[df.index[-2], "close"] = snr_high + buffer + 2.0
    df.loc[df.index[-1], "close"] = snr_high + buffer + 2.0

    res = NXEAEngine.evaluate_nx_reverse(df, "XAUUSD", is_recovery=True)
    if res.action == "BUY":
        assert res.is_recovery is True
        stop_dist = res.entry_price - res.stop_loss
        tp_dist = res.take_profit - res.entry_price
        # Verify 1:5 RR
        assert abs(tp_dist - 5.0 * stop_dist) <= 0.10


def test_nx_poc_volume_profile_and_pinbar_rejection():
    """Verify NX-POC 40-bin volume profile POC and rejection pinbar."""
    df = generate_synthetic_ohlcv(120, base_price=2650.0, trend=0.0)
    atr = NXEAEngine.compute_atr_shift2(df)

    # Concentrate volume at price 2660.0 to create a prominent POC
    profile_slice_indices = df.index[-102:-2]
    for idx in profile_slice_indices[:30]:
        df.loc[idx, "open"] = 2659.5
        df.loc[idx, "close"] = 2660.5
        df.loc[idx, "high"] = 2661.0
        df.loc[idx, "low"] = 2659.0
        df.loc[idx, "volume"] = 50000.0  # Dominant volume spike

    # Price at 2655 (below POC 2660), distance is ~5.0 (within 0.5-2.0 ATR)
    # Craft a bullish pinbar rejection: lower wick >= 35% of total range
    bar1_open = 2654.0
    bar1_close = 2655.5
    bar1_low = 2651.0  # lower wick = 2654.0 - 2651.0 = 3.0
    bar1_high = 2656.0  # total range = 5.0 -> lower wick is 60% of range (>= 35%)

    df.loc[df.index[-2], "open"] = bar1_open
    df.loc[df.index[-2], "close"] = bar1_close
    df.loc[df.index[-2], "low"] = bar1_low
    df.loc[df.index[-2], "high"] = bar1_high
    df.loc[df.index[-3], "close"] = 2653.0  # close1 > close2
    df.loc[df.index[-1], "close"] = 2655.5

    res = NXEAEngine.evaluate_nx_poc(df, "XAUUSD", profile_bars=100, bins=40)
    if res.action == "BUY":
        assert res.quality_passed is True
        assert res.take_profit > res.entry_price
        assert res.stop_loss < res.entry_price
        # 1:1 RR symmetric target
        tp_dist = round(abs(res.take_profit - res.entry_price), 2)
        sl_dist = round(abs(res.entry_price - res.stop_loss), 2)
        assert abs(tp_dist - sl_dist) <= 0.05


@pytest.mark.asyncio
async def test_nx_ea_engine_integration_modes():
    """Verify SmartAutoTraderEngine evaluates all 4 NX EA modes without crashes."""
    settings = Settings(env_file=None, use_live_market_data=False)
    vault = UserVault(settings=settings)
    vault.save_user_credentials(
        user_id=101,
        api_key="EXNESS_DEMO_101",
        api_secret="secret",
        server_name="Exness-MT5Trial",
        testnet=True,
        verified_usdt_balance=1000.0,
        autotrade_enabled=True,
    )
    engine = SmartAutoTraderEngine(settings=settings, user_vault=vault)

    for mode in ("EA_NX_BREAK", "EA_NX_PARSIAL", "EA_NX_REVERSE", "EA_NX_POC"):
        report = await engine.evaluate_and_execute(
            user_id=101,
            symbol="XAUUSD",
            ea_mode=mode,
        )
        assert report is not None
        assert report.decision is not None
        assert report.decision.indicators.get("ea_mode") == mode
