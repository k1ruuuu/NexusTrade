from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class ConfluenceReport:
    """
    Multi-Factor Confluence & Mathematical Expectancy assessment based on
    'Panduan Teknik Analisis Trading' (Sections 1-6).
    """

    direction: Literal["BUY", "SELL", "HOLD"]
    pillars_passed: int
    total_pillars: int
    confluence_ratio: float
    cvd_delta_20: float
    cvd_divergence: Literal["NONE", "BULLISH_DIV", "BEARISH_DIV"]
    cvd_status: str
    relative_volume: float
    pct_above_ma20: float
    drawdown_pct: float
    max_drawdown_pct: float
    expected_move_1d: float
    expected_move_pct: float
    reward_risk_ratio: float
    win_probability: float
    expectancy_r: float
    expectancy_usd: float
    confluence_adjustment: float
    invalidation_note: str
    adx_14: float = 25.0
    is_sideways: bool = False
    market_structure_status: str = "Trending Normal (ADX 25.0)"

class MultiFactorConfluenceAnalyzer:
    """
    Implements the 6 Independent Analytical Pillars from 'Panduan Teknik Analisis Trading':
      1. Trend & Momentum Structure (SMA/EMA slope, RSI Wilder, MACD Histogram)
      2. Order Flow & Activity Confirmation (Cumulative Volume Delta CVD, Relative Volume RVOL, Spread bps)
      3. Support/Resistance Zone & 1-Sigma Expected Move (P * sigma_annual * sqrt(d/365))
      4. Market Breadth & Relative Strength (Advance-Decline Line ADL, % Above MA20, Drawdown/MDD)
      5. Fundamental / News Sentiment Catalyst (FinBERT / Online SGD)
      6. Risk & Mathematical Expectancy: E = (p_w * AvgWin) - (p_l * AvgLoss)
    """

    @staticmethod
    def compute_confluence_series(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute vectorized Order Flow (CVD), Relative Volume (RVOL), Market Breadth (ADL, % Above MA20),
        Drawdown / Maximum Drawdown (MDD), and Expected Move features on an OHLCV DataFrame.
        """
        out = df.copy()
        close = pd.to_numeric(out["close"], errors="coerce").astype(np.float64)
        open_p = pd.to_numeric(out["open"], errors="coerce").astype(np.float64)
        high = pd.to_numeric(out["high"], errors="coerce").astype(np.float64)
        low = pd.to_numeric(out["low"], errors="coerce").astype(np.float64)
        volume = (
            pd.to_numeric(out["volume"], errors="coerce").fillna(1.0).astype(np.float64)
            if "volume" in out.columns
            else pd.Series(1.0, index=out.index, dtype=np.float64)
        )

        # 1. Section 1.3: Order Flow — Aggressive Buy vs Sell Volume & Cumulative Volume Delta (CVD)
        hl_span = (high - low).replace(0.0, np.nan)
        clv = (((close - low) - (high - close)) / hl_span).fillna(0.0).clip(-1.0, 1.0)
        vol_delta = clv * volume
        cvd = vol_delta.cumsum()

        vol_sum_20 = volume.rolling(window=20, min_periods=1).sum().replace(0.0, np.nan)
        out["cvd_norm_20"] = (
            vol_delta.rolling(window=20, min_periods=1).sum() / vol_sum_20
        ).fillna(0.0)

        price_chg_10 = close.pct_change(10).fillna(0.0)
        cvd_chg_10 = (
            vol_delta.rolling(window=10, min_periods=1).sum()
            / volume.rolling(window=10, min_periods=1).sum().replace(0.0, np.nan)
        ).fillna(0.0)
        # Divergence score: +1 when price drops but CVD accumulates (bullish), -1 when price rises but CVD distributes (bearish)
        div_signal = np.where(
            (price_chg_10 > 0.002) & (cvd_chg_10 < -0.05),
            -1.0,
            np.where((price_chg_10 < -0.002) & (cvd_chg_10 > 0.05), 1.0, 0.0),
        )
        out["cvd_divergence"] = pd.Series(div_signal, index=out.index, dtype=np.float64)
        out["cvd_raw"] = cvd

        # 2. Section 3.1: Relative Volume (RVOL = V_t / avg(V_20))
        vol_avg_20 = volume.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan)
        out["rvol_20"] = (volume / vol_avg_20).fillna(1.0).clip(0.1, 10.0)

        # 3. Section 4.1: Drawdown & Maximum Drawdown (Drawdown_t = (P_t - Peak_t) / Peak_t)
        rolling_peak_50 = close.rolling(window=50, min_periods=1).max().replace(0.0, np.nan)
        out["drawdown_50"] = ((close - rolling_peak_50) / rolling_peak_50).fillna(0.0)
        out["mdd_50"] = out["drawdown_50"].rolling(window=50, min_periods=1).min().fillna(0.0)

        # 4. Section 4.2: Market Breadth / Internal Participation (AD Difference, ADL, % Above MA20)
        bar_adv_dec = np.sign(close - open_p)
        out["adl_norm_20"] = (
            pd.Series(bar_adv_dec, index=out.index).rolling(window=20, min_periods=1).mean().fillna(0.0)
        )
        ma_20 = close.rolling(window=20, min_periods=1).mean()
        above_ma20 = (close > ma_20).astype(np.float64)
        out["pct_above_ma20"] = above_ma20.rolling(window=20, min_periods=1).mean().fillna(0.5)

        # 5. Section 4.5: Expected Move (P * sigma_annual * sqrt(d / 365))
        ret_1 = close.pct_change(1).fillna(0.0)
        bar_std_20 = ret_1.rolling(window=20, min_periods=2).std(ddof=0).fillna(0.01)
        out["expected_move_1bar_pct"] = bar_std_20.clip(lower=1e-4)
        out["expected_move_ratio"] = (
            (close - close.shift(5)).abs() / (close * bar_std_20 * np.sqrt(5.0)).replace(0.0, np.nan)
        ).fillna(0.5).clip(0.0, 5.0)

        # 6. Wilder's ADX(14) (Average Directional Index) for Sideways vs Trending Regime Detection
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_14_w = tr.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean().replace(0.0, np.nan)
        plus_di = (
            100.0
            * pd.Series(plus_dm, index=out.index).ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()
            / atr_14_w
        ).fillna(0.0)
        minus_di = (
            100.0
            * pd.Series(minus_dm, index=out.index).ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()
            / atr_14_w
        ).fillna(0.0)
        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        if len(dx) > 1:
            dx.iloc[0] = np.nan
        out["adx_14"] = (
            dx.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean().fillna(22.0).clip(0.0, 100.0)
        )
        return out

    @classmethod
    def evaluate_confluence(
        cls,
        df: Optional[pd.DataFrame],
        action: Literal["BUY", "SELL", "HOLD"],
        s_tech: float,
        s_sent: float,
        s_vol: float,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        support_1: float,
        resistance_1: float,
        annualized_volatility: float = 0.45,
        risk_capital_usd: float = 100.0,
    ) -> ConfluenceReport:
        """
        Evaluate the 6 Independent Pillars from Section 5 & Section 6 of
        'Panduan Teknik Analisis Trading' to boost win-rate and filter out false signals.
        """
        price = max(float(entry_price), 1e-6)
        sl_dist = abs(price - float(stop_loss))
        tp_dist = abs(float(take_profit_1) - price)
        rrr = round(tp_dist / sl_dist, 2) if sl_dist > 1e-9 else 2.0

        # Section 4.5: 1-Day Expected Move = P * sigma_annual * sqrt(1 / 365)
        sigma_ann = max(0.05, min(float(annualized_volatility), 3.0))
        exp_move_1d = price * sigma_ann * float(np.sqrt(1.0 / 365.0))
        exp_move_pct = (exp_move_1d / price) * 100.0

        if df is not None and len(df) >= 15:
            enriched = cls.compute_confluence_series(df.tail(120))
            last = enriched.iloc[-1]
            cvd_delta_20 = float(last["cvd_norm_20"])
            div_val = float(last["cvd_divergence"])
            rvol = float(last["rvol_20"])
            pct_above_ma20 = float(last["pct_above_ma20"]) * 100.0
            dd_pct = float(last["drawdown_50"]) * 100.0
            mdd_pct = float(last["mdd_50"]) * 100.0
            adl_norm = float(last["adl_norm_20"])
            adx_14 = float(last.get("adx_14", 25.0))
        else:
            cvd_delta_20 = 0.25 if s_tech > 0.15 else (-0.25 if s_tech < -0.15 else 0.0)
            div_val = 0.0
            rvol = 1.20
            pct_above_ma20 = 65.0 if s_tech > 0 else (35.0 if s_tech < 0 else 50.0)
            dd_pct = -1.2
            mdd_pct = -4.5
            adl_norm = 0.20 if s_tech > 0 else (-0.20 if s_tech < 0 else 0.0)
            adx_14 = 28.0 if abs(s_tech) >= 0.35 else 15.5

        # Sideways Regime Detection: ADX < 20 without high-volume breakout conviction
        is_sideways = bool(adx_14 < 20.0 and abs(s_tech) < 0.78 and rvol < 1.15)
        if is_sideways:
            market_structure_status = f"Sideways / Ranging (ADX {adx_14:.1f} < 20 • Rawan Fakeout)"
        elif adx_14 < 20.0:
            market_structure_status = f"Breakout Awal dari Sideways (ADX {adx_14:.1f} • RVOL {rvol:.2f}x)"
        elif adx_14 >= 25.0:
            market_structure_status = f"Trending Kuat (ADX {adx_14:.1f})"
        else:
            market_structure_status = f"Transisi Tren (ADX {adx_14:.1f})"
        if div_val > 0.5:
            cvd_div: Literal["NONE", "BULLISH_DIV", "BEARISH_DIV"] = "BULLISH_DIV"
        elif div_val < -0.5:
            cvd_div = "BEARISH_DIV"
        else:
            cvd_div = "NONE"

        if cvd_delta_20 >= 0.05:
            cvd_status = f"Beli Agresif (+{cvd_delta_20:.2f})" + (
                " • Divergensi Bullish" if cvd_div == "BULLISH_DIV" else ""
            )
        elif cvd_delta_20 <= -0.05:
            cvd_status = f"Jual Agresif ({cvd_delta_20:.2f})" + (
                " • Divergensi Bearish" if cvd_div == "BEARISH_DIV" else ""
            )
        else:
            cvd_status = f"Seimbang ({cvd_delta_20:+.2f})"

        # Evaluate the 6 Independent Pillars (Section 5 Table)
        pillars_passed = 0

        # Pillar 1: Trend & Momentum Structure (Technical Analysis + Not Trapped in Low-Volume Sideways Chop)
        p1_ok = (not is_sideways) and (
            (action == "BUY" and s_tech >= 0.20) or (action == "SELL" and s_tech <= -0.20)
        )
        if p1_ok:
            pillars_passed += 1
        # Pillar 2: Order Flow (CVD) & Relative Volume (RVOL) Confirmation (No adverse divergence)
        if action == "BUY":
            p2_ok = cvd_div != "BEARISH_DIV" and (cvd_delta_20 >= -0.08 or rvol >= 0.85)
        elif action == "SELL":
            p2_ok = cvd_div != "BULLISH_DIV" and (cvd_delta_20 <= 0.08 or rvol >= 0.85)
        else:
            p2_ok = False
        if p2_ok:
            pillars_passed += 1

        # Pillar 3: Support/Resistance Zone Clearance (Treated as a zone, not a single thin line)
        zone_buffer = max(sl_dist * 0.25, price * 0.0012)
        if action == "BUY":
            p3_ok = float(resistance_1) <= 0 or (float(resistance_1) - price) >= zone_buffer
        elif action == "SELL":
            p3_ok = float(support_1) <= 0 or (price - float(support_1)) >= zone_buffer
        else:
            p3_ok = False
        if p3_ok:
            pillars_passed += 1

        # Pillar 4: Market Breadth & Participation (ADL & % Above MA20)
        if action == "BUY":
            p4_ok = pct_above_ma20 >= 40.0 or adl_norm >= -0.10
        elif action == "SELL":
            p4_ok = pct_above_ma20 <= 60.0 or adl_norm <= 0.10
        else:
            p4_ok = False
        if p4_ok:
            pillars_passed += 1

        # Pillar 5: Fundamental / News Catalyst Alignment (Neutral band is (-0.15, +0.15))
        if action == "BUY":
            p5_ok = s_sent > -0.15
        elif action == "SELL":
            p5_ok = s_sent < 0.15
        else:
            p5_ok = False
        if p5_ok:
            pillars_passed += 1

        # Pillar 6: Volatility Regime & Realistic Reward-to-Risk Ratio >= 1.15
        p6_ok = rrr >= 1.15 and s_vol < 0.85
        if p6_ok:
            pillars_passed += 1

        confluence_ratio = round(pillars_passed / 6.0, 4)

        # Compute Confluence Adjustment to boost high-confluence setups and penalize divergences
        adj = 0.0
        if action == "BUY":
            if cvd_div == "BEARISH_DIV":
                adj -= 0.12
            elif cvd_delta_20 > 0.08 and rvol >= 1.10:
                adj += 0.06
        elif action == "SELL":
            if cvd_div == "BULLISH_DIV":
                adj -= 0.12
            elif cvd_delta_20 < -0.08 and rvol >= 1.10:
                adj += 0.06

        # Section 6.3: Mathematical Strategy Expectancy: E = (p_w * AvgWin) - (p_l * AvgLoss)
        base_prob = 0.42 + 0.32 * min(1.0, abs(s_tech) * 0.75 + abs(s_sent) * 0.25)
        win_prob = float(np.clip(base_prob + 0.12 * (confluence_ratio - 0.5) + adj, 0.25, 0.88))
        loss_prob = 1.0 - win_prob
        # Subtract estimated 8 bps spread+fee drag (0.04R)
        expectancy_r = round((win_prob * rrr) - (loss_prob * 1.0) - 0.04, 3)
        expectancy_usd = round(expectancy_r * max(risk_capital_usd, 10.0), 2)

        if action == "BUY":
            invalidation_note = f"Batal jika tembus ke bawah Zona Support S1 ({support_1:,.2f}) / SL ({stop_loss:,.2f})"
        elif action == "SELL":
            invalidation_note = f"Batal jika tembus ke atas Zona Resistance R1 ({resistance_1:,.2f}) / SL ({stop_loss:,.2f})"
        else:
            invalidation_note = "Tunggu konfirmasi penembusan Support S1 atau Resistance R1 dengan RVOL > 1.1x"

        return ConfluenceReport(
            direction=action,
            pillars_passed=pillars_passed,
            total_pillars=6,
            confluence_ratio=confluence_ratio,
            cvd_delta_20=round(cvd_delta_20, 4),
            cvd_divergence=cvd_div,
            cvd_status=cvd_status,
            relative_volume=round(rvol, 2),
            pct_above_ma20=round(pct_above_ma20, 1),
            drawdown_pct=round(dd_pct, 2),
            max_drawdown_pct=round(mdd_pct, 2),
            expected_move_1d=round(exp_move_1d, 4),
            expected_move_pct=round(exp_move_pct, 2),
            reward_risk_ratio=rrr,
            win_probability=round(win_prob, 4),
            expectancy_r=expectancy_r,
            expectancy_usd=expectancy_usd,
            confluence_adjustment=round(adj, 4),
            invalidation_note=invalidation_note,
            adx_14=round(adx_14, 2),
            is_sideways=is_sideways,
            market_structure_status=market_structure_status,
        )
