"""
Test Suite for Institutional Risk Management Principles:
1. Tentukan Stop-Loss (SL) dulu sebelum entry (Structural Invalidation beyond HL/LH).
2. Hitung Risk Reward (RR) nya dulu, jelek (< 1.5R)? Lewati!
3. Jaga risiko per trade tetap kecil & konsisten (Fixed Risk % of Equity):
   - SL Jauh -> Ukuran Posisi Diperkecil.
   - SL Dekat -> Ukuran Posisi Diperbesar.
   - Risiko Dolar ($) di SL tetap sama & terukur.
4. Validasi Matematika: Trader A (Winrate 80%, RR 0.25R = Net 0) vs Trader B (Winrate 40%, RR 2R = Net +2R).
"""
from __future__ import annotations

import pytest
from ai_engine.risk_management.position_sizer import PositionSizer, PositionSizeResult
from ai_engine.risk_management.stop_loss_manager import StopLossManager, StopLossTargets
from config.settings import Settings


def test_rule_1_structural_invalidation_stop_loss():
    """Rule 1: Tentukan Stop-Loss dulu sebelum entry di luar struktur invalidasi (HL / LH)."""
    sl_mgr = StopLossManager()

    # Skenario 1: BUY di Pullback (Ide: Higher Low 2640.0 terjaga)
    entry_buy = 2650.0
    latest_hl = 2640.0
    targets_buy = sl_mgr.calculate_structural_targets(
        entry_price=entry_buy,
        action="BUY",
        latest_hl=latest_hl,
        atr_14=5.0,
        min_rrr=2.0,
    )
    # SL harus berada di bawah Higher Low (latest_hl - buffer)
    assert targets_buy.stop_loss < latest_hl
    # Risk 1R adalah jarak entry ke SL
    risk_1r = entry_buy - targets_buy.stop_loss
    assert risk_1r > 0.0
    # Reward TP1 adalah 2x risiko (2R / rasio 1:2)
    reward_2r = targets_buy.take_profit_1 - entry_buy
    assert abs(reward_2r - (2.0 * risk_1r)) <= 0.02
    assert targets_buy.reward_risk_ratio_1 == 2.0

    # Skenario 2: SELL di Pullback (Ide: Lower High 2660.0 terjaga)
    entry_sell = 2650.0
    latest_lh = 2660.0
    targets_sell = sl_mgr.calculate_structural_targets(
        entry_price=entry_sell,
        action="SELL",
        latest_lh=latest_lh,
        atr_14=5.0,
        min_rrr=2.0,
    )
    # SL harus berada di atas Lower High (latest_lh + buffer)
    assert targets_sell.stop_loss > latest_lh
    risk_1r_s = targets_sell.stop_loss - entry_sell
    assert risk_1r_s > 0.0
    reward_2r_s = entry_sell - targets_sell.take_profit_1
    assert abs(reward_2r_s - (2.0 * risk_1r_s)) <= 0.02


def test_rule_2_unfavorable_risk_reward_rejection():
    """Rule 2: Hitung Risk-Reward (RR) nya dulu. Kalau jelek (< 1.5R), lewati / reject!"""
    sizer = PositionSizer()

    # Skenario A: RR Jelek (Entry 100, SL 90 -> Risk 10, TP 105 -> Reward 5 -> RR 0.5R < 1.5R)
    bad_rr = sizer.calculate_position(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=105.0,  # 0.5R
        confidence=0.85,
        s_vol=0.10,
        portfolio_equity=1000.0,
    )
    assert bad_rr.approved is False
    assert "Risk-Reward Ratio tidak menguntungkan" in (bad_rr.rejection_reason or "")
    assert bad_rr.position_units == 0.0

    # Skenario B: RR Bagus (Entry 100, SL 95 -> Risk 5, TP 110 -> Reward 10 -> RR 2.0R >= 1.5R)
    good_rr = sizer.calculate_position(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,  # 2.0R
        confidence=0.85,
        s_vol=0.10,
        portfolio_equity=1000.0,
    )
    assert good_rr.approved is True
    assert good_rr.position_units > 0.0


def test_rule_3_consistent_fixed_risk_sizing():
    """
    Rule 3: Jaga risiko per trade tetap kecil & konsisten (Fixed Risk % of Equity):
    - SL Jauh -> Posisi diperkecil.
    - SL Dekat -> Posisi diperbesar.
    - Besar risiko ($) yang dijaga tetap konsisten.
    """
    sizer = PositionSizer()
    equity = 100.0  # Modal $100

    # Skenario 1: SL Jauh (Jarak SL = $10.0)
    res_far = sizer.calculate_position(
        entry_price=100.0,
        stop_loss=90.0,    # Stop distance = 10.0
        take_profit=120.0, # TP 2R
        confidence=1.0,
        s_vol=0.0,
        portfolio_equity=equity,
    )
    units_far = res_far.position_units
    dollar_risk_far = units_far * 10.0

    # Skenario 2: SL Dekat (Jarak SL = $2.0)
    res_near = sizer.calculate_position(
        entry_price=100.0,
        stop_loss=98.0,    # Stop distance = 2.0
        take_profit=104.0, # TP 2R
        confidence=1.0,
        s_vol=0.0,
        portfolio_equity=equity,
    )
    units_near = res_near.position_units
    dollar_risk_near = units_near * 2.0

    # Validasi Prinsip:
    # 1. SL dekat -> posisi (units) harus lebih besar dari SL jauh!
    assert units_near > units_far
    # 2. Yang dijaga bukan ukuran posisinya, tapi besar risikonya ($ dollar risk konsisten)!
    assert abs(dollar_risk_near - dollar_risk_far) <= 0.05
    # 3. Risiko per trade kecil sesuai 1.5% modal ($1.50 dari $100)
    assert abs(dollar_risk_near - 1.50) <= 0.05


def test_trader_a_vs_trader_b_expectancy_math():
    """
    Validasi Matematika:
    Trader A: Winrate 80% (8/10), tapi tiap menang cuman dapat 0.25R. Net = +2R - 2R = 0 net.
    Trader B: Winrate 40% (4/10), tapi tiap menang dapat 2.0R. Net = +8R - 6R = +2R net!
    """
    risk_1r = 10.0  # $10 risk per trade

    # Trader A: 8 wins of 0.25R ($2.50) and 2 losses of 1R ($10.0)
    trader_a_wins = 8 * (0.25 * risk_1r)   # +$20.0 (+2R)
    trader_a_losses = 2 * (1.0 * risk_1r)  # -$20.0 (-2R)
    trader_a_net = trader_a_wins - trader_a_losses
    assert trader_a_net == 0.0  # Hasilnya NOL meskipun winrate 80%!

    # Trader B: 4 wins of 2.0R ($20.0) and 6 losses of 1R ($10.0)
    trader_b_wins = 4 * (2.0 * risk_1r)    # +$80.0 (+8R)
    trader_b_losses = 6 * (1.0 * risk_1r)  # -$60.0 (-6R)
    trader_b_net = trader_b_wins - trader_b_losses
    assert trader_b_net == 20.0  # Hasilnya PROFIT +$20 (+2R) meskipun winrate hanya 40%!

    assert trader_b_net > trader_a_net
