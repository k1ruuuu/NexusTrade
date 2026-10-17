from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from config.settings import Settings, get_settings

TradeDirection = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class StopLossTargets:
    """Institutional ATR-based Stop Loss, Take Profit (TP1/TP2), and Trailing Stop levels."""

    action: TradeDirection
    entry_price: float
    atr_14: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    trailing_stop_distance: float
    risk_per_unit: float
    reward_risk_ratio_1: float
    reward_risk_ratio_2: float


class StopLossManager:
    """
    Calculates dynamic 2x ATR(14) Stop-Loss, minimum 1:2 RRR Take-Profit targets,
    and ratcheting trailing stops.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def calculate_targets(
        self,
        entry_price: float,
        atr_14: float,
        action: TradeDirection,
        atr_multiplier: Optional[float] = None,
        min_rrr: Optional[float] = None,
    ) -> StopLossTargets:
        """
        Compute Entry, Stop-Loss (2 * ATR(14)), TP1 (min 1:2 RRR), and TP2 (1:3 RRR).
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        effective_atr = atr_14 if atr_14 > 0 else entry_price * 0.012
        mult = atr_multiplier if atr_multiplier is not None else self.settings.atr_stop_loss_multiplier
        rrr_1 = max(min_rrr if min_rrr is not None else self.settings.min_risk_reward_ratio, 2.0)
        rrr_2 = round(rrr_1 * 1.5, 2)

        stop_distance = mult * effective_atr
        # Guard against stop_loss <= 0 on extreme ATR inputs
        stop_distance = min(stop_distance, entry_price * 0.45)

        if action == "BUY":
            stop_loss = entry_price - stop_distance
            tp1 = entry_price + (stop_distance * rrr_1)
            tp2 = entry_price + (stop_distance * rrr_2)
        elif action == "SELL":
            stop_loss = entry_price + stop_distance
            tp1 = max(entry_price * 0.05, entry_price - (stop_distance * rrr_1))
            tp2 = max(entry_price * 0.02, entry_price - (stop_distance * rrr_2))
        else:
            stop_loss = entry_price - stop_distance
            tp1 = entry_price + (stop_distance * rrr_1)
            tp2 = entry_price + (stop_distance * rrr_2)

        return StopLossTargets(
            action=action,
            entry_price=round(entry_price, 4),
            atr_14=round(effective_atr, 4),
            stop_loss=round(stop_loss, 4),
            take_profit_1=round(tp1, 4),
            take_profit_2=round(tp2, 4),
            trailing_stop_distance=round(stop_distance, 4),
            risk_per_unit=round(stop_distance, 4),
            reward_risk_ratio_1=round(rrr_1, 2),
            reward_risk_ratio_2=round(rrr_2, 2),
        )

    def calculate_realistic_targets(
        self,
        entry_price: float,
        atr_14: float,
        action: TradeDirection,
        support_1: float = 0.0,
        support_2: float = 0.0,
        resistance_1: float = 0.0,
        resistance_2: float = 0.0,
    ) -> StopLossTargets:
        """
        Compute realistic, tight intraday Stop-Loss and strict 1:2 (TP1) / 1:3 (TP2)
        Take-Profit targets calibrated for 1m-5m execution:
          - Stop-Loss: 0.65 * effective_atr (tight protection outside 1m-5m noise)
          - TP1: 2.0 * Stop-Loss distance (Strict 1:2 Reward-to-Risk Ratio)
          - TP2: 3.0 * Stop-Loss distance (Strict 1:3 Reward-to-Risk Ratio)
          - Trailing Distance: 0.55 * Stop-Loss distance for responsive profit locking
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        effective_atr = atr_14 if atr_14 > 0 else entry_price * 0.001
        stop_distance = min(0.65 * effective_atr, entry_price * 0.04)
        trail_distance = max(0.55 * stop_distance, effective_atr * 0.35)

        tp1_dist = stop_distance * 2.0
        tp2_dist = stop_distance * 3.0

        if action == "BUY":
            stop_loss = entry_price - stop_distance
            tp1 = entry_price + tp1_dist
            tp2 = entry_price + tp2_dist
        elif action == "SELL":
            stop_loss = entry_price + stop_distance
            tp1 = max(entry_price * 0.05, entry_price - tp1_dist)
            tp2 = max(entry_price * 0.02, entry_price - tp2_dist)
        else:
            stop_loss = entry_price - stop_distance
            tp1 = entry_price + tp1_dist
            tp2 = entry_price + tp2_dist

        return StopLossTargets(
            action=action,
            entry_price=round(entry_price, 4),
            atr_14=round(effective_atr, 4),
            stop_loss=round(stop_loss, 4),
            take_profit_1=round(tp1, 4),
            take_profit_2=round(tp2, 4),
            trailing_stop_distance=round(trail_distance, 4),
            risk_per_unit=round(stop_distance, 4),
            reward_risk_ratio_1=2.0,
            reward_risk_ratio_2=3.0,
        )

    def calculate_structural_targets(
        self,
        entry_price: float,
        action: TradeDirection,
        latest_hl: float = 0.0,
        latest_lh: float = 0.0,
        atr_14: float = 0.0,
        min_rrr: float = 2.0,
        buffer_atr_mult: float = 0.15,
    ) -> StopLossTargets:
        """
        Hitung Stop-Loss Berdasarkan Invalidation Point Struktur Market (Pullback):
        - BUY di Pullback: Ide kita Higher Low (HL) terjaga.
          Stop-Loss ditaruh sedikit di bawah lembah itu (latest_hl - buffer).
          Jika harga menembus ke bawah, ide salah -> Cut loss kecil, terukur, dan diterima dari awal.
        - SELL di Pullback: Ide kita Lower High (LH) terjaga.
          Stop-Loss ditaruh sedikit di atas puncak itu (latest_lh + buffer).
          Jika harga menembus ke atas, ide salah -> Cut loss kecil, terukur, dan diterima dari awal.
        - Jarak Entry ke Stop-Loss adalah Risiko = 1R.
        - Jarak Entry ke Take-Profit 1 = 2x Risiko = 2R (Rasio 1R:2R). Satu kali benar menutup 2x salah!
        - Take-Profit 2 = 3R (Rasio 1R:3R).
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        effective_atr = atr_14 if atr_14 > 0 else entry_price * 0.002
        buffer = max(buffer_atr_mult * effective_atr, entry_price * 0.0005)
        min_risk = max(0.50 * effective_atr, entry_price * 0.001)
        max_risk = min(3.0 * effective_atr, entry_price * 0.05)

        if action == "BUY":
            if latest_hl > 0 and latest_hl < entry_price:
                raw_sl = latest_hl - buffer
                raw_dist = entry_price - raw_sl
                risk_dist = min(max(raw_dist, min_risk), max_risk)
                stop_loss = entry_price - risk_dist
            else:
                risk_dist = min(0.65 * effective_atr, entry_price * 0.04)
                stop_loss = entry_price - risk_dist
            tp1 = entry_price + (risk_dist * min_rrr)
            tp2 = entry_price + (risk_dist * (min_rrr + 1.0))
        elif action == "SELL":
            if latest_lh > 0 and latest_lh > entry_price:
                raw_sl = latest_lh + buffer
                raw_dist = raw_sl - entry_price
                risk_dist = min(max(raw_dist, min_risk), max_risk)
                stop_loss = entry_price + risk_dist
            else:
                risk_dist = min(0.65 * effective_atr, entry_price * 0.04)
                stop_loss = entry_price + risk_dist
            tp1 = max(entry_price * 0.05, entry_price - (risk_dist * min_rrr))
            tp2 = max(entry_price * 0.02, entry_price - (risk_dist * (min_rrr + 1.0)))
        else:
            risk_dist = min(0.65 * effective_atr, entry_price * 0.04)
            stop_loss = entry_price - risk_dist
            tp1 = entry_price + (risk_dist * min_rrr)
            tp2 = entry_price + (risk_dist * (min_rrr + 1.0))

        return StopLossTargets(
            action=action,
            entry_price=round(entry_price, 4),
            atr_14=round(effective_atr, 4),
            stop_loss=round(stop_loss, 4),
            take_profit_1=round(tp1, 4),
            take_profit_2=round(tp2, 4),
            trailing_stop_distance=round(risk_dist * 0.55, 4),
            risk_per_unit=round(risk_dist, 4),
            reward_risk_ratio_1=round(min_rrr, 2),
            reward_risk_ratio_2=round(min_rrr + 1.0, 2),
        )

    def calculate_event_mode_targets(
        self,
        entry_price: float,
        action: TradeDirection,
        account_balance: float,
        lot_size: float = 0.01,
        symbol: str = "XAUUSD",
        risk_pct: float = 0.015,
    ) -> StopLossTargets:
        """
        Special Macro Event Sniper Targets:
          1. NO Take Profit (TP1 = 0.0, TP2 = 0.0) -> Let runners ride explosive news trends without ceiling.
          2. Stop Loss strictly bounded to Account Capital (account_balance * risk_pct) -> Prevents Margin Call (MC)!
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        bal = max(10.0, float(account_balance))
        lot = max(0.01, float(lot_size))
        eff_risk_pct = max(0.005, min(0.05, float(risk_pct)))
        max_dollar_risk = bal * eff_risk_pct

        clean_sym = symbol.upper().replace("/", "").replace("-", "")
        if any(k in clean_sym for k in ("XAU", "GOLD")):
            point_mult = 100.0  # 1.0 lot = 100 oz -> $1.00 move is $100
        elif any(k in clean_sym for k in ("BTC", "ETH", "SOL", "CRYPTO")):
            point_mult = 1.0    # 1.0 lot = 1 coin -> $1.00 move is $1
        else:
            point_mult = 100000.0  # Standard FX lot

        dollar_per_point = max(1e-6, lot * point_mult)
        base_distance = max_dollar_risk / dollar_per_point
        # Strict Anti-MC Ceiling: Maximum risk distance must never exceed 5% of total account capital
        max_mc_safe_dist = (bal * 0.05) / dollar_per_point
        min_room = min(entry_price * 0.003, max_mc_safe_dist)
        stop_distance = max(min_room, min(base_distance, max_mc_safe_dist, entry_price * 0.10))
        if action == "BUY":
            stop_loss = entry_price - stop_distance
        elif action == "SELL":
            stop_loss = entry_price + stop_distance
        else:
            stop_loss = entry_price - stop_distance

        return StopLossTargets(
            action=action,
            entry_price=round(entry_price, 4),
            atr_14=round(stop_distance, 4),
            stop_loss=round(stop_loss, 4),
            take_profit_1=0.0,  # TP REMOVED
            take_profit_2=0.0,  # TP REMOVED
            trailing_stop_distance=round(stop_distance * 0.50, 4),
            risk_per_unit=round(stop_distance, 4),
            reward_risk_ratio_1=0.0,
            reward_risk_ratio_2=0.0,
        )

    @staticmethod
    def update_trailing_stop(
        action: TradeDirection,
        current_price: float,
        current_stop_loss: float,
        trailing_distance: float,
    ) -> float:
        """
        Ratchet trailing stop-loss in the direction of profitable price movement.
        Never widens risk beyond current_stop_loss.
        """
        if trailing_distance <= 0:
            return current_stop_loss

        if action == "BUY":
            candidate = current_price - trailing_distance
            return round(max(current_stop_loss, candidate), 4)
        if action == "SELL":
            candidate = current_price + trailing_distance
            return round(min(current_stop_loss, candidate), 4)
        return current_stop_loss
