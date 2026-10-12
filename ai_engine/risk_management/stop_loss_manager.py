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
