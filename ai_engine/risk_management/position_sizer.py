from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.risk_management.position_sizer")


@dataclass
class PositionSizeResult:
    """Institutional position sizing & circuit breaker assessment."""

    approved: bool
    rejection_reason: Optional[str]
    portfolio_equity: float
    base_risk_capital: float
    adjusted_risk_capital: float
    position_units: float
    position_notional: float
    account_risk_pct: float
    kelly_fraction: float
    risk_multiplier: float
    confidence: float
    s_vol: float
    circuit_breaker_active: bool
    volatility_suppressed: bool


class PositionSizer:
    """
    Kelly Criterion & Volatility-adjusted institutional position sizer with
    4% Daily Drawdown Circuit Breaker and Extreme Volatility Suppression.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.daily_start_equity: float = self.settings.portfolio_initial_capital
        self.current_equity: float = self.settings.portfolio_initial_capital
    @property
    def current_daily_drawdown(self) -> float:
        """Return today's drawdown fraction in [0.0, 1.0]."""
        if self.daily_start_equity <= 0:
            return 0.0
        loss = max(0.0, self.daily_start_equity - self.current_equity)
        return round(loss / self.daily_start_equity, 6)

    def is_circuit_breaker_tripped(
        self,
        daily_drawdown_override: Optional[float] = None,
    ) -> bool:
        """Return True if daily loss reaches or exceeds max_daily_drawdown_limit (4%)."""
        dd = (
            daily_drawdown_override
            if daily_drawdown_override is not None
            else self.current_daily_drawdown
        )
        return dd >= self.settings.max_daily_drawdown_limit

    @staticmethod
    def compute_kelly_fraction(
        win_probability: float,
        reward_risk_ratio: float = 2.0,
        half_kelly: bool = True,
    ) -> float:
        """
        Compute Kelly Criterion optimal fraction: f* = (p * b - q) / b.
        Uses Half-Kelly dampening by default for institutional drawdown control.
        """
        p = min(max(win_probability, 0.0), 1.0)
        q = 1.0 - p
        b = max(reward_risk_ratio, 1e-6)
        raw_kelly = max(0.0, (p * b - q) / b)
        dampened = 0.5 * raw_kelly if half_kelly else raw_kelly
        return round(min(dampened, 0.25), 4)

    def calculate_position(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        s_vol: float,
        portfolio_equity: Optional[float] = None,
        reward_risk_ratio: Optional[float] = None,
        daily_drawdown_override: Optional[float] = None,
        enable_kelly: bool = False,
    ) -> PositionSizeResult:
        """
          Position Size = Base Size * (1 - S_vol) * Confidence
          Max Account Risk Per Trade <= 1.5% of total portfolio capital
          Daily Drawdown Circuit Breaker < 4%
          Volatility Suppression < max_volatility_threshold
        """
        equity = portfolio_equity if portfolio_equity is not None else self.current_equity
        rrr = reward_risk_ratio if reward_risk_ratio is not None else self.settings.min_risk_reward_ratio

        clamped_vol = min(max(float(s_vol), 0.0), 1.0)
        clamped_conf = min(max(float(confidence), 0.0), 1.0)
        risk_multiplier = round(1.0 - clamped_vol, 4)

        circuit_tripped = self.is_circuit_breaker_tripped(daily_drawdown_override)
        vol_suppressed = clamped_vol >= self.settings.max_volatility_threshold

        kelly = self.compute_kelly_fraction(clamped_conf, reward_risk_ratio=rrr, half_kelly=True) if enable_kelly else 0.0
        base_risk_capital = equity * self.settings.max_account_risk_per_trade

        if circuit_tripped:
            logger.warning(
                "Circuit breaker tripped: daily drawdown limit exceeded",
                daily_drawdown=daily_drawdown_override or self.current_daily_drawdown,
                limit=self.settings.max_daily_drawdown_limit,
            )
            return PositionSizeResult(
                approved=False,
                rejection_reason="Daily drawdown circuit breaker exceeded (4.0% limit)",
                portfolio_equity=round(equity, 2),
                base_risk_capital=round(base_risk_capital, 2),
                adjusted_risk_capital=0.0,
                position_units=0.0,
                position_notional=0.0,
                account_risk_pct=0.0,
                kelly_fraction=kelly,
                risk_multiplier=risk_multiplier,
                confidence=clamped_conf,
                s_vol=clamped_vol,
                circuit_breaker_active=True,
                volatility_suppressed=vol_suppressed,
            )

        if vol_suppressed:
            logger.warning(
                "High volatility safety suppression triggered",
                s_vol=clamped_vol,
                threshold=self.settings.max_volatility_threshold,
            )
            return PositionSizeResult(
                approved=False,
                rejection_reason=(
                    f"Realized volatility S_vol ({clamped_vol:.2f}) exceeds safe threshold "
                    f"({self.settings.max_volatility_threshold:.2f})"
                ),
                portfolio_equity=round(equity, 2),
                base_risk_capital=round(base_risk_capital, 2),
                adjusted_risk_capital=0.0,
                position_units=0.0,
                position_notional=0.0,
                account_risk_pct=0.0,
                kelly_fraction=kelly,
                risk_multiplier=risk_multiplier,
                confidence=clamped_conf,
                s_vol=clamped_vol,
                circuit_breaker_active=False,
                volatility_suppressed=True,
            )

        # AGENT.md 5.A: Position Size = Base Size * (1 - S_vol) * Confidence
        scaling_factor = (1.0 - clamped_vol) * clamped_conf
        adjusted_risk_capital = min(
            base_risk_capital,
            base_risk_capital * scaling_factor,
        )

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 1e-9 or entry_price <= 0:
            stop_distance = max(entry_price * 0.02, 1e-4)

        raw_units = adjusted_risk_capital / stop_distance
        # Cap total notional exposure to 100% of unlevered portfolio equity
        max_units_by_equity = equity / max(entry_price, 1e-6)
        position_units = min(raw_units, max_units_by_equity)
        position_notional = position_units * entry_price
        actual_risk_capital = position_units * stop_distance
        account_risk_pct = (actual_risk_capital / equity) if equity > 0 else 0.0

        return PositionSizeResult(
            approved=True,
            rejection_reason=None,
            portfolio_equity=round(equity, 2),
            base_risk_capital=round(base_risk_capital, 2),
            adjusted_risk_capital=round(actual_risk_capital, 2),
            position_units=round(position_units, 6),
            position_notional=round(position_notional, 2),
            account_risk_pct=round(account_risk_pct, 6),
            kelly_fraction=kelly,
            risk_multiplier=risk_multiplier,
            confidence=clamped_conf,
            s_vol=clamped_vol,
            circuit_breaker_active=False,
            volatility_suppressed=False,
        )
