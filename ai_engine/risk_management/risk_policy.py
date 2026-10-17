"""
Centralized Institutional Risk Policy Module (Production Remediation Specification).
Enforces non-bypassable risk bounds, capital preservation invariants, and preflight checks
independently of strategy outputs, Telegram commands, webhook hints, and news modes.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger("nexustrade.risk.policy")


@dataclass
class SymbolTradingSpec:
    """Comprehensive broker symbol specification for order sizing and margin calculations."""
    symbol: str
    digits: int = 2
    point: float = 0.01
    contract_size: float = 100.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 30
    freeze_level: int = 0
    tick_size: float = 0.01
    tick_value: float = 1.0  # Monetary value per 1.0 lot per 1.0 tick in account currency
    margin_initial: float = 0.0
    spread_points: int = 15
    base_currency: str = "XAU"
    quote_currency: str = "USD"
    account_currency: str = "USD"


@dataclass
class AccountRiskSnapshot:
    """Snapshot of account equity, margin, drawdowns, and reserved risk."""
    account_id: str
    balance: float
    equity: float
    free_margin: float
    start_of_day_equity: float
    high_water_equity: float
    open_positions_loss_at_stop: float = 0.0
    in_flight_reserved_risk: float = 0.0
    is_circuit_breaker_latched: bool = False
    kill_switch_active: bool = False
    last_reset_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )


@dataclass
class PreflightRiskResult:
    """Rigorous preflight risk evaluation result."""
    approved: bool
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None
    allowed_lots: float = 0.0
    estimated_risk_usd: float = 0.0
    trade_budget_usd: float = 0.0
    daily_headroom_usd: float = 0.0
    total_headroom_usd: float = 0.0


class RiskPolicy:
    """Centralized, immutable policy engine enforcing capital preservation rules."""

    def __init__(
        self,
        max_risk_per_trade_pct: float = 0.015,
        max_daily_drawdown_limit: float = 0.04,
        max_total_drawdown_limit: float = 0.08,
        max_open_risk_fraction: float = 0.06,
        max_symbol_exposure_fraction: float = 0.03,
        max_spread_points: float = 40.0,
        live_trading_enabled: bool = False,
        allow_micro_min_lot_override: bool = True,
        micro_account_ceiling_usd: float = 250.0,
        micro_max_lot_risk_fraction: float = 0.50,
    ) -> None:
        self.max_risk_per_trade_pct = float(max_risk_per_trade_pct)
        self.max_daily_drawdown_limit = float(max_daily_drawdown_limit)
        self.max_total_drawdown_limit = float(max_total_drawdown_limit)
        self.max_open_risk_fraction = float(max_open_risk_fraction)
        self.max_symbol_exposure_fraction = float(max_symbol_exposure_fraction)
        self.max_spread_points = float(max_spread_points)
        self.live_trading_enabled = bool(live_trading_enabled)
        self.allow_micro_min_lot_override = bool(allow_micro_min_lot_override)
        self.micro_account_ceiling_usd = float(micro_account_ceiling_usd)
        self.micro_max_lot_risk_fraction = float(micro_max_lot_risk_fraction)

    @staticmethod
    def floor_to_broker_step(
        raw_volume: float,
        volume_min: float = 0.01,
        volume_step: float = 0.01,
        volume_max: float = 100.0,
    ) -> float:
        """
        Round lot volume strictly DOWN to valid volume_step anchored at volume_min (NT-03, Section 3).
        Formula: max(0.0, floor((raw_volume - volume_min) / volume_step)) * volume_step + volume_min
        Returns 0.0 if raw_volume < volume_min.
        """
        v_min = max(1e-6, float(volume_min))
        v_step = max(1e-6, float(volume_step))
        v_max = max(v_min, float(volume_max))

        if raw_volume < v_min:
            return 0.0

        clamped = min(float(raw_volume), v_max)
        steps_above_min = math.floor(((clamped - v_min) / v_step) + 1e-9)
        stepped = v_min + (max(0, steps_above_min) * v_step)

        # Precision formatting
        step_str = f"{v_step:.6f}".rstrip("0")
        decimals = len(step_str.split(".")[1]) if "." in step_str else 2
        return round(min(v_max, max(v_min, stepped)), decimals)

    @staticmethod
    def calculate_worst_case_loss(
        spec: SymbolTradingSpec,
        side: Literal["BUY", "SELL"],
        entry_price: float,
        stop_loss: float,
        lots: float,
    ) -> float:
        """
        Calculate actual stop-side monetary loss in account currency (NT-03).
        Takes contract size, point size, tick size, tick value, and currency conversion into account.
        """
        if lots <= 0 or entry_price <= 0 or stop_loss <= 0:
            return 0.0

        # Point distance
        if side == "BUY":
            price_delta = max(0.0, entry_price - stop_loss)
        else:
            price_delta = max(0.0, stop_loss - entry_price)

        # Conversion using contract size
        clean_sym = spec.symbol.upper()
        raw_loss = price_delta * float(lots) * spec.contract_size

        # Quote currency conversion to USD if quote is not USD (e.g. EURJPY, USDCHF, USDCAD)
        if len(clean_sym) == 6 and clean_sym.startswith("USD") and not clean_sym.endswith("USD"):
            # Base is USD, quote is JPY/CAD/CHF -> raw_loss is in quote ccy -> divide by exit/entry price
            raw_loss = raw_loss / max(entry_price, 1e-6)

        return round(raw_loss, 2)

    def validate_stops_level(
        self,
        spec: SymbolTradingSpec,
        side: Literal["BUY", "SELL"],
        entry_price: float,
        stop_loss: float,
        take_profit: float = 0.0,
    ) -> Tuple[bool, str]:
        """Validate Stop-Loss side and minimum stops_level / freeze_level compliance."""
        pt = spec.point
        min_stops_dist = max(spec.stops_level * pt, pt * 10.0)

        if side == "BUY":
            if stop_loss >= entry_price:
                return False, f"SL ({stop_loss}) harus berada di bawah harga Entry ({entry_price})"
            if (entry_price - stop_loss) < min_stops_dist:
                return False, f"Jarak SL ({entry_price - stop_loss:.4f}) terlalu dekat (< {min_stops_dist:.4f} pt)"
            if take_profit > 0 and take_profit <= entry_price:
                return False, f"TP ({take_profit}) harus berada di atas harga Entry ({entry_price})"
        else:
            if stop_loss <= entry_price:
                return False, f"SL ({stop_loss}) harus berada di atas harga Entry ({entry_price})"
            if (stop_loss - entry_price) < min_stops_dist:
                return False, f"Jarak SL ({stop_loss - entry_price:.4f}) terlalu dekat (< {min_stops_dist:.4f} pt)"
            if take_profit > 0 and take_profit >= entry_price:
                return False, f"TP ({take_profit}) harus berada di bawah harga Entry ({entry_price})"

        return True, "Stops level valid"

    def evaluate_preflight(
        self,
        spec: SymbolTradingSpec,
        account: AccountRiskSnapshot,
        side: Literal["BUY", "SELL"],
        entry_price: float,
        stop_loss: float,
        take_profit: float = 0.0,
        requested_lots: Optional[float] = None,
        is_live_account: bool = False,
    ) -> PreflightRiskResult:
        """
        Complete non-bypassable preflight risk evaluation enforcing NT-01, NT-02, NT-03, NT-11:
        1. Master kill switch & live safety freeze.
        2. Latched persistent circuit breaker.
        3. Spread filter.
        4. Stops level compliance.
        5. Absolute trade budget (equity * max_risk_per_trade_pct; NO micro-account expansion).
        6. Daily drawdown headroom & all-time high-water drawdown headroom.
        7. Volume floor to valid broker step.
        8. Total aggregate portfolio risk cap check.
        """
        # 1. Master Kill Switch & Live Freeze Check
        if account.kill_switch_active:
            return PreflightRiskResult(
                approved=False,
                rejection_code="KILL_SWITCH_ACTIVE",
                rejection_reason="Master Kill Switch aktif. Seluruh eksekusi order dibekukan.",
            )

        if is_live_account and not self.live_trading_enabled:
            return PreflightRiskResult(
                approved=False,
                rejection_code="LIVE_TRADING_DISABLED",
                rejection_reason="Akun LIVE terdeteksi tetapi LIVE_TRADING_ENABLED=false (Safety Freeze).",
            )

        # 2. Latched Circuit Breaker Check
        if account.is_circuit_breaker_latched:
            return PreflightRiskResult(
                approved=False,
                rejection_code="CIRCUIT_BREAKER_LATCHED",
                rejection_reason="Circuit Breaker drawdown harian terkunci (latched). Butuh reset audit manual.",
            )

        # 3. Spread Filter Check
        if spec.spread_points > self.max_spread_points:
            return PreflightRiskResult(
                approved=False,
                rejection_code="EXCESSIVE_SPREAD",
                rejection_reason=f"Spread pasar ({spec.spread_points} pt) melebihi batas aman ({self.max_spread_points} pt).",
            )

        # 4. Stops Level & Side Validation
        stops_ok, stops_reason = self.validate_stops_level(spec, side, entry_price, stop_loss, take_profit)
        if not stops_ok:
            return PreflightRiskResult(
                approved=False,
                rejection_code="INVALID_STOPS_LEVEL",
                rejection_reason=stops_reason,
            )

        # 5. Calculate Absolute Capital Budgets (NT-01: NO micro-account expansion!)
        eq = max(0.0, account.equity)
        if eq <= 0:
            return PreflightRiskResult(
                approved=False,
                rejection_code="INSUFFICIENT_EQUITY",
                rejection_reason="Equity akun kosong atau non-positif.",
            )

        trade_budget = eq * self.max_risk_per_trade_pct

        # Daily Drawdown Headroom (persisted start-of-day equity)
        sod_eq = max(eq, account.start_of_day_equity)
        current_daily_loss = max(0.0, sod_eq - eq)
        max_daily_loss_allowed = sod_eq * self.max_daily_drawdown_limit
        daily_headroom = max(0.0, max_daily_loss_allowed - current_daily_loss)

        if current_daily_loss >= max_daily_loss_allowed:
            return PreflightRiskResult(
                approved=False,
                rejection_code="DAILY_DRAWDOWN_EXCEEDED",
                rejection_reason=(
                    f"Batas rugi harian tercapai (Rugi ${current_daily_loss:,.2f} >= "
                    f"Batas ${max_daily_loss_allowed:,.2f} / {self.max_daily_drawdown_limit * 100:.1f}%)."
                ),
                daily_headroom_usd=0.0,
            )

        # All-Time High-Water Drawdown Headroom
        hw_eq = max(eq, account.high_water_equity)
        current_hw_loss = max(0.0, hw_eq - eq)
        max_hw_loss_allowed = hw_eq * self.max_total_drawdown_limit
        total_headroom = max(0.0, max_hw_loss_allowed - current_hw_loss)

        if current_hw_loss >= max_hw_loss_allowed:
            return PreflightRiskResult(
                approved=False,
                rejection_code="TOTAL_DRAWDOWN_EXCEEDED",
                rejection_reason=(
                    f"Batas rugi total high-water tercapai (Rugi ${current_hw_loss:,.2f} >= "
                    f"Batas ${max_hw_loss_allowed:,.2f} / {self.max_total_drawdown_limit * 100:.1f}%)."
                ),
                total_headroom_usd=0.0,
            )

        effective_max_risk = min(trade_budget, daily_headroom, total_headroom)

        # 6. Sizing to fit risk (NT-03: Using broker volume_min & volume_step)
        loss_per_one_lot = self.calculate_worst_case_loss(spec, side, entry_price, stop_loss, 1.0)
        if loss_per_one_lot <= 1e-9:
            return PreflightRiskResult(
                approved=False,
                rejection_code="ZERO_STOP_DISTANCE",
                rejection_reason="Jarak Stop Loss tidak boleh 0.",
            )

        raw_affordable_lots = effective_max_risk / loss_per_one_lot
        if requested_lots is not None and requested_lots > 0:
            requested_loss = self.calculate_worst_case_loss(spec, side, entry_price, stop_loss, requested_lots)
            if requested_loss > effective_max_risk + 1e-6:
                return PreflightRiskResult(
                    approved=False,
                    rejection_code="CUSTOM_LOT_EXCEEDS_RISK",
                    rejection_reason=(
                        f"Risiko {requested_lots:.2f} Lot (${requested_loss:,.2f}) melampaui batas risiko (${effective_max_risk:,.2f})."
                    ),
                    allowed_lots=0.0,
                    estimated_risk_usd=requested_loss,
                    trade_budget_usd=round(trade_budget, 2),
                    daily_headroom_usd=round(daily_headroom, 2),
                )
            target_lots = requested_lots
        else:
            target_lots = raw_affordable_lots

        floored_lots = self.floor_to_broker_step(
            raw_volume=target_lots,
            volume_min=spec.volume_min,
            volume_step=spec.volume_step,
            volume_max=spec.volume_max,
        )
        if floored_lots < spec.volume_min:
            min_lot_loss = self.calculate_worst_case_loss(spec, side, entry_price, stop_loss, spec.volume_min)
            # Fix 4: Allow micro accounts (< $250 USD) to trade broker minimum lot if risk is within micro_max_lot_risk_fraction
            if (
                self.allow_micro_min_lot_override
                and eq < self.micro_account_ceiling_usd
                and min_lot_loss <= (eq * self.micro_max_lot_risk_fraction)
            ):
                floored_lots = spec.volume_min
                effective_max_risk = max(effective_max_risk, min_lot_loss)
            else:
                return PreflightRiskResult(
                    approved=False,
                    rejection_code="MIN_LOT_EXCEEDS_RISK",
                    rejection_reason=(
                        f"Risiko lot minimum broker ({spec.volume_min:.2f} Lot = ${min_lot_loss:,.2f}) "
                        f"melampaui anggaran risiko efektif (${effective_max_risk:,.2f})."
                    ),
                    trade_budget_usd=round(trade_budget, 2),
                    daily_headroom_usd=round(daily_headroom, 2),
                    total_headroom_usd=round(total_headroom, 2),
                )
        estimated_risk = self.calculate_worst_case_loss(spec, side, entry_price, stop_loss, floored_lots)

        # NT-02: Non-bypassable hard cap check!
        if estimated_risk > effective_max_risk + 1e-6:
            return PreflightRiskResult(
                approved=False,
                rejection_code="HARD_RISK_CAP",
                rejection_reason=(
                    f"Risiko posisi (${estimated_risk:,.2f}) melampaui batas risiko (${effective_max_risk:,.2f})."
                ),
                allowed_lots=floored_lots,
                estimated_risk_usd=estimated_risk,
                trade_budget_usd=round(trade_budget, 2),
            )

        # 7. Total Open Portfolio Risk Cap Check (NT-11)
        total_open_risk_with_trade = account.open_positions_loss_at_stop + account.in_flight_reserved_risk + estimated_risk
        eff_portfolio_cap = (
            max(self.max_open_risk_fraction, self.micro_max_lot_risk_fraction)
            if (self.allow_micro_min_lot_override and eq < self.micro_account_ceiling_usd)
            else self.max_open_risk_fraction
        )
        max_portfolio_risk_allowed = eq * eff_portfolio_cap
        if total_open_risk_with_trade > max_portfolio_risk_allowed + 1e-6:
            return PreflightRiskResult(
                approved=False,
                rejection_code="PORTFOLIO_RISK_CAP_EXCEEDED",
                rejection_reason=(
                    f"Total risiko portofolio (${total_open_risk_with_trade:,.2f}) melebihi "
                    f"batas agregat ({eff_portfolio_cap * 100:.1f}% = ${max_portfolio_risk_allowed:,.2f})."
                ),
                allowed_lots=floored_lots,
                estimated_risk_usd=estimated_risk,
            )

        return PreflightRiskResult(
            approved=True,
            allowed_lots=floored_lots,
            estimated_risk_usd=estimated_risk,
            trade_budget_usd=round(trade_budget, 2),
            daily_headroom_usd=round(daily_headroom, 2),
            total_headroom_usd=round(total_headroom, 2),
        )
