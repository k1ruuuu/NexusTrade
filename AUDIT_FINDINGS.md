# AUDIT_FINDINGS.md — NexusTrade Production Audit & Remediation Log

**Target:** `NexusTrade-feat-nexustrade-ai-core`  
**Date:** October 2026  
**Auditor:** Quantitative Systems Architect & Security Engineering Lead  
**Governing Standard:** `NEXUSTRADE_PRODUCTION_REMEDIATION_PLAN.md`

---

## 1. Executive Summary

This document records the exact root cause, reproduction scenario, severity, remediation, and residual risk assessment for findings NT-01 through NT-12 identified in the production audit. All P0 and P1 vulnerabilities have been systematically remediated and verified via the 20-scenario acceptance test suite (`tests/test_production_remediation.py`).

---

## 2. Granular Remediation Findings Log

### NT-01: Micro-Account Risk Budget Inflation & Drawdown Relaxation
- **Severity:** P0 (Critical Capital Preservation)
- **File & Line:** `ai_engine/auto_trader/engine.py` (formerly lines 746–756, 856–860)
- **Original Behavior:** For accounts under $100 equity, the trade budget was artificially expanded up to 45% ($4.50), and the daily drawdown limit was relaxed from 4% to 20% to force 0.01 lot orders to pass.
- **Risk Impact:** Rapid account liquidation on small balances upon 2–3 consecutive losses.
- **Remediation:** Removed micro-account special-casing completely. Enforced strict linear risk budgeting: $\text{Budget} = \text{Equity} \times \text{eff\_risk\_rate}$. If the broker's minimum lot creates worst-case stop loss loss exceeding budget, the trade is rejected with `MIN_LOT_EXCEEDS_RISK`.
- **Test Evidence:** `tests/test_production_remediation.py::test_r01_micro_account_no_budget_expansion_rejects_min_lot` (PASSED).
- **Residual Risk:** Small accounts (<$50) trading high-notional assets (like Gold) will frequently receive `MIN_LOT_EXCEEDS_RISK`. This is the mathematically correct and desired capital preservation behavior.

---

### NT-02: Macro Event Sniper Mode Bypassing Stop-Loss Risk Cap
- **Severity:** P0 (Critical Capital Preservation)
- **File & Line:** `ai_engine/auto_trader/engine.py` (formerly line 889)
- **Original Behavior:** The conditional check `if not macro_event_mode and loss_at_stop > max_risk_budget:` allowed any trade executed with `macro_event_mode` active to completely bypass the risk ceiling.
- **Risk Impact:** Uncapped losses on volatile news releases (CPI/NFP/FOMC) risking margin call.
- **Remediation:** Removed `not macro_event_mode` exclusion. The centralized `RiskPolicy.evaluate_preflight` now strictly gates all order requests regardless of macro mode, EA mode, or admin flags. If risk exceeds budget, rejects with `HARD_RISK_CAP`.
- **Test Evidence:** `tests/test_production_remediation.py::test_r02_macro_event_mode_enforces_non_bypassable_hard_risk_cap` (PASSED).
- **Residual Risk:** Zero risk budget bypass.

---

### NT-03: Arbitrary 0.01 Lot Step and USD-Centric PnL Assumptions
- **Severity:** P0 (Contract Specification Correctness)
- **File & Line:** `ai_engine/auto_trader/engine.py` & `mt5_symbols.py`
- **Original Behavior:** Lot sizing assumed generic 2-decimal 0.01 step rounding for all symbols, and PnL conversions assumed USD quote currency for all pairs.
- **Risk Impact:** `TRADE_RETCODE_INVALID_VOLUME` on symbols with 0.1 or 0.001 lot steps, and wrong position sizing on non-USD quote currency pairs (e.g. USDJPY, EURGBP).
- **Remediation:** Implemented `RiskPolicy.floor_to_broker_step` using formula $\text{volume\_min} + \lfloor \frac{\text{raw} - \text{min}}{\text{step}} \rfloor \times \text{step}$. Integrated quote currency normalization in `calculate_worst_case_loss`.
- **Test Evidence:** `tests/test_production_remediation.py::test_r04_volume_step_and_min_lot_flooring` (PASSED).
- **Residual Risk:** Exotic cross currencies must have broker tick value available.

---

### NT-04: In-Memory Execution Locks Lost on Worker Restart
- **Severity:** P0 (Durable Order Idempotency)
- **File & Line:** `ai_engine/auto_trader/engine.py` & `user_vault.py`
- **Original Behavior:** Concurrency protection relied solely on an in-process `_execution_locks` dictionary. Multi-process deployments or worker restarts allowed duplicate orders.
- **Remediation:** Implemented persistent SQLite tables `order_lifecycle_intents` and `risk_reservations`. Risk is atomically reserved in SQLite before order submission and released after confirmation. State transitions are tracked durably (`SIGNAL_CREATED -> RISK_RESERVED -> SUBMITTING -> PROTECTION_VERIFIED`).
- **Test Evidence:** `tests/test_production_remediation.py::test_r06_concurrent_execution_prevents_duplicate_orders` & `test_r08_order_intent_lifecycle_persistence` (PASSED).
- **Residual Risk:** Distributed clustering requires database file locking or Redis distributed locks.

---

### NT-05: Unverified Broker-Side Protection & Blind Retries
- **Severity:** P0 (Execution Safety)
- **File & Line:** `ai_engine/auto_trader/engine.py`, `position_monitor.py`, `mt5_exness_client.py`
- **Original Behavior:** Transport timeouts caused unconfirmed blind retries or marked orders as protected without verifying MT5 broker ticket SL status.
- **Remediation:** 
  1. Post-timeout reconciliation in `MT5ExnessClient.place_market_order` queries live open broker orders to detect if the ticket was filled before returning a status.
  2. Positions lacking confirmed SL are recorded as `FILLED_UNPROTECTED` and transition to `INCIDENT_UNPROTECTED`.
  3. Gate 5 halts all new trades across the entire account if any unprotected position exists.
- **Test Evidence:** `tests/test_production_remediation.py::test_r07_order_timeout_reconciliation_finds_ticket` & `test_r10_unprotected_position_halts_new_trading` (PASSED).
- **Residual Risk:** Broker network partitions require out-of-band monitoring alerts.

---

### NT-06: EMA Scalping State Mistaken for Crossover
- **Severity:** P1 (Signal Integrity)
- **File & Line:** `ai_engine/auto_trader/engine.py` (lines 475–488)
- **Original Behavior:** Fired buy signals continuously while `EMA9 > EMA21`, confusing an ongoing trend state with a fresh crossing event.
- **Remediation:** Required two consecutive closed candles: `prev_ema9 <= prev_ema21 and last_ema9 > last_ema21` for bullish crossovers; bearish reverse. Returns `action="HOLD"` when trend is continuing without a fresh cross.
- **Test Evidence:** `tests/test_production_remediation.py::test_r11_ema_no_repeated_crossover_without_cross` (PASSED).
- **Residual Risk:** Trend following continuation trades should use separate named strategy.

---

### NT-07: Fixed 0.85 Confidence Score Fabrications
- **Severity:** P1 (Model Calibration)
- **File & Line:** `ai_engine/auto_trader/engine.py`
- **Original Behavior:** Hardcoded fixed confidence `ea_conf = 0.85` for EA strategies.
- **Remediation:** Removed fixed 0.85. Replaced with dynamic heuristic score derived from indicator alignment and EMA spread ratio.
- **Test Evidence:** `tests/test_production_remediation.py::test_r11_ema_no_repeated_crossover_without_cross` (PASSED).
- **Residual Risk:** Heuristic confidence is not a calibrated win probability.

---

### NT-08: Blind RSI 70/30 Counter-Trend Entries
- **Severity:** P1 (Strategy Robustness)
- **File & Line:** `ai_engine/auto_trader/engine.py` (lines 420–443)
- **Original Behavior:** Fired SELL blindly whenever RSI touched 70, shorting strong bull trends with green candles.
- **Remediation:** Required closed-candle reversal confirmation: RSI $\ge$ 70 AND closed red candle (`close < prev_close`). If price is still making green candles, holds and waits for confirmation.
- **Test Evidence:** `tests/test_production_remediation.py::test_r12_rsi_overbought_requires_reversal_confirmation` (PASSED).
- **Residual Risk:** Extreme parabolic climaxes may experience delayed entry.

---

### NT-09: Uncalibrated Confidence Fed to Kelly Criterion
- **Severity:** P1 (Risk Sizing)
- **File & Line:** `ai_engine/risk_management/position_sizer.py`
- **Original Behavior:** Fed raw heuristic confidence directly into Kelly formula by default.
- **Remediation:** Disabled Kelly sizing by default (`enable_kelly=False`). Position sizing uses fixed-fractional risk budget by default.
- **Test Evidence:** `tests/test_production_remediation.py::test_r13_kelly_disabled_by_default` (PASSED).
- **Residual Risk:** Enabling Kelly requires independent probability calibration.

---

### NT-10: Unrealistic Backtester Fills & Zero Cost Modeling
- **Severity:** P1 (Backtest Fidelity)
- **File & Line:** `ai_engine/backtester/engine.py`
- **Original Behavior:** Backtester executed stop loss at exact stop price even during adverse gaps, and assumed winning trades during intrabar ambiguity.
- **Remediation:** Modeled adverse gap slippage (fills at `open_p - slippage` on gaps), conservative intrabar ambiguity (assumes SL hit first if both TP and SL touched in same bar), and broker commissions.
- **Test Evidence:** `tests/test_production_remediation.py::test_r14_backtest_gap_and_intrabar_ambiguity` (PASSED).
- **Residual Risk:** Historical 1-minute data still lacks true tick-level liquidity microstructure.

---

### NT-11: Account-Wide Aggregate Risk and Correlated Exposure
- **Severity:** P0 (Portfolio Safety)
- **File & Line:** `ai_engine/risk_management/risk_policy.py` & `engine.py`
- **Original Behavior:** Only checked risk per trade on individual symbols without aggregating open position risk.
- **Remediation:** Preflight verifies that total open risk (existing positions worst-case SL loss + active risk reservations + proposed trade) $\le \text{max\_open\_risk\_fraction} \times \text{Equity}$ (default 6%). Rejects with `PORTFOLIO_RISK_CAP_EXCEEDED` if breached.
- **Test Evidence:** `tests/test_production_remediation.py::test_r15_portfolio_aggregate_exposure_cap_rejection` (PASSED).
- **Residual Risk:** Cross-asset correlations may spike during macro black swan events.

---

### NT-12: News/Event Trading Bypassing Risk Gates
- **Severity:** P1 (Event Mode Safeguards)
- **File & Line:** `ai_engine/auto_trader/engine.py`
- **Original Behavior:** Macro event mode bypassed standard risk gates.
- **Remediation:** Event modes now pass identical centralized preflight risk gates, plus stricter spread filters and volatility caps.
- **Test Evidence:** `tests/test_production_remediation.py::test_r02_macro_event_mode_enforces_non_bypassable_hard_risk_cap` (PASSED).
- **Residual Risk:** High-slippage news releases must be traded with wide broker stop distances.
