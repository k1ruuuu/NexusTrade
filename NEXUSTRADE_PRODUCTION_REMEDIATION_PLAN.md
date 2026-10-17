# NexusTrade — Production Remediation & Release Gate Plan

**Target:** repository `NexusTrade-feat-nexustrade-ai-core` (ZIP supplied, 17 October 2026).  
**Audience:** engineering / coding agent / QA / operations.  
**Status:** implementation specification, **NOT** a claim that production readiness has been achieved.  
**Priority:** capital protection > correctness > reproducibility > signal quality > performance.

## 0. Mandatory instructions for implementing agent

1. Inspect the existing architecture and `AGENT.md`, `README.md`, `TASKS.md`, `PRODUCTION_AUDIT_REMEDIATION_SPEC.md`, and `ADVANCED_AI_TRADING_ENGINE_SPEC.md` before edits. Treat existing documents as proposals, not proof that issues are fixed. Verify actual runtime behavior.
2. Work in a new branch, preserve existing public interfaces where possible, create small reviewable commits, and **do not** place credentials or real-account identifiers in code, fixtures, screenshots, or logs.
3. For every item below: locate all call sites, add a regression test that fails on the original behavior, implement the fix, rerun tests, and record evidence. Never weaken a test to make it pass.
4. Default **LIVE trading OFF**. No real orders during development, CI, unit tests, or integration tests. Broker tests use stubs or demo credentials explicitly isolated from live.
5. Fail closed: stale quotes, missing symbol metadata, invalid stop, unavailable broker, unknown fill state, missing balance, corrupted model, or unverified risk budget => **NO NEW ORDER**. Existing positions must still be monitored/managed safely.
6. Do not promise profitable strategies or 100% safety. Make thresholds configurable and backtest/walk-forward validate strategy parameters; hard capital-loss limits cannot be bypassed by a strategy, operator convenience, or news mode.

## 1. Confirmed findings and code targets

| ID | Severity | Location | Required fix |
|---|---|---|---|
| NT-01 | P0 | `ai_engine/auto_trader/engine.py` micro-account `trade_budget`, daily DD logic | Remove special-case risk inflation (up to 45% equity / $4.50) and relaxed drawdown. Minimum broker lot that breaches budget => reject. |
| NT-02 | P0 | `ai_engine/auto_trader/engine.py` `if not macro_event_mode and loss_at_stop > max_risk_budget` | Enforce identical non-bypassable risk check for **all** modes, including macro/news, EA, custom lots. |
| NT-03 | P0 | `ai_engine/auto_trader/engine.py` `candidate_lots`, `loss_at_stop` | Use broker symbol `volume_min`, `volume_step`, `volume_max`, tick size/value and currency conversion; validate actual stop-side monetary loss and margin. Never assume 0.01 steps or USD PnL for all instruments. |
| NT-04 | P0 | `ai_engine/auto_trader/engine.py` in-memory `_execution_locks` | Durable idempotency + account/symbol order coordination across workers, restarts and retries. |
| NT-05 | P0 | `ai_engine/auto_trader/engine.py`, `position_monitor.py`, `mt5_exness_client.py` | Reconcile requested/filled position and broker-side SL/TP; do not mark protected unless verified. |
| NT-06 | P1 | `ai_engine/auto_trader/engine.py` scalping EMA condition | Distinguish EMA *state* from actual crossover. Use two **closed** candles for crossing signals, or explicitly rename/test as trend continuation. |
| NT-07 | P1 | `ai_engine/auto_trader/engine.py` `ea_conf = 0.85` | Remove fixed probability-like confidence; keep raw heuristic scores distinct from calibrated win probabilities. |
| NT-08 | P1 | `ai_engine/auto_trader/engine.py` RSI 70/30 reversal | Do not treat RSI thresholds alone as reversal confirmation. Add strategy-specific trend/regime + entry confirmation and HOLD rules, validated out-of-sample. |
| NT-09 | P1 | `ai_engine/risk_management/position_sizer.py` `compute_kelly_fraction` | Never feed uncalibrated signal confidence into Kelly. Default Kelly OFF; only enable on robust, cost-adjusted, independently calibrated probabilities with capped fraction. |
| NT-10 | P1 | `ai_engine/backtester/engine.py` | Conservative gap-through-stop fills, spread, commission, swap/financing, slippage, partial fills, latency, and intrabar ambiguity. |
| NT-11 | P0 | engine + broker + position monitor | Recompute and enforce account-wide risk, correlated exposure, margin and loss caps immediately before each order; handle concurrent orders atomically. |
| NT-12 | P1 | news/event mode | Event/news trading must use normal risk limits plus stricter spread/volatility/liquidity/staleness checks; no bypass. |

**Existing related audit:** `PRODUCTION_AUDIT_REMEDIATION_SPEC.md` additionally alleges feature/target leakage, forced quantile labels, fabricated sentiment, webhook conviction overrides and HTF regime errors. **Reproduce each claim against the current code and tests** before implementing; add independent issues NT-13 onward. Do not blindly copy proposed triple-barrier math: enforce causal features, execution timing, purging/embargo, and deterministic same-bar barrier tie handling.

## 2. Required execution state machine

Implement an explicit, persisted order lifecycle:

`SIGNAL_CREATED -> DATA_VALIDATED -> STRATEGY_APPROVED -> RISK_RESERVED -> PREFLIGHT_PASSED -> SUBMITTING -> (FILLED | PARTIALLY_FILLED | REJECTED | UNKNOWN) -> PROTECTION_VERIFIED -> MONITORING -> CLOSED -> RECONCILED`.

- Every transition has a unique event ID, UTC timestamp, account scope, symbol, reason code, config/model version and audit trail.
- **UNKNOWN** is not safe to retry blindly: query broker order/deal/position history first. Distinguish transport timeout from broker rejection.
- Submit with a stable client intent ID where broker supports it; where not supported, reconcile using a durable local intent ledger and broker tickets/magic/comment within their actual limitations. Do not assume broker deduplicates client comments.
- Reserve risk atomically per account, across all symbols, prior to submission; release/update reservation on rejection, partial fill, close and reconciliation. Use a database transaction/advisory lock or equivalent multi-worker-safe primitive; document failover semantics.
- Broker-side SL is mandatory for newly opened positions unless broker restrictions force an explicit **reject/no-trade**; never rely solely on an in-process trailing stop or Telegram availability.
- After fill, verify position size, side, entry, SL, TP, margin and broker ticket. On protection failure: immediately attempt a bounded, broker-compliant repair; if unsuccessful, trigger incident policy (e.g., emergency close where executable) and stop new trading. Log actual outcome; never assume an emergency close succeeded.
- Support partial fills, netting vs hedging account types, market closure, freeze levels, stop levels, requotes, order expiration, disconnects and restart recovery.

## 3. Risk engine — mandatory invariants

Define all risk controls centrally in a `RiskPolicy` and enforce them independently of strategy output, Telegram commands, webhook hints and news modes:

- Per-trade risk <= configured fraction of **current equity**, after estimated spread, fees and conservative slippage; no micro-account exception. Suggested **initial demo-only example**: 0.25–0.5% per trade, 1–2% daily equity drawdown, 3–5% overall drawdown. These are configuration examples, not guarantees or universally suitable real-account limits.
- Calculate daily drawdown from an explicitly defined broker-time or UTC reset and a persisted start-of-day equity snapshot; include floating PnL and commissions. Use a separate all-time/equity-high-water circuit breaker. Restart must not reset a tripped breaker.
- Total open risk = worst-case SL loss of all existing positions + pending orders + in-flight reservations + proposed trade, with netting/hedging and currency conversion accounted for. Apply configurable symbol, asset-class, directional and correlated-exposure caps.
- For BUY use executable ask for entry and bid for exit; reverse for SELL. Estimate risk with broker `order_calc_profit` (or tested equivalent), `order_calc_margin`, contract size, tick size, tick value and account currency. Reject on unavailable/invalid metadata or calculation failure.
- Round volume **down** to valid `volume_step` anchored at `volume_min`, not generic 2 decimals. Reject if below `volume_min`; check `volume_max`, volume limits and free margin. Never move SL closer solely to make a minimum lot fit.
- Validate SL is on correct side of executable entry, respects `trade_stops_level`, `trade_freeze_level`, price digits/tick size, and broker session. Use post-fill risk recheck; if actual exposure exceeds policy, reduce/close according to documented emergency policy.
- Custom lot is a *request*, never an override. `macro_event_mode`, `ea_mode`, `is_micro_account` and administrator settings cannot disable hard caps.
- Circuit breakers: stale data, excessive spread, news blackout, broker errors, consecutive execution failures, unprotected position, reconciliation mismatch, daily/total loss, margin stress. Latching breakers require explicit audited reset; Telegram cannot silently bypass.

### Reference pseudocode

```python
intent = persist_signal(signal)
if not data_is_fresh(intent) or not strategy_is_valid(intent):
    return reject(intent, "INVALID_SIGNAL_OR_DATA")
with atomic_account_risk_reservation(intent.account_id):
    account = broker.get_account_state()
    positions = broker.get_positions_and_pending_orders()
    symbol = broker.get_symbol_spec(intent.symbol)
    quote = broker.get_fresh_quote(intent.symbol)
    if any_missing_or_invalid(account, positions, symbol, quote):
        return reject(intent, "PREFLIGHT_UNAVAILABLE")
    volume = floor_to_broker_step(size_for_stop_risk(intent, account, symbol, quote))
    if volume < symbol.volume_min:
        return reject(intent, "MIN_LOT_EXCEEDS_RISK")
    worst_case_loss = conservative_broker_loss(intent, volume, quote, symbol)
    if not all_hard_caps_pass(account, positions, intent, worst_case_loss):
        return reject(intent, "HARD_RISK_CAP")
    reserve_risk(intent.id, worst_case_loss)
submit_once_and_reconcile(intent)  # Never blind-retry an ambiguous response
verify_broker_side_protection_or_trigger_incident(intent)
```

## 4. Signal / AI / ML corrections

1. **Scalping:** if using crossover, compute EMA9/EMA21 on two consecutive fully closed candles: bullish `prev_ema9 <= prev_ema21 and last_ema9 > last_ema21`; bearish reverse. Prevent duplicate signals for same symbol/timeframe/candle/strategy version. If using trend continuation instead, create a separately named strategy with independent entry criteria.
2. **RSI:** overbought/oversold means stretched momentum, not guaranteed reversal. Require tested confirmation such as closed-bar reversal structure and trend filter; allow HOLD when unclear. Avoid claiming an arbitrary filter is profitable.
3. **Scores vs probabilities:** name raw ensemble output `signal_score`, calibrated `p_win` only after calibration on strictly held-out chronological data. Document target event and time horizon; measure Brier score, log loss and reliability curves by market regime. Remove fixed `0.85` or other arbitrary probability claims.
4. **Kelly:** disable by default. If later enabled, require calibrated out-of-sample `p_win`, realistic net payoff distribution, minimum sample criteria, capped fractional Kelly and hard portfolio caps; otherwise fixed fractional sizing.
5. **ML leakage:** inspect `ohlcv_predictor.py`, `ensemble_agent.py`, `meta_label_engine.py`, structural extractors and macro-regime pipeline. No future candle, post-entry outcome, revised future macro release, global scaling, centered rolling window, or future-trained regime parameters in features at decision time. Split chronologically with purging/embargo appropriate to label horizons; fit preprocessing only on train folds.
6. **Webhooks:** external TradingView payloads are untrusted hints, never an authority to force BUY/SELL or confidence. Authenticate, check timestamp/replay nonce, bound numerical inputs, validate schema, rate limit, and still pass every risk gate.
7. **News:** keep source timestamp, publication timestamp, ingestion timestamp and event timezone; distinguish missing news from neutral sentiment. Never manufacture directional sentiment from technical indicators. Fail closed for required missing event data.
8. **Model governance:** version model, features, training dataset and thresholds; validate artifact checksum and schema on load; define safe fallback HOLD on errors/drift; shadow-mode before promotion and rollback capability.

## 5. Backtesting and validation

- Use only information available at each decision timestamp. If signal uses candle close, earliest simulated fill must be after that close (e.g. next tradable tick/open), never the same candle's earlier high/low.
- Model bid/ask spread, commissions, overnight fees/swap, variable slippage, contract sizing, margin, execution delays, rejected/partial orders and broker trading sessions.
- Gap beyond SL: execute at first realistically available tradable price with adverse slippage, **not** always the stop price. If TP and SL are both inside one OHLC bar and tick order is unknown, use a conservative deterministic rule or mark ambiguous; disclose assumptions. Do not assert a guaranteed worst-case loss when gaps can exceed stops.
- Account for survivorship bias, missing data, symbol rollovers, timezone/DST, exchange holidays, duplicate bars, stale news and corporate actions where applicable.
- Chronological train/validation/test with purged walk-forward evaluation, untouched final holdout, parameter search only within training folds. Report performance by symbol, regime, timeframe and period.
- Metrics: net PnL after costs, max equity drawdown, expected shortfall/tail risk, Sharpe/Sortino with appropriate frequency, profit factor, expectancy per trade, win rate, average win/loss, trade count, turnover, slippage and calibration. Include bootstrap uncertainty, stress tests and sensitivity to transaction costs.
- Run paper/demo forward test long enough to cover varied regimes and a statistically meaningful trade sample; there is no universal number of days/trades guaranteeing readiness.

## 6. Operational production hardening

- **Configuration:** explicit `ENV=dev|test|demo|live`, separate broker credentials/DB/queues per environment, `LIVE_TRADING_ENABLED=false` default, account type verification at broker, explicit two-step operator approval for live enablement. Never infer demo vs live from a display label alone.
- **Secrets:** env/secret manager; encryption at rest for user vault, secure key rotation, least privilege, never log account passwords, Telegram tokens, API keys or personally identifying trade/account data. Scan git history and dependencies.
- **Telegram/admin:** RBAC, owner/account authorization for every callback, signed/expiring actions where applicable, replay protection, rate limits, audit logs, and confirmation for start/stop/close actions. Telegram outage must not prevent SL or incident handling.
- **Reliability:** supervised processes, bounded retries with jitter for reads only, broker-aware reconciliation before order retries, queue backpressure, DB migrations/backups/restore drills, graceful shutdown, restart recovery and singleton/leader ownership of account execution.
- **Observability:** structured JSON logs with correlation IDs, metrics for stale feeds, rejected risk gates, unknown order outcomes, reconciliation mismatches, unprotected positions, broker latency, drawdown and breaker state; actionable alerts and runbooks. No sensitive values in telemetry.
- **Dependencies:** pin and audit versions; test OS/architecture compatibility of MetaTrader5 terminal/client (including Linux deployment constraints); verify broker account mode, symbols and trade permissions on target host before enabling trading.
- **Deployment:** reproducible build, CI lint/type/static analysis/tests, staging demo soak, controlled canary with smallest risk allowed by policy, explicit rollback and operator kill switch. No autonomous transition from demo to live.

## 7. Required tests (minimum acceptance suite)

| Test ID | Scenario | Expected |
|---|---|---|
| R-01 | $10 equity; broker min lot risk $4.50 | Reject `MIN_LOT_EXCEEDS_RISK`, never expand budget |
| R-02 | `macro_event_mode=True` and SL loss > budget | Reject `HARD_RISK_CAP` |
| R-03 | `custom_lot` exceeds risk or margin | Reject, no broker order |
| R-04 | Volume step 0.1 / 0.001 and min != 0.01 | Correct step rounding or reject |
| R-05 | Broker quote stale, symbol missing, PnL calc error | Reject, fail closed |
| R-06 | Two workers same signal simultaneously | At most one live intent/order, account risk reserved once |
| R-07 | Order submission timeout but broker filled | Reconcile ticket; no duplicate retry |
| R-08 | Partial fill, requote, restart mid-submit | Consistent persisted state and correct remaining risk |
| R-09 | Fill price slips and actual SL risk > cap | Trigger defined corrective action, alert, reconcile |
| R-10 | Broker SL modification rejected | Position flagged unprotected; emergency workflow; no new trades |
| R-11 | EMA9 above EMA21 for 10 bars without crossing | No repeated crossover BUY |
| R-12 | RSI > 70 in strong uptrend without reversal | No automatic SELL solely on RSI |
| R-13 | Missing calibration artifact / fixed confidence | HOLD or heuristic-only path; Kelly disabled |
| R-14 | Candle hits SL and TP / gap through SL | Conservative deterministic fills, cost-adjusted PnL |
| R-15 | Portfolio near drawdown/exposure cap | Reject across all symbols and all modes |
| R-16 | UTC rollover/restart during drawdown breaker | Breaker remains latched until authorized reset |
| R-17 | Malformed/forged/replayed TradingView webhook | Reject without affecting live orders |
| R-18 | Demo credentials used in live config or vice versa | Startup blocked |
| R-19 | Broker offline, Telegram offline | No new order; independent protection/alerts/recovery |
| R-20 | Training pipeline includes future-derived feature | Leakage detection fails CI |

Add property-based tests for lot-step arithmetic and risk invariants, concurrency tests, chaos/fault-injection tests and demo-only end-to-end tests. Never run live-money CI.

## 8. Implementation phases / checkboxes

### Phase 0 — Inventory and freeze (P0)
- [ ] Build complete call graph: Telegram/webhook -> decision -> sizing -> broker -> monitor -> ledger.
- [ ] Record baseline tests, current config defaults, symbols/account modes, open issues and reproducible failing tests.
- [ ] Disable live execution until all release gates pass.

### Phase 1 — Capital preservation (P0)
- [ ] Fix NT-01, NT-02, NT-03, NT-11 and centralize RiskPolicy.
- [ ] Enforce hard limits before submit and after fill; add R-01 to R-05, R-09, R-15, R-16.
- [ ] Implement kill switch and latched breaker with persistence.

### Phase 2 — Execution correctness (P0)
- [ ] Fix NT-04 and NT-05; durable intents, reservations, order state machine, reconciliation.
- [ ] Broker-side SL verification, partial fills, timeout handling, restart recovery.
- [ ] Add R-06 to R-10, R-19.

### Phase 3 — Signal integrity and AI (P1)
- [ ] Fix NT-06 to NT-09, inspect additional claims NT-13+.
- [ ] Calibrate probabilities, eliminate leakage, audit news/webhook signal sources.
- [ ] Add R-11 to R-13, R-17, R-20.

### Phase 4 — Backtest and model QA (P1)
- [ ] Fix NT-10 and add R-14.
- [ ] Produce reproducible cost-aware walk-forward report, stress results and uncertainty intervals.
- [ ] Freeze strategy version and parameter provenance.

### Phase 5 — Demo staging (P0 release gate)
- [ ] Run full tests/CI; demo integration with verified broker account type.
- [ ] Exercise disconnect, restart, gaps, high spread, news spike, missing quotes, concurrent orders, rejected SL and kill switch.
- [ ] Record zero **unexplained** duplicate orders, zero known risk-gate bypasses, reconciled orders and correct alerts; investigate every exception.
- [ ] Independent reviewer signs off architecture, tests, risk policy and operations runbook.

### Phase 6 — Conditional real-money rollout (human-controlled)
- [ ] Operator explicitly accepts financial risk and broker/regulatory constraints.
- [ ] Verify account is live, currency, leverage, margin rules and trade permissions; enforce approved risk caps.
- [ ] Enable live only with deliberate authorization, small initial exposure, continuous monitoring and instant kill/rollback procedures.
- [ ] Suspend automatically upon any protection/reconciliation discrepancy; revert to demo when uncertain.

## 9. Definition of done and expected deliverables

A change is **not done** merely because Python compiles or tests pass in mocks. Provide:

1. Code changes with filenames and reviewable commits/PR.
2. `AUDIT_FINDINGS.md`: each NT issue, exact original behavior, severity, reproduction, remediation and residual risk.
3. `RISK_POLICY.md`: formal invariants, units, caps, precedence, timezone/reset semantics and override policy.
4. `ORDER_LIFECYCLE.md`: state diagram, failure modes, broker reconciliation and emergency procedures.
5. `BACKTEST_VALIDATION.md`: datasets, chronological splits, costs, metrics, assumptions and limitations.
6. `PRODUCTION_RUNBOOK.md`: demo/live separation, setup, monitoring, incident response, kill switch, recovery and rollback.
7. Automated regression/integration/chaos tests with CI results and broker demo evidence.
8. `RELEASE_CHECKLIST.md` with named approver and explicit **NO-GO** status until every P0 gate is verified.

### Final release decision

**NO-GO for real accounts** if *any* of the following remains: risk cap bypass; unbounded lot sizing; stale/unknown broker data allowed to place orders; duplicate/ambiguous order retries; unprotected positions without emergency handling; missing durable drawdown breaker; no restart reconciliation; secrets exposure; no broker demo integration evidence; no human-approved live enablement.

**GO for demo** only after critical safety controls and demo-specific operational checks pass. **GO for live** requires all P0 controls, independent review, broker-specific validation and explicit human authorization. Passing these gates reduces engineering risk but does **not** guarantee profits or prevent losses from market gaps, broker outages or extreme volatility.
