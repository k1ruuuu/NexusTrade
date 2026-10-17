# ORDER_LIFECYCLE.md — NexusTrade Order State Machine & Reconciliation

**Specification:** Persisted Order Intent Lifecycle  
**Database Table:** `order_lifecycle_intents` & `risk_reservations` in `user_vault.db`

---

## 1. Lifecycle State Machine Diagram

```
[SIGNAL_CREATED]
       │
       ▼
[DATA_VALIDATED] (Quotes fresh, spread within limits)
       │
       ▼
[STRATEGY_APPROVED] (Technical / MTF / RSI confirmed)
       │
       ▼
[RISK_RESERVED] (Risk allocation locked in SQLite risk_reservations)
       │
       ▼
[PREFLIGHT_PASSED] (RiskPolicy invariants verified)
       │
       ▼
  [SUBMITTING] (Dispatched to MT5 broker API with throttle delay)
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
   [FILLED]                     [TRANSPORT_TIMEOUT]                   [REJECTED]
       │                                 │                                 │
       │                                 ▼                                 ▼
       │                    [RECONCILIATION_ATTEMPT]             (Reservation Released)
       │                                 │
       │                   ┌─────────────┴─────────────┐
       │                   ▼                           ▼
       │            (Ticket Found)             (No Ticket Found)
       │                   │                           │
       │                   ▼                           ▼
       │             [RECONCILED]             [MT5_REJECTED]
       │                   │                           │
       ▼                   ▼                           ▼
[PROTECTION_VERIFIED] ◄────┴───────────────────► (Released)
       │
       ▼ (If SL missing on broker)
[INCIDENT_UNPROTECTED] ──► (Emergency Close Attempted)
       │
       ▼
  [MONITORING] (Trailing stop ratcheting + Basket monitoring)
       │
       ▼
   [CLOSED] (TP, SL, or Panic Close executed)
       │
       ▼
 [RECONCILED] (Audited PnL locked in SQLite user_positions & daily_risk)
```

---

## 2. State Descriptions & Transition Invariants

| State | Description | Invariants & Transition Rules |
|:---|:---|:---|
| `SIGNAL_CREATED` | Strategy or webhook emits actionable trading bias | Valid symbol and timeframe verified. |
| `DATA_VALIDATED` | Fresh tick quote and OHLCV received | Stale data (>60s) rejects immediately. |
| `STRATEGY_APPROVED` | All strategy rules confirmed | Strict 2-bar crossover, RSI confirmation. |
| `RISK_RESERVED` | Pre-allocation of risk dollars in SQLite | Prevents concurrent race condition double-allocation. |
| `PREFLIGHT_PASSED` | `RiskPolicy` approved trade | Volume step floored, margin verified. |
| `SUBMITTING` | Outbound request to broker active | 500ms–1500ms throttle delay enforced. |
| `FILLED` | Broker returned success `retcode == 10009` | Order ticket logged in intent ledger. |
| `PROTECTION_VERIFIED`| Broker-side SL confirmed active on MT5 | Guarantees position will not run unprotected. |
| `INCIDENT_UNPROTECTED`| Order filled but broker SL failed | Trading on account halted immediately. |
| `MONITORING` | Position active in session manager loop | Trailing stop ratcheting subject to Step Filter. |
| `CLOSED` | Trade closed by TP, SL, or Panic Close | Exit price and net PnL recorded. |
| `RECONCILED` | Final PnL credited to daily risk ledger | Full round-trip audit complete. |

---

## 3. Failure Modes & Emergency Handling Procedures

### Failure Mode 1: Transport Timeout During Submit
- **Problem:** The HTTP request to the broker MT5 REST server timed out while the broker may have already filled the order.
- **Handling:** **DO NOT RETRY BLINDLY.**
- The engine calls `mt5_client.get_live_open_orders()`. If an order matching the requested symbol, side, and timestamp within 60s is found, it transitions to `RECONCILED_FILLED`.
- If no ticket is found after multiple reconciliation checks, it transitions to `REJECTED` and releases the risk reservation.

### Failure Mode 2: Broker Rejects Stop-Loss Modification
- **Problem:** The broker accepted the market fill but rejected the subsequent SL modification (`ERR_INVALID_STOPS`).
- **Handling:** Position is immediately flagged as `FILLED_UNPROTECTED`.
- An immediate emergency close is attempted via `_close_position_on_broker`.
- If emergency close fails, Gate 5 enters `INCIDENT_UNPROTECTED` state, halting all new order placement on the account until human intervention resolves the position.

### Failure Mode 3: Disconnect or Terminal Crash
- **Problem:** Bot process restarts while positions are open.
- **Handling:** On startup, `sync_live_mt5_positions()` queries the terminal, re-populates open tickets into `UserVault`, and verifies that all active positions have broker-side stops intact.
