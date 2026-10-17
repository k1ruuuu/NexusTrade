# RISK_POLICY.md — NexusTrade Capital Preservation & Risk Invariants

**Authority:** Centralized Risk Governance Engine  
**Module:** `ai_engine/risk_management/risk_policy.py`  
**Compliance Standard:** Institutional Quantitative Risk Management

---

## 1. Core Invariants (Non-Bypassable)

The NexusTrade risk engine enforces strict capital preservation principles. No trading strategy, Telegram inline command, webhook payload, macro news event, or micro-account parameter can bypass these rules.

| Rule ID | Invariant | Formal Specification | Violation Action |
|:---:|:---|:---|:---|
| **RP-01** | **Single Trade Risk Cap** | $\text{Loss}_{\text{SL}} \le \text{Current Equity} \times \text{max\_risk\_per\_trade\_pct}$ | Reject `HARD_RISK_CAP` / `MIN_LOT_EXCEEDS_RISK` |
| **RP-02** | **No Micro-Account Expansion** | $\text{Trade Budget} \le \text{Equity} \times 1.5\%$ (No 45% or $4.50 floor) | Reject `MIN_LOT_EXCEEDS_RISK` |
| **RP-03** | **Daily Drawdown Breaker** | $\text{Loss}_{\text{day}} \ge \text{StartOfDay Equity} \times 4.0\%$ | Latch Circuit Breaker `DAILY_DRAWDOWN_EXCEEDED` |
| **RP-04** | **High-Water Drawdown Breaker** | $\text{Loss}_{\text{HW}} \ge \text{HighWater Equity} \times 8.0\%$ | Latch Circuit Breaker `TOTAL_DRAWDOWN_EXCEEDED` |
| **RP-05** | **Portfolio Aggregate Cap** | $\sum \text{Open Risk} + \text{Trade Risk} \le \text{Equity} \times 6.0\%$ | Reject `PORTFOLIO_RISK_CAP_EXCEEDED` |
| **RP-06** | **Broker Volume Step Flooring** | $\text{Lot} = \text{volume\_min} + \lfloor \frac{\text{raw} - \text{min}}{\text{step}} \rfloor \times \text{step}$ | Rejection if $\text{Lot} < \text{volume\_min}$ |
| **RP-07** | **Stops Level Compliance** | $|\text{Entry} - \text{SL}| \ge \text{stops\_level} \times \text{point}$ | Reject `INVALID_STOPS_LEVEL` |
| **RP-08** | **Spread Filter** | $\text{Spread} \le \text{max\_spread\_points}$ (Default 40 pt) | Reject `EXCESSIVE_SPREAD` |
| **RP-09** | **Safety Freeze Default** | $\text{LIVE\_TRADING\_ENABLED} = \text{False}$ by default | Reject `LIVE_TRADING_DISABLED` on Live |
| **RP-10** | **Incident Halting** | Any open position with `FILLED_UNPROTECTED` | Halt trading `INCIDENT_UNPROTECTED` |

---

## 2. Mathematical Formulations

### A. Volume Step Flooring Formula (Section 3.A)
$$\text{Normalized Lot} = \max\left(\text{volume\_min}, \min\left(\text{volume\_max}, \text{volume\_min} + \left\lfloor \frac{\text{raw\_lot} - \text{volume\_min}}{\text{volume\_step}} \right\rfloor \times \text{volume\_step}\right)\right)$$

### B. Stop-Loss Monetary Loss Calculation
For asset with contract size $C$, entry price $P_{\text{entry}}$, stop loss price $P_{\text{SL}}$, and lot volume $L$:
$$\text{Loss}_{\text{USD}} = |P_{\text{entry}} - P_{\text{SL}}| \times L \times C \times \text{FX\_Conversion}$$

### C. Daily Drawdown Reset Semantics
- Start of Day Equity ($\text{SOD}$) is captured at `00:00:00 UTC`.
- If UTC date rolls over, a fresh daily baseline is established.
- **Persistence Guarantee:** If the circuit breaker is latched, a UTC date rollover or engine restart **DOES NOT** unlatch the breaker. An audited manual reset (`/reset_breaker` or `vault.reset_circuit_breaker`) is required.

---

## 3. Order Execution Preflight State Machine

```
Preflight Check Triggered
   │
   ├─► [1] Master Kill Switch Active? ──► YES ──► REJECT (KILL_SWITCH_ACTIVE)
   │
   ├─► [2] Live Account without Flag? ──► YES ──► REJECT (LIVE_TRADING_DISABLED)
   │
   ├─► [3] Circuit Breaker Latched? ───► YES ──► REJECT (CIRCUIT_BREAKER_LATCHED)
   │
   ├─► [4] Unprotected Position Open? ─► YES ──► REJECT (INCIDENT_UNPROTECTED)
   │
   ├─► [5] Spread > max_spread_points? ─► YES ──► REJECT (EXCESSIVE_SPREAD)
   │
   ├─► [6] SL violates stops_level? ──► YES ──► REJECT (INVALID_STOPS_LEVEL)
   │
   ├─► [7] Min lot loss > budget? ─────► YES ──► REJECT (MIN_LOT_EXCEEDS_RISK)
   │
   ├─► [8] Custom lot loss > budget? ──► YES ──► REJECT (CUSTOM_LOT_EXCEEDS_RISK)
   │
   ├─► [9] Total open risk > 6% cap? ──► YES ──► REJECT (PORTFOLIO_RISK_CAP_EXCEEDED)
   │
   └─► ALL CHECKS PASSED ──────────────► APPROVE PREFLIGHT (Allowed Lots Allocated)
```
