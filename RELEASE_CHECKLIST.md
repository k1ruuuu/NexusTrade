# RELEASE_CHECKLIST.md — NexusTrade Production Release Gate Sign-Off

**Target Branch:** `feat/production-remediation-gate`  
**Governing Plan:** `NEXUSTRADE_PRODUCTION_REMEDIATION_PLAN.md`  
**Current Release Status:** **GO FOR DEMO** | **NO-GO FOR REAL-MONEY LIVE** (Awaiting Human Operator Sign-Off)

---

## 1. Phase Gate Verification Matrix

| Gate | Category | Description | Status | Evidence |
|:---:|:---|:---|:---:|:---|
| **G-01** | **Risk Policy** | Centralized `RiskPolicy` module enforces hard caps | **VERIFIED** | `ai_engine/risk_management/risk_policy.py` |
| **G-02** | **NT-01** | Micro-account budget expansion eliminated | **VERIFIED** | `tests/test_production_remediation.py::test_r01` |
| **G-03** | **NT-02** | Macro event mode non-bypassable risk check | **VERIFIED** | `tests/test_production_remediation.py::test_r02` |
| **G-04** | **NT-03** | Volume step flooring & multi-currency conversion | **VERIFIED** | `tests/test_production_remediation.py::test_r04` |
| **G-05** | **NT-04** | Durable SQLite order intents & risk reservations | **VERIFIED** | `tests/test_production_remediation.py::test_r06, r08` |
| **G-06** | **NT-05** | Unprotected position detection & broker reconciliation | **VERIFIED** | `tests/test_production_remediation.py::test_r07, r10` |
| **G-07** | **NT-06** | Closed-candle crossover requirement (2 bars) | **VERIFIED** | `tests/test_production_remediation.py::test_r11` |
| **G-08** | **NT-07** | Removal of fixed 0.85 confidence claims | **VERIFIED** | `ai_engine/auto_trader/engine.py` |
| **G-09** | **NT-08** | RSI 70/30 closed-candle reversal confirmation | **VERIFIED** | `tests/test_production_remediation.py::test_r12` |
| **G-10** | **NT-09** | Kelly Criterion disabled by default | **VERIFIED** | `tests/test_production_remediation.py::test_r13` |
| **G-11** | **NT-10** | Backtest gap slippage & intrabar ambiguity | **VERIFIED** | `tests/test_production_remediation.py::test_r14` |
| **G-12** | **NT-11** | Account-wide aggregate risk & exposure caps | **VERIFIED** | `tests/test_production_remediation.py::test_r15` |
| **G-13** | **Circuit Breaker** | Persistent SQLite latched breaker across restart | **VERIFIED** | `tests/test_production_remediation.py::test_r16` |
| **G-14** | **Safety Freeze** | `LIVE_TRADING_ENABLED=false` default enforced | **VERIFIED** | `tests/test_production_remediation.py::test_r18` |
| **G-15** | **Test Suite** | 100% Green on all acceptance tests (R-01 to R-20) | **VERIFIED** | 20 passed in pytest |

---

## 2. Final Release Decision

- **DEMO DEPLOYMENT:** **APPROVED (GO)**  
  All critical capital preservation controls, preflight risk checks, and execution state machines are verified in code and tests.
- **LIVE REAL-MONEY DEPLOYMENT:** **HOLD (NO-GO)**  
  Requires:
  1. Human operator explicit written sign-off of financial risk.
  2. Setting `LIVE_TRADING_ENABLED=true` in `.env` after operator review.
  3. Minimum 14-day soak test on demo account without any unhandled exceptions.

**Authorized Lead Sign-Off:**  
*Lead Quantitative Systems Engineer & Security Architect*  
*NexusTrade Engineering Core*
