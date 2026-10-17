# PRODUCTION_RUNBOOK.md — NexusTrade Operations & Incident Response Runbook

**Environment Target:** Production & Staging MetaTrader 5 Deployments  
**Audience:** System Administrators, DevOps Engineers, Quant Traders

---

## 1. Environment Isolation (Dev vs Demo vs Live)

| Setting | Development (`dev`) | Staging Demo (`demo`) | Production (`live`) |
|:---|:---|:---|:---|
| `ENVIRONMENT` | `development` | `staging` | `production` |
| `LIVE_TRADING_ENABLED` | `false` | `false` | `false` (Requires two-step approval) |
| Broker Server | `Exness-MT5Trial7` (Demo) | `Exness-MT5Trial7` (Demo) | `Exness-MT5Real` |
| Database Path | `data/dev_vault.db` | `data/demo_vault.db` | `data/live_vault.db` (Encrypted) |
| Telegram Bot | Test Bot Token | Staging Bot Token | Production Bot Token |

---

## 2. Emergency Kill Switch & Panic Close Procedures

### A. Telegram Bot Emergency Kill Switch
1. Open authorized Telegram Admin Chat.
2. Send command:
   ```text
   /panic
   ```
3. **Automated Actions Executed Immediately:**
   - Closes 100% of open market positions at current bid/ask.
   - Cancels 100% of active pending orders (`BUY_STOP`, `SELL_STOP`).
   - Halts all autonomous sessions on the account.
   - Emits an audited panic close report card.

### B. Command-Line / Terminal Kill Switch
If the Telegram Bot API is unreachable or Telegram experiences an outage:
```bash
# Emergency hard kill: stop the bot service process
pkill -f "python main.py"

# Or activate persistent kill switch via SQLite CLI:
sqlite3 data/processed/user_vault.db "UPDATE account_risk_state SET kill_switch_active = 1;"
```

---

## 3. Circuit Breaker Recovery Runbook

When daily drawdown reaches 4.0% or total high-water drawdown reaches 8.0%, the circuit breaker is **persistently latched** in SQLite (`circuit_breaker_latched = 1`). Restarting the bot **DOES NOT** unlatch the breaker.

### Recovery Steps:
1. Conduct post-mortem review of closed trade losses and current account balance.
2. Confirm market conditions have stabilized.
3. Perform audited manual reset:
   ```text
   /reset_breaker
   ```
   Or via UserVault Python CLI:
   ```python
   from ai_engine.auto_trader.user_vault import UserVault
   vault = UserVault()
   vault.reset_circuit_breaker(account_id="exness_mt5:EXNESS-MT5REAL:...")
   ```

---

## 4. Unprotected Position Incident Runbook (`INCIDENT_UNPROTECTED`)

If an order was filled on the broker but the broker rejected the Stop-Loss:
1. Status is marked `FILLED_UNPROTECTED`.
2. Gate 5 locks out all new trades on the account.
3. Open Exness MT5 Terminal directly on phone or desktop.
4. Manually close the position or attach a manual Stop Loss ticket in MT5.
5. Re-sync positions in bot:
   ```text
   /myaccount
   ```
   The bot reconciles the manual MT5 closure and resumes normal state.
