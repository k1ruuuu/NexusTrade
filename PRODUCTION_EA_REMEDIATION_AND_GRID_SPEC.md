# 🛠️ NEXUSTRADE AI — PRODUCTION AUDIT REMEDIATION & MULTI-ASSET GRID EA SPECIFICATION

**Target Repository:** `NexusTrade-feat-nexustrade-ai-core`  
**Objective:**  
1. Remediate all **9 critical bugs and flow logic flaws** identified across `engine.py`, `risk_policy.py`, `mt5_exness_client.py`, `ensemble_agent.py`, `sentiment_analyzer.py`, `position_monitor.py`, `session_manager.py`, and `user_vault.py`.  
2. Implement the **High-Frequency M1 Pending Stop Grid Scalper (`EA_GRID_SCALPER`)** matching the exact execution mechanics of the reference MT5 EA video (`BUY_STOP` / `SELL_STOP` layering, dynamic equity lot compounding, multi-asset pip/point normalization for Gold, Crypto, Indices, and Forex, plus mass order/position cleanup).  
3. Ensure **100% production readiness** for both **Real (Live)** and **Demo (Trial)** MT5 accounts.

---

## PART 1: REMEDIATION OF 9 CRITICAL FLOW LOGIC BUGS

### Fix 1: Resolve `NameError` in `EnsembleDecisionAgent._score_timeframe_trend`
* **File:** `ai_engine/models/ensemble_agent.py`
* **Root Cause:** `mom_score` and `flow_score` are referenced in `score = float(np.clip(...))` without being defined, causing every call to `_score_timeframe_trend()` to throw a `NameError`, get caught by `except Exception:`, and silently return `0.0, 0.0, "🟡 FLAT"`.
* **Required Change:** Define `mom_score` and `flow_score` before computing `score` inside `_score_timeframe_trend()`:

```python
            rsi14 = float(last["rsi_14"])
            macd_h = float(last["macd_hist"])
            atr14 = float(last["atr_14"])
            cvd20 = float(last.get("cvd_norm_20", 0.0))

            # Define momentum and order-flow scores before blending
            rsi_comp = np.tanh((rsi14 - 50.0) / 15.0)
            macd_comp = np.sign(macd_h) * min(1.0, abs(macd_h) / max(atr14 * 0.25, 1e-6))
            mom_score = float(np.clip(0.60 * rsi_comp + 0.40 * macd_comp, -1.0, 1.0))
            flow_score = float(np.clip(cvd20 * 4.0, -1.0, 1.0))
```

---

### Fix 2: Fix `s_sent` Sentiment Overwrite Bug
* **File:** `ai_engine/models/sentiment_analyzer.py`
* **Root Cause:** In `analyze_symbol()`, `s_sent` is computed using `0.65 * composite_sentiment + 0.35 * fb_mean` inside `if live_used and live_items:`, but immediately overwritten after the `if/else` block by `s_sent = float(np.clip(weighted_polarity, -1.0, 1.0))`.
* **Required Change:** Preserve the blended `s_sent` when `composite_sentiment` is present:

```python
        total_weight = sum(item.weight for item in scored_items) or 1.0
        weighted_polarity = sum(item.polarity * item.weight for item in scored_items) / total_weight
        if live_used and composite_sentiment is not None and len(scored_items) > 0:
            blended_sent = 0.65 * float(composite_sentiment) + 0.35 * weighted_polarity
            s_sent = float(np.clip(blended_sent, -1.0, 1.0))
        else:
            s_sent = float(np.clip(weighted_polarity, -1.0, 1.0))
```

---

### Fix 3: Fix Timeout Reconciliation Crash & `notional_usdt` Calculation in `MT5ExnessClient`
* **File:** `ai_engine/auto_trader/mt5_exness_client.py`
* **Root Causes:**
  1. In `place_market_order()`, `est_notional`, the retry block, and the timeout reconciliation block compute `notional_usdt` as `round(quantity * open_px, 2)` without multiplying by `get_mt5_contract_size(symbol)`. Because `engine.py` passes `quantity=candidate_lots`, `notional_usdt` is understated by `100x` on Gold (`XAUUSD`) and `100,000x` on Forex.
  2. In the `except Exception` timeout reconciliation block, `live_orders` can be `None` (raising `TypeError`), and `mt5.mtapi.io/OpenedOrders` returns `orderType` (or `type`) and `openPrice`, NOT `side` or `price_open`.
* **Required Change:**
  1. Calculate `est_notional = round(lots * get_mt5_contract_size(symbol) * reference_price, 2)`.
  2. In the retry success return block, set:
     ```python
     notional_usdt=round(exec_lots * get_mt5_contract_size(candidate_sym) * open_px, 2),
     ```
  3. Replace the timeout reconciliation block in `place_market_order()` with:
     ```python
            try:
                live_orders = await self.get_live_open_orders(
                    account_id_or_login, token_or_password, server_name
                )
                for cand in (live_orders or []):
                    if not isinstance(cand, dict):
                        continue
                    cand_sym = str(cand.get("symbol") or "").strip()
                    raw_type = str(cand.get("orderType") or cand.get("type") or cand.get("side") or "").upper()
                    cand_side = "BUY" if ("BUY" in raw_type or raw_type == "0") else ("SELL" if ("SELL" in raw_type or raw_type == "1") else "")
                    if cand_sym in (mt5_symbol_m, mt5_symbol_raw) and cand_side == side:
                        ticket_str = str(cand.get("ticket") or cand.get("order") or "")
                        open_px = float(cand.get("openPrice") or cand.get("price_open") or reference_price)
                        exec_lots = float(cand.get("lots") or lots)
                        if ticket_str:
                            logger.info("Reconciled filled order after transport timeout", ticket=ticket_str)
                            return BinanceOrderResult(
                                executed=True,
                                order_id=ticket_str,
                                symbol=cand_sym or mt5_symbol_m,
                                side=side,
                                executed_qty=exec_lots,
                                avg_price=open_px,
                                notional_usdt=round(exec_lots * get_mt5_contract_size(cand_sym or symbol) * open_px, 2),
                                testnet=testnet,
                                status=f"RECONCILED_FILLED (Ticket #{ticket_str})",
                            )
            except Exception as rec_err:
                logger.debug("Reconciliation query after timeout failed", error=str(rec_err))
     ```

---

### Fix 4: Resolve Micro-Account / EA Mode Rejection Deadlock in `RiskPolicy` & `SmartAutoTraderEngine`
* **Files:** `ai_engine/risk_management/risk_policy.py` & `ai_engine/auto_trader/engine.py`
* **Root Cause:** Gate 1 allows accounts with `$10.00+` USD, but Gate 6 (`RiskPolicy.evaluate_preflight`) strictly enforces `trade_budget = eq * max_risk_per_trade_pct` (1% of $10 = $0.10). On `XAUUSD`, `0.01` lot with a 20–50 pip SL risks `$2.00–$5.00`, causing 100% of accounts under `$200–$500` to fail with `MIN_LOT_EXCEEDS_RISK`. Furthermore, `decision.position_units` is left as `0.0` in `EA_SCALPING` and `EA_EXPERT_ADVISOR` modes.
* **Required Change:**
  1. In `RiskPolicy.__init__()` (`ai_engine/risk_management/risk_policy.py`), add `allow_micro_min_lot_override: bool = True` and `micro_account_ceiling_usd: float = 250.0`, `micro_max_lot_risk_fraction: float = 0.50`.
  2. In `RiskPolicy.evaluate_preflight()`, when `eq < self.micro_account_ceiling_usd` and `self.allow_micro_min_lot_override` is `True`:
     * If `floored_lots < spec.volume_min`, compute `min_lot_loss = self.calculate_worst_case_loss(spec, side, entry_price, stop_loss, spec.volume_min)`.
     * If `min_lot_loss <= min(eq * self.micro_max_lot_risk_fraction, daily_headroom, total_headroom)`:
       approve `floored_lots = spec.volume_min` and set `effective_max_risk = max(effective_max_risk, min_lot_loss)`.
  3. In `engine.py`, when constructing `SignalDecision` for `EA_EXPERT_ADVISOR`, `EA_SCALPING`, or `EA_GRID_SCALPER`, set `position_units = 0.01 * get_mt5_contract_size(clean_sym)` so `decision.position_units` is never `0.0`.

---

### Fix 5: Fix Session Dictionary Cleanup Memory Leak & Duplicate Symbol Sessions
* **File:** `ai_engine/auto_trader/session_manager.py`
* **Root Causes:**
  1. `_session_loop()` calls `self._sessions.pop((sess.user_id, sess.symbol), None)`, whereas `start_timed_session()` keys `self._sessions` by `sess.session_id` (`str`). Completed/cancelled sessions are never removed from memory.
  2. Calling `start_timed_session()` multiple times for the same `(user_id, symbol)` spawns duplicate concurrent background tasks.
* **Required Change:**
  1. In `start_timed_session()`, before creating a new session, stop any existing running session for `(uid, clean_sym)`:
     ```python
     for existing_sess in self.get_user_sessions(uid):
         if existing_sess.symbol == clean_sym:
             self.stop_session_by_id(uid, existing_sess.session_id)
     ```
  2. In `_session_loop()`, replace all 3 occurrences of `self._sessions.pop((sess.user_id, sess.symbol), None)` with:
     ```python
     self._sessions.pop(sess.session_id, None)
     ```

---

### Fix 6: Fix BUY & SELL Trailing Stop Ratcheting in `PositionMonitorMixin`
* **File:** `ai_engine/auto_trader/position_monitor.py`
* **Root Causes:**
  1. For `BUY` positions, `self._apply_trailing_stop_update()` is only called inside `if bar_high >= pos.entry_price + 0.60 * est_atr:`. If `candidate_sl` was ratcheted by `locked_60` (70% of TP1) or moderate adverse news (`-0.30 < news_s_sent <= -0.18`), it is ignored unless `bar_high >= entry + 0.60 * est_atr`.
  2. For `SELL` positions, if `pos.stop_loss <= 0.0` (e.g., `FILLED_UNPROTECTED`), `candidate_sl = pos.stop_loss` starts at `0.0` and `min(candidate_sl, ...)` keeps it at `0.0`, so `if candidate_sl < pos.stop_loss - 1e-6:` is always `False`.
* **Required Change:**
  * For **BUY**:
    ```python
            if bar_high >= pos.entry_price + 0.60 * est_atr:
                be_plus = round(pos.entry_price + 0.20 * est_atr, 4)
                candidate_sl = max(candidate_sl, be_plus)
                trailed = self.engine.ensemble_agent.stop_loss_manager.update_trailing_stop(
                    "BUY", bar_high, candidate_sl, trail_dist
                )
                candidate_sl = max(candidate_sl, trailed)
            if candidate_sl > pos.stop_loss + 1e-6:
                return await self._apply_trailing_stop_update(
                    pos, candidate_sl, curr_price, unr_pnl, unr_pct
                )
    ```
  * For **SELL**:
    ```python
        candidate_sl = pos.stop_loss if pos.stop_loss > 0.0 else round(pos.entry_price + initial_stop_dist, 4)
        # ... [existing news & tp1_span_s checks using min(candidate_sl, ...)] ...
        if bar_low <= pos.entry_price - 0.60 * est_atr:
            be_plus = round(pos.entry_price - 0.20 * est_atr, 4)
            candidate_sl = min(candidate_sl, be_plus)
            trailed = self.engine.ensemble_agent.stop_loss_manager.update_trailing_stop(
                "SELL", bar_low, candidate_sl, trail_dist
            )
            candidate_sl = min(candidate_sl, trailed)
        if pos.stop_loss <= 0.0 or candidate_sl < pos.stop_loss - 1e-6:
            return await self._apply_trailing_stop_update(
                pos, candidate_sl, curr_price, unr_pnl, unr_pct
            )
    ```

---

### Fix 7: Separate Realized Balance vs Floating Equity in `UserVault`
* **Files:** `ai_engine/auto_trader/mt5_symbols.py`, `ai_engine/auto_trader/mt5_exness_client.py`, `ai_engine/auto_trader/user_vault.py`
* **Root Cause:** `verify_credentials()` sets `total_usdt = round(equity, 2)` and discards `balance`. When `refresh_live_exness_balance()` updates SQLite, `user_daily_risk.start_equity` and `close_position()` use floating equity as the base balance, distorting daily drawdown calculations when positions are open.
* **Required Change:**
  1. Add `balance_usdt: float = 0.0` to `BinanceAccountVerification` in `mt5_symbols.py`.
  2. In `MT5ExnessClient.verify_credentials()`, populate `balance_usdt=round(balance, 2)` and `total_usdt=round(equity, 2)`.
  3. In `UserVault.get_account_risk_snapshot()`, accept `live_balance: float` and `live_equity: Optional[float] = None`, setting `balance=eff_bal`, `equity=eff_eq`, and anchoring `start_of_day_equity` to `eff_bal` on daily UTC rollover.

---

### Fix 8: Enforce `RiskPolicy` Preflight Inside `_open_news_reversal_position`
* **File:** `ai_engine/auto_trader/position_monitor.py`
* **Root Cause:** `_open_news_reversal_position()` bypasses `RiskPolicy.evaluate_preflight()` and `account_risk_state.kill_switch_active` / `circuit_breaker_latched`, placing a reversal order with `closed_pos.quantity` directly.
* **Required Change:** Before calling `place_market_order` in `_open_news_reversal_position()`, query `acct_snapshot = self.engine.vault.get_account_risk_snapshot(...)` and run `policy.evaluate_preflight(...)`. Abort reversal if `not preflight.approved`, and use `preflight.allowed_lots` as the reversal lot size.

---

### Fix 9: Accurate Symbol Digits & Point Specification in `engine.py`
* **File:** `ai_engine/auto_trader/engine.py` & `ai_engine/auto_trader/mt5_symbols.py`
* **Root Cause:** `engine.py` hardcodes `digits = 2 if decision.entry_price >= 10.0 else (4 if decision.entry_price >= 0.1 else 5)`. For `USDJPY` (price ~155.0), `digits` becomes `2` (`point = 0.01`), which is 1 pip instead of 1 point (`0.001`), distorting `stops_level` validation in `RiskPolicy`.
* **Required Change:** Add a helper `get_mt5_digits_and_point(symbol: str, price: float) -> tuple[int, float]` in `mt5_symbols.py`:
  * `XAUUSD` / `XAGUSD` / `USOIL` / `BTCUSD` / `ETHUSD` / `US30` / `US500` / `USTEC`: `digits = 2, point = 0.01` (`XAGUSD` `digits = 3, point = 0.001`)
  * 6-letter FX ending in `JPY`: `digits = 3, point = 0.001`
  * Standard 6-letter FX (`EURUSD`, `GBPUSD`, etc.): `digits = 5, point = 0.00001`

---

## PART 2: NEW FEATURE SPECIFICATION — HIGH-FREQUENCY M1 PENDING STOP GRID EA (`EA_GRID_SCALPER`)

To replicate the exact behavior shown in the reference MT5 EA video (**$10 initial deposit, M1 timeframe, bidirectional `BUY_STOP` & `SELL_STOP` trap grid, dynamic auto-lot compounding up to 2.00+ lots, and mass order/position operations**), implement the following architecture across **all MT5-supported symbols** (Gold `XAUUSD`, Crypto `BTCUSD`/`ETHUSD`, Indices `US30`/`USTEC`/`US500`, and Forex):

### 1. Multi-Asset Grid Scaling & Dynamic Compounding (`ai_engine/auto_trader/mt5_symbols.py`)
Add `GridScalperConfig` and helper functions to `mt5_symbols.py`:

```python
@dataclass
class GridScalperConfig:
    """Configuration for High-Frequency M1 Pending Stop Grid EA (Video Mode)."""
    layers_per_side: int = 5              # 5 Buy Stops + 5 Sell Stops per cycle
    first_offset_pips: float = 10.0       # Distance from current Bid/Ask to Layer 1 (e.g. $1.00 on Gold)
    grid_step_pips: float = 10.0          # Distance between consecutive stop layers ($1.00 on Gold)
    take_profit_pips: float = 15.0        # Tight scalping TP per pending order ($1.50 on Gold)
    stop_loss_pips: float = 25.0          # Protective SL per pending order ($2.50 on Gold)
    base_lot: float = 0.01
    max_lot: float = 2.00                 # Video caps at 2.00 lots per order
    compounding_enabled: bool = True
    equity_per_001_lot: float = 15.0      # Dynamic compounding: $10->$0.01, $45->0.03, $120->0.08, $1500->1.00, $3000+->2.00
    target_equity_usd: float = 5000.0     # Auto-stop & close all when equity target is reached
    max_equity_drawdown_pct: float = 0.35 # Emergency basket equity stop


def calculate_compounded_grid_lot(
    equity_usd: float,
    cfg: GridScalperConfig,
    volume_min: float = 0.01,
    volume_step: float = 0.01,
) -> float:
    """
    Replicate the video's aggressive equity-based lot compounding progression:
    - Equity $10 - $29     -> 0.01 lot
    - Equity $30 - $60     -> 0.03 lot
    - Equity $120 - $250   -> 0.08 lot
    - Equity $1,000+       -> 1.00 lot
    - Equity $3,000+       -> 2.00 lot
    """
    if not cfg.compounding_enabled or equity_usd <= 20.0:
        return cfg.base_lot
    raw_lot = (equity_usd / max(cfg.equity_per_001_lot, 1.0)) * 0.01
    stepped = math.floor((raw_lot + 1e-9) / volume_step) * volume_step
    return round(max(volume_min, min(stepped, cfg.max_lot)), 2)


def build_pending_stop_grid_levels(
    symbol: str,
    bid_price: float,
    ask_price: float,
    cfg: GridScalperConfig,
) -> list[dict[str, Any]]:
    """
    Generate symmetric BUY_STOP (above Ask) and SELL_STOP (below Bid) trap levels
    normalized for any MT5 instrument (XAUUSD, BTCUSD, US30, EURUSD, etc.).
    """
    pip_sz = get_mt5_pip_size(symbol, ask_price)
    _, point_sz = get_mt5_digits_and_point(symbol, ask_price)
    decimals = 2 if point_sz >= 0.01 else (3 if point_sz >= 0.001 else 5)

    first_dist = max(cfg.first_offset_pips * pip_sz, 3.0 * pip_sz)
    step_dist = max(cfg.grid_step_pips * pip_sz, 2.0 * pip_sz)
    tp_dist = cfg.take_profit_pips * pip_sz
    sl_dist = cfg.stop_loss_pips * pip_sz

    orders: list[dict[str, Any]] = []
    for i in range(cfg.layers_per_side):
        # BUY_STOP above current Ask
        buy_entry = round(ask_price + first_dist + (i * step_dist), decimals)
        buy_tp = round(buy_entry + tp_dist, decimals) if tp_dist > 0 else 0.0
        buy_sl = round(buy_entry - sl_dist, decimals) if sl_dist > 0 else 0.0
        orders.append({
            "side": "BUY_STOP",
            "op_code": 4,  # MT5 OP_BUYSTOP / ORDER_TYPE_BUY_STOP
            "layer": i + 1,
            "price": buy_entry,
            "stop_loss": buy_sl,
            "take_profit": buy_tp,
        })

        # SELL_STOP below current Bid
        sell_entry = round(bid_price - first_dist - (i * step_dist), decimals)
        sell_tp = round(sell_entry - tp_dist, decimals) if tp_dist > 0 else 0.0
        sell_sl = round(sell_entry + sl_dist, decimals) if sl_dist > 0 else 0.0
        orders.append({
            "side": "SELL_STOP",
            "op_code": 5,  # MT5 OP_SELLSTOP / ORDER_TYPE_SELL_STOP
            "layer": i + 1,
            "price": sell_entry,
            "stop_loss": sell_sl,
            "take_profit": sell_tp,
        })
    return orders
```

---

### 2. Pending Order & Mass Operations in `MT5ExnessClient` (`ai_engine/auto_trader/mt5_exness_client.py`)
Add the following methods to `MT5ExnessClient`:

1. **`place_pending_stop_order(...)`:**
   * Sends `operation=4` (`BUY_STOP`) or `operation=5` (`SELL_STOP`) with explicit `price`, `stoploss`, `takeprofit`, and `volume` to `{MT5_REST_DIRECT_URL}/OrderSendSafe`.
2. **`delete_pending_order(account_id_or_login, token_or_password, ticket, server_name)`:**
   * Cancels an untriggered pending order via `{MT5_REST_DIRECT_URL}/OrderDelete` (or `/OrderCloseSafe` fallback).
3. **`close_all_and_delete_pending(account_id_or_login, token_or_password, symbol, server_name)`:**
   * Replicates the video's end-of-session **"Tutup Semua Posisi" (Close All Positions)** and **"Hapus Semua Order" (Delete All Pending Orders)** bulk actions:
     * Fetches `/OpenedOrders`.
     * Separates filled market positions (`BUY`/`SELL`) from untriggered pending orders (`BUY_STOP`/`SELL_STOP`).
     * Executes concurrent or throttled `close_mt5_ticket()` for all active positions and `delete_pending_order()` for all pending orders on that symbol.

---

### 3. Grid Cycle Management in `SmartAutoTraderEngine` & `AutoTradeSessionManager`
* **Files:** `ai_engine/auto_trader/engine.py`, `ai_engine/auto_trader/session_manager.py`, `ai_engine/auto_trader/session_models.py`
* **Implementation Flow (`ea_mode == "EA_GRID_SCALPER"`):**
  1. **Extend `ea_mode` Literal:** Update `ea_mode` type annotations to `Optional[Literal["EA_SCALPING", "EA_EXPERT_ADVISOR", "EA_GRID_SCALPER"]]`.
  2. **Cycle Execution (`evaluate_and_execute_grid_cycle`):**
     * **Step A (Live Account & Equity Sync):** Query `/AccountSummary` & `/OpenedOrders` from MT5.
     * **Step B (Target Profit / Max Drawdown Kill-Switch):**
       * If `live_equity >= cfg.target_equity_usd` (e.g., `$5,000` target from `$10` deposit) OR `live_equity <= session_peak_equity * (1.0 - cfg.max_equity_drawdown_pct)`:
         * Immediately call `mt5_client.close_all_and_delete_pending()`.
         * Mark the session `COMPLETED` and send the final Telegram summary report.
     * **Step C (Dynamic Lot Calculation):**
       * Compute `current_lot = calculate_compounded_grid_lot(live_equity, cfg)` (or use `sess.custom_lot` if locked by user).
     * **Step D (Pending Trap Maintenance & Replenishment):**
       * Inspect `/OpenedOrders` for the symbol.
       * Count active `BUY_STOP` and `SELL_STOP` orders and check how far price has moved from the existing grid center.
       * If price has broken out through the grid (leaving stale opposite-side pending stops behind) or if pending orders on either side `< 2`:
         * Delete stale untriggered pending orders that are more than `2.5 * first_offset_pips * layers_per_side` away from current price.
         * Fetch live `Bid` and `Ask` via `/GetQuote` (or latest M1 candle close +/- spread).
         * Deploy missing `BUY_STOP` and `SELL_STOP` layers around current price using `build_pending_stop_grid_levels()`.
     * **Step E (Active Position Trailing & Profit Locking):**
       * For any pending order that has triggered into an active `BUY` or `SELL` position, monitor via `PositionMonitorMixin` to ratchet SL into break-even+ once in profit.

---

### 4. Telegram Bot Commands & Callback Integration (`telegram_bot/handlers/autotrade.py`)
Add user-friendly triggers for the new mode across all MT5 symbols:
* `/ea_grid <SYMBOL> [DURATION] [LOT_OR_AUTO]`
  * Examples:
    * `/ea_grid XAUUSD 20m` (Runs M1 Grid Stop Scalper on Gold for 20 minutes with auto-compounding lot starting at `0.01`)
    * `/ea_grid BTCUSD 1h 0.02` (Runs on Bitcoin with fixed `0.02` lot)
    * `/ea_grid US30 30m` (Runs on Dow Jones Index)
    * `/ea_grid EURUSD 1h` (Runs on Forex)
* Add Inline Keyboard buttons on the live session card:
  * `🛑 Tutup Semua Posisi & Hapus Pending Order` (Triggers `close_all_and_delete_pending` immediately).

---

## PART 3: VERIFICATION & TEST SUITE REQUIREMENTS

After applying all changes above, run `pytest` and verify:
1. `test_score_timeframe_trend_no_name_error`: Confirm `_score_timeframe_trend()` returns non-zero scores on trending OHLCV DataFrames and never raises `NameError`.
2. `test_sentiment_composite_not_overwritten`: Confirm `analyze_symbol()` preserves the `65% composite_sentiment + 35% FinBERT` blend when `composite_sentiment` is returned by `MultiSourceNewsAggregator`.
3. `test_micro_account_min_lot_execution`: Confirm a `$10.00 - $50.00` account running `EA_SCALPING`, `EA_EXPERT_ADVISOR`, or `EA_GRID_SCALPER` on `XAUUSD`, `BTCUSD`, and `EURUSD` is approved for `0.01` lot when within `micro_max_lot_risk_fraction`.
4. `test_session_cleanup_and_deduplication`: Confirm `_session_loop` pops `sess.session_id` cleanly upon completion/cancellation and starting a session on the same symbol stops the previous session first.
5. `test_grid_scalper_multi_asset_levels`: Confirm `build_pending_stop_grid_levels()` generates valid `BUY_STOP` (op_code 4) and `SELL_STOP` (op_code 5) prices and `calculate_compounded_grid_lot()` scales lots accurately from `0.01` at `$10` equity up to `2.00` at `$3,000+` equity.