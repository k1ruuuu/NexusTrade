# ADVANCED_AI_TRADING_ENGINE_SPEC.md
# Advanced AI Trading Engine: Triple-Barrier Labeling, Meta-Labeling, and Precision Execution Specification

## 1. Scope & Core Architecture
This document provides the definitive mathematical and programmatic specifications for upgrading **NexusTrade AI** from a basic predictive classifier into an institutional-grade, multi-stage trading agent. 

To eliminate false positives, reduce lookahead bias, and maximize risk-adjusted returns (Sharpe ratio), the engine replaces traditional binary labeling with **Triple-Barrier Labeling** and implements a **Two-Stage Meta-Labeling Architecture**.

---

## 2. Triple-Barrier Labeling Methodology

Traditional machine learning in trading assigns binary labels ($1$ for up, $0$ or $-1$ for down) based on a fixed future time horizon $t + k$. This ignores path dependency, volatility changes, and stop-loss/take-profit mechanics.

### 2.1 Mathematical Formulation
For each bar $t$, define three potential barriers:
1. **Upper Barrier (Take Profit - $y_{upper}$):**
   $$Barrier_{upper}(t) = Close_t + (k_{tp} \cdot ATR_{14}(t))$$
2. **Lower Barrier (Stop Loss - $y_{lower}$):**
   $$Barrier_{lower}(t) = Close_t - (k_{sl} \cdot ATR_{14}(t))$$
3. **Vertical Barrier (Time Expiry - $t_{max}$):**
   $$t_{max} = t + L_{horizon}$$
   Where $L_{horizon}$ is the maximum holding period (e.g., 60 bars on $M1$).

### 2.2 Label Assignment Rule
Let $P(\tau)$ be the path of prices at subsequent bars $\tau \in [t+1, t_{max}]$. The label $y_t \in \{-1, 0, 1\}$ is determined by whichever barrier is touched *first*:
- $y_t = +1$ (BUY Success) if $P(\tau) \ge Barrier_{upper}(t)$ before touching $Barrier_{lower}(t)$.
- $y_t = -1$ (SELL Success) if $P(\tau) \le Barrier_{lower}(t)$ before touching $Barrier_{upper}(t)$.
- $y_t = 0$ (Neutral / Time Expired) if neither barrier is touched by $t_{max}$.

### 2.3 Concrete Numerical Example
- **Current Bar ($t$):** $Close_t = 65,000.00$, $ATR_{14}(t) = 150.00$
- **Multipliers:** $k_{tp} = 2.0$, $k_{sl} = 1.5$, $L_{horizon} = 30$ bars.
- **Computed Barriers:**
  - $Barrier_{upper} = 65,000 + (2.0 \cdot 150) = 65,300.00$
  - $Barrier_{lower} = 65,000 - (1.5 \cdot 150) = 64,775.00$
  - $t_{max} = t + 30$
- **Path Evaluation:** If price hits $65,300.00$ at bar $t+12$ before dipping to $64,775.00$, the label $y_t = +1$.

---

## 3. Meta-Labeling Architecture (Two-Stage Model)

To prevent false positives, we decouple the **Directional Model** (which predicts *where* price is heading) from the **Meta-Model** (which predicts *whether* the directional signal will be profitable).

### 3.1 Stage 1: Primary Directional Model
- **Algorithm:** LightGBM / XGBoost multi-class classifier.
- **Input Features ($X_1$):** Candlestick anatomy ratios, S/R distances, FVG zones, and multi-timeframe trend alignment.
- **Output:** Predicted side $\hat{y} \in \{-1, +1\}$.

### 3.2 Stage 2: Secondary Meta-Model
- **Target Variable ($m_t$):** Binary outcome indicating if the Primary Model's prediction was correct.
  $$m_t = 1 \quad \text{if} \quad \hat{y}_t == y_t \quad \text{else} \quad 0$$
- **Input Features ($X_2$):** Macro intermarket correlation (DXY, VIX), Volume Z-Score, Volatility regime state, and session hour (London/New York).
- **Output:** Probability of trade success $P(m_t = 1 | X_2)$.

### 3.3 Execution Gate
A trade order is transmitted to Telegram and executed only if:
$$\hat{y}_t \neq 0 \quad \text{AND} \quad P(m_t = 1 | X_2) \ge 0.72$$

---

## 4. Multi-Dimensional Feature Vector Specification

All features must strictly prevent lookahead bias by using `.shift(1)` or rolling computations up to $t-1$.

| Dimension | Feature Name | Mathematical Definition | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Trend & Regime** | EMA Slope Ratio | $Slope_{50}(t) = \frac{EMA_{50}(t) - EMA_{50}(t-5)}{5 \cdot ATR(t)}$ | Macro trend direction |
| **2. Candle Anatomy** | Wick Asymmetry Index | $WAI(t) = \frac{UpperWick(t) - LowerWick(t)}{High(t) - Low(t) + \epsilon}$ | Rejection & exhaustion |
| **3. Structural Location**| Normalized S/R Distance | $Dist_{SR}(t) = \frac{Close(t) - \mu_{SR}}{ATR(t)}$ | Proximity to support/resistance |
| **4. Volatility & Volume**| Volume Z-Score | $VolZ(t) = \frac{Volume(t) - \mu_{Vol}(20)}{\sigma_{Vol}(20)}$ | Institutional participation |

---

## 5. Macro Intermarket & Event Risk Filters

1. **DXY Trend Filter:** If 1-hour percentage change of Dollar Index ($DXY$) > $+0.15\%$, suppress all crypto and gold BUY signals.
2. **Economic Event Lockout:**
   - Query ForexFactory JSON endpoint: `https://nfs.faireconomy.media/ff_calendar_thisweek.json`
   - If an event with `impact == "High"` occurs within $\pm 15$ minutes of current timestamp $t$, set `SystemState = LOCKED` and reject all incoming webhook alerts.

---

## 6. Python Implementation Reference (Production Engine)

Save as `ai_engine/models/meta_label_engine.py`:

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TripleBarrierConfig:
    k_tp: float = 2.0
    k_sl: float = 1.5
    horizon_bars: int = 30


class TripleBarrierLabeler:
    @staticmethod
    def apply_triple_barrier(
        df: pd.DataFrame, config: TripleBarrierConfig
    ) -> pd.Series:
        labels = pd.Series(index=df.index, dtype=int)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        n = len(df)

        for i in range(n - config.horizon_bars):
            entry_price = closes[i]
            atr = atrs[i]
            if np.isnan(atr) or atr <= 0:
                labels.iloc[i] = 0
                continue

            upper_barrier = entry_price + (config.k_tp * atr)
            lower_barrier = entry_price - (config.k_sl * atr)

            end_idx = min(i + config.horizon_bars, n)
            path_highs = highs[i + 1 : end_idx]
            path_lows = lows[i + 1 : end_idx]

            hit_upper = np.where(path_highs >= upper_barrier)[0]
            hit_lower = np.where(path_lows <= lower_barrier)[0]

            first_upper = hit_upper[0] if len(hit_upper) > 0 else 999999
            first_lower = hit_lower[0] if len(hit_lower) > 0 else 999999

            if first_upper < first_lower:
                labels.iloc[i] = 1
            elif first_lower < first_upper:
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0

        return labels
```
