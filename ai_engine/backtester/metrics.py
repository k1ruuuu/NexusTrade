from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    """Institutional performance metrics for event-driven backtest evaluation."""

    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float


def compute_sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe Ratio from periodic strategy returns."""
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    rf_per_period = risk_free_rate / max(periods_per_year, 1)
    excess = arr - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std <= 1e-12:
        return 0.0
    return round(float((np.mean(excess) / std) * np.sqrt(periods_per_year)), 4)


def compute_sortino_ratio(
    returns: Sequence[float] | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino Ratio penalizing only downside volatility."""
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    rf_per_period = risk_free_rate / max(periods_per_year, 1)
    excess = arr - rf_per_period
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return round(float(np.mean(excess) * np.sqrt(periods_per_year) * 10.0), 4)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 1e-12:
        return 0.0
    return round(float((np.mean(excess) / downside_std) * np.sqrt(periods_per_year)), 4)


def compute_max_drawdown(equity_curve: Sequence[float] | pd.Series) -> float:
    """Compute maximum peak-to-trough drawdown fraction in [0.0, 1.0]."""
    arr = np.asarray(equity_curve, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    running_max = np.where(running_max <= 0, 1e-9, running_max)
    drawdowns = (running_max - arr) / running_max
    return round(float(np.max(drawdowns)), 4)


def compute_profit_factor(trade_pnls: Sequence[float]) -> float:
    """Compute ratio of gross winning trade PnL to gross losing trade PnL."""
    arr = np.asarray(trade_pnls, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    gross_profit = float(np.sum(arr[arr > 0]))
    gross_loss = float(np.abs(np.sum(arr[arr < 0])))
    if gross_loss <= 1e-9:
        return round(gross_profit if gross_profit > 0 else 0.0, 4)
    return round(gross_profit / gross_loss, 4)


def summarize_performance(
    equity_curve: Sequence[float] | pd.Series,
    trade_pnls: Sequence[float],
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Aggregate complete financial performance metrics from equity curve and trade list."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) == 0:
        raise ValueError("equity_curve cannot be empty")

    initial_cap = float(eq[0])
    final_eq = float(eq[-1])
    total_return = ((final_eq - initial_cap) / max(initial_cap, 1e-9)) * 100.0

    n_periods = max(len(eq) - 1, 1)
    growth_ratio = max(final_eq / max(initial_cap, 1e-9), 1e-6)
    annualized_ret = ((growth_ratio ** (periods_per_year / n_periods)) - 1.0) * 100.0

    periodic_returns = np.diff(eq) / np.where(eq[:-1] <= 0, 1e-9, eq[:-1])
    sharpe = compute_sharpe_ratio(periodic_returns, periods_per_year=periods_per_year)
    sortino = compute_sortino_ratio(periodic_returns, periods_per_year=periods_per_year)
    max_dd = compute_max_drawdown(eq) * 100.0

    pnls = np.asarray(trade_pnls, dtype=np.float64)
    total_trades = int(len(pnls))
    winning_trades = int(np.sum(pnls > 0))
    losing_trades = int(np.sum(pnls <= 0))
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    pf = compute_profit_factor(pnls)
    avg_pnl = float(np.mean(pnls)) if total_trades > 0 else 0.0

    return BacktestMetrics(
        initial_capital=round(initial_cap, 2),
        final_equity=round(final_eq, 2),
        total_return_pct=round(total_return, 2),
        annualized_return_pct=round(annualized_ret, 2),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=round(max_dd, 2),
        profit_factor=pf,
        win_rate_pct=round(win_rate, 2),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        avg_trade_pnl=round(avg_pnl, 2),
    )
