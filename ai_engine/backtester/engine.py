from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_engine.backtester.metrics import BacktestMetrics, summarize_performance
from ai_engine.risk_management.position_sizer import PositionSizer
from ai_engine.risk_management.stop_loss_manager import StopLossManager
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor

logger = get_logger("nexustrade.ai_engine.backtester")


@dataclass
class SimulatedTrade:
    """Record of an executed trade during event-driven simulation."""

    entry_time: str
    exit_time: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestReport:
    """Complete backtest simulation report with metrics, trade log, equity curve, and chart."""

    symbol: str
    days: int
    metrics: BacktestMetrics
    equity_curve: list[float]
    timestamps: list[str]
    trades: list[SimulatedTrade] = field(default_factory=list)
    chart_path: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "days": self.days,
            "metrics": asdict(self.metrics),
            "equity_curve": self.equity_curve,
            "timestamps": self.timestamps,
            "trades": [asdict(t) for t in self.trades],
            "chart_path": str(self.chart_path) if self.chart_path else None,
        }


class EventDrivenBacktester:
    """
    Event-driven bar-by-bar backtesting engine with zero lookahead bias,
    2x ATR(14) stop-loss, 1:2 RRR take-profit, and trailing stop execution.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        preprocessor: Optional[OHLCVPreprocessor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = preprocessor or OHLCVPreprocessor(self.settings)
        self.stop_loss_manager = StopLossManager(self.settings)
        self.position_sizer = PositionSizer(self.settings)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        days: int = 90,
        initial_capital: Optional[float] = None,
        generate_plot: bool = True,
    ) -> BacktestReport:
        """
        Execute event-driven backtest over OHLCV DataFrame using strictly lagged indicators.
        """
        featured = self.preprocessor.engineer_features(
            df, symbol=symbol, timeframe="1d", enforce_lag=True
        )
        if days > 0 and len(featured) > days:
            featured = featured.tail(days).copy()

        if len(featured) < 10:
            raise ValueError(f"Insufficient bars ({len(featured)}) for backtest on {symbol}")

        capital = initial_capital or self.settings.portfolio_initial_capital
        equity = float(capital)
        equity_curve: list[float] = [round(equity, 2)]
        timestamps: list[str] = [str(featured["timestamp_iso"].iloc[0])]
        trades: list[SimulatedTrade] = []

        open_position: Optional[dict[str, Any]] = None

        for idx in range(1, len(featured)):
            row = featured.iloc[idx]
            bar_time = str(row["timestamp_iso"])
            open_p = float(row["open"])
            high_p = float(row["high"])
            low_p = float(row["low"])
            close_p = float(row["close"])
            atr_val = max(float(row["atr_14"]), close_p * 0.008)

            # 1. Manage active position against current bar's High/Low
            if open_position is not None:
                direction = open_position["direction"]
                stop_loss = open_position["stop_loss"]
                tp1 = open_position["tp1"]
                units = open_position["units"]
                entry_p = open_position["entry_price"]

                exit_price: Optional[float] = None
                exit_reason = ""

                if direction == "BUY":
                    if low_p <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "STOP_LOSS"
                    elif high_p >= tp1:
                        exit_price = tp1
                        exit_reason = "TAKE_PROFIT"
                    else:
                        open_position["stop_loss"] = self.stop_loss_manager.update_trailing_stop(
                            "BUY", close_p, stop_loss, open_position["trail_dist"]
                        )
                elif direction == "SELL":
                    if high_p >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "STOP_LOSS"
                    elif low_p <= tp1:
                        exit_price = tp1
                        exit_reason = "TAKE_PROFIT"
                    else:
                        open_position["stop_loss"] = self.stop_loss_manager.update_trailing_stop(
                            "SELL", close_p, stop_loss, open_position["trail_dist"]
                        )

                if exit_price is not None:
                    raw_pnl = (
                        (exit_price - entry_p) * units
                        if direction == "BUY"
                        else (entry_p - exit_price) * units
                    )
                    fee = (entry_p + exit_price) * units * 0.0005
                    net_pnl = raw_pnl - fee
                    equity = max(0.0, equity + net_pnl)
                    ret_pct = (net_pnl / max(entry_p * units, 1e-6)) * 100.0
                    trades.append(
                        SimulatedTrade(
                            entry_time=open_position["entry_time"],
                            exit_time=bar_time,
                            symbol=symbol.upper(),
                            direction=direction,
                            entry_price=round(entry_p, 4),
                            exit_price=round(exit_price, 4),
                            units=round(units, 6),
                            pnl=round(net_pnl, 2),
                            return_pct=round(ret_pct, 2),
                            exit_reason=exit_reason,
                        )
                    )
                    open_position = None

            # 2. Evaluate new entry using strictly lagged t-1 features stored on row
            if open_position is None:
                rsi_prev = float(row["rsi_14"])
                ema_20_prev = float(row["ema_20"])
                ema_50_prev = float(row["ema_50"])
                macd_hist_prev = float(row["macd_hist"])

                signal_dir: Optional[str] = None
                confidence = 0.78
                if ema_20_prev > ema_50_prev and 48.0 <= rsi_prev <= 72.0 and macd_hist_prev >= 0:
                    signal_dir = "BUY"
                elif ema_20_prev < ema_50_prev and 28.0 <= rsi_prev <= 52.0 and macd_hist_prev < 0:
                    signal_dir = "SELL"

                if signal_dir in ("BUY", "SELL"):
                    targets = self.stop_loss_manager.calculate_targets(
                        entry_price=open_p,
                        atr_14=atr_val,
                        action=signal_dir,  # type: ignore[arg-type]
                    )
                    sizing = self.position_sizer.calculate_position(
                        entry_price=open_p,
                        stop_loss=targets.stop_loss,
                        confidence=confidence,
                        s_vol=0.25,
                        portfolio_equity=equity,
                        daily_drawdown_override=0.0,
                    )
                    if sizing.approved and sizing.position_units > 0:
                        open_position = {
                            "entry_time": bar_time,
                            "direction": signal_dir,
                            "entry_price": open_p,
                            "stop_loss": targets.stop_loss,
                            "tp1": targets.take_profit_1,
                            "trail_dist": targets.trailing_stop_distance,
                            "units": sizing.position_units,
                        }

            # Mark-to-market equity
            mtm_pnl = 0.0
            if open_position is not None:
                if open_position["direction"] == "BUY":
                    mtm_pnl = (close_p - open_position["entry_price"]) * open_position["units"]
                else:
                    mtm_pnl = (open_position["entry_price"] - close_p) * open_position["units"]

            equity_curve.append(round(max(0.0, equity + mtm_pnl), 2))
            timestamps.append(bar_time)

        # Close any remaining position at final bar close
        if open_position is not None:
            final_row = featured.iloc[-1]
            final_close = float(final_row["close"])
            direction = open_position["direction"]
            entry_p = open_position["entry_price"]
            units = open_position["units"]
            raw_pnl = (
                (final_close - entry_p) * units
                if direction == "BUY"
                else (entry_p - final_close) * units
            )
            fee = (entry_p + final_close) * units * 0.0005
            net_pnl = raw_pnl - fee
            equity = max(0.0, equity + net_pnl)
            equity_curve[-1] = round(equity, 2)
            trades.append(
                SimulatedTrade(
                    entry_time=open_position["entry_time"],
                    exit_time=str(final_row["timestamp_iso"]),
                    symbol=symbol.upper(),
                    direction=direction,
                    entry_price=round(entry_p, 4),
                    exit_price=round(final_close, 4),
                    units=round(units, 6),
                    pnl=round(net_pnl, 2),
                    return_pct=round((net_pnl / max(entry_p * units, 1e-6)) * 100.0, 2),
                    exit_reason="END_OF_PERIOD",
                )
            )

        metrics = summarize_performance(
            equity_curve=equity_curve,
            trade_pnls=[t.pnl for t in trades],
            periods_per_year=252,
        )

        chart_path: Optional[Path] = None
        if generate_plot:
            chart_path = self.render_equity_curve_plot(
                symbol=symbol.upper(),
                equity_curve=equity_curve,
                metrics=metrics,
            )

        return BacktestReport(
            symbol=symbol.upper(),
            days=days,
            metrics=metrics,
            equity_curve=equity_curve,
            timestamps=timestamps,
            trades=trades,
            chart_path=chart_path,
        )

    def render_equity_curve_plot(
        self,
        symbol: str,
        equity_curve: list[float],
        metrics: BacktestMetrics,
    ) -> Path:
        """Render institutional dark-theme equity & drawdown curve PNG."""
        out_path = self.settings.cache_data_dir / f"backtest_{symbol.lower()}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax_eq, ax_dd) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            dpi=100,
            gridspec_kw={"height_ratios": [3, 1]},
            facecolor="#0b0f19",
        )
        for ax in (ax_eq, ax_dd):
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#9ca3af")
            for spine in ax.spines.values():
                spine.set_color("#374151")
            ax.grid(True, color="#1f2937", linestyle="--", linewidth=0.6)

        x_vals = np.arange(len(equity_curve))
        eq_arr = np.asarray(equity_curve, dtype=np.float64)
        peaks = np.maximum.accumulate(eq_arr)
        dd_pct = -((peaks - eq_arr) / np.where(peaks <= 0, 1.0, peaks)) * 100.0

        ax_eq.plot(x_vals, eq_arr, color="#10b981", linewidth=2.0, label="Pertumbuhan Saldo Portofolio ($)")
        ax_eq.fill_between(x_vals, eq_arr[0], eq_arr, color="#10b981", alpha=0.12)
        ax_eq.set_title(
            f"Uji Strategi (Backtest) {symbol} — Hasil: {metrics.total_return_pct:+.2f}% | "
            f"Menang (Win Rate): {metrics.win_rate_pct:.1f}% | Sharpe: {metrics.sharpe_ratio:.2f}",
            color="#f9fafb",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
        ax_eq.set_ylabel("Saldo Modal ($)", color="#e5e7eb")
        ax_eq.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151", labelcolor="#f9fafb")

        ax_dd.plot(x_vals, dd_pct, color="#ef4444", linewidth=1.4, label="Persentase Penurunan Modal / Drawdown (%)")
        ax_dd.fill_between(x_vals, 0, dd_pct, color="#ef4444", alpha=0.22)
        ax_dd.set_ylabel("Penurunan (%)", color="#e5e7eb")
        ax_dd.set_xlabel("Periode Waktu (Bar)", color="#e5e7eb")
        ax_dd.legend(loc="lower left", facecolor="#1f2937", edgecolor="#374151", labelcolor="#f9fafb")

        fig.tight_layout()
        fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path
