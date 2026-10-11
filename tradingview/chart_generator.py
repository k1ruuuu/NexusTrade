from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from ai_engine.models.ensemble_agent import SignalDecision
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor
from data_pipeline.market_symbols import normalize_symbol, normalize_timeframe

logger = get_logger("nexustrade.tradingview.chart_generator")


class ChartSnapshotGenerator:
    """
    High-resolution 1200x800 TradingView-style dark chart snapshot generator.
    Renders candlesticks, EMA 20/50/200, Bollinger Bands, Volume, RSI(14),
    and annotated Entry / 2x ATR Stop-Loss / Take-Profit (TP1, TP2) levels.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        preprocessor: Optional[OHLCVPreprocessor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = preprocessor or OHLCVPreprocessor(self.settings)

    def generate_chart(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        decision: Optional[SignalDecision] = None,
        output_path: Optional[Path] = None,
        max_bars: int = 90,
    ) -> Path:
        """
        Render a 1200x800 high-resolution PNG chart with candlesticks, MAs, and trade levels.
        """
        safe_symbol = normalize_symbol(symbol)
        safe_timeframe = normalize_timeframe(timeframe)
        cleaned = self.preprocessor.clean_ohlcv(df, symbol=safe_symbol, timeframe=safe_timeframe)
        enriched = self.preprocessor.compute_raw_indicators(cleaned).tail(max_bars).copy()

        if enriched.empty:
            raise ValueError(f"Cannot render chart for empty OHLCV DataFrame ({safe_symbol})")

        target_path = output_path or (
            self.settings.cache_data_dir / f"chart_{safe_symbol.lower()}_{safe_timeframe}.png"
        )

        dpi = 100
        fig_w = self.settings.chart_width / dpi
        fig_h = self.settings.chart_height / dpi

        fig, (ax_price, ax_vol, ax_rsi) = plt.subplots(
            3,
            1,
            figsize=(fig_w, fig_h),
            dpi=dpi,
            sharex=True,
            gridspec_kw={"height_ratios": [3.4, 0.9, 1.1], "hspace": 0.06},
            facecolor="#0b0e14",
        )

        for ax in (ax_price, ax_vol, ax_rsi):
            ax.set_facecolor("#131722")
            ax.tick_params(colors="#9ca3af", labelsize=9)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            for spine in ax.spines.values():
                spine.set_color("#2a2e39")
            ax.grid(True, color="#1e222d", linestyle="--", linewidth=0.6, alpha=0.8)

        x = np.arange(len(enriched))
        opens = enriched["open"].to_numpy(dtype=np.float64)
        highs = enriched["high"].to_numpy(dtype=np.float64)
        lows = enriched["low"].to_numpy(dtype=np.float64)
        closes = enriched["close"].to_numpy(dtype=np.float64)
        volumes = enriched["volume"].to_numpy(dtype=np.float64)

        bull_color = "#089981"
        bear_color = "#f23645"
        candle_width = 0.62

        for idx in range(len(enriched)):
            is_bull = closes[idx] >= opens[idx]
            color = bull_color if is_bull else bear_color
            ax_price.plot(
                [x[idx], x[idx]],
                [lows[idx], highs[idx]],
                color=color,
                linewidth=1.1,
                zorder=2,
            )
            body_bottom = min(opens[idx], closes[idx])
            body_height = max(abs(closes[idx] - opens[idx]), (highs[idx] - lows[idx]) * 0.04, 1e-4)
            rect = Rectangle(
                (x[idx] - candle_width / 2.0, body_bottom),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
            ax_price.add_patch(rect)

        # Moving Averages & Bollinger Bands
        ax_price.plot(x, enriched["ema_20"], color="#3b82f6", linewidth=1.3, label="Tren Cepat (EMA 20)", zorder=4)
        ax_price.plot(x, enriched["ema_50"], color="#f59e0b", linewidth=1.3, label="Tren Menengah (EMA 50)", zorder=4)
        ax_price.plot(x, enriched["ema_200"], color="#a855f7", linewidth=1.5, label="Tren Utama (EMA 200)", zorder=4)
        ax_price.plot(
            x,
            enriched["bb_upper"],
            color="#64748b",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
            label="Rentang Harga (Bollinger)",
        )
        ax_price.plot(
            x,
            enriched["bb_lower"],
            color="#64748b",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
        )
        ax_price.fill_between(
            x,
            enriched["bb_lower"],
            enriched["bb_upper"],
            color="#3b82f6",
            alpha=0.05,
        )

        # Annotate Signal Entry, Stop Loss, and Take Profit levels
        latest_x = x[-1]
        latest_close = float(closes[-1])
        if decision is not None:
            entry_p = decision.entry_price or latest_close
            ax_price.axhline(
                entry_p,
                color="#38bdf8",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label=f"Harga Masuk (Entry): {entry_p:,.2f}",
            )
            if decision.stop_loss > 0:
                ax_price.axhline(
                    decision.stop_loss,
                    color="#ef4444",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"Batas Rugi (Stop Loss): {decision.stop_loss:,.2f}",
                )
            if decision.take_profit_1 > 0:
                ax_price.axhline(
                    decision.take_profit_1,
                    color="#10b981",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"Target Untung 1 (TP1): {decision.take_profit_1:,.2f}",
                )
            if decision.take_profit_2 > 0:
                ax_price.axhline(
                    decision.take_profit_2,
                    color="#22c55e",
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.85,
                    label=f"Target Untung 2 (TP2): {decision.take_profit_2:,.2f}",
                )

            marker_color = (
                "#10b981"
                if decision.action == "BUY"
                else ("#ef4444" if decision.action == "SELL" else "#eab308")
            )
            marker_style = "^" if decision.action == "BUY" else ("v" if decision.action == "SELL" else "o")
            ax_price.scatter(
                [latest_x],
                [entry_p],
                color=marker_color,
                s=130,
                marker=marker_style,
                zorder=6,
                edgecolors="#ffffff",
                linewidths=1.0,
            )

        badge_str = decision.badge if decision is not None else "GRAFIK PASAR"
        conf_str = f"{decision.confidence * 100:.1f}%" if decision is not None else "N/A"
        ax_price.set_title(
            f"NexusTrade AI  •  {symbol.upper()} ({timeframe})  •  {badge_str} (Keyakinan: {conf_str})  •  Harga: {latest_close:,.2f}",
            color="#f8fafc",
            fontsize=12.5,
            fontweight="bold",
            loc="left",
            pad=10,
        )
        ax_price.set_ylabel("Harga", color="#cbd5e1", fontsize=10)
        ax_price.legend(
            loc="upper left",
            ncol=3,
            fontsize=8.5,
            facecolor="#1e222d",
            edgecolor="#2a2e39",
            labelcolor="#e2e8f0",
        )

        # Volume panel
        vol_colors = [bull_color if closes[i] >= opens[i] else bear_color for i in range(len(enriched))]
        ax_vol.bar(x, volumes, width=candle_width, color=vol_colors, alpha=0.75)
        ax_vol.set_ylabel("Volume", color="#cbd5e1", fontsize=9)

        # RSI(14) panel
        rsi_vals = enriched["rsi_14"].to_numpy(dtype=np.float64)
        ax_rsi.plot(x, rsi_vals, color="#38bdf8", linewidth=1.3, label="RSI (14)")
        ax_rsi.axhline(70.0, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_rsi.axhline(30.0, color="#10b981", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_rsi.axhline(50.0, color="#64748b", linestyle=":", linewidth=0.7, alpha=0.6)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("Momentum (RSI)", color="#cbd5e1", fontsize=9)

        # Format x-axis ticks with UTC timestamps
        tick_indices = np.linspace(0, len(enriched) - 1, min(8, len(enriched)), dtype=int)
        tick_labels = [
             enriched.index[i].strftime("%m-%d %H:%M")
            if hasattr(enriched.index[i], "strftime")
            else str(enriched.index[i])[:16]
            for i in tick_indices
        ]
        ax_rsi.set_xticks(tick_indices)
        ax_rsi.set_xticklabels(tick_labels, color="#9ca3af", fontsize=8.5)

        fig.subplots_adjust(left=0.04, right=0.91, top=0.93, bottom=0.06)
        fig.savefig(target_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)

        return target_path

    async def generate_chart_async(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        decision: Optional[SignalDecision] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Asynchronously render chart snapshot without blocking the event loop."""
        return await asyncio.to_thread(
            self.generate_chart,
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            decision=decision,
            output_path=output_path,
        )
