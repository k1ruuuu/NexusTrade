from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.features.structure_extractor import CandleStructuralExtractor
@dataclass
class SupportResistanceLevels:
    """
    Structural Support (bawah) and Resistance (atas) levels detected from OHLCV history.
    Guarantees invariant: support_2 < support_1 < current_price < resistance_1 < resistance_2.
    """

    current_price: float
    pivot_point: float
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    dist_to_r1_pct: float
    dist_to_s1_pct: float
    channel_position_pct: float
    sr_bias: float
    status_label: str


class SupportResistanceDetector:
    """
    Automatic multi-method Support & Resistance detector combining:
    1. Volume-weighted Swing High / Swing Low fractals (3-bar and 5-bar causal pivots)
    2. Multi-window Donchian structural highs/lows (20, 50, 100 bars)
    3. Classic Floor Pivot Points (P, R1, R2, S1, S2)
    Clusters nearby candidate zones within 0.35 * ATR(14) tolerance.
    """
    def __init__(self, cluster_eps_atr: float = 0.35) -> None:
        self._structural_extractor = CandleStructuralExtractor(cluster_eps_atr=cluster_eps_atr)

    def detect_levels(
        self,
        df: Optional[pd.DataFrame],
        current_price: float,
        atr_14: Optional[float] = None,
    ) -> SupportResistanceLevels:
        """Detect structural S1, S2, R1, R2 levels around current_price."""
        price = float(current_price) if current_price > 0 else 100.0
        eff_atr = float(atr_14) if (atr_14 is not None and atr_14 > 0) else price * 0.012
        min_gap = max(eff_atr * 0.35, price * 0.0015)

        if df is None or len(df) < 10:
            return self._fallback_from_atr(price, eff_atr)

        highs = pd.to_numeric(df["high"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        lows = pd.to_numeric(df["low"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        closes = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        volumes = (
            pd.to_numeric(df["volume"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
            if "volume" in df.columns
            else np.ones_like(closes)
        )

        n = min(len(highs), len(lows), len(closes))
        if n < 10:
            return self._fallback_from_atr(price, eff_atr)

        highs = highs[-min(n, 240) :]
        lows = lows[-min(n, 240) :]
        closes = closes[-min(n, 240) :]
        volumes = volumes[-min(n, 240) :]
        m = len(closes)

        # 1. Classic Pivot Point from recent 20-bar structural window
        win = min(m, 20)
        period_high = float(np.max(highs[-win:]))
        period_low = float(np.min(lows[-win:]))
        period_close = float(closes[-1])
        pivot = (period_high + period_low + period_close) / 3.0
        piv_r1 = (2.0 * pivot) - period_low
        piv_s1 = (2.0 * pivot) - period_high
        piv_r2 = pivot + (period_high - period_low)
        piv_s2 = pivot - (period_high - period_low)

        # 2. Collect candidate (level, weight) pairs from DBSCAN Extrema Clusters, Swing Highs/Lows & Donchian windows
        candidates: list[tuple[float, float]] = [
            (piv_r1, 2.2),
            (piv_r2, 1.8),
            (piv_s1, 2.2),
            (piv_s2, 1.8),
        ]
        try:
            _, _, db_clusters = self._structural_extractor.detect_support_resistance_clusters(
                df, window=5, lookback=min(m, 160)
            )
            for cl in db_clusters:
                candidates.append((float(cl.level), min(6.0, 2.0 + 0.5 * float(cl.touches))))
        except Exception:
            pass
        for d_win, w in ((20, 2.5), (50, 3.0), (120, 3.2)):
            if m >= d_win:
                candidates.append((float(np.max(highs[-d_win:])), w))
                candidates.append((float(np.min(lows[-d_win:])), w))

        vol_median = float(np.median(volumes)) if float(np.median(volumes)) > 0 else 1.0
        for k in (2, 4):
            if m > 2 * k + 1:
                for i in range(k, m - k):
                    recency_w = 0.6 + 0.8 * (i / m)
                    vol_w = min(2.5, max(0.7, float(volumes[i]) / vol_median))
                    window_h = highs[i - k : i + k + 1]
                    window_l = lows[i - k : i + k + 1]
                    if highs[i] >= np.max(window_h):
                        candidates.append((float(highs[i]), recency_w * vol_w * (1.2 if k == 4 else 0.9)))
                    if lows[i] <= np.min(window_l):
                        candidates.append((float(lows[i]), recency_w * vol_w * (1.2 if k == 4 else 0.9)))

        clustered = self._cluster_levels(candidates, tolerance=min_gap)

        above = sorted(
            [(lvl, w) for lvl, w in clustered if lvl >= price + min_gap],
            key=lambda x: x[0],
        )
        below = sorted(
            [(lvl, w) for lvl, w in clustered if lvl <= price - min_gap],
            key=lambda x: x[0],
            reverse=True,
        )

        # Select R1 & R2 strictly above current_price
        if len(above) >= 1:
            r1 = above[0][0]
        else:
            r1 = price + 1.5 * eff_atr

        r2_candidates = [lvl for lvl, _ in above[1:] if lvl >= r1 + min_gap]
        if r2_candidates:
            r2 = r2_candidates[0]
        else:
            r2 = max(r1 + 1.25 * eff_atr, price + 2.8 * eff_atr)

        # Select S1 & S2 strictly below current_price
        if len(below) >= 1:
            s1 = below[0][0]
        else:
            s1 = max(price * 0.05, price - 1.5 * eff_atr)

        s2_candidates = [lvl for lvl, _ in below[1:] if lvl <= s1 - min_gap and lvl > 0]
        if s2_candidates:
            s2 = s2_candidates[0]
        else:
            s2 = max(price * 0.02, min(s1 - 1.25 * eff_atr, price - 2.8 * eff_atr))

        return self._build_result(price, pivot, s1, s2, r1, r2)

    @staticmethod
    def _cluster_levels(
        candidates: list[tuple[float, float]],
        tolerance: float,
    ) -> list[tuple[float, float]]:
        if not candidates:
            return []
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        clusters: list[list[tuple[float, float]]] = [[sorted_cands[0]]]
        for lvl, weight in sorted_cands[1:]:
            prev_mean = sum(x[0] * x[1] for x in clusters[-1]) / sum(x[1] for x in clusters[-1])
            if abs(lvl - prev_mean) <= tolerance:
                clusters[-1].append((lvl, weight))
            else:
                clusters.append([(lvl, weight)])

        result: list[tuple[float, float]] = []
        for group in clusters:
            total_w = sum(x[1] for x in group)
            weighted_lvl = sum(x[0] * x[1] for x in group) / total_w
            result.append((weighted_lvl, total_w))
        return result

    def _fallback_from_atr(self, price: float, eff_atr: float) -> SupportResistanceLevels:
        s1 = max(price * 0.05, price - 1.5 * eff_atr)
        s2 = max(price * 0.02, price - 3.0 * eff_atr)
        r1 = price + 1.5 * eff_atr
        r2 = price + 3.0 * eff_atr
        return self._build_result(price, price, s1, s2, r1, r2)

    @staticmethod
    def _build_result(
        price: float,
        pivot: float,
        s1: float,
        s2: float,
        r1: float,
        r2: float,
    ) -> SupportResistanceLevels:
        decimals = 2 if price >= 10.0 else (4 if price >= 0.1 else 6)
        dist_r1_pct = ((r1 - price) / price) * 100.0
        dist_s1_pct = ((s1 - price) / price) * 100.0
        span = max(r1 - s1, 1e-9)
        pos_pct = float(np.clip(((price - s1) / span) * 100.0, 0.0, 100.0))

        # Structural bias: positive near S1 support (+0.08 max), negative near R1 resistance (-0.08 max)
        sr_bias = float(np.clip((50.0 - pos_pct) / 625.0, -0.08, 0.08))

        if pos_pct <= 25.0:
            status_label = "Dekat Support Bawah S1 (Area Pantul Beli)"
        elif pos_pct >= 75.0:
            status_label = "Dekat Resistance Atas R1 (Rawan Tertahan)"
        else:
            status_label = "Tengah Jalur S1–R1 (Ruang Gerak Lega)"

        return SupportResistanceLevels(
            current_price=round(price, decimals),
            pivot_point=round(pivot, decimals),
            support_1=round(s1, decimals),
            support_2=round(s2, decimals),
            resistance_1=round(r1, decimals),
            resistance_2=round(r2, decimals),
            dist_to_r1_pct=round(dist_r1_pct, 2),
            dist_to_s1_pct=round(dist_s1_pct, 2),
            channel_position_pct=round(pos_pct, 1),
            sr_bias=round(sr_bias, 4),
            status_label=status_label,
        )
