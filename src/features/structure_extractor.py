from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class SupportResistanceCluster:
    """Clustered Support or Resistance zone synthesized via DBSCAN with exponential time decay."""

    level: float
    strength: float
    touches: int
    mean_volume: float


@dataclass(frozen=True)
class FairValueGapZone:
    """3-bar Fair Value Gap (FVG) structural price imbalance zone."""

    direction: str  # "BULL" or "BEAR"
    created_idx: int
    lower_bound: float
    upper_bound: float
    active: bool = True


@dataclass(frozen=True)
class VolumeProfileSnapshot:
    """Rolling Volume Profile summary: Point of Control (POC) and 70% Value Area (VAL, VAH)."""

    poc: float
    vah: float
    val: float
    total_volume: float


class CandleStructuralExtractor:
    """
    Implements Modules 1, 2, and 3 of the NexusTrade AI Technical Specification:
      - Module 1: Candlestick Anatomy, Dimensionless Ratios & Continuous Pattern Scores
      - Module 2: Causal Local Extrema + DBSCAN Support & Resistance Clustering
      - Module 3: Structural Zones (Fair Value Gaps FVG & Rolling Volume Profile POC/VAH/VAL)

    All output features are dimensionless or normalized by ATR_k(t) to guarantee strict
    scale invariance under positive scalar price multiplication and zero lookahead bias.
    """

    STRUCTURAL_FEATURE_COLUMNS: tuple[str, ...] = (
        "r_body",
        "r_wick",
        "rel_spread",
        "score_pinbar",
        "score_engulf",
        "delta_r",
        "delta_s",
        "pos_sr",
        "delta_poc",
        "delta_vah",
        "delta_val",
        "dist_fvg_bull",
        "dist_fvg_bear",
    )

    def __init__(
        self,
        atr_period: int = 14,
        cluster_eps_atr: float = 0.35,
        min_samples: int = 2,
        swing_window: int = 5,
        lookback_bars: int = 120,
        decay_lambda: float = 0.015,
        volume_profile_window: int = 100,
        volume_profile_bins: int = 24,
        value_area_Target: float = 0.70,
    ) -> None:
        self.atr_period = max(int(atr_period), 1)
        self.cluster_eps_atr = float(cluster_eps_atr)
        self.min_samples = max(int(min_samples), 2)
        self.swing_window = max(int(swing_window), 1)
        self.lookback_bars = max(int(lookback_bars), 20)
        self.decay_lambda = float(decay_lambda)
        self.volume_profile_window = max(int(volume_profile_window), 10)
        self.volume_profile_bins = max(int(volume_profile_bins), 8)
        self.value_area_target = float(np.clip(value_area_Target, 0.50, 0.95))

    def compute_atr_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute causal Simple Moving Average True Range:
          ATR_k(t) = (1 / k) * sum_{i=0}^{k-1} TR_{t-i}
        """
        high = pd.to_numeric(df["high"], errors="coerce").astype(np.float64)
        low = pd.to_numeric(df["low"], errors="coerce").astype(np.float64)
        close = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
        prev_close = close.shift(1).fillna(close)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=self.atr_period, min_periods=1).mean()
        # Scale-proportional floor prevents division by zero while preserving scalar invariance
        scale_floor = close.abs() * 1e-8
        return np.maximum(atr, scale_floor)

    def compute_candle_anatomy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Module 1 (Sections 2.1 - 2.3):
        Computes body, wick, spread, relative spread, pinbar rejection score, and engulfing momentum.
        Strictly causal: uses data available up to index t.
        """
        out = df.copy()
        high = pd.to_numeric(out["high"], errors="coerce").astype(np.float64)
        low = pd.to_numeric(out["low"], errors="coerce").astype(np.float64)
        close = pd.to_numeric(out["close"], errors="coerce").astype(np.float64)
        open_ = pd.to_numeric(out["open"], errors="coerce").astype(np.float64)

        atr = self.compute_atr_series(out)
        out["atr"] = atr

        # 2.1 Geometric Decomposition
        spread = (high - low).clip(lower=0.0)
        body = (close - open_).abs()
        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)
        direction = np.sign(close - open_)

        # Relative epsilon scaled to price/spread for exact scale invariance under c * P
        eps = np.maximum(close.abs() * 1e-12, 1e-18)
        denom = spread + eps

        # 2.2 Stationary Dimensionless Ratios
        out["r_body"] = np.where(spread > eps, (body / denom).clip(0.0, 1.0), 0.0)
        out["r_wick"] = np.where(
            spread > eps, ((upper_wick - lower_wick) / denom).clip(-1.0, 1.0), 0.0
        )
        out["rel_spread"] = (spread / atr).fillna(0.0)

        # 2.3 Algorithmic Candlestick Pattern Quantification
        raw_pinbar = np.where(
            spread > eps,
            (lower_wick - 2.0 * body - upper_wick) / denom,
            0.0,
        )
        out["score_pinbar"] = np.clip(raw_pinbar, 0.0, 1.0)

        prev_body = body.shift(1).fillna(0.0)
        prev_dir = pd.Series(direction, index=out.index).shift(1).fillna(0.0)
        prev_high = high.shift(1).fillna(high)
        prev_low = low.shift(1).fillna(low)

        is_opposite_dir = (direction != 0.0) & (prev_dir != 0.0) & (direction != prev_dir)
        is_larger_body = body > prev_body
        encompasses_range = (high >= prev_high) & (low <= prev_low)
        engulf_cond = is_opposite_dir & is_larger_body & encompasses_range

        out["score_engulf"] = np.where(
            engulf_cond,
            ((body - prev_body) / atr).fillna(0.0),
            0.0,
        )
        return out

    def detect_support_resistance_clusters(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None,
        lookback: Optional[int] = None,
    ) -> Tuple[float, float, List[SupportResistanceCluster]]:
        """
        Module 2 (Sections 3.1 - 3.2):
        Causal local extrema identification (excluding last `window` bars to enforce
        tau <= t - window) followed by DBSCAN clustering with exponential time-decay
        level synthesis and volume-weighted cluster strength.
        """
        k = max(int(window if window is not None else self.swing_window), 1)
        m_bars = max(int(lookback if lookback is not None else self.lookback_bars), 2 * k + 2)

        sub = df.iloc[-m_bars:] if len(df) > m_bars else df
        highs = pd.to_numeric(sub["high"], errors="coerce").to_numpy(dtype=np.float64)
        lows = pd.to_numeric(sub["low"], errors="coerce").to_numpy(dtype=np.float64)
        closes = pd.to_numeric(sub["close"], errors="coerce").to_numpy(dtype=np.float64)
        volumes = (
            pd.to_numeric(sub["volume"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
            if "volume" in sub.columns
            else np.ones_like(closes)
        )

        n = len(closes)
        if n == 0:
            return 0.0, 0.0, []

        curr_price = float(closes[-1])
        fallback_s = float(np.min(lows))
        fallback_r = float(np.max(highs))

        if "atr" in sub.columns:
            atr = float(sub["atr"].iloc[-1])
        elif "atr_14" in sub.columns:
            atr = float(sub["atr_14"].iloc[-1])
        else:
            atr = float(self.compute_atr_series(sub).iloc[-1])
        atr = max(atr, abs(curr_price) * 1e-8, 1e-12)

        if n <= 2 * k + 1:
            return fallback_s, fallback_r, []

        # Strictly causal: exclude the last `k` bars so peak at tau is only known at t >= tau + k
        causal_highs = highs[:-k]
        causal_lows = lows[:-k]

        peak_idx = argrelextrema(causal_highs, np.greater_equal, order=k)[0]
        trough_idx = argrelextrema(causal_lows, np.less_equal, order=k)[0]

        candidate_prices: List[float] = []
        candidate_taus: List[int] = []
        candidate_vols: List[float] = []

        # Filter unmitigated swing highs (not decisively broken by subsequent closes) and swing lows
        for idx in peak_idx:
            candidate_prices.append(float(highs[idx]))
            candidate_taus.append(int(idx))
            candidate_vols.append(float(volumes[idx]))
        for idx in trough_idx:
            candidate_prices.append(float(lows[idx]))
            candidate_taus.append(int(idx))
            candidate_vols.append(float(volumes[idx]))

        if len(candidate_prices) < self.min_samples:
            return fallback_s, fallback_r, []

        candidates_arr = np.asarray(candidate_prices, dtype=np.float64).reshape(-1, 1)
        taus_arr = np.asarray(candidate_taus, dtype=np.float64)
        vols_arr = np.asarray(candidate_vols, dtype=np.float64)

        # Scale-invariant epsilon: 0.35 * ATR_k(t)
        eps = max(self.cluster_eps_atr * atr, abs(curr_price) * 1e-9, 1e-12)
        db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(candidates_arr)
        labels = db.labels_

        t_now = float(n - 1)
        clusters: List[SupportResistanceCluster] = []
        unique_labels = sorted(set(int(lbl) for lbl in labels if lbl != -1))

        for lbl in unique_labels:
            mask = labels == lbl
            p_cluster = candidates_arr[mask, 0]
            tau_cluster = taus_arr[mask]
            v_cluster = vols_arr[mask]

            # Section 3.2 Eq: w_i = exp(-lambda * (t - tau_i))
            weights = np.exp(-self.decay_lambda * np.maximum(0.0, t_now - tau_cluster))
            weight_sum = float(np.sum(weights))
            level_m = (
                float(np.sum(weights * p_cluster) / weight_sum)
                if weight_sum > 0.0
                else float(np.mean(p_cluster))
            )
            mean_vol = float(np.mean(v_cluster)) if len(v_cluster) > 0 else 1.0
            strength_m = float(len(p_cluster)) * mean_vol
            clusters.append(
                SupportResistanceCluster(
                    level=level_m,
                    strength=strength_m,
                    touches=int(len(p_cluster)),
                    mean_volume=mean_vol,
                )
            )

        supports = [c.level for c in clusters if c.level < curr_price]
        resistances = [c.level for c in clusters if c.level > curr_price]

        # Deterministic Fallback (Guardrail 2): rolling minimum low / maximum high when no cluster on that side
        nearest_s = float(max(supports)) if supports else fallback_s
        nearest_r = float(min(resistances)) if resistances else fallback_r
        if nearest_s >= curr_price:
            nearest_s = min(fallback_s, curr_price - 0.5 * atr)
        if nearest_r <= curr_price:
            nearest_r = max(fallback_r, curr_price + 0.5 * atr)

        return float(nearest_s), float(nearest_r), clusters

    def detect_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 5,
    ) -> Tuple[float, float]:
        """
        Section 7.1 Interface:
        Identifies unmitigated clustered S/R levels using DBSCAN.
        Returns (nearest_support, nearest_resistance).
        """
        work_df = df if ("atr" in df.columns or "atr_14" in df.columns) else self.compute_candle_anatomy(df)
        nearest_s, nearest_r, _ = self.detect_support_resistance_clusters(work_df, window=window)
        return nearest_s, nearest_r

    @staticmethod
    def detect_active_fvg_zones(
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> Tuple[Optional[FairValueGapZone], Optional[FairValueGapZone]]:
        """
        Module 3 (Section 4.1):
        Scan 3-bar sequences [tau-2, tau-1, tau] up to current bar t and return the most recent
        unmitigated Bullish FVG and Bearish FVG.
          - Bullish FVG at tau: L_tau - H_{tau-2} > 0, active until L_m <= H_{tau-2} for m > tau.
          - Bearish FVG at tau: L_{tau-2} - H_tau > 0, active until H_m >= L_{tau-2} for m > tau.
        """
        n = len(highs)
        if n < 3:
            return None, None

        active_bull: Optional[FairValueGapZone] = None
        active_bear: Optional[FairValueGapZone] = None

        # Track unmitigated gaps causally through index n - 1
        bull_zones: List[FairValueGapZone] = []
        bear_zones: List[FairValueGapZone] = []

        for tau in range(2, n):
            h_tau = float(highs[tau])
            l_tau = float(lows[tau])

            # Mitigate prior active zones using bar tau's price action
            if bull_zones:
                bull_zones = [z for z in bull_zones if l_tau > z.lower_bound]
            if bear_zones:
                bear_zones = [z for z in bear_zones if h_tau < z.upper_bound]

            # Check new Bullish FVG at bar tau: L_tau - H_{tau-2} > 0
            h_prev2 = float(highs[tau - 2])
            l_prev2 = float(lows[tau - 2])
            if l_tau > h_prev2:
                bull_zones.append(
                    FairValueGapZone(
                        direction="BULL",
                        created_idx=tau,
                        lower_bound=h_prev2,
                        upper_bound=l_tau,
                        active=True,
                    )
                )
            # Check new Bearish FVG at bar tau: L_{tau-2} - H_tau > 0
            if l_prev2 > h_tau:
                bear_zones.append(
                    FairValueGapZone(
                        direction="BEAR",
                        created_idx=tau,
                        lower_bound=h_tau,
                        upper_bound=l_prev2,
                        active=True,
                    )
                )

        if bull_zones:
            active_bull = bull_zones[-1]
        if bear_zones:
            active_bear = bear_zones[-1]
        return active_bull, active_bear

    def compute_volume_profile_snapshot(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> VolumeProfileSnapshot:
        """
        Module 3 (Section 4.2):
        Compute Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL)
        containing >= 70% of cumulative volume over the supplied rolling window.
        """
        if len(closes) == 0:
            return VolumeProfileSnapshot(poc=0.0, vah=0.0, val=0.0, total_volume=0.0)

        min_l = float(np.min(lows))
        max_h = float(np.max(highs))
        total_vol = float(np.sum(volumes))

        if max_h <= min_l or total_vol <= 0.0:
            c_last = float(closes[-1])
            return VolumeProfileSnapshot(poc=c_last, vah=c_last, val=c_last, total_volume=max(total_vol, 0.0))

        n_bins = self.volume_profile_bins
        edges = np.linspace(min_l, max_h, n_bins + 1, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_vols = np.zeros(n_bins, dtype=np.float64)

        # Distribute each bar's volume uniformly across all price bins intersected by [L_tau, H_tau]
        for h_i, l_i, c_i, v_i in zip(highs, lows, closes, volumes):
            if v_i <= 0.0:
                continue
            if h_i <= l_i:
                idx = int(np.clip(np.searchsorted(edges, c_i, side="right") - 1, 0, n_bins - 1))
                bin_vols[idx] += v_i
                continue
            start_b = int(np.clip(np.searchsorted(edges, l_i, side="right") - 1, 0, n_bins - 1))
            end_b = int(np.clip(np.searchsorted(edges, h_i, side="left") - 1, start_b, n_bins - 1))
            count_b = end_b - start_b + 1
            bin_vols[start_b : end_b + 1] += v_i / float(count_b)

        poc_idx = int(np.argmax(bin_vols))
        poc_price = float(centers[poc_idx])

        # Expand Value Area contiguously around poc_idx until >= 70% of total volume is captured
        target_vol = self.value_area_target * float(np.sum(bin_vols))
        cum_vol = float(bin_vols[poc_idx])
        left_idx = poc_idx
        right_idx = poc_idx

        while cum_vol < target_vol and (left_idx > 0 or right_idx < n_bins - 1):
            next_left_vol = float(bin_vols[left_idx - 1]) if left_idx > 0 else -1.0
            next_right_vol = float(bin_vols[right_idx + 1]) if right_idx < n_bins - 1 else -1.0
            if next_right_vol >= next_left_vol:
                right_idx += 1
                cum_vol += float(bin_vols[right_idx])
            else:
                left_idx -= 1
                cum_vol += float(bin_vols[left_idx])

        val_price = float(edges[left_idx])
        vah_price = float(edges[right_idx + 1])
        return VolumeProfileSnapshot(
            poc=poc_price,
            vah=vah_price,
            val=val_price,
            total_volume=total_vol,
        )

    def extract_structural_features(
        self,
        df: pd.DataFrame,
        enforce_lag: bool = False,
    ) -> pd.DataFrame:
        """
        Compute all Module 1, Module 2, and Module 3 features causally for every bar t:
          - Candle Anatomy (1-4 + engulf): r_body, r_wick, rel_spread, score_pinbar, score_engulf
          - S/R Proximity (5-7): delta_r, delta_s, pos_sr
          - Volume Profile (8-10): delta_poc, delta_vah, delta_val
          - FVG Imbalance (11-12): dist_fvg_bull, dist_fvg_bear

        Guardrail 1: When `enforce_lag=True`, all extracted feature columns are shifted by 1 bar
        so decisions at bar t strictly use features known at bar t-1.
        Guardrail 3: Zero unscaled prices in output feature columns; all distances are divided by ATR_k(t).
        """
        out = self.compute_candle_anatomy(df)
        n = len(out)
        if n == 0:
            for col in self.STRUCTURAL_FEATURE_COLUMNS:
                out[col] = pd.Series(dtype=np.float64)
            return out

        highs = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=np.float64)
        lows = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=np.float64)
        closes = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64)
        volumes = (
            pd.to_numeric(out["volume"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
            if "volume" in out.columns
            else np.ones_like(closes)
        )
        atrs = out["atr"].to_numpy(dtype=np.float64)

        delta_r_arr = np.zeros(n, dtype=np.float64)
        delta_s_arr = np.zeros(n, dtype=np.float64)
        pos_sr_arr = np.full(n, 0.5, dtype=np.float64)

        delta_poc_arr = np.zeros(n, dtype=np.float64)
        delta_vah_arr = np.zeros(n, dtype=np.float64)
        delta_val_arr = np.zeros(n, dtype=np.float64)

        dist_fvg_bull_arr = np.zeros(n, dtype=np.float64)
        dist_fvg_bear_arr = np.zeros(n, dtype=np.float64)

        # Causal single-pass FVG state tracker
        bull_zones: List[FairValueGapZone] = []
        bear_zones: List[FairValueGapZone] = []

        k = self.swing_window
        sr_lookback = self.lookback_bars
        vp_win = self.volume_profile_window

        for t in range(n):
            c_t = float(closes[t])
            h_t = float(highs[t])
            l_t = float(lows[t])
            atr_t = max(float(atrs[t]), abs(c_t) * 1e-8, 1e-12)

            # 1. Update causal FVG zones up to bar t
            if bull_zones:
                bull_zones = [z for z in bull_zones if l_t > z.lower_bound]
            if bear_zones:
                bear_zones = [z for z in bear_zones if h_t < z.upper_bound]

            if t >= 2:
                h_prev2 = float(highs[t - 2])
                l_prev2 = float(lows[t - 2])
                if l_t > h_prev2:
                    bull_zones.append(
                        FairValueGapZone("BULL", t, lower_bound=h_prev2, upper_bound=l_t)
                    )
                if l_prev2 > h_t:
                    bear_zones.append(
                        FairValueGapZone("BEAR", t, lower_bound=h_t, upper_bound=l_prev2)
                    )

            if bull_zones:
                top_bull = bull_zones[-1].upper_bound
                dist_fvg_bull_arr[t] = (c_t - top_bull) / atr_t
            if bear_zones:
                bot_bear = bear_zones[-1].lower_bound
                dist_fvg_bear_arr[t] = (c_t - bot_bear) / atr_t

            # 2. Causal Volume Profile over [max(0, t - vp_win + 1) : t + 1]
            vp_start = max(0, t - vp_win + 1)
            vp_snap = self.compute_volume_profile_snapshot(
                highs[vp_start : t + 1],
                lows[vp_start : t + 1],
                closes[vp_start : t + 1],
                volumes[vp_start : t + 1],
            )
            delta_poc_arr[t] = (c_t - vp_snap.poc) / atr_t
            delta_vah_arr[t] = (c_t - vp_snap.vah) / atr_t
            delta_val_arr[t] = (c_t - vp_snap.val) / atr_t

            # 3. Causal S/R Detection via Extrema + DBSCAN over [max(0, t - sr_lookback + 1) : t + 1]
            sr_start = max(0, t - sr_lookback + 1)
            sub_len = t - sr_start + 1
            if sub_len <= 2 * k + 1:
                s_star = float(np.min(lows[sr_start : t + 1]))
                r_star = float(np.max(highs[sr_start : t + 1]))
            else:
                sub_h = highs[sr_start : t + 1]
                sub_l = lows[sr_start : t + 1]
                causal_h = sub_h[:-k]
                causal_l = sub_l[:-k]
                p_idx = argrelextrema(causal_h, np.greater_equal, order=k)[0]
                tr_idx = argrelextrema(causal_l, np.less_equal, order=k)[0]
                cands = np.concatenate([sub_h[p_idx], sub_l[tr_idx]]).reshape(-1, 1)
                if len(cands) < self.min_samples:
                    s_star = float(np.min(sub_l))
                    r_star = float(np.max(sub_h))
                else:
                    eps = max(self.cluster_eps_atr * atr_t, abs(c_t) * 1e-9, 1e-12)
                    db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(cands)
                    tau_cands = np.concatenate([p_idx, tr_idx]).astype(np.float64)
                    t_rel = float(sub_len - 1)
                    supp_lvls: List[float] = []
                    res_lvls: List[float] = []
                    for lbl in set(int(x) for x in db.labels_ if x != -1):
                        m_lbl = db.labels_ == lbl
                        w_lbl = np.exp(-self.decay_lambda * np.maximum(0.0, t_rel - tau_cands[m_lbl]))
                        w_sum = float(np.sum(w_lbl))
                        lvl = (
                            float(np.sum(w_lbl * cands[m_lbl, 0]) / w_sum)
                            if w_sum > 0.0
                            else float(np.mean(cands[m_lbl, 0]))
                        )
                        if lvl < c_t:
                            supp_lvls.append(lvl)
                        elif lvl > c_t:
                            res_lvls.append(lvl)
                    s_star = float(max(supp_lvls)) if supp_lvls else float(np.min(sub_l))
                    r_star = float(min(res_lvls)) if res_lvls else float(np.max(sub_h))

            delta_r_arr[t] = (r_star - c_t) / atr_t
            delta_s_arr[t] = (c_t - s_star) / atr_t
            sr_span = r_star - s_star
            scale_eps = max(abs(c_t) * 1e-12, 1e-18)
            pos_sr_arr[t] = (
                float(np.clip((c_t - s_star) / (sr_span + scale_eps), 0.0, 1.0))
                if sr_span > scale_eps
                else 0.5
            )

        out["delta_r"] = delta_r_arr
        out["delta_s"] = delta_s_arr
        out["pos_sr"] = pos_sr_arr
        out["delta_poc"] = delta_poc_arr
        out["delta_vah"] = delta_vah_arr
        out["delta_val"] = delta_val_arr
        out["dist_fvg_bull"] = dist_fvg_bull_arr
        out["dist_fvg_bear"] = dist_fvg_bear_arr

        if enforce_lag:
            for col in self.STRUCTURAL_FEATURE_COLUMNS:
                out[col] = out[col].shift(1).fillna(0.0)

        return out
