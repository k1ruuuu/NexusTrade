from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence
import warnings

from hmmlearn import hmm
import numpy as np
import pandas as pd

from src.features.structure_extractor import CandleStructuralExtractor

RegimeLabel = Literal[
    "Risk-On Expansion",
    "Mean-Reverting / Consolidation",
    "Liquidity Contraction / Crisis",
]


@dataclass(frozen=True)
class MacroRegimeResult:
    """Runtime output of the 3-state Global Macro HMM Regime Engine."""

    prob_bull_expansion: float  # State 0: P(z_t = 0 | y_{1:t})
    prob_consolidation: float   # State 1: P(z_t = 1 | y_{1:t})
    prob_crisis_bear: float     # State 2: P(z_t = 2 | y_{1:t})
    roro_index: float
    dominant_state: int
    regime_label: RegimeLabel


class MacroRegimeDetector:
    """
    Module 4 (Sections 5.1 - 5.3 & 7.2):
    Global Macro Regime & Intermarket Prediction Engine using a 3-state Gaussian HMM:
      - State 0: Risk-On Expansion (Bullish Trend) — High positive mean return, low-to-moderate variance
      - State 1: Mean-Reverting / Consolidation (Chop) — Near-zero mean return, lowest variance
      - State 2: Liquidity Contraction / Crisis (Bearish / High Vol) — Negative mean return, elevated variance

    All inputs (log returns, realized volatility, rolling 60-bar RoRo z-scores) are strictly
    scale-invariant under scalar price multiplication c * P_t.
    """

    RORO_WEIGHTS: tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15)
    REGIME_NAMES: dict[int, RegimeLabel] = {
        0: "Risk-On Expansion",
        1: "Mean-Reverting / Consolidation",
        2: "Liquidity Contraction / Crisis",
    }

    def __init__(
        self,
        n_regimes: int = 3,
        zscore_window: int = 60,
        corr_window: int = 30,
    ) -> None:
        self.n_regimes = max(int(n_regimes), 2)
        self.zscore_window = max(int(zscore_window), 10)
        self.corr_window = max(int(corr_window), 5)
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            min_covar=1e-4,
        )
        self._state_permutation: np.ndarray = np.arange(self.n_regimes, dtype=np.int64)
        self._is_fitted: bool = False

    @staticmethod
    def compute_log_returns(prices: pd.Series | np.ndarray) -> pd.Series:
        """Compute causal log returns r_t = ln(P_t / P_{t-1})."""
        s = pd.Series(prices, dtype=np.float64).replace(0.0, np.nan)
        ratio = (s / s.shift(1)).replace([0.0, np.inf, -np.inf], np.nan)
        return np.log(ratio).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
        """Compute causal rolling z-score z(x_t) = (x_t - mu_x) / sigma_x over `window` bars."""
        s = pd.Series(series, dtype=np.float64).fillna(0.0)
        mean = s.rolling(window=window, min_periods=2).mean()
        std = s.rolling(window=window, min_periods=2).std(ddof=0)
        eps = np.maximum(std.abs() * 1e-9, 1e-12)
        z = np.where(std > eps, (s - mean) / (std + eps), 0.0)
        return pd.Series(np.clip(z, -5.0, 5.0), index=s.index, dtype=np.float64)

    def compute_intermarket_correlations(
        self,
        target_prices: pd.Series,
        benchmarks: Dict[str, pd.Series],
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Section 5.1:
        Compute moving Pearson correlation vector rho_{i, t} between log returns of
        target trading asset P_target and benchmark series (SPX, DXY, US10Y, VIX).
        """
        win = max(int(window if window is not None else self.corr_window), 5)
        r_target = self.compute_log_returns(target_prices)
        out = pd.DataFrame(index=r_target.index)

        for name, bench_prices in benchmarks.items():
            aligned_bench = pd.Series(bench_prices, index=r_target.index, dtype=np.float64)
            r_bench = self.compute_log_returns(aligned_bench)
            corr = r_target.rolling(window=win, min_periods=5).corr(r_bench)
            out[f"corr_{name.lower()}"] = corr.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)

        return out

    def compute_roro_index(
        self,
        df: pd.DataFrame,
        intermarket_df: Optional[pd.DataFrame] = None,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Section 5.2:
        Construct Macro Risk-On / Risk-Off (RoRo) Composite Index:
          RoRo_t = 0.35 * z(dSPX_t) - 0.25 * z(dDXY_t) - 0.25 * z(dVIX_t) + 0.15 * z(dEM_t)
        When external intermarket series are not supplied, derives the 4 orthogonal macro proxies
        causally from the asset's scale-invariant log returns and realized volatility shocks.
        """
        win = max(int(window if window is not None else self.zscore_window), 10)
        w1, w2, w3, w4 = self.RORO_WEIGHTS

        if intermarket_df is not None and all(
            col in intermarket_df.columns for col in ("SPX", "DXY", "VIX", "EM")
        ):
            d_spx = self.compute_log_returns(intermarket_df["SPX"])
            d_dxy = self.compute_log_returns(intermarket_df["DXY"])
            d_vix = self.compute_log_returns(intermarket_df["VIX"])
            d_em = self.compute_log_returns(intermarket_df["EM"])
        else:
            close = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
            high = pd.to_numeric(df["high"], errors="coerce").astype(np.float64)
            low = pd.to_numeric(df["low"], errors="coerce").astype(np.float64)

            log_ret_1 = self.compute_log_returns(close)
            log_ret_5 = log_ret_1.rolling(window=5, min_periods=1).sum()
            hl_vol = np.log((high / low.replace(0.0, np.nan)).replace(0.0, np.nan)).fillna(0.0)
            rv_10 = log_ret_1.rolling(window=10, min_periods=2).std(ddof=0).fillna(0.0)

            # Scale-invariant internal proxies when external benchmark feed is absent
            d_spx = log_ret_5
            d_dxy = -log_ret_1.rolling(window=10, min_periods=1).mean()
            d_vix = hl_vol.diff(1).fillna(0.0) + rv_10.diff(1).fillna(0.0)
            d_em = log_ret_1.ewm(span=8, adjust=False).mean()

        z_spx = self.rolling_zscore(d_spx, window=win)
        z_dxy = self.rolling_zscore(d_dxy, window=win)
        z_vix = self.rolling_zscore(d_vix, window=win)
        z_em = self.rolling_zscore(d_em, window=win)

        roro = (w1 * z_spx) - (w2 * z_dxy) - (w3 * z_vix) + (w4 * z_em)
        return pd.Series(roro, index=df.index, dtype=np.float64).fillna(0.0)

    def build_observation_matrix(
        self,
        df: pd.DataFrame,
        intermarket_df: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, pd.Series]:
        """
        Section 5.3:
        Construct observation matrix Y = [r_{target, t}, sigma_{realized, t}, RoRo_t]
        where every column is strictly dimensionless and invariant to c * P_t.
        """
        close = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
        r_target = self.compute_log_returns(close)
        sigma_realized = (
            r_target.rolling(window=14, min_periods=2)
            .std(ddof=0)
            .fillna(r_target.abs())
            .fillna(1e-4)
            .clip(lower=1e-6)
        )
        roro = self.compute_roro_index(df, intermarket_df=intermarket_df)

        obs = np.column_stack(
            [
                r_target.to_numpy(dtype=np.float64),
                sigma_realized.to_numpy(dtype=np.float64),
                roro.to_numpy(dtype=np.float64),
            ]
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, roro

    def fit_regimes(self, returns_matrix: np.ndarray) -> MacroRegimeDetector:
        """
        Section 7.2 Interface:
        Fits 3-state Gaussian HMM on historical intermarket/observation returns.
        returns_matrix shape: [T_samples, N_features]
        Canonicalizes hidden states so:
          - State 0 = Risk-On Expansion (Bullish)
          - State 1 = Mean-Reverting / Consolidation (Chop)
          - State 2 = Liquidity Contraction / Crisis (Bearish / High Vol)
        """
        x_mat = np.asarray(returns_matrix, dtype=np.float64)
        if x_mat.ndim == 1:
            x_mat = x_mat.reshape(-1, 1)
        x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples, n_features = x_mat.shape
        if n_samples < self.n_regimes * 4:
            self._is_fitted = False
            self._state_permutation = np.arange(self.n_regimes, dtype=np.int64)
            return self

        # Add tiny deterministic jitter only if a column has zero variance
        col_std = np.std(x_mat, axis=0)
        if np.any(col_std < 1e-10):
            rng = np.random.default_rng(42)
            jitter = rng.normal(0.0, 1e-6, size=x_mat.shape)
            x_fit = x_mat + jitter
        else:
            x_fit = x_mat

        fitted_model = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cand = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=200,
                    random_state=42,
                    min_covar=1e-4,
                )
                cand.fit(x_fit)
                if hasattr(cand, "means_"):
                    fitted_model = cand
            except Exception:
                pass
            if fitted_model is None:
                try:
                    cand = hmm.GaussianHMM(
                        n_components=self.n_regimes,
                        covariance_type="diag",
                        n_iter=200,
                        random_state=42,
                        min_covar=1e-3,
                    )
                    cand.fit(x_fit)
                    if hasattr(cand, "means_"):
                        fitted_model = cand
                except Exception:
                    pass

        if fitted_model is None or not hasattr(fitted_model, "means_"):
            self._is_fitted = False
            return self

        self.model = fitted_model
        self._is_fitted = True
        # Canonical state ordering:
        # Score each state by (mean_return + 0.25 * mean_roro - 0.50 * mean_vol)
        means = self.model.means_  # shape [K, D]
        scores = np.zeros(self.n_regimes, dtype=np.float64)
        for k in range(self.n_regimes):
            mean_ret = float(means[k, 0])
            mean_vol = float(means[k, 1]) if n_features >= 2 else float(np.trace(self.model.covars_[k]))
            mean_roro = float(means[k, 2]) if n_features >= 3 else 0.0
            scores[k] = mean_ret + 0.25 * mean_roro - 0.50 * mean_vol

        # Highest score -> State 0 (Bullish), Middle -> State 1 (Chop), Lowest -> State 2 (Bearish/Crisis)
        desc_order = np.argsort(scores)[::-1]
        self._state_permutation = desc_order.astype(np.int64)
        return self

    def _reorder_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Permute raw HMM state probabilities into canonical [Bullish(0), Chop(1), Bearish(2)] order."""
        reordered = raw_probs[..., self._state_permutation]
        row_sums = np.sum(reordered, axis=-1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        return reordered / row_sums

    def predict_regime_probabilities(self, current_features: np.ndarray) -> np.ndarray:
        """
        Section 7.2 Interface:
        Returns continuous posterior probabilities for [Bullish, Choppy, Bearish] at the latest step
        using the Forward algorithm over the supplied observation sequence.
        """
        seq_probs = self.filter_regime_probabilities_sequence(current_features)
        return seq_probs[-1]

    def filter_regime_probabilities_sequence(self, features_matrix: np.ndarray) -> np.ndarray:
        """
        Section 5.3 Runtime Inference:
        Causal Forward filtering recursion computing:
          pi_t = [P(z_t = 0 | y_{1:t}), P(z_t = 1 | y_{1:t}), P(z_t = 2 | y_{1:t})]
        at every timestep t = 0 .. T-1 without looking ahead to t+1 .. T.
        """
        x_mat = np.asarray(features_matrix, dtype=np.float64)
        if x_mat.ndim == 1:
            x_mat = x_mat.reshape(1, -1)
        x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)
        n_steps = len(x_mat)

        if n_steps == 0:
            return np.full((1, self.n_regimes), 1.0 / self.n_regimes, dtype=np.float64)

        if not self._is_fitted:
            if n_steps >= self.n_regimes * 4:
                self.fit_regimes(x_mat)
            else:
                # Deterministic analytical softmax fallback for very short sequences
                out_short = np.zeros((n_steps, self.n_regimes), dtype=np.float64)
                for t in range(n_steps):
                    ret_t = float(x_mat[t, 0])
                    vol_t = float(x_mat[t, 1]) if x_mat.shape[1] >= 2 else 0.01
                    roro_t = float(x_mat[t, 2]) if x_mat.shape[1] >= 3 else 0.0
                    logits = np.array(
                        [
                            25.0 * ret_t + 0.6 * roro_t,
                            -10.0 * abs(ret_t) - 5.0 * vol_t,
                            -25.0 * ret_t - 0.6 * roro_t + 8.0 * vol_t,
                        ],
                        dtype=np.float64,
                    )
                    logits -= np.max(logits)
                    exp_l = np.exp(logits)
                    out_short[t] = exp_l / np.sum(exp_l)
                return out_short

        # Compute log emission likelihoods log P(y_t | z_t = k) via fitted GaussianHMM
        log_prob = self.model._compute_log_likelihood(x_mat)  # shape [T, K]
        startprob = np.clip(self.model.startprob_, 1e-12, 1.0)
        startprob = startprob / np.sum(startprob)
        transmat = np.clip(self.model.transmat_, 1e-12, 1.0)
        transmat = transmat / np.sum(transmat, axis=1, keepdims=True)

        filtered_raw = np.zeros((n_steps, self.n_regimes), dtype=np.float64)

        # Forward recursion step t = 0
        log_alpha = np.log(startprob) + log_prob[0]
        log_alpha -= np.max(log_alpha)
        alpha = np.exp(log_alpha)
        alpha /= np.sum(alpha)
        filtered_raw[0] = alpha

        # Forward recursion steps t = 1 .. T - 1 (strictly causal: uses only y_{0:t})
        for t in range(1, n_steps):
            prior_t = np.dot(alpha, transmat)
            prior_t = np.clip(prior_t, 1e-15, 1.0)
            log_alpha = np.log(prior_t) + log_prob[t]
            log_alpha -= np.max(log_alpha)
            alpha = np.exp(log_alpha)
            alpha /= np.sum(alpha)
            filtered_raw[t] = alpha

        return self._reorder_probs(filtered_raw)

    def detect_regime_series(
        self,
        df: pd.DataFrame,
        intermarket_df: Optional[pd.DataFrame] = None,
        enforce_lag: bool = False,
    ) -> pd.DataFrame:
        """
        Compute causal HMM posterior probabilities [regime_prob_bull, regime_prob_chop, regime_prob_bear]
        and Macro Risk-On/Risk-Off Composite Index [roro_index] for every row in `df`.
        """
        out = df.copy()
        if len(out) == 0:
            for col in ("regime_prob_bull", "regime_prob_chop", "regime_prob_bear", "roro_index"):
                out[col] = pd.Series(dtype=np.float64)
            return out

        obs_mat, roro = self.build_observation_matrix(out, intermarket_df=intermarket_df)
        if not self._is_fitted and len(obs_mat) >= self.n_regimes * 4:
            # Fit on initial training portion or full history prior to forward filtering
            self.fit_regimes(obs_mat)

        probs = self.filter_regime_probabilities_sequence(obs_mat)
        out["regime_prob_bull"] = probs[:, 0]
        out["regime_prob_chop"] = probs[:, 1]
        out["regime_prob_bear"] = probs[:, 2]
        out["roro_index"] = roro.to_numpy(dtype=np.float64)

        if enforce_lag:
            out["regime_prob_bull"] = out["regime_prob_bull"].shift(1).fillna(1.0 / 3.0)
            out["regime_prob_chop"] = out["regime_prob_chop"].shift(1).fillna(1.0 / 3.0)
            out["regime_prob_bear"] = out["regime_prob_bear"].shift(1).fillna(1.0 / 3.0)
            out["roro_index"] = out["roro_index"].shift(1).fillna(0.0)

        return out

    def evaluate_latest(
        self,
        df: pd.DataFrame,
        intermarket_df: Optional[pd.DataFrame] = None,
    ) -> MacroRegimeResult:
        """Evaluate the latest bar's macro regime state and posterior probability simplex."""
        enriched = self.detect_regime_series(df, intermarket_df=intermarket_df, enforce_lag=False)
        if len(enriched) == 0:
            return MacroRegimeResult(
                prob_bull_expansion=1.0 / 3.0,
                prob_consolidation=1.0 / 3.0,
                prob_crisis_bear=1.0 / 3.0,
                roro_index=0.0,
                dominant_state=1,
                regime_label="Mean-Reverting / Consolidation",
            )

        p0 = float(enriched["regime_prob_bull"].iloc[-1])
        p1 = float(enriched["regime_prob_chop"].iloc[-1])
        p2 = float(enriched["regime_prob_bear"].iloc[-1])
        roro_val = float(enriched["roro_index"].iloc[-1])
        probs = [p0, p1, p2]
        dom = int(np.argmax(probs))
        return MacroRegimeResult(
            prob_bull_expansion=round(p0, 6),
            prob_consolidation=round(p1, 6),
            prob_crisis_bear=round(p2, 6),
            roro_index=round(roro_val, 4),
            dominant_state=dom,
            regime_label=self.REGIME_NAMES.get(dom, "Mean-Reverting / Consolidation"),
        )


class UnifiedFeaturePipeline:
    """
    Section 6:
    Constructs the 18-dimensional Unified State Vector X_t in R^18 combining:
      - Dimensions 1-4:   Candle Anatomy (r_body, r_wick, rel_spread, score_pinbar)
      - Dimensions 5-7:   Support / Resistance (delta_r, delta_s, pos_sr)
      - Dimensions 8-10:  Market Structure Volume Profile (delta_poc, delta_vah, delta_val)
      - Dimensions 11-12: Fair Value Gaps (dist_fvg_bull, dist_fvg_bear)
      - Dimensions 13-15: Macro Regimes HMM Posteriors (regime_prob_bull, regime_prob_chop, regime_prob_bear)
      - Dimension 16:     Intermarket Liquidity (roro_index)
      - Dimension 17:     Sentiment Polarity (s_sent)
      - Dimension 18:     Log-Transformed Realized Volatility (log_vol_optiver)
    """

    UNIFIED_VECTOR_COLUMNS: tuple[str, ...] = (
        "r_body",
        "r_wick",
        "rel_spread",
        "score_pinbar",
        "delta_r",
        "delta_s",
        "pos_sr",
        "delta_poc",
        "delta_vah",
        "delta_val",
        "dist_fvg_bull",
        "dist_fvg_bear",
        "regime_prob_bull",
        "regime_prob_chop",
        "regime_prob_bear",
        "roro_index",
        "s_sent",
        "log_vol_optiver",
    )

    def __init__(
        self,
        structural_extractor: Optional[CandleStructuralExtractor] = None,
        macro_detector: Optional[MacroRegimeDetector] = None,
    ) -> None:
        self.structural_extractor = structural_extractor or CandleStructuralExtractor()
        self.macro_detector = macro_detector or MacroRegimeDetector()

    def transform(
        self,
        df: pd.DataFrame,
        intermarket_df: Optional[pd.DataFrame] = None,
        s_sent: float = 0.0,
        optiver_sigma: Optional[float] = None,
        enforce_lag: bool = True,
    ) -> pd.DataFrame:
        """
        Transform raw OHLCV (+ optional intermarket data) into a DataFrame containing
        all 18 Unified Feature Vector columns with strict shift-by-1 causality when `enforce_lag=True`.
        """
        struct_df = self.structural_extractor.extract_structural_features(
            df, enforce_lag=enforce_lag
        )
        macro_df = self.macro_detector.detect_regime_series(
            struct_df, intermarket_df=intermarket_df, enforce_lag=enforce_lag
        )

        macro_df["s_sent"] = float(np.clip(s_sent, -1.0, 1.0))

        close = pd.to_numeric(macro_df["close"], errors="coerce").astype(np.float64)
        log_ret = MacroRegimeDetector.compute_log_returns(close)
        rv_20 = (
            log_ret.rolling(window=20, min_periods=2)
            .std(ddof=0)
            .fillna(1e-3)
            .clip(lower=1e-6)
        )
        if optiver_sigma is not None and optiver_sigma > 0:
            sigma_series = pd.Series(float(optiver_sigma), index=macro_df.index, dtype=np.float64)
        else:
            sigma_series = rv_20 if not enforce_lag else rv_20.shift(1).fillna(1e-3)

        macro_df["log_vol_optiver"] = np.log(np.maximum(sigma_series, 1e-6))
        return macro_df

    def extract_latest_vector(
        self,
        df: pd.DataFrame,
        intermarket_df: Optional[pd.DataFrame] = None,
        s_sent: float = 0.0,
        optiver_sigma: Optional[float] = None,
        enforce_lag: bool = False,
    ) -> np.ndarray:
        """Return the 18-dimensional vector x_t in R^18 for the latest bar."""
        transformed = self.transform(
            df=df,
            intermarket_df=intermarket_df,
            s_sent=s_sent,
            optiver_sigma=optiver_sigma,
            enforce_lag=enforce_lag,
        )
        latest = transformed[list(self.UNIFIED_VECTOR_COLUMNS)].iloc[-1]
        return latest.to_numpy(dtype=np.float64)
