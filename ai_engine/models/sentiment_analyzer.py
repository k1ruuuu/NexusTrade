from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from ai_engine.models.sentiment_lexicon import (
    BEARISH_LEXICON,
    BOOTSTRAP_TRAINING_CORPUS,
    BULLISH_LEXICON,
    DEFAULT_ASSET_HEADLINES,
    SentimentLabel,
    classify_sentiment_polarity,
)
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from data_pipeline.live_news_feed import LiveNewsFeed, LiveNewsItem
from data_pipeline.preprocessor_sentiment import SentimentNLPPreprocessor

logger = get_logger("nexustrade.ai_engine.sentiment")

@dataclass
class HeadlineSentimentScore:
    """Individual headline polarity breakdown."""

    headline: str
    polarity: float
    label: SentimentLabel
    confidence: float
    weight: float = 1.0
    source: str = "Local"
    published_at: str = ""


@dataclass
class SentimentAnalysisResult:
    """Aggregated financial sentiment analysis output for an asset."""

    symbol: str
    s_sent: float
    label: SentimentLabel
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    model_name: str
    headline_scores: list[HeadlineSentimentScore] = field(default_factory=list)
    live_news_used: bool = False
    newly_learned_count: int = 0
    total_corpus_size: int = 0
    online_updates_count: int = 0


class FinancialSentimentAnalyzer:
    """
    FinBERT & Online Continual Learning Financial Sentiment Engine.
    Combines HuggingFace FinBERT / FinancialPhraseBank lexicon with an online
    incremental SGDClassifier (partial_fit) that continuously learns from live
    internet news and market price reactions ("Learn by Doing").
    """

    CLASSES = np.array([-1, 0, 1], dtype=np.int32)

    def __init__(
        self,
        settings: Optional[Settings] = None,
        preprocessor: Optional[SentimentNLPPreprocessor] = None,
        live_news_feed: Optional[LiveNewsFeed] = None,
        use_transformers: Optional[bool] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = preprocessor or SentimentNLPPreprocessor(self.settings)
        self.live_news_feed = live_news_feed or LiveNewsFeed(self.settings, self.preprocessor)
        self.model_name = self._normalize_model_name(self.settings.finbert_model_name)
        self.use_transformers = (
            use_transformers
            if use_transformers is not None
            else self.settings.use_transformers_pipeline
        )
        self._hf_pipeline: Optional[Any] = None

        # Online incremental learning components (Learn-by-Doing)
        self.vectorizer = HashingVectorizer(
            n_features=4096,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
        )
        self.online_classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            random_state=42,
        )
        self._seen_headline_ids: set[str] = set()
        self.online_updates_count: int = 0
        self.corpus_path: Path = self.settings.processed_data_dir / "online_news_corpus.parquet"
        self._initialize_online_learner()

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        if name.strip().lower() in ("prosusde/finbert", "prosusai/finbert"):
            return "ProsusAI/finbert"
        return name.strip()

    def _initialize_online_learner(self) -> None:
        """Bootstrap incremental SGDClassifier on seed corpus and persisted online news Parquet."""
        texts = [self.preprocessor.clean_text(t) for t, _ in BOOTSTRAP_TRAINING_CORPUS]
        labels = np.array([lbl for _, lbl in BOOTSTRAP_TRAINING_CORPUS], dtype=np.int32)
        x_boot = self.vectorizer.transform(texts)
        self.online_classifier.partial_fit(x_boot, labels, classes=self.CLASSES)
        self.online_updates_count = 1

        if self.corpus_path.exists():
            try:
                stored_df = pd.read_parquet(self.corpus_path)
                if not stored_df.empty and "headline" in stored_df.columns and "learned_label" in stored_df.columns:
                    stored_texts = stored_df["headline"].astype(str).tolist()
                    stored_labels = stored_df["learned_label"].astype(np.int32).to_numpy()
                    x_stored = self.vectorizer.transform(stored_texts)
                    self.online_classifier.partial_fit(x_stored, stored_labels, classes=self.CLASSES)
                    if "headline_id" in stored_df.columns:
                        self._seen_headline_ids.update(stored_df["headline_id"].astype(str).tolist())
                    self.online_updates_count += 1
            except Exception as exc:
                logger.warning("Failed to load persisted online news corpus", error=str(exc))

    def _get_hf_pipeline(self) -> Optional[Any]:
        if not self.use_transformers:
            return None
        if self._hf_pipeline is not None:
            return self._hf_pipeline
        try:
            from transformers import pipeline

            self._hf_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,
            )
            return self._hf_pipeline
        except Exception as exc:
            logger.warning(
                "FinBERT transformers pipeline unavailable; using calibrated financial NLP engine",
                model=self.model_name,
                error=str(exc),
            )
            self.use_transformers = False
            return None

    def _compute_lexicon_polarity(self, tokens: list[str]) -> tuple[float, float, int]:
        """Return (lexicon_polarity, confidence, matched_count)."""
        raw_sum = 0.0
        matched_count = 0
        negation_active = False

        for tok in tokens:
            if tok in ("not", "no", "never", "neither", "without", "fails", "failed"):
                negation_active = True
                continue

            if tok in BULLISH_LEXICON:
                val = BULLISH_LEXICON[tok]
                raw_sum += -val * 0.8 if negation_active else val
                matched_count += 1
                negation_active = False
            elif tok in BEARISH_LEXICON:
                val = BEARISH_LEXICON[tok]
                raw_sum += -val * 0.6 if negation_active else val
                matched_count += 1
                negation_active = False

        if matched_count == 0:
            return 0.0, 0.55, 0
        polarity = float(np.tanh(raw_sum / np.sqrt(matched_count)))
        confidence = float(min(0.98, 0.60 + 0.38 * abs(polarity)))
        return polarity, confidence, matched_count

    def _compute_online_model_polarity(self, cleaned_text: str) -> float:
        """Predict polarity in [-1.0, +1.0] from the online incremental SGDClassifier."""
        x_vec = self.vectorizer.transform([cleaned_text])
        probs = self.online_classifier.predict_proba(x_vec)[0]
        # Classes are [-1, 0, 1] -> probs[2] is P(+1), probs[0] is P(-1)
        return float(np.clip(probs[2] - probs[0], -1.0, 1.0))

    def score_headline(
        self,
        headline: str,
        weight: float = 1.0,
        source: str = "Local",
        published_at: str = "",
    ) -> HeadlineSentimentScore:
        """Compute sentiment polarity score in [-1.0, +1.0] blending Lexicon/FinBERT and Online Learner."""
        tokenized = self.preprocessor.tokenize_text(headline)
        if not tokenized.cleaned_text:
            return HeadlineSentimentScore(
                headline=headline,
                polarity=0.0,
                label="NEUTRAL",
                confidence=0.5,
                weight=weight,
                source=source,
                published_at=published_at,
            )

        hf_pipe = self._get_hf_pipeline()
        if hf_pipe is not None:
            raw_preds = hf_pipe(tokenized.cleaned_text[:512])[0]
            scores_map = {item["label"].lower(): float(item["score"]) for item in raw_preds}
            p_pos = scores_map.get("positive", 0.0)
            p_neg = scores_map.get("negative", 0.0)
            p_neu = scores_map.get("neutral", 0.0)
            polarity = float(np.clip(p_pos - p_neg, -1.0, 1.0))
            confidence = max(p_pos, p_neg, p_neu)
        else:
            lex_pol, lex_conf, matched = self._compute_lexicon_polarity(tokenized.tokens)
            online_pol = self._compute_online_model_polarity(tokenized.cleaned_text)
            if matched > 0:
                polarity = float(np.clip(0.75 * lex_pol + 0.25 * online_pol, -1.0, 1.0))
                confidence = lex_conf
            else:
                # When a headline contains novel vocabulary not in static lexicon, rely on online learned model
                polarity = float(np.clip(online_pol * 0.65, -1.0, 1.0))
                confidence = float(min(0.90, 0.55 + 0.35 * abs(polarity)))

        label = classify_sentiment_polarity(polarity)

        return HeadlineSentimentScore(
            headline=tokenized.cleaned_text,
            polarity=round(polarity, 4),
            label=label,
            confidence=round(confidence, 4),
            weight=round(float(weight), 4),
            source=source,
            published_at=published_at,
        )

    @staticmethod
    def _estimate_market_reaction_return(
        published_at_iso: str,
        market_df: Optional[pd.DataFrame],
    ) -> float:
        """
        Compute realized market return following headline publication timestamp
        to provide real-world price-action feedback for online learning.
        """
        if market_df is None or market_df.empty or "close" not in market_df.columns:
            return 0.0
        try:
            pub_ts = pd.to_datetime(published_at_iso, utc=True)
            idx = market_df.index
            if not isinstance(idx, pd.DatetimeIndex):
                return 0.0
            after_bars = market_df.loc[idx >= pub_ts]
            if len(after_bars) >= 2:
                horizon_bars = after_bars.iloc[:5]
                entry_p = float(horizon_bars["open"].iloc[0])
                latest_p = float(horizon_bars["close"].iloc[-1])
                if entry_p > 0:
                    return float((latest_p - entry_p) / entry_p)
        except Exception:
            return 0.0
        return 0.0

    def ingest_and_learn_live_news(
        self,
        symbol: str = "BTCUSDT",
        market_df: Optional[pd.DataFrame] = None,
        force_refresh: bool = False,
    ) -> tuple[list[LiveNewsItem], int]:
        """
        Harvest latest breaking news from the internet for `symbol`, label unseen
        headlines via teacher NLP + realized market return feedback ("Learn by Doing"),
        update the online SGDClassifier with partial_fit, and persist to Parquet.
        """
        live_items = self.live_news_feed.fetch_live_news(
            symbol=symbol, max_items=30, force_refresh=force_refresh, lookback_days=90
        )
        if not live_items or not self.settings.online_learning_enabled:
            return live_items, 0

        new_items = [
            item for item in live_items if item.headline_id not in self._seen_headline_ids
        ]
        if not new_items:
            return live_items, 0

        new_texts: list[str] = []
        new_labels: list[int] = []
        new_records: list[dict[str, Any]] = []

        for item in new_items:
            tokenized = self.preprocessor.tokenize_text(item.headline)
            lex_pol, _, _ = self._compute_lexicon_polarity(tokenized.tokens)
            mkt_ret = self._estimate_market_reaction_return(item.published_at, market_df)
            mkt_signal = float(np.tanh(mkt_ret * 120.0))

            # Blend teacher NLP polarity (70%) with realized market price reaction (30%)
            combined_signal = 0.70 * lex_pol + 0.30 * mkt_signal
            if combined_signal >= 0.12:
                learned_label = 1
            elif combined_signal <= -0.12:
                learned_label = -1
            else:
                learned_label = 0

            new_texts.append(tokenized.cleaned_text)
            new_labels.append(learned_label)
            self._seen_headline_ids.add(item.headline_id)
            new_records.append(
                {
                    "headline_id": item.headline_id,
                    "symbol": item.symbol,
                    "headline": tokenized.cleaned_text,
                    "source": item.source,
                    "published_at": item.published_at,
                    "teacher_polarity": round(lex_pol, 4),
                    "market_return_feedback": round(mkt_ret, 6),
                    "learned_label": learned_label,
                    "learned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        x_new = self.vectorizer.transform(new_texts)
        y_new = np.asarray(new_labels, dtype=np.int32)
        self.online_classifier.partial_fit(x_new, y_new, classes=self.CLASSES)
        self.online_updates_count += 1

        try:
            new_df = pd.DataFrame(new_records)
            if self.corpus_path.exists():
                existing_df = pd.read_parquet(self.corpus_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["headline_id"], keep="last")
            else:
                combined_df = new_df
            self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_parquet(self.corpus_path, engine="pyarrow", index=False)
        except Exception as exc:
            logger.warning("Could not persist online news corpus to Parquet", error=str(exc))

        logger.info(
            "Online sentiment learner updated from live internet news",
            symbol=symbol.upper(),
            newly_learned=len(new_items),
            total_corpus_size=len(self._seen_headline_ids),
            online_updates=self.online_updates_count,
        )
        return live_items, len(new_items)

    def get_default_headlines_for_symbol(self, symbol: str) -> list[str]:
        upper = symbol.upper()
        for prefix, headlines in DEFAULT_ASSET_HEADLINES.items():
            if prefix in upper:
                return list(headlines)
        return [
            f"Institutional order flow in {upper} shows constructive momentum and steady liquidity",
            f"Options market positioning for {upper} reflects balanced hedging near key technical levels",
            f"Sector allocation models maintain constructive overweight rating on {upper}",
        ]

    def analyze_headlines(
        self,
        headlines: Optional[list[str]] = None,
        symbol: str = "BTCUSDT",
        weights: Optional[list[float]] = None,
        market_df: Optional[pd.DataFrame] = None,
    ) -> SentimentAnalysisResult:
        """
        Score a batch of news headlines (or harvest & learn live internet news when
        headlines is None) and return unified polarity S_sent in [-1.0, +1.0].
        """
        live_used = False
        newly_learned = 0
        live_items: list[LiveNewsItem] = []

        if headlines is None and self.settings.use_live_news_feed:
            live_items, newly_learned = self.ingest_and_learn_live_news(
                symbol=symbol, market_df=market_df
            )
            if live_items:
                live_used = True

        if live_used and live_items:
            # Weight recent breaking news higher while preserving 3-month (90-day) macro news tail
            eff_weights = [
                float(max(0.18, np.exp(-0.10 * idx))) for idx in range(len(live_items))
            ]
            scored_items = [
                self.score_headline(
                    item.headline,
                    weight=w,
                    source=item.source,
                    published_at=item.published_at,
                )
                for item, w in zip(live_items, eff_weights)
            ]
        else:
            if headlines is not None:
                effective_headlines = list(headlines)
            elif self.settings.use_live_news_feed:
                effective_headlines = []
            else:
                effective_headlines = self.get_default_headlines_for_symbol(symbol)
            if not effective_headlines:
                return SentimentAnalysisResult(
                    symbol=symbol.upper(),
                    s_sent=0.0,
                    label="NEUTRAL",
                    bullish_ratio=0.0,
                    bearish_ratio=0.0,
                    neutral_ratio=1.0,
                    model_name=self.model_name,
                    headline_scores=[],
                    live_news_used=False,
                    newly_learned_count=0,
                    total_corpus_size=len(self._seen_headline_ids),
                    online_updates_count=self.online_updates_count,
                )

            if weights is None or len(weights) != len(effective_headlines):
                eff_weights = [float(np.exp(-0.18 * idx)) for idx in range(len(effective_headlines))]
            else:
                eff_weights = [max(0.0, float(w)) for w in weights]

            scored_items = [
                self.score_headline(text, weight=w)
                for text, w in zip(effective_headlines, eff_weights)
            ]

        total_weight = sum(item.weight for item in scored_items) or 1.0
        weighted_polarity = sum(item.polarity * item.weight for item in scored_items) / total_weight
        s_sent = float(np.clip(weighted_polarity, -1.0, 1.0))

        n_items = len(scored_items)
        bullish_ratio = sum(1 for item in scored_items if item.label == "POSITIVE") / n_items
        bearish_ratio = sum(1 for item in scored_items if item.label == "NEGATIVE") / n_items
        neutral_ratio = max(0.0, 1.0 - bullish_ratio - bearish_ratio)

        agg_label = classify_sentiment_polarity(s_sent)

        return SentimentAnalysisResult(
            symbol=symbol.upper(),
            s_sent=round(s_sent, 4),
            label=agg_label,
            bullish_ratio=round(bullish_ratio, 4),
            bearish_ratio=round(bearish_ratio, 4),
            neutral_ratio=round(neutral_ratio, 4),
            model_name=f"{self.model_name} + OnlineSGD",
            headline_scores=scored_items,
            live_news_used=live_used,
            newly_learned_count=newly_learned,
            total_corpus_size=len(self._seen_headline_ids),
            online_updates_count=self.online_updates_count,
        )

