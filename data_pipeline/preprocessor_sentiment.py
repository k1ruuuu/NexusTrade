from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.data_pipeline.sentiment")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
TICKER_CASH_PATTERN = re.compile(r"\$([A-Za-z]{1,6})\b")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+(?:\.[0-9]+)?%?|\$[A-Za-z]+")

LABEL_TO_POLARITY: dict[str, int] = {
    "positive": 1,
    "pos": 1,
    "bullish": 1,
    "1": 1,
    "neutral": 0,
    "neu": 0,
    "0": 0,
    "negative": -1,
    "neg": -1,
    "bearish": -1,
    "-1": -1,
}


@dataclass(frozen=True)
class TokenizedHeadline:
    """Cleaned and tokenized financial headline ready for transformer inference."""

    raw_text: str
    cleaned_text: str
    tokens: list[str]
    input_ids: list[int]
    attention_mask: list[int]
    tickers: list[str]
    label: Optional[int] = None


class SentimentNLPPreprocessor:
    """
    FinancialPhraseBank and Massive Stock News DB cleaner, normalizer, and tokenizer.
    Prepares labeled and unlabeled financial news text for FinBERT inference.
    """

    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    PAD_TOKEN_ID = 0
    VOCAB_MODULO = 28996

    def __init__(self, settings: Optional[Settings] = None, max_length: int = 64) -> None:
        self.settings = settings or get_settings()
        self.max_length = max_length

    @staticmethod
    def extract_tickers(text: str) -> list[str]:
        """Extract explicit cashtags ($AAPL, $BTC) and known asset symbols from headline."""
        if not text:
            return []
        cashtags = [m.upper() for m in TICKER_CASH_PATTERN.findall(text)]
        known_keywords = {
            "BITCOIN": "BTC",
            "BTC": "BTC",
            "ETHEREUM": "ETH",
            "ETH": "ETH",
            "SOLANA": "SOL",
            "SOL": "SOL",
            "GOLD": "XAUUSD",
            "XAUUSD": "XAUUSD",
            "XAU": "XAUUSD",
            "AAPL": "AAPL",
            "NVDA": "NVDA",
            "SPX": "SPX",
        }
        upper_words = set(re.findall(r"\b[A-Z]{3,10}\b", text.upper()))
        for word in upper_words:
            if word in known_keywords:
                cashtags.append(known_keywords[word])
        return sorted(set(cashtags))

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove URLs, HTML tags, normalize cashtags and whitespace."""
        if not isinstance(text, str):
            return ""
        cleaned = HTML_TAG_PATTERN.sub(" ", text)
        cleaned = URL_PATTERN.sub(" ", cleaned)
        cleaned = TICKER_CASH_PATTERN.sub(lambda m: f" {m.group(1).upper()} ", cleaned)
        cleaned = re.sub(r"[^\w\s\.\,\%\-\+\$]", " ", cleaned)
        cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    @classmethod
    def normalize_label(cls, raw_label: object) -> int:
        """Map string or numeric sentiment label into {-1, 0, +1}."""
        if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
            return 0
        key = str(raw_label).strip().lower().lstrip("@")
        if key in LABEL_TO_POLARITY:
            return LABEL_TO_POLARITY[key]
        try:
            numeric = float(key)
            if numeric > 0.15:
                return 1
            if numeric < -0.15:
                return -1
            return 0
        except ValueError:
            return 0

    def tokenize_text(
        self,
        text: str,
        label: Optional[object] = None,
    ) -> TokenizedHeadline:
        """Tokenize cleaned text into tokens, deterministic input_ids, and attention_mask."""
        cleaned = self.clean_text(text)
        tickers = self.extract_tickers(text)
        raw_tokens = [tok.lower() for tok in TOKEN_PATTERN.findall(cleaned)]
        truncated_tokens = raw_tokens[: max(1, self.max_length - 2)]

        token_ids = [self.CLS_TOKEN_ID]
        for tok in truncated_tokens:
            digest = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            token_ids.append(1000 + (digest % (self.VOCAB_MODULO - 1000)))
        token_ids.append(self.SEP_TOKEN_ID)

        attention_mask = [1] * len(token_ids)
        pad_len = max(0, self.max_length - len(token_ids))
        if pad_len > 0:
            token_ids.extend([self.PAD_TOKEN_ID] * pad_len)
            attention_mask.extend([0] * pad_len)

        normalized_label = self.normalize_label(label) if label is not None else None
        return TokenizedHeadline(
            raw_text=text,
            cleaned_text=cleaned,
            tokens=truncated_tokens,
            input_ids=token_ids,
            attention_mask=attention_mask,
            tickers=tickers,
            label=normalized_label,
        )

    @staticmethod
    def compute_headline_decay_weights(
        timestamps: pd.Series,
        reference_time: Optional[pd.Timestamp] = None,
        half_life_hours: float = 12.0,
    ) -> pd.Series:
        """
        Compute exponential time-decay weights for news headlines so recent headlines
        carry higher weight than stale news (zero lookahead relative to reference_time).
        """
        parsed_ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
        ref = reference_time or parsed_ts.max()
        if pd.isna(ref):
            return pd.Series(np.ones(len(timestamps), dtype=np.float32), index=timestamps.index)

        if ref.tzinfo is None:
            ref = ref.tz_localize("UTC")

        delta_hours = ((ref - parsed_ts).dt.total_seconds() / 3600.0).clip(lower=0.0).fillna(0.0)
        decay_lambda = np.log(2.0) / max(half_life_hours, 1e-3)
        weights = np.exp(-decay_lambda * delta_hours)
        # Exclude future headlines relative to reference_time
        weights = np.where(parsed_ts > ref, 0.0, weights)
        return pd.Series(weights.astype(np.float32), index=timestamps.index)

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Clean, label, and tokenize a FinancialPhraseBank or Massive News DB DataFrame.
        """
        out = df.copy()
        cols_lower = {c.lower(): c for c in out.columns}

        resolved_text_col = text_col
        if resolved_text_col is None:
            for candidate in ("sentence", "headline", "title", "text", "news"):
                if candidate in cols_lower:
                    resolved_text_col = cols_lower[candidate]
                    break
        if resolved_text_col is None or resolved_text_col not in out.columns:
            raise ValueError(
                f"Could not locate text column in DataFrame columns: {list(out.columns)}"
            )

        resolved_label_col = label_col
        if resolved_label_col is None:
            for candidate in ("sentiment", "label", "polarity"):
                if candidate in cols_lower:
                    resolved_label_col = cols_lower[candidate]
                    break

        tokenized_records = [
            self.tokenize_text(
                str(row[resolved_text_col]),
                row[resolved_label_col] if resolved_label_col and resolved_label_col in out.columns else None,
            )
            for _, row in out.iterrows()
        ]

        out["cleaned_text"] = [rec.cleaned_text for rec in tokenized_records]
        out["tokens"] = [rec.tokens for rec in tokenized_records]
        out["token_count"] = [len(rec.tokens) for rec in tokenized_records]
        out["input_ids"] = [rec.input_ids for rec in tokenized_records]
        out["attention_mask"] = [rec.attention_mask for rec in tokenized_records]
        out["extracted_tickers"] = [rec.tickers for rec in tokenized_records]

        if resolved_label_col and resolved_label_col in out.columns:
            out["sentiment_label"] = [
                int(rec.label) if rec.label is not None else 0 for rec in tokenized_records
            ]

        resolved_ts_col = timestamp_col
        if resolved_ts_col is None:
            for candidate in ("timestamp", "date", "published_at", "time"):
                if candidate in cols_lower:
                    resolved_ts_col = cols_lower[candidate]
                    break
        if resolved_ts_col and resolved_ts_col in out.columns:
            out["decay_weight"] = self.compute_headline_decay_weights(out[resolved_ts_col])

        out = out[out["cleaned_text"].str.len() > 0].reset_index(drop=True)
        return out
