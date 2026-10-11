from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Settings
from data_pipeline.feature_store import FeatureStore
from data_pipeline.kaggle_downloader import (
    KAGGLE_DATASET_REGISTRY,
    KaggleDatasetDownloader,
)
from data_pipeline.preprocessor_microstructure import (
    MicrostructureVolatilityPreprocessor,
)
from data_pipeline.preprocessor_ohlcv import OHLCVPreprocessor
from data_pipeline.preprocessor_sentiment import SentimentNLPPreprocessor


def _make_sample_ohlcv(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(101)
    idx = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    rets = rng.normal(0.0005, 0.004, size=rows)
    close = 50000.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = close[0] * 0.999
    high = np.maximum(open_p, close) * 1.002
    low = np.minimum(open_p, close) * 0.998
    volume = rng.uniform(100.0, 1500.0, size=rows)
    return pd.DataFrame(
        {
            "timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_kaggle_registry_and_downloader_verification(tmp_path: Path) -> None:
    assert len(KAGGLE_DATASET_REGISTRY) == 8
    datasets = [s for s in KAGGLE_DATASET_REGISTRY.values() if s.resource_type == "dataset"]
    competitions = [s for s in KAGGLE_DATASET_REGISTRY.values() if s.resource_type == "competition"]
    assert len(datasets) == 6
    assert len(competitions) == 2

    settings = Settings(data_dir=tmp_path / "data")

    def fake_runner(
        cmd: list[str], env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        dest_dir = Path(cmd[-1])
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive_path = dest_dir / "sample.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("prices.csv", "timestamp,open,high,low,close,volume\n2026-01-01T00:00:00Z,10,12,9,11,100\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    downloader = KaggleDatasetDownloader(settings=settings, command_runner=fake_runner)
    report = downloader.download_resource("global_indices", force=True)

    assert report.verified is True
    assert len(report.files) == 1
    assert "prices.csv" in report.md5_checksums
    assert len(report.md5_checksums["prices.csv"]) == 32


def test_ohlcv_preprocessor_zero_lookahead_and_parquet_export(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    store = FeatureStore(settings=settings)
    prep = OHLCVPreprocessor(settings=settings, feature_store=store)

    df_original = _make_sample_ohlcv(120)
    df_shocked = df_original.copy()
    # Mutate the final bar's close price drastically to test zero lookahead bias
    df_shocked.loc[df_shocked.index[-1], "close"] = df_shocked.loc[df_shocked.index[-1], "close"] * 3.5
    df_shocked.loc[df_shocked.index[-1], "high"] = df_shocked.loc[df_shocked.index[-1], "close"] * 1.1

    feat_orig, parquet_path = prep.process_and_export(
        df_original, symbol="BTCUSDT", timeframe="15m", output_name="test_ohlcv", enforce_lag=True
    )
    feat_shocked = prep.engineer_features(
        df_shocked, symbol="BTCUSDT", timeframe="15m", enforce_lag=True
    )

    assert parquet_path.exists()
    assert isinstance(feat_orig.index, pd.DatetimeIndex)
    assert str(feat_orig.index.tz) == "UTC"

    # Zero lookahead invariant: features at the final bar t must be identical even when bar t's close is shocked
    for col in prep.FEATURE_COLUMNS:
        val_orig = float(feat_orig[col].iloc[-1])
        val_shock = float(feat_shocked[col].iloc[-1])
        assert np.isclose(val_orig, val_shock, rtol=1e-5, atol=1e-5), f"Lookahead leak detected in {col}"

    # Verify float32 downcasting and Parquet reload
    loaded = store.load("test_ohlcv", symbol="BTCUSDT")
    assert len(loaded) == len(feat_orig)
    assert loaded["rsi_14"].dtype == np.float32

    # Verify multi-timeframe resampling to 1h
    resampled_1h = prep.resample_ohlcv(df_original, target_timeframe="1h", symbol="BTCUSDT")
    assert len(resampled_1h) == 30
    assert (resampled_1h["high"] >= resampled_1h["low"]).all()


def test_microstructure_and_jane_street_preprocessor(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    micro = MicrostructureVolatilityPreprocessor(settings=settings)

    rng = np.random.default_rng(42)
    n = 150
    book_df = pd.DataFrame(
        {
            "bid_price1": 100.0 + np.cumsum(rng.normal(0, 0.1, n)),
            "ask_price1": 100.05 + np.cumsum(rng.normal(0, 0.1, n)),
            "bid_size1": rng.uniform(10.0, 200.0, n),
            "ask_size1": rng.uniform(10.0, 200.0, n),
        }
    )
    rv_out = micro.process_optiver_order_book(book_df, windows=(10, 20, 60), enforce_lag=True)
    assert "wap" in rv_out.columns
    assert "rv_window_20" in rv_out.columns
    assert "book_depth_imbalance" in rv_out.columns
    assert ((rv_out["rv_regime_score"] >= 0.0) & (rv_out["rv_regime_score"] <= 1.0)).all()

    base_feat = rng.normal(0, 1, 100)
    js_df = pd.DataFrame(
        {
            "feature_0": base_feat,
            "feature_1": base_feat * 0.999 + rng.normal(0, 0.001, 100),  # Highly collinear with feature_0
            "feature_2": rng.normal(0, 1, 100),
            "weight": rng.uniform(0.5, 2.0, 100),
            "resp": rng.normal(0.001, 0.01, 100),
        }
    )
    filtered_df, dropped = micro.filter_jane_street_correlations(js_df, correlation_threshold=0.90)
    assert "feature_1" in dropped
    assert "js_cross_interaction" in filtered_df.columns
    assert "utility_score" in filtered_df.columns


def test_sentiment_nlp_preprocessor() -> None:
    prep = SentimentNLPPreprocessor(max_length=32)
    sample_df = pd.DataFrame(
        {
            "sentence": [
                "Spot $BTC inflows surge 45% as institutional accumulation hits record high https://example.com",
                "Company reports severe Q3 loss and downward guidance for $AAPL",
                "Market trading volume remains flat ahead of central bank holiday",
            ],
            "sentiment": ["positive", "negative", "neutral"],
            "timestamp": [
                "2026-10-11T04:00:00Z",
                "2026-10-11T02:00:00Z",
                "2026-10-10T12:00:00Z",
            ],
        }
    )
    processed = prep.preprocess_dataframe(sample_df)
    assert len(processed) == 3
    assert list(processed["sentiment_label"]) == [1, -1, 0]
    assert "BTC" in processed["extracted_tickers"].iloc[0]
    assert "AAPL" in processed["extracted_tickers"].iloc[1]
    assert len(processed["input_ids"].iloc[0]) == 32
    assert processed["decay_weight"].iloc[0] > processed["decay_weight"].iloc[2]
