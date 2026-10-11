from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger("nexustrade.data_pipeline.kaggle")


@dataclass(frozen=True)
class KaggleResourceSpec:
    """Metadata specification for a required Kaggle dataset or competition."""

    key: str
    slug: str
    resource_type: Literal["dataset", "competition"]
    category: str
    purpose: str
    key_features: tuple[str, ...] = field(default_factory=tuple)


KAGGLE_DATASET_REGISTRY: dict[str, KaggleResourceSpec] = {
    "global_indices": KaggleResourceSpec(
        key="global_indices",
        slug="jacksaleeby/global-stock-market-indices-4-6m-rows",
        resource_type="dataset",
        category="Multi-Asset OHLCV",
        purpose="Global equity trend, cross-market regime detection",
        key_features=("Index correlations", "Rolling Returns", "SMA/EMA (50, 200)", "RSI", "MACD"),
    ),
    "us_stocks": KaggleResourceSpec(
        key="us_stocks",
        slug="borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
        resource_type="dataset",
        category="US Stocks OHLCV",
        purpose="Individual asset swing trading & momentum models",
        key_features=("High/Low ranges", "VWAP", "Bollinger Bands"),
    ),
    "crypto_intraday": KaggleResourceSpec(
        key="crypto_intraday",
        slug="novandraanugrah/bitcoin-historical-datasets-2018-2024",
        resource_type="dataset",
        category="Crypto Intraday",
        purpose="15m, 1h, 4h, 1d multi-timeframe trend & momentum",
        key_features=("Multi-timeframe trend alignment", "ATR", "Donchian Channels"),
    ),
    "crypto_high_freq": KaggleResourceSpec(
        key="crypto_high_freq",
        slug="beannguyen94/binance-full-history",
        resource_type="dataset",
        category="Crypto High-Freq",
        purpose="1-minute microstructure & scalping validation",
        key_features=("1m order flow proxies", "micro-breakouts", "volume surges"),
    ),
    "jane_street": KaggleResourceSpec(
        key="jane_street",
        slug="jane-street-market-prediction",
        resource_type="competition",
        category="Microstructure",
        purpose="Ultra-short term directional opportunity scoring",
        key_features=("Feature embeddings", "cross-feature interactions", "trade weights"),
    ),
    "optiver_volatility": KaggleResourceSpec(
        key="optiver_volatility",
        slug="optiver-realized-volatility-prediction",
        resource_type="competition",
        category="Volatility",
        purpose="Realized volatility forecasting & dynamic risk sizing",
        key_features=("Book depth imbalance", "Bid-Ask spread volatility", "log returns variance"),
    ),
    "financial_phrasebank": KaggleResourceSpec(
        key="financial_phrasebank",
        slug="ankurzing/sentiment-analysis-for-financial-news",
        resource_type="dataset",
        category="Sentiment NLP",
        purpose="Financial text sentiment classifier training (FinBERT)",
        key_features=("Positive / Neutral / Negative sentiment classification scores",),
    ),
    "massive_news_db": KaggleResourceSpec(
        key="massive_news_db",
        slug="miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests",
        resource_type="dataset",
        category="News DB",
        purpose="News-impact backtesting & event-driven trading signals",
        key_features=("Headline sentiment decay", "event-driven return correlation"),
    ),
}


@dataclass
class DownloadArtifactReport:
    """Integrity and extraction report for a downloaded or synced dataset."""

    key: str
    slug: str
    target_dir: Path
    files: list[Path]
    md5_checksums: dict[str, str]
    verified: bool
    source: Literal["kaggle_api", "local_workspace", "cached"]
    error: Optional[str] = None


class KaggleDatasetDownloader:
    """Automated Kaggle dataset & competition ingestion with MD5 integrity verification."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        command_runner: Optional[Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[str]]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.raw_dir = self.settings.raw_data_dir
        self._command_runner = command_runner or self._default_command_runner

    def _build_kaggle_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.settings.kaggle_username:
            env["KAGGLE_USERNAME"] = self.settings.kaggle_username
        if self.settings.kaggle_key:
            env["KAGGLE_KEY"] = self.settings.kaggle_key
        return env

    def has_valid_credentials(self) -> bool:
        if (
            self.settings.kaggle_username
            and self.settings.kaggle_key
            and self.settings.kaggle_username != "your_kaggle_username"
            and self.settings.kaggle_key != "your_kaggle_api_key"
        ):
            return True
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists() and kaggle_json.stat().st_size > 0

    @staticmethod
    def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
        """Compute hexadecimal MD5 digest for file integrity validation."""
        hasher = hashlib.md5()
        with file_path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def verify_directory_integrity(
        cls,
        directory: Path,
        expected_md5: Optional[dict[str, str]] = None,
    ) -> tuple[bool, list[Path], dict[str, str]]:
        """Verify non-empty extracted files and optional expected MD5 checksums."""
        if not directory.exists():
            return False, [], {}

        files = sorted(
            p for p in directory.rglob("*") if p.is_file() and p.name != ".gitkeep"
        )
        if not files:
            return False, [], {}

        checksums: dict[str, str] = {}
        for file_path in files:
            if file_path.stat().st_size == 0:
                return False, files, checksums
            rel_name = file_path.relative_to(directory).as_posix()
            checksums[rel_name] = cls.compute_md5(file_path)

        if expected_md5:
            for rel_name, expected_hash in expected_md5.items():
                actual_hash = checksums.get(rel_name)
                if actual_hash != expected_hash:
                    return False, files, checksums

        return True, files, checksums

    @staticmethod
    def extract_archives(target_dir: Path, remove_zip: bool = True) -> list[Path]:
        """Extract all ZIP archives in target_dir safely preventing path traversal."""
        extracted_files: list[Path] = []
        resolved_target = target_dir.resolve()

        for archive_path in sorted(target_dir.glob("*.zip")):
            if not zipfile.is_zipfile(archive_path):
                raise ValueError(f"Corrupted ZIP archive detected: {archive_path}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    destination = (resolved_target / member.filename).resolve()
                    if not str(destination).startswith(str(resolved_target)):
                        raise ValueError(
                            f"Archive member attempted path traversal: {member.filename}"
                        )
                    zf.extract(member, resolved_target)
                    if not member.is_dir():
                        extracted_files.append(destination)
            if remove_zip:
                archive_path.unlink()

        return extracted_files

    @staticmethod
    def _default_command_runner(
        cmd: list[str], env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )

    def build_cli_command(self, spec: KaggleResourceSpec, target_dir: Path) -> list[str]:
        """Construct the Kaggle CLI command for a dataset or competition."""
        if spec.resource_type == "dataset":
            return [
                "kaggle",
                "datasets",
                "download",
                "-d",
                spec.slug,
                "-p",
                str(target_dir),
            ]
        return [
            "kaggle",
            "competitions",
            "download",
            "-c",
            spec.slug,
            "-p",
            str(target_dir),
        ]

    def sync_local_workspace_datasets(self) -> dict[str, Path]:
        """Link or copy available workspace datasets (XAUUSD, 50 Crypto) into data/raw/."""
        synced: dict[str, Path] = {}
        xau_src = self.settings.local_xauusd_path
        if xau_src.exists():
            xau_dest_dir = self.raw_dir / "xauusd"
            xau_dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = xau_dest_dir / xau_src.name
            if not dest_file.exists():
                shutil.copy2(xau_src, dest_file)
            synced["xauusd"] = dest_file

        crypto_dir = self.settings.local_crypto_dir
        if crypto_dir.exists():
            crypto_dest_dir = self.raw_dir / "crypto_daily"
            crypto_dest_dir.mkdir(parents=True, exist_ok=True)
            for csv_file in sorted(crypto_dir.glob("*.csv")):
                dest_file = crypto_dest_dir / csv_file.name
                if not dest_file.exists():
                    shutil.copy2(csv_file, dest_file)
            synced["crypto_daily"] = crypto_dest_dir

        return synced

    def download_resource(
        self,
        key: str,
        force: bool = False,
        expected_md5: Optional[dict[str, str]] = None,
    ) -> DownloadArtifactReport:
        """Download, extract, and verify a single Kaggle dataset or competition."""
        if key not in KAGGLE_DATASET_REGISTRY:
            raise KeyError(
                f"Unknown dataset key '{key}'. Available keys: {list(KAGGLE_DATASET_REGISTRY)}"
            )

        spec = KAGGLE_DATASET_REGISTRY[key]
        target_dir = self.raw_dir / spec.key
        target_dir.mkdir(parents=True, exist_ok=True)

        if not force:
            verified, files, checksums = self.verify_directory_integrity(
                target_dir, expected_md5=expected_md5
            )
            if verified:
                logger.info(
                    "Dataset already cached and verified",
                    key=spec.key,
                    slug=spec.slug,
                    file_count=len(files),
                )
                return DownloadArtifactReport(
                    key=spec.key,
                    slug=spec.slug,
                    target_dir=target_dir,
                    files=files,
                    md5_checksums=checksums,
                    verified=True,
                    source="cached",
                )

        cmd = self.build_cli_command(spec, target_dir)
        env = self._build_kaggle_env()

        try:
            logger.info("Executing Kaggle download", key=spec.key, slug=spec.slug)
            self._command_runner(cmd, env)
            self.extract_archives(target_dir)
            verified, files, checksums = self.verify_directory_integrity(
                target_dir, expected_md5=expected_md5
            )
            return DownloadArtifactReport(
                key=spec.key,
                slug=spec.slug,
                target_dir=target_dir,
                files=files,
                md5_checksums=checksums,
                verified=verified,
                source="kaggle_api",
            )
        except Exception as exc:
            logger.warning(
                "Kaggle download failed; checking local fallback data",
                key=spec.key,
                slug=spec.slug,
                error=str(exc),
            )
            return DownloadArtifactReport(
                key=spec.key,
                slug=spec.slug,
                target_dir=target_dir,
                files=[],
                md5_checksums={},
                verified=False,
                source="kaggle_api",
                error=str(exc),
            )

    async def download_all_async(
        self, keys: Optional[list[str]] = None, force: bool = False
    ) -> dict[str, DownloadArtifactReport]:
        """Asynchronously download and verify all registered Kaggle datasets."""
        target_keys = keys or list(KAGGLE_DATASET_REGISTRY.keys())
        self.sync_local_workspace_datasets()
        tasks = [
            asyncio.to_thread(self.download_resource, key, force)
            for key in target_keys
        ]
        results = await asyncio.gather(*tasks)
        return {report.key: report for report in results}
