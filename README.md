# NexusTrade AI - Institutional Telegram Trading Bot

NexusTrade AI is an autonomous, institutional-grade AI-powered Telegram Trading Bot integrating multi-asset Kaggle datasets, real-time TradingView Webhooks, automated `1200x800` chart snapshot rendering, and a multimodal quantitative ensemble engine.

## Key Features
- **Multimodal AI Ensemble (`ai_engine/`):**
  - **Technical Predictor ($S_{\text{tech}} \in [-1, 1]$):** LightGBM multi-timeframe classifier trained on zero-lookahead OHLCV indicators (`RSI(14)`, `MACD(12,26,9)`, `Bollinger Bands(20,2)`, `ATR(14)`, `EMA 20/50/200`, `SMA 50/200`, `VWAP`, `Donchian(20)`).
  - **Realized Volatility Estimator ($S_{\text{vol}} \in [0, 1]$):** Optiver-inspired HAR-RV regressor modeling Level-1 WAP log-return variance, Parkinson high-low volatility, and bid-ask spread volatility.
  - **Financial Sentiment Engine ($S_{\text{sent}} \in [-1, 1]$):** FinBERT (`ProsusAI/finbert`) sentiment scorer with a FinancialPhraseBank-calibrated financial NLP lexicon engine and exponential headline time-decay weighting.
- **Institutional Risk Engine (`ai_engine/risk_management/`):**
  - Dynamic $2 \times \text{ATR(14)}$ Stop-Loss, minimum `1:2` (`TP1`) and `1:3` (`TP2`) Reward-to-Risk targets, and ratcheting trailing stops.
  - Half-Kelly Criterion position sizing scaled by $(1 - S_{\text{vol}}) \times \text{Confidence}$, capped at `1.5%` max account risk per trade.
  - `4.0%` Daily Drawdown Circuit Breaker and Extreme Volatility ($S_{\text{vol}} \ge 0.85$) safety suppression.
- **Kaggle & Local Data Pipeline (`data_pipeline/`):**
  - Automated downloader (`KaggleDatasetDownloader`) with MD5 integrity verification and path-traversal-safe archive extraction for all 6 datasets and 2 competitions, plus local `XAUUSD` (`15m`) and `50 Crypto` (`1d`) datasets.
  - Apache Parquet `FeatureStore` with automatic `float64 -> float32` and `int64 -> int32` memory optimization.
- **TradingView Real-Time Integration (`tradingview/`):**
  - FastAPI webhook server (`POST /api/v1/tradingview/webhook`) protected by constant-time `X-TradingView-Secret` verification.
  - High-resolution `1200x800` dark-theme candlestick, EMA/BB, Volume, and RSI(14) chart snapshot generator (`ChartSnapshotGenerator`).
  - Ready-to-use Pine Script v5 strategy (`tradingview/pine_script_sample.pine`).
- **Telegram Bot Interface (`telegram_bot/`):**
  - Built on `aiogram` v3 with `WhitelistAuthMiddleware`, sliding-window `RateLimitMiddleware`, interactive inline keyboards, and rich HTML signal cards.

---

## Quick Start Guide

### 1. Setup Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Populate TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USER_IDS, TRADINGVIEW_WEBHOOK_SECRET, KAGGLE_USERNAME, KAGGLE_KEY
```

### 3. Run Self-Check & Test Suite
```bash
# Run system self-check
python main.py --check

# Run full pytest suite
pytest tests/ -v
```

### 4. Launch Services
```bash
# Launch both FastAPI Webhook Server and Aiogram v3 Telegram Bot
python main.py

# Or launch with Docker Compose (Bot + Redis Cache)
docker compose up --build -d
```

---

## License
MIT License. For educational, research, and algorithmic trading simulation purposes.
