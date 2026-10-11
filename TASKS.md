# TASKS.md - Implementation Roadmap & Task Checklist

Track your development progress using this step-by-step checklist. Each task has defined acceptance criteria.

---

## Phase 1: Environment Setup & Project Foundation
- [x] **1.1. Repository Scaffolding**
  - Create full directory tree matching `AGENT.md`.
  - Create `.env.example`, `.gitignore`, `pyproject.toml` / `requirements.txt`.
  - *Criteria:* Directory structure is created and valid.
- [x] **1.2. Configuration & Logging Architecture**
  - Implement `config/settings.py` using `pydantic-settings` to parse environment variables.
  - Implement `config/logging_config.py` with structured colored output (Rich or structlog).
  - *Criteria:* Application logs startup info cleanly with type-validated configs.
- [x] **1.3. Docker Containerization**
  - Write `Dockerfile` and `docker-compose.yml` (services: `nexustrade-bot`, `redis-cache`).
  - *Criteria:* Container builds cleanly and runs entrypoint script without errors.

---

## Phase 2: Kaggle Dataset Ingestion & Preprocessing
- [x] **2.1. Automated Kaggle Downloader**
  - Implement `data_pipeline/kaggle_downloader.py` using Kaggle API credentials.
  - Create download scripts for all 6 target datasets and 2 competitions.
  - *Criteria:* Script verifies md5/integrity and extracts raw files into `data/raw/`.
- [x] **2.2. OHLCV Data Preprocessor**
  - Implement `data_pipeline/preprocessor_ohlcv.py`.
  - Resample and clean multi-timeframe data (1m, 15m, 1h, 1d) with zero lookahead bias.
  - Generate technical indicators (RSI, MACD, Bollinger Bands, ATR, EMA 20/50/200).
  - Export to `data/processed/ohlcv.parquet`.
  - *Criteria:* Parquet dataset is saved with datetime indices and validated features.
- [x] **2.3. Microstructure & Volatility Preprocessor**
  - Implement `data_pipeline/preprocessor_microstructure.py`.
  - Extract realized volatility formulas from Optiver competition dataset.
  - Extract feature correlation filters from Jane Street dataset.
  - *Criteria:* Produces realized volatility metric rolling window benchmarks.
- [x] **2.4. Sentiment & News NLP Preprocessor**
  - Implement `data_pipeline/preprocessor_sentiment.py`.
  - Clean and tokenize FinancialPhraseBank and Daily Financial News data.
  - *Criteria:* Text is tokenized, labeled, and prepared for transformer inference.

---

## Phase 3: AI Engine & Ensemble Modeling
- [x] **3.1. Technical Directional Model**
  - Implement `ai_engine/models/ohlcv_predictor.py` using LightGBM / XGBoost.
  - Train binary/multiclass classifier: Buy, Sell, Hold based on forward-looking returns.
  - *Criteria:* Achieves ROC-AUC > 0.58 on validation split without lookahead bias.
- [x] **3.2. Realized Volatility Predictor**
  - Implement `ai_engine/models/volatility_estimator.py`.
  - Predict next-period volatility to calibrate dynamic risk sizing.
  - *Criteria:* Outputs scalar volatility coefficient between 0.0 and 1.0.
- [x] **3.3. FinBERT Financial Sentiment Engine**
  - Implement `ai_engine/models/sentiment_analyzer.py` using Hugging Face `ProsusDE/finbert`.
  - Fine-tune or benchmark against FinancialPhraseBank.
  - *Criteria:* Returns sentiment polarity score (-1.0 to +1.0) for news headlines.
- [x] **3.4. Ensemble Decision Agent**
  - Implement `ai_engine/models/ensemble_agent.py`.
  - Synthesize technical + volatility + sentiment outputs into unified signal confidence score.
  - *Criteria:* Outputs `SignalDecision(action="BUY", confidence=0.82, risk_multiplier=0.75)`.
- [x] **3.5. Event-Driven Backtesting Engine**
  - Implement `ai_engine/backtester/engine.py`.
  - Compute Sharpe Ratio, Maximum Drawdown, Profit Factor, Win Rate, and equity curve.
  - *Criteria:* Generates a backtest report dictionary and performance plot.

---

## Phase 4: TradingView Webhook & Real-Time Charting
- [x] **4.1. FastAPI Webhook Server**
  - Implement `tradingview/webhook_server.py`.
  - Create endpoint `POST /api/v1/tradingview/webhook` with `X-TradingView-Secret` validation.
  - Parse and validate payload via `tradingview/alert_parser.py`.
  - *Criteria:* Valid alerts return 200 OK and queue signal processing; invalid secrets return 401.
- [x] **4.2. Pine Script Alert Strategy Generator**
  - Write sample Pine Script v5 strategy snippet in `tradingview/pine_script_sample.pine`.
  - Configure Pine Script alert trigger with webhook URL and JSON payload formatting.
  - *Criteria:* Pine script compiles successfully in TradingView Pine Editor without errors.
- [x] **4.3. Real-Time Chart Snapshot Service**
  - Implement `tradingview/chart_generator.py`.
  - Generate chart graphics with candlestick, MA lines, and entry/exit signal annotations.
  - *Criteria:* Saves 1200x800 high-resolution PNG chart ready for Telegram dispatch.

---

## Phase 5: Telegram Bot UI/UX Development
- [x] **5.1. Aiogram Bot Foundation & Middleware**
  - Implement `telegram_bot/bot.py`.
  - Set up `auth_middleware.py` (whitelist check) and `rate_limit.py` (throttling).
  - *Criteria:* Bot responds to authorized users and politely drops unlisted user commands.
- [x] **5.2. Core Command Handlers**
  - Implement `/start`, `/help`, `/status` in `telegram_bot/handlers/commands.py`.
  - Create responsive inline keyboards for quick navigation.
  - *Criteria:* Smooth user onboarding with interactive buttons.
- [x] **5.3. Signal & Analysis Handlers**
  - Implement `/signal <symbol>`, `/sentiment <symbol>`, `/volatility <symbol>`.
  - Format response cards using Telegram HTML styling with visual progress bars.
  - Attach generated chart images directly to the signal message.
  - *Criteria:* Complete signal card with chart arrives within 3 seconds of command trigger.
- [x] **5.4. Webhook-to-Telegram Dispatcher**
  - Connect TradingView webhook receiver directly to the Telegram bot dispatcher.
  - Broadcast confirmed trading signals to the configured admin Telegram channel or group.
  - *Criteria:* Webhook alert triggers immediate Telegram push notification with trade card.

---

## Phase 6: Risk Management & Production Hardening
- [x] **6.1. Position Sizing & Stop-Loss Engine**
  - Implement Kelly Criterion, ATR-based Stop Loss, and Take Profit levels in `risk_management/`.
  - *Criteria:* Signal cards always include strict Entry, Stop Loss, and TP1/TP2 targets.
- [x] **6.2. Circuit Breaker & Safety Safeguards**
  - Max drawdown cut-off logic implemented.
  - High volatility safety suppression implemented.
  - *Criteria:* Bot rejects signals if market volatility exceeds safe thresholds.
- [x] **6.3. Comprehensive Test Suite**
  - Write unit and integration tests in `tests/`.
  - Test webhook authentication, data pipeline transforms, and Telegram message formatters.
  - *Criteria:* `pytest tests/` passes with 100% green status.
