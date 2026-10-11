# AGENT.md - Autonomous AI Developer Instructions

## 1. Project Mission & Identity
You are an expert Quant Engineer and Full-Stack Python Architect building **NexusTrade AI** — an institutional-grade, AI-powered Telegram Trading Bot with real-time TradingView integration and multimodal intelligence.

The bot ingests multi-asset historical datasets from Kaggle for model training and backtesting, connects to TradingView via Webhooks and chart rendering engines for real-time market data, and delivers algorithmic signals, sentiment scoring, and portfolio risk management directly through an interactive Telegram interface.

---

## 2. System Architecture & Directory Structure

Your implementation must strictly follow this modular structure:

```
nexustrade-ai/
├── AGENT.md                          # Master AI developer rules (this document)
├── TASKS.md                          # Granular implementation checklist
├── PROMPT_GOAL.md                    # Master prompt for LLM agent sessions
├── README.md                         # Quick start, architecture, & run guide
├── .env.example                      # Template for secrets and environment vars
├── requirements.txt                  # Python dependencies
├── docker-compose.yml                # Multi-container orchestration (Bot + DB + Redis)
├── Dockerfile                        # Production container image
├── config/
│   ├── settings.py                   # Pydantic BaseSettings management
│   ├── logging_config.py             # Structured JSON / Rich logging
│   └── tradingview_config.json       # Webhook verification and ticker mappings
├── data_pipeline/
│   ├── kaggle_downloader.py          # Kaggle API automated dataset ingestion
│   ├── preprocessor_ohlcv.py         # Multi-timeframe OHLCV cleaning & alignment
│   ├── preprocessor_microstructure.py# Volatility & order book features (Jane St / Optiver)
│   ├── preprocessor_sentiment.py     # NLP news text tokenization & cleaning
│   └── feature_store.py              # Parquet-based feature storage & cache
├── ai_engine/
│   ├── models/
│   │   ├── ohlcv_predictor.py        # LightGBM / XGBoost multi-timeframe directional model
│   │   ├── volatility_estimator.py   # Realized volatility regressor (Optiver-style)
│   │   ├── sentiment_analyzer.py     # FinBERT / HuggingFace financial sentiment scorer
│   │   └── ensemble_agent.py         # Meta-learner combining signals into final confidence
│   ├── backtester/
│   │   ├── engine.py                 # Event-driven backtesting engine (Sharpe, Drawdown, Win Rate)
│   │   └── metrics.py                # Financial performance metrics computation
│   └── risk_management/
│       ├── position_sizer.py         # Kelly Criterion & Volatility-adjusted position sizing
│       └── stop_loss_manager.py      # Trailing Stop, ATR-based Stop Loss, Take Profit
├── tradingview/
│   ├── webhook_server.py             # FastAPI server listening for TradingView webhook alerts
│   ├── alert_parser.py               # Payload schema validation & sanitization
│   └── chart_generator.py            # TradingView Lightweight Charts / Puppeteer chart snapshot generator
├── telegram_bot/
│   ├── bot.py                        # Aiogram v3 Dispatcher & Bot lifecycle
│   ├── handlers/
│   │   ├── commands.py               # /start, /help, /status, /analyze, /portfolio
│   │   ├── signals.py                # /signal, /sentiment, /backtest
│   │   └── callbacks.py              # Inline keyboard callbacks and interactive menus
│   ├── middlewares/
│   │   ├── auth_middleware.py        # Whitelist / admin authorization guard
│   │   └── rate_limit.py             # Anti-spam throttling
│   └── views/
│       └── formatters.py             # Telegram HTML / MarkdownV2 message styling & cards
└── tests/
    ├── test_data_pipeline.py
    ├── test_ai_engine.py
    ├── test_webhook.py
    └── test_telegram.py
```

---

## 3. Kaggle Datasets Integration Matrix

You must download, process, and map the following Kaggle datasets using the Kaggle API (`kaggle datasets download -d ...` and `kaggle competitions download -c ...`):

| Category | Dataset Name & Slug | Primary Purpose in Model Pipeline | Key Features to Extract |
| :--- | :--- | :--- | :--- |
| **Multi-Asset OHLCV** | `jacksaleeby/global-stock-market-indices-4-6m-rows` | Global equity trend, cross-market regime detection | Index correlations, Rolling Returns, SMA/EMA (50, 200), RSI, MACD |
| **US Stocks OHLCV** | `borismarjanovic/price-volume-data-for-all-us-stocks-etfs` | Individual asset swing trading & momentum models | High/Low ranges, Volume Weighted Average Price (VWAP), Bollinger Bands |
| **Crypto Intraday** | `novandraanugrah/bitcoin-historical-datasets-2018-2024` | 15m, 1h, 4h, 1d multi-timeframe trend & momentum | Multi-timeframe trend alignment, ATR, Donchian Channels |
| **Crypto High-Freq** | `beannguyen94/binance-full-history` | 1-minute microstructure & scalping validation | 1m order flow proxies, micro-breakouts, volume surges |
| **Microstructure** | `c/jane-street-market-prediction` | Ultra-short term directional opportunity scoring | Feature embeddings, anonymized cross-feature interactions, trade weights |
| **Volatility** | `c/optiver-realized-volatility-prediction` | Realized volatility forecasting & dynamic risk sizing | Book depth imbalance, Bid-Ask spread volatility, log returns variance |
| **Sentiment NLP** | `ankurzing/sentiment-analysis-for-financial-news` | Financial text sentiment classifier training (FinBERT) | Positive / Neutral / Negative sentiment classification scores |
| **News DB** | `miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests` | News-impact backtesting & event-driven trading signals | Headline sentiment decay, event-driven return correlation |

### Data Guardrails:
- **No Lookahead Bias:** All rolling features, z-scores, and technical indicators must strictly use `shift(1)` or rolling windows computed up to $t-1$.
- **Standardized Timestamp:** Align all timestamps to UTC ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`).
- **Memory Optimization:** Downcast numerical data types (`float64` $	o$ `float32`, `int64` $	o$ `int32`), store cleaned tabular datasets in Apache Parquet format.

---

## 4. TradingView Real-Time Integration Architecture

Real-time market feeds and visual charts must be integrated through two primary mechanisms:

### A. TradingView Webhook Receiver (FastAPI)
- Build an asynchronous endpoint `POST /api/v1/tradingview/webhook` protected by a secret passphrase header (`X-TradingView-Secret`).
- **Standard Alert Payload Schema (JSON):**
  ```json
  {
    "secret": "YOUR_ENCRYPTED_WEBHOOK_SECRET",
    "ticker": "BINANCE:BTCUSDT",
    "interval": "15m",
    "action": "BUY",
    "price": 67420.50,
    "strategy": "TrendBreakout_v2",
    "indicators": {
      "rsi": 62.4,
      "macd": 142.1,
      "atr": 450.0
    },
    "timestamp": "2026-10-11T04:50:00Z"
  }
  ```
- **Execution Flow:**
  1. Validate incoming webhook secret and payload integrity.
  2. Query `ai_engine.ensemble_agent` to validate signal confidence (combining Technical + Volatility + FinBERT Sentiment).
  3. If confidence $\ge 	ext{threshold}$ (e.g. 0.72) and risk parameters pass, format and push an alert message to Telegram.

### B. Dynamic Real-Time Chart Generation
- Implement `tradingview/chart_generator.py`:
  - **Option 1 (Primary):** Headless Chromium (Playwright/Puppeteer) capturing high-resolution TradingView chart widgets or Lightweight Charts canvas.
  - **Option 2 (Fallback):** Render interactive candlestick and indicator plots via `mplfinance` or Plotly saved as high-definition PNG images.
- Images are dispatched directly to Telegram chat using `bot.send_photo()` with formatted caption cards.

---

## 5. AI Ensemble & Risk Engine Specifications

### A. Ensemble Decision Matrix
The final trading recommendation is formulated by combining 3 distinct sub-models:
1. **$S_{\text{tech}} \in [-1, 1]$ (Technical Model):** Directional signal derived from multi-timeframe OHLCV indicators trained on stock & crypto datasets.
2. **$S_{\text{vol}} \in [0, 1]$ (Volatility Filter):** Optiver-derived realized volatility scaler. If volatility is extreme, reduce position size or pause signals.
3. **$S_{\text{sent}} \in [-1, 1]$ (Sentiment Model):** Sentiment score from FinBERT trained on FinancialPhraseBank & financial news.
4. **Final Decision Metric:**
   $$\text{Signal Confidence} = w_1 S_{\text{tech}} + w_2 S_{\text{sent}}$$
   $$\text{Position Size} = \text{Base Size} \times (1 - S_{\text{vol}}) \times \text{Confidence}$$

### B. Institutional Risk Guardrails
- **Max Account Risk Per Trade:** 1.5% of total portfolio capital.
- **Dynamic Stop-Loss:** Calculated as $2 \times \text{ATR(14)}$ from entry price.
- **Risk-to-Reward Ratio (RRR):** Minimum 1 : 2 target.
- **Daily Drawdown Circuit Breaker:** If daily losses exceed 4%, halt all automated alert signals until 00:00 UTC.

---

## 6. Telegram Bot UI/UX Requirements

- **Framework:** `aiogram` v3 (Modern Asyncio architecture with FSM & routers).
- **Security:** Whitelist `TELEGRAM_ALLOWED_USER_IDS`. Block unauthenticated users.
- **Command Set:**
  - `/start`: Interactive dashboard with quick-action inline buttons.
  - `/signal <symbol>`: Run real-time AI ensemble prediction, generate TradingView chart image, and return signal breakdown card (Action, Confidence, Stop Loss, Take Profit).
  - `/sentiment <symbol>`: Displays current news sentiment index, bullish/bearish ratio, and recent headlines.
  - `/volatility <symbol>`: Current realized volatility vs historical distribution.
  - `/portfolio`: Show active paper/live positions, unrealized PnL, and Sharpe ratio.
  - `/backtest <symbol> <days>`: Triggers backtest simulation and returns performance report with equity curve.
- **Message Format:** Clean HTML or MarkdownV2 cards with status badges (🟢 `STRONG BUY`, 🔴 `STRONG SELL`, 🟡 `NEUTRAL / HOLD`).

---

## 7. Developer Rules & Conventions
1. **Asynchronous First:** All network I/O, database queries, and Telegram requests must use `asyncio`, `aiohttp`, or `httpx`.
2. **Strict Typing:** Python 3.11+ type hints (`typing.Optional`, `Union`, `Literal`, Pydantic models) everywhere.
3. **Defensive Error Handling:** Webhook endpoints must never crash or return 500 without logging context; fail gracefully with structured error messages.
4. **Environment Isolation:** Zero hardcoded credentials. All API tokens, chat IDs, and secrets must load from `.env`.
