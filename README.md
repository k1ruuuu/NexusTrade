# NexusTrade AI — Institutional Quantitative Telegram Trading Bot

**NexusTrade AI** is an autonomous, institutional-grade quantitative trading terminal for Telegram that combines **5-year historical + real-time market memory**, **online continual news sentiment learning**, **TradingView webhook ingestion**, and **automated 1200x800 chart rendering**.

---

## Key Capabilities

### 1. Multimodal AI Ensemble & 5-Year Memory (`ai_engine/`)
- **Technical Directional Classifier ($S_{\text{tech}} \in [-1, 1]$):**
  - Powered by LightGBM (`OHLCVPredictor`) trained continuously on merged **5-year daily OHLCV history** (`~1,260–1,825` daily bars) and **live intraday candlesticks** (`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1W`).
  - Zero-lookahead (`shift(1)`) multi-horizon feature engineering across `RSI(14)`, `MACD(12,26,9)`, `Bollinger Bands(20,2)`, `ATR(14)`, `EMA 20/50/200`, `SMA 50/200`, `VWAP`, `Donchian(20)`, efficiency ratios, and multi-horizon momentum/volatility ratios (`>83%–88%` chronological validation accuracy).
- **Realized Volatility Estimator ($S_{\text{vol}} \in [0, 1]$):**
  - Optiver-inspired Heterogeneous Autoregressive Realized Volatility (`HAR-RV`) regressor (`RealizedVolatilityEstimator`) modeling Level-1 WAP log-return variance, Parkinson high-low range estimator, and bid-ask spread dynamics to classify regimes (`LOW`, `NORMAL`, `ELEVATED`, `EXTREME`).
- **Continual Learning Financial Sentiment Engine ($S_{\text{sent}} \in [-1, 1]$):**
  - Hybrid FinBERT (`ProsusAI/finbert`) + FinancialPhraseBank lexicon + Online Incremental `SGDClassifier(loss="log_loss")` (`partial_fit`) that continuously harvests live financial headlines (Yahoo Finance + Google News RSS) and updates weights against real market price reaction.

### 2. Institutional Risk & Portfolio Protection (`ai_engine/risk_management/`)
- **Dynamic ATR Stop-Loss & Reward-to-Risk Targets:** Automatic $2 \times \text{ATR(14)}$ Stop-Loss, minimum `1:2` (`TP1`) and `1:3` (`TP2`) Reward-to-Risk targets, and ratcheting trailing stops (`StopLossManager`).
- **Automatic Support & Resistance Detection (`SupportResistanceDetector`):** Combines volume-weighted Swing High/Low fractals, multi-window Donchian channels (`20`/`50`/`120` bars), and Classic Floor Pivot Points to compute structural `Support Bawah (S1, S2)` and `Resistance Atas (R1, R2)` levels, displayed in Telegram signal cards and plotted on `1200x800` charts.
- **Volatility-Adjusted Half-Kelly Position Sizing:** Scales risk allocation by $(1 - S_{\text{vol}}) \times \text{Confidence}$, strictly capped at `1.5%` maximum equity risk per trade (`PositionSizer`).
- **Circuit Breakers:** Automatic trade suppression when daily drawdown reaches `4.0%` or market volatility enters `EXTREME` regime ($S_{\text{vol}} \ge 0.85$).

### 3. Multi-Asset Live & Kaggle Data Pipeline (`data_pipeline/`)
- **Live Market & 5-Year Historical Feeds (`LiveMarketDataFeed`):** Real-time OHLCV and 5-year historical Parquet caching (`data/processed/memory_5y_<symbol>.parquet`) via Binance Spot REST API and Yahoo Finance v8 API, with offline fallback to local `XAUUSD` (`15m`) and `50 Crypto` (`1d`) datasets.
- **Live News Harvester (`LiveNewsFeed`):** Concurrent async RSS aggregation from Yahoo Finance and Google News with exponential time-decay weighting ($w_i = \exp(-\lambda \Delta t_i)$).
- **Kaggle Ingestion & Parquet Feature Store:** Automated downloader (`KaggleDatasetDownloader`) with MD5 checksum verification and Zip-Slip protection, backed by `FeatureStore` (`float64 -> float32`, `int64 -> int32` downcasting).

### 4. Real-Time TradingView Integration (`tradingview/`)
- **Hardened FastAPI Webhook Server (`POST /api/v1/tradingview/webhook`):**
  - Constant-time `hmac.compare_digest` verification via `X-TradingView-Secret` header or payload `secret`.
  - Per-IP sliding-window rate limiting (`30 req / 60s`), `64 KB` payload cap, timestamp freshness replay protection, and security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Cache-Control: no-store`).
- **1200x800 Dark-Theme Chart Renderer (`ChartSnapshotGenerator`):** Renders candlesticks, EMA 20/50/200, Bollinger Bands, Volume, RSI(14), and annotated Entry / Stop-Loss / TP1 / TP2 levels.
- **Pine Script v5 Strategy (`tradingview/pine_script_sample.pine`):** Ready-to-attach TradingView strategy emitting structured JSON webhook alerts.

### 5. Interactive Telegram Bot Terminal (`telegram_bot/`)
- Built with `aiogram` v3 featuring `WhitelistAuthMiddleware`, sliding-window `RateLimitMiddleware`, human-friendly Indonesian HTML cards with ASCII progress bars, and interactive inline keyboards.
- **Multi-User `/setup` Vault & 6-Gate Smart Auto-Trader (`ai_engine/auto_trader/`):**
  - **Multi-Provider 2-Way Execution (`Exness Go / MetaTrader 5` & `Binance`):** Supports connecting **Exness Go / MetaTrader 5** (`/setup exness <login> <password> [Exness-MT5Trial|Exness-MT5Real]`) for **2-way `BUY` (Long) & `SELL` (Short)** across **all asset classes** (Crypto `BTCUSDm`, Gold `XAUUSDm`, Forex `EURUSDm`, US Stocks `NVDAm`, Indices `US500m`) with native server-side `Stop Loss` & `Take Profit`, alongside **Binance Spot / Futures**.
  - **Strict Admin vs. User Role Isolation:** Credentials in `.env` (`BINANCE_API_KEY` / `BINANCE_API_SECRET`) apply **strictly to Admin users**. Regular users must run `/setup` in Telegram to connect their own personal account (encrypted per user via `PBKDF2-HMAC-SHA256`, with automatic secret chat deletion).
  - **6-Gate Pre-Trade Checklist (`SmartAutoTraderEngine`):** Never opens a position blindly — requires (1) verified `/setup` account, (2) aligned breaking news & FinBERT sentiment, (3) symmetric high-conviction 5Y+live technical signal (`BUY` or `SELL`), (4) structural Support/Resistance clearance (`S1`/`R1`), (5) normal volatility & zero duplicate symbol position, and (6) automatic `2xATR` Stop-Loss & `TP1`/`TP2` locking.
- **Admin Control Panel (`/admin`):**
  - **Management User:** Add, edit expiration (`1 Hari`, `1 Minggu`, `1 Bulan`), and delete/revoke users via Telegram ID, with automatic middleware expiration enforcement.
  - **Broadcast:** Send custom official announcements to all active users.
  - **Maintenance Mode (`ON`/`OFF`):** Toggle server maintenance state; when `ON`, all non-admin user requests receive a maintenance notice while Admin retains full access.
  - **Audit Trail:** Full activity logging of every command, callback button press, admin operation, and denied/expired/maintenance event.
---

## Supported Markets & Symbols

| Asset Class | Example Tickers | Data Source |
| :--- | :--- | :--- |
| **Cryptocurrencies** | `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `AVAXUSDT`, `LINKUSDT` | Binance Spot API + Yahoo Finance 5Y + Local CSV |
| **Precious Metals & Commodities** | `XAUUSD` (Gold), `XAGUSD` (Silver), `WTI` / `USOIL` (Crude Oil), `NATGAS` | Yahoo Finance (`GC=F`, `SI=F`, `CL=F`, `NG=F`) + Local `XAUUSD_15m.csv` |
| **US Equities** | `NVDA`, `AAPL`, `MSFT`, `TSLA`, `AMZN`, `META`, `GOOGL`, `AMD` | Yahoo Finance v8 Chart & Headline API |
| **Global Indices** | `SPX` (`^GSPC`), `NDX` (`^NDX`), `DJI` (`^DJI`), `VIX` (`^VIX`) | Yahoo Finance v8 Chart & Headline API |
| **Major Forex Pairs** | `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `USDCAD`, `USDCHF` | Yahoo Finance (`=X` ISO FX pairs) |

---

## Project Structure

```text
NexusTrade/
├── AGENT.md                                  # System architecture & quantitative specifications
├── TASKS.md                                  # Phased implementation roadmap (Phases 1-6)
├── README.md                                 # Documentation & operational guide
├── main.py                                   # Unified entrypoint (FastAPI + Aiogram + Online Learner)
├── pyproject.toml                            # Python package, pytest, and ruff configuration
├── requirements.txt                          # Pinned production dependencies
├── .env.example                              # Environment variable template
├── Dockerfile                                # Production container image (Python 3.11-slim)
├── docker-compose.yml                        # Bot + localhost-bound Redis 7.2 orchestration
├── config/
│   ├── settings.py                           # Pydantic v2 BaseSettings & fail-fast secret validation
│   ├── logging_config.py                     # Structured Rich / JSON logging setup
│   └── tradingview_config.json               # Ticker normalization & webhook configuration
├── data_pipeline/
│   ├── market_symbols.py                     # Symbol/timeframe sanitization & Yahoo/Binance maps
│   ├── kaggle_downloader.py                  # Async Kaggle downloader with MD5 & Zip-Slip checks
│   ├── preprocessor_ohlcv.py                 # Zero-lookahead OHLCV indicators & multi-horizon features
│   ├── preprocessor_microstructure.py        # Optiver WAP/RV & Jane Street utility preprocessors
│   ├── preprocessor_sentiment.py             # Financial NLP cleaner, ticker mapper, & time-decay
│   ├── feature_store.py                      # Downcasted Apache Parquet storage & reader
│   ├── live_market_feed.py                   # Real-time + 5-year historical OHLCV harvester
│   └── live_news_feed.py                     # Async Yahoo Finance & Google News RSS harvester
├── ai_engine/
│   ├── models/
│   │   ├── ohlcv_predictor.py                # LightGBM multi-horizon directional classifier
│   │   ├── volatility_estimator.py           # Optiver-style HAR-RV realized volatility regressor
│   │   ├── sentiment_lexicon.py              # Financial lexicon, bootstrap corpus, & label mapper
│   │   ├── sentiment_analyzer.py             # FinBERT + Online Incremental SGD sentiment engine
│   │   ├── market_data_loader.py             # Local dataset & synthetic OHLCV loader
│   │   └── ensemble_agent.py                 # Multimodal ensemble orchestrator & 5Y memory trainer
│   ├── auto_trader/
│   │   ├── vault_crypto.py                       # PBKDF2-HMAC-SHA256 cipher & key-masking helpers
│   │   ├── vault_positions.py                    # Position records, SQLite position helpers, & live balance sync
│   │   ├── user_vault.py                         # Encrypted SQLite multi-user vault (Admin vs User isolation)
│   │   ├── admin_models.py                       # Managed user, expiry preset, & audit dataclasses
│   │   ├── admin_store.py                        # SQLite Admin store (User expiry, Maintenance, Broadcast, Audit)
│   │   ├── binance_client.py                     # Async HMAC-SHA256 Binance Spot & Futures client
│   │   ├── mt5_symbols.py                        # MT5 symbol mapper, lot calculator, & SL/TP anchor helper
│   │   ├── mt5_exness_client.py                  # MetaTrader 5 / Exness Go all-asset 2-way BUY/SELL client
│   │   ├── session_models.py                     # Duration/lot token parsers & session state dataclasses
│   │   ├── session_manager.py                    # Timed autonomous trading session & trailing-stop position holder
│   │   └── engine.py                             # 6-Gate Smart Auto-Trader execution engine
│   ├── risk_management/
│   │   ├── stop_loss_manager.py              # 2x ATR(14) Stop-Loss, TP1/TP2, & trailing stop engine
│   │   ├── support_resistance.py             # Swing High/Low, Donchian & Pivot S1/S2 & R1/R2 detector
│   │   └── position_sizer.py                 # Half-Kelly position sizer & drawdown circuit breaker
│   └── backtester/
│       ├── engine.py                         # Event-driven backtester & equity curve plotter
│       └── metrics.py                        # Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor
├── tradingview/
│   ├── alert_parser.py                       # Strict Pydantic v2 TradingView webhook validator
│   ├── chart_generator.py                    # 1200x800 dark-theme candlestick & indicator renderer
│   ├── webhook_server.py                     # Hardened FastAPI webhook receiver & dispatcher
│   └── pine_script_sample.pine               # Pine Script v5 institutional alert strategy
├── telegram_bot/
│   ├── bot.py                                # Aiogram v3 Bot & Dispatcher factory
│   ├── middlewares/
│   │   ├── auth_middleware.py                # TELEGRAM_ALLOWED_USER_IDS whitelist guard
│   │   └── rate_limit.py                     # Sliding-window anti-spam rate limiter
│   ├── handlers/
│   │   ├── handler_utils.py                  # Shared user identity, command args, & wizard state helpers
│   │   ├── commands.py                       # /start, /help, /status, /portfolio handlers
│   │   ├── signals.py                        # /signal, /analyze, /sentiment, /volatility, /backtest, /learn
│   │   ├── setup.py                          # /setup, /cancel, /myaccount, & /unlink handlers
│   │   ├── autotrade.py                      # /autotrade handler & setup/autotrade inline callbacks
│   │   ├── admin.py                          # /admin command router (User, Broadcast, Maintenance, Audit)
│   │   ├── admin_callbacks.py                # Interactive /admin wizard inputs & inline callbacks
│   │   └── callbacks.py                      # Interactive inline keyboard callback router
│   └── views/
│       ├── formatter_utils.py                # Progress bars, badges, & plain-Indonesian helpers
│       ├── formatters.py                         # HTML cards for Telegram messages
│       ├── setup_formatters.py                   # HTML cards for /setup wizard & 6-Gate Auto-Trade reports
│       └── admin_formatters.py                   # HTML cards for Admin Panel, User Management, & Audit logs
└── tests/
    ├── test_data_pipeline.py                 # Kaggle, zero-lookahead OHLCV, Optiver, & NLP tests
    ├── test_ai_engine.py                     # LightGBM, HAR-RV, FinBERT, Ensemble, & Backtester tests
    ├── test_webhook.py                       # AlertParser, 1200x800 chart, & FastAPI webhook tests
    └── test_telegram.py                      # Whitelist auth, rate limit, HTML cards, & handler tests
```

---

## Telegram Bot Commands

| Command | Description | Example |
| :--- | :--- | :--- |
| `/start` | Open the main interactive trading terminal & quick-action buttons | `/start` |
| `/help` | Display plain-language guide to all commands and indicators | `/help` |
| `/status` | Inspect AI engine readiness, 5Y memory cache, and circuit breakers | `/status` |
| `/signal <symbol> [tf]` | Generate full AI trade signal (`BUY`/`SELL`/`HOLD`), Entry/SL/TP, & chart | `/signal BTCUSDT 15m` |
| `/analyze <symbol> [tf]` | Alias for `/signal` with full technical + sentiment + volatility breakdown | `/analyze XAUUSD 1h` |
| `/sentiment <symbol>` | Run live news harvesting & FinBERT/SGD sentiment analysis | `/sentiment NVDA` |
| `/volatility <symbol> [tf]` | Compute Optiver HAR-RV realized volatility forecast & regime | `/volatility ETHUSDT 15m` |
| `/portfolio [equity]` | View position sizing limits, drawdown buffer, & Kelly parameters | `/portfolio 25000` |
| `/backtest [symbol] [days]` | Run event-driven simulation with fees/slippage & equity curve chart | `/backtest BTCUSDT 90` |
| `/learn [symbol] [tf]` | Trigger an immediate online learning cycle from live news & 5Y + live prices | `/learn XAUUSD 15m` |
| `/setup` | Connect personal Binance account (Testnet/Live) with auto-deleted secret chat | `/setup testnet KEY SECRET` |
| `/myaccount` | View personal Binance connection, role (`Admin` vs `User`), balance, & open positions | `/myaccount` |
| `/unlink` | Delete personal Binance credentials from the encrypted vault | `/unlink` |
| `/autotrade <sym> [dur]` | Run timed autonomous trading session (`5H`, `2H`, `30M`, `1D`) holding positions to TP1/TP2 with Trailing Stop | `/autotrade XAUUSD 5H` |
| `/admin` | Open Admin Control Panel (User Add/Edit/Delete `1d`/`1w`/`1m`, Broadcast, Maintenance `ON`/`OFF`, Audit) | `/admin` |

---

## Quick Start Guide

### 1. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` and set your credentials:
- `TELEGRAM_BOT_TOKEN` — Token from `@BotFather`
- `TELEGRAM_ALLOWED_USER_IDS` — Comma-separated Telegram numeric user IDs
- `TRADINGVIEW_WEBHOOK_SECRET` — Strong secret (`>= 16` chars in production) matching your Pine Script alert header
- `KAGGLE_USERNAME` / `KAGGLE_KEY` — Optional Kaggle API credentials for automated dataset downloads

### 3. Run Self-Check & Automated Test Suite
```bash
# Run full system self-check (Settings + 5Y AI Ensemble + 1200x800 Chart Render)
.venv/bin/python main.py --check

# Run the complete pytest suite
.venv/bin/pytest tests/ -v
```

### 4. Launch NexusTrade AI
```bash
# Run unified FastAPI Webhook Server + Aiogram v3 Bot + Background Online Learner
.venv/bin/python main.py

# Run only the FastAPI TradingView Webhook Server
.venv/bin/python main.py --webhook-only

# Run only the Telegram Bot polling loop
.venv/bin/python main.py --bot-only

# Or deploy via Docker Compose (NexusTrade Bot + Redis Cache)
docker compose up --build -d
```

---

## License
MIT License. Built for quantitative research, educational exploration, and algorithmic trading simulation.
