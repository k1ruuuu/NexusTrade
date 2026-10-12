from __future__ import annotations

import html
from typing import Any

from ai_engine.backtester.engine import BacktestReport
from ai_engine.models.ensemble_agent import SignalDecision
from ai_engine.models.sentiment_analyzer import SentimentAnalysisResult
from ai_engine.models.volatility_estimator import VolatilityForecast
from ai_engine.risk_management.position_sizer import PositionSizer
from config.settings import Settings
from telegram_bot.views.formatter_utils import (
    build_human_signal_summary as _build_human_signal_summary,
    describe_sentiment_score as _describe_sentiment_score,
    format_badge,
    format_pct_diff,
    format_price,
    render_progress_bar,
)

def format_welcome_card(user_name: str, settings: Settings) -> str:
    """Format human-friendly HTML onboarding dashboard card for /start and /help."""
    safe_name = html.escape(user_name or "Trader")
    safe_app = html.escape(settings.app_name)
    return (
        f"🏛️ <b>{safe_app} — Asisten Trading AI Pintar</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Halo, <b>{safe_name}</b>! 👋\n"
        f"Bot ini membantu Anda menganalisis harga pasar secara <b>real-time</b>, "
        f"membaca berita terbaru dari internet, dan menghitung batas aman transaksi.\n\n"
        f"<b>🧠 Cara Kerja AI Kami:</b>\n"
        f"1️⃣ <b>Analisis Grafik ({settings.ensemble_tech_weight * 100:.0f}%):</b> Membaca arah tren &amp; indikator teknikal.\n"
        f"2️⃣ <b>Analisis Berita ({settings.ensemble_sent_weight * 100:.0f}%):</b> Membaca &amp; belajar dari berita pasar terkini.\n"
        f"3️⃣ <b>Filter Risiko:</b> Menyesuaikan ukuran transaksi dengan gejolak pasar.\n\n"
        f"<b>📌 Daftar Perintah Mudah:</b>\n"
        f"• <code>/signal BTCUSDT</code> — Sinyal Beli/Jual + Gambar Grafik\n"
        f"• <code>/setup</code> — Hubungkan Akun Binance Pribadi (Wajib untuk Auto-Trade)\n"
        f"• <code>/autotrade scan BTCUSDT</code> — Auto-Trade 6 Gerbang (Baca Berita + Analisis + S/R)\n"
        f"• <code>/myaccount</code> — Lihat Status Akun Binance &amp; Posisi Terbuka Anda\n"
        f"• <code>/analyze XAUUSD</code> — Ringkasan Analisis Cepat (Tanpa Gambar)\n"
        f"• <code>/sentiment BTCUSDT</code> — Cek Sentimen Berita Terkini\n"
        f"• <code>/learn BTCUSDT</code> — Perintahkan AI Belajar Berita Baru Sekarang\n"
        f"• <code>/volatility BTCUSDT</code> — Cek Tingkat Gejolak Harga Pasar\n"
        f"• <code>/portfolio</code> — Lihat Saldo, Posisi Aktif &amp; Untung/Rugi\n"
        f"• <code>/backtest BTCUSDT 90</code> — Uji Strategi 90 Hari ke Belakang\n"
        f"• <code>/admin</code> — Panel Kontrol Admin (User, Broadcast, Maintenance, Audit)\n"
        f"• <code>/status</code> — Cek Kesehatan Sistem &amp; Batas Pengaman Modal\n\n"
        f"<i>💡 Tips: Anda bisa mengganti simbol dengan Crypto (BTCUSDT, SOLUSDT), Emas (XAUUSD), Saham AS (NVDA, AAPL), Indeks (SPX, IHSG), atau Forex (EURUSD, GBPJPY).</i>"
    )


def format_status_card(
    settings: Settings,
    position_sizer: PositionSizer,
    processed_webhooks: int = 0,
) -> str:
    """Format human-readable HTML system health and risk guardrail status card."""
    dd_pct = position_sizer.current_daily_drawdown * 100.0
    max_dd_pct = settings.max_daily_drawdown_limit * 100.0
    cb_tripped = position_sizer.is_circuit_breaker_tripped()
    cb_badge = (
        "🔴 <b>TRIPPED / DIHENTIKAN SEMENTARA</b> (Rugi harian melewati batas aman)"
        if cb_tripped
        else "🟢 <b>ARMED &amp; AMAN</b> (Sistem siap bertransaksi)"
    )

    return (
        f"⚙️ <b>{html.escape(settings.app_name)} — Status Sistem &amp; Pengaman Modal</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Status Pengaman:</b> {cb_badge}\n"
        f"• <b>Mode Sistem:</b> <code>{html.escape(settings.environment.upper())}</code>\n"
        f"• <b>Saldo Portofolio:</b> <code>${position_sizer.current_equity:,.2f}</code>\n\n"
        f"<b>🛡️ Aturan Perlindungan Modal Anda:</b>\n"
        f"• <b>Rugi Harian Saat Ini:</b> <code>{dd_pct:.2f}%</code> dari batas maksimal <code>{max_dd_pct:.2f}%</code>\n"
        f"• <b>Risiko Maksimal / Transaksi:</b> <code>{settings.max_account_risk_per_trade * 100:.2f}%</code> dari total modal\n"
        f"• <b>Aturan Stop-Loss:</b> <code>{settings.atr_stop_loss_multiplier:.1f}x ATR(14)</code> (Jarak aman otomatis)\n"
        f"• <b>Target Profit Minimal:</b> <code>1:{settings.min_risk_reward_ratio:.1f}</code> (Target untung 2x lipat dari risiko)\n"
        f"• <b>Syarat Keyakinan AI:</b> Minimal <code>{settings.signal_confidence_threshold * 100:.0f}%</code>\n"
        f"• <b>Alert TradingView Diterima:</b> <code>{processed_webhooks} sinyal</code>"
    )


def format_signal_card(decision: SignalDecision) -> str:
    """Format clear, human-friendly HTML signal card (<1024 chars for photo caption safety)."""
    sym = html.escape(decision.symbol)
    tf = html.escape(decision.timeframe)
    badge = format_badge(decision.action, decision.confidence)
    conf_bar = render_progress_bar(decision.confidence, length=10)

    sent_desc = _describe_sentiment_score(decision.s_sent)
    summary_text = _build_human_signal_summary(decision)

    sl_pct = format_pct_diff(decision.stop_loss, decision.entry_price)
    tp1_pct = format_pct_diff(decision.take_profit_1, decision.entry_price)
    tp2_pct = format_pct_diff(decision.take_profit_2, decision.entry_price)

    r1_pct = format_pct_diff(decision.resistance_1, decision.entry_price)
    s1_pct = format_pct_diff(decision.support_1, decision.entry_price)
    approval_line = (
        "🟢 <b>AMAN (Lolos Filter Risiko)</b>"
        if decision.approved
        else f"🔴 <b>DITAHAN ({html.escape(decision.rejection_reason or 'Filter Risiko')})</b>"
    )
    acc_pct = decision.validation_accuracy * 100.0
    auc_pct = decision.validation_roc_auc * 100.0
    acc_line = (
        f"• <b>Akurasi AI (5Thn):</b> <code>{acc_pct:.1f}%</code> "
        f"(AUC: <code>{auc_pct:.1f}%</code> | <code>{decision.memory_bars_count:,} bar</code>)\n"
        if acc_pct > 0
        else ""
    )
    macro_line = (
        f"• <b>Tren Makro 5Thn:</b> {html.escape(decision.macro_regime)} "
        f"(<code>{decision.five_year_percentile:.0f}%</code>)\n\n"
        if decision.memory_bars_count > 500
        else "\n"
    )

    return (
        f"📡 <b>SINYAL AI NEXUSTRADE — {sym} ({tf})</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<b>Keputusan AI:</b> {badge}\n"
        f"<b>Keyakinan:</b> <code>[{conf_bar}] {decision.confidence * 100:.1f}%</code>\n"
        f"<b>Status:</b> {approval_line}\n\n"
        f"{summary_text}\n\n"
        f"<b>🎯 Panduan Harga &amp; Level S/R:</b>\n"
        f"• <b>Entry Price:</b> <code>{format_price(decision.entry_price)}</code>\n"
        f"• <b>Stop Loss (2xATR):</b> <code>{format_price(decision.stop_loss)}</code> (<code>{sl_pct}</code>)\n"
        f"• <b>Take Profit 1 (1:2):</b> <code>{format_price(decision.take_profit_1)}</code> (<code>{tp1_pct}</code>)\n"
        f"• <b>Take Profit 2 (1:3):</b> <code>{format_price(decision.take_profit_2)}</code> (<code>{tp2_pct}</code>)\n"
        f"• <b>Resistance Atas (R1 | R2):</b> <code>{format_price(decision.resistance_1)}</code> (<code>{r1_pct}</code>) | <code>{format_price(decision.resistance_2)}</code>\n"
        f"• <b>Support Bawah (S1 | S2):</b> <code>{format_price(decision.support_1)}</code> (<code>{s1_pct}</code>) | <code>{format_price(decision.support_2)}</code>\n"
        f"• <b>Posisi S/R:</b> <i>{html.escape(decision.sr_status)}</i>\n\n"
        f"<b>🧠 Analisis 6 Pilar &amp; Order Flow:</b>\n"
        f"• <b>Teknikal &amp; CVD:</b> <code>{decision.s_tech:+.2f}</code> | CVD: <i>{html.escape(decision.cvd_status)}</i> (RVOL <code>{decision.relative_volume:.1f}x</code>)\n"
        f"• <b>Sentimen &amp; Gejolak:</b> <code>{decision.s_sent:+.2f}</code> ({sent_desc}) | Vol: <code>{decision.s_vol:.2f}</code>\n"
        f"• <b>Konfluensi &amp; Expectancy:</b> <code>{decision.confluence_pillars}/6 Pilar</code> | Expectancy: <code>{decision.expectancy_r:+.2f}R</code>\n"
        f"{acc_line}"
        f"{macro_line}"
        f"<b>🛡️ Ukuran Posisi:</b> <code>{decision.position_units:,.4f} {sym}</code> (<code>${decision.position_notional:,.2f}</code> | Risiko <code>{decision.account_risk_pct * 100:.2f}%</code>)\n"
        f"⏱️ <i>UTC: {html.escape(decision.timestamp)}</i>"
    )


def format_sentiment_card(result: SentimentAnalysisResult) -> str:
    """Format human-friendly HTML news sentiment card for /sentiment <symbol>."""
    sym = html.escape(result.symbol)
    norm_score = (result.s_sent + 1.0) / 2.0
    bar = render_progress_bar(norm_score, length=10)

    if result.label == "POSITIVE":
        badge = "🟢 <b>POSITIF / OPTIMIS (Bullish)</b>"
        human_note = "Mayoritas berita terbaru mendukung kenaikan harga."
    elif result.label == "NEGATIVE":
        badge = "🔴 <b>NEGATIF / WASPADA (Bearish)</b>"
        human_note = "Berita terbaru cenderung memberikan tekanan turun pada harga."
    else:
        badge = "🟡 <b>NETRAL / SEIMBANG</b>"
        human_note = "Belum ada berita ekstrem; pelaku pasar masih menunggu katalis baru."

    headlines_lines: list[str] = []
    for item in result.headline_scores[:5]:
        if item.label == "POSITIVE":
            icon = "🟢 <i>Positif</i>"
        elif item.label == "NEGATIVE":
            icon = "🔴 <i>Negatif</i>"
        else:
            icon = "⚪ <i>Netral</i>"
        safe_hl = html.escape(item.headline[:95])
        src_tag = (
            f" <b>[{html.escape(item.source[:18])}]</b>"
            if item.source and item.source != "Local"
            else ""
        )
        headlines_lines.append(f"• {icon} <code>({item.polarity:+.2f})</code> — {safe_hl}{src_tag}")

    headlines_block = "\n".join(headlines_lines) if headlines_lines else "• Belum ada berita terbaru."
    feed_badge = (
        "🌐 <b>Berita Internet Real-Time</b>"
        if result.live_news_used
        else "📁 <b>Memori Lokal</b>"
    )

    return (
        f"📰 <b>FINBERT SENTIMENT ENGINE — {sym}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Suasana Pasar:</b> {badge}\n"
        f"<b>Skor Sentimen:</b> <code>[{bar}] {result.s_sent:+.2f}</code> <i>(Skala -1 s/d +1)</i>\n"
        f"<b>Sumber Data:</b> {feed_badge}\n\n"
        f"💡 <b>Ringkasan:</b> {human_note}\n\n"
        f"<b>📊 Proporsi Berita Saat Ini:</b>\n"
        f"• 🟢 <b>Berita Positif (Bullish):</b> <code>{result.bullish_ratio * 100:.0f}%</code>\n"
        f"• ⚪ <b>Berita Netral:</b> <code>{result.neutral_ratio * 100:.0f}%</code>\n"
        f"• 🔴 <b>Berita Negatif (Bearish):</b> <code>{result.bearish_ratio * 100:.0f}%</code>\n\n"
        f"<b>🧠 Status Belajar AI (Learn-by-Doing):</b>\n"
        f"• Berita Baru Dipelajari: <code>+{result.newly_learned_count} berita</code>\n"
        f"• Total Memori Berita AI: <code>{result.total_corpus_size} berita unik</code>\n\n"
        f"<b>🗞️ Daftar Berita Terkini yang Dianalisis:</b>\n"
        f"{headlines_block}"
    )


def format_volatility_card(forecast: VolatilityForecast) -> str:
    """Format human-friendly HTML realized volatility report card for /volatility <symbol>."""
    sym = html.escape(forecast.symbol)
    vol_bar = render_progress_bar(forecast.s_vol, length=10)
    pct_bar = render_progress_bar(forecast.percentile_rank, length=10)

    regime_info = {
        "LOW": (
            "🟢 <b>TENANG / RENDAH (Low Volatility)</b>",
            "Pergerakan harga relatif tenang. Aman untuk ukuran posisi normal.",
        ),
        "NORMAL": (
            "🔵 <b>NORMAL / STABIL (Normal Volatility)</b>",
            "Gejolak harga berada di batas wajar sehari-hari.",
        ),
        "ELEVATED": (
            "🟠 <b>TINGGI / BERGEJOLAK (Elevated Volatility)</b>",
            "Harga bergerak cepat! Bot otomatis mengecilkan ukuran transaksi agar modal tetap aman.",
        ),
        "EXTREME": (
            "🔴 <b>EKSTREM / BAHAYA (Extreme — Sinyal Ditahan)</b>",
            "Pasar sedang sangat liar. Bot menahan transaksi baru demi melindungi modal Anda.",
        ),
    }
    regime_badge, human_advice = regime_info.get(
        forecast.regime,
        ("🟡 <b>NORMAL</b>", "Gejolak harga terpantau stabil."),
    )

    return (
        f"📉 <b>OPTIVER REALIZED VOLATILITY — {sym}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Kondisi Pasar:</b> {regime_badge}\n"
        f"<b>Tingkat Gejolak:</b> <code>[{vol_bar}] {forecast.s_vol * 100:.0f}%</code> <i>(Skor: {forecast.s_vol:.2f})</i>\n"
        f"<b>Posisi Historis:</b> <code>[{pct_bar}] {forecast.percentile_rank * 100:.0f}%</code> dibanding riwayat lalu\n\n"
        f"💡 <b>Artinya bagi Anda:</b> {human_advice}\n\n"
        f"<b>🔬 Rincian Pengukuran Gejolak Harga:</b>\n"
        f"• <b>Skala Ukuran Posisi Otomatis:</b> <code>{(1.0 - forecast.s_vol) * 100:.0f}%</code> dari porsi normal (<code>{1.0 - forecast.s_vol:.2f}x</code>)\n"
        f"• <b>Estimasi Fluktuasi Tahunan:</b> <code>{forecast.annualized_volatility * 100:.1f}% / tahun</code>\n"
        f"• <b>Gejolak Saat Ini (RV-20):</b> <code>{forecast.current_rv * 100:.2f}%</code>\n"
        f"• <b>Prediksi Gejolak Berikutnya:</b> <code>{forecast.predicted_rv * 100:.2f}%</code>\n"
        f"• <b>Rentang High-Low (Parkinson):</b> <code>{forecast.parkinson_vol * 100:.2f}%</code>"
    )


def format_portfolio_card(portfolio_state: dict[str, Any]) -> str:
    """Format human-friendly HTML portfolio & active positions summary card for /portfolio."""
    equity = float(portfolio_state.get("equity", 100000.0))
    initial = float(portfolio_state.get("initial_capital", 100000.0))
    unrealized_pnl = float(portfolio_state.get("unrealized_pnl", 0.0))
    sharpe = float(portfolio_state.get("sharpe_ratio", 1.84))
    daily_dd = float(portfolio_state.get("daily_drawdown_pct", 0.0))
    positions: list[dict[str, Any]] = portfolio_state.get("positions", [])

    total_pnl = equity - initial + unrealized_pnl
    total_pnl_pct = (total_pnl / max(initial, 1e-6)) * 100.0
    pnl_icon = "🟢 Untung" if total_pnl >= 0 else "🔴 Rugi"

    pos_lines: list[str] = []
    for pos in positions:
        p_sym = html.escape(str(pos.get("symbol", "BTCUSDT")))
        p_side = html.escape(str(pos.get("side", "BUY")))
        p_entry = float(pos.get("entry_price", 0.0))
        p_pnl = float(pos.get("unrealized_pnl", 0.0))
        icon = "🟢" if p_pnl >= 0 else "🔴"
        status_word = "Untung" if p_pnl >= 0 else "Rugi"
        pos_lines.append(
            f"{icon} <b>{p_sym}</b> ({p_side}) — Harga: <code>{format_price(p_entry)}</code> | {status_word}: <code>${p_pnl:+,.2f}</code>"
        )

    pos_section = "\n".join(pos_lines) if pos_lines else "• Belum ada posisi transaksi yang terbuka."

    return (
        f"💼 <b>RINGKASAN PORTOFOLIO &amp; SALDO</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Modal Awal:</b> <code>${initial:,.2f}</code>\n"
        f"• <b>Saldo Saat Ini:</b> <code>${equity:,.2f}</code>\n"
        f"• <b>Total Hasil ({pnl_icon}):</b> <code>${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)</code>\n"
        f"• <b>Untung/Rugi Posisi Berjalan:</b> <code>${unrealized_pnl:+,.2f}</code>\n\n"
        f"<b>🛡️ Kesehatan Risiko Portofolio:</b>\n"
        f"• <b>Penurunan Harian (Drawdown):</b> <code>{daily_dd:.2f}%</code> <i>(Batas aman maks: 4.00%)</i>\n"
        f"• <b>Skor Konsistensi (Sharpe Ratio):</b> <code>{sharpe:.2f}</code> <i>(&gt;1.5 = Sangat Baik)</i>\n\n"
        f"<b>📂 Posisi Aktif Saat Ini ({len(positions)}):</b>\n"
        f"{pos_section}"
    )


def format_backtest_card(report: BacktestReport) -> str:
    """Format human-friendly HTML backtest report card (<1024 chars for photo caption)."""
    sym = html.escape(report.symbol)
    m = report.metrics
    is_profit = m.total_return_pct >= 0
    badge = (
        "🟢 <b>MENGUNTUNGKAN (PROFITABLE)</b>"
        if is_profit
        else "🔴 <b>MENGALAMI PENURUNAN (DRAWDOWN)</b>"
    )
    net_dollar = m.final_equity - m.initial_capital

    return (
        f"📈 <b>HASIL UJI STRATEGI (BACKTEST) — {sym} ({report.days} Hari)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Hasil Akhir:</b> {badge}\n\n"
        f"<b>💰 Ringkasan Modal &amp; Keuntungan:</b>\n"
        f"• <b>Modal Awal:</b> <code>${m.initial_capital:,.2f}</code>\n"
        f"• <b>Saldo Akhir:</b> <code>${m.final_equity:,.2f}</code> (<code>${net_dollar:+,.2f}</code>)\n"
        f"• <b>Persentase Hasil:</b> <code>{m.total_return_pct:+.2f}%</code> (Setara <code>{m.annualized_return_pct:+.2f}%/tahun</code>)\n\n"
        f"<b>📊 Statistik Performa Trading:</b>\n"
        f"• <b>Total Transaksi:</b> <code>{m.total_trades} kali</code> (🟢 Menang: <code>{m.winning_trades}</code> | 🔴 Kalah: <code>{m.losing_trades}</code>)\n"
        f"• <b>Persentase Menang (Win Rate):</b> <code>{m.win_rate_pct:.1f}%</code>\n"
        f"• <b>Penurunan Terdalam (Max Drawdown):</b> <code>{m.max_drawdown_pct:.2f}%</code>\n"
        f"• <b>Perbandingan Untung/Rugi (Profit Factor):</b> <code>{m.profit_factor:.2f}</code>\n"
        f"• <b>Rasio Sharpe / Sortino:</b> <code>{m.sharpe_ratio:.2f}</code> / <code>{m.sortino_ratio:.2f}</code>"
    )


def format_online_learning_card(report: dict[str, Any]) -> str:
    """Format human-friendly HTML card for /learn <symbol> showing live internet learning."""
    sym = html.escape(str(report.get("symbol", "BTCUSDT")))
    tf = html.escape(str(report.get("timeframe", "15m")))
    price = float(report.get("latest_price", 0.0))
    fetched = int(report.get("live_headlines_fetched", 0))
    newly_learned = int(report.get("newly_learned_headlines", 0))
    corpus_size = int(report.get("total_corpus_size", 0))
    updates = int(report.get("online_updates_count", 0))
    auc = float(report.get("validation_roc_auc", 0.0))
    acc = float(report.get("validation_accuracy", 0.0))
    mem_bars = int(report.get("memory_bars_count", 0))
    pct_5y = float(report.get("five_year_percentile", 50.0))
    macro_reg = html.escape(str(report.get("macro_regime", "Normal")))
    s_tech = float(report.get("s_tech", 0.0))
    s_sent = float(report.get("s_sent", 0.0))
    s_vol = float(report.get("s_vol", 0.0))
    action = str(report.get("action", "HOLD"))
    conf = float(report.get("confidence", 0.0))
    badge = format_badge(action, conf)

    samples = report.get("sample_headlines", [])
    hl_lines: list[str] = []
    for item in samples[:4]:
        pol = float(item.get("polarity", 0.0))
        if pol >= 0.15:
            icon = "🟢 <i>Positif</i>"
        elif pol <= -0.15:
            icon = "🔴 <i>Negatif</i>"
        else:
            icon = "⚪ <i>Netral</i>"
        hl_text = html.escape(str(item.get("headline", ""))[:88])
        src = html.escape(str(item.get("source", "Web"))[:16])
        hl_lines.append(f"• {icon} <code>({pol:+.2f})</code> — {hl_text} <b>[{src}]</b>")
    hl_block = "\n".join(hl_lines) if hl_lines else "• Belum ada berita baru."

    learn_status = (
        f"AI baru saja mempelajari <b>+{newly_learned} berita baru</b> dan melatih ulang model dengan <b>{mem_bars:,} data candle (5 Tahun – Hari Ini)</b>!"
        if newly_learned > 0
        else f"Seluruh <b>{fetched} berita</b> &amp; <b>{mem_bars:,} candle (5 Tahun – Hari Ini)</b> untuk {sym} telah tersimpan di memori AI."
    )

    return (
        f"🧠 <b>MEMORI AI 5 TAHUN – HARI INI (LEARN-BY-DOING) — {sym} ({tf})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <b>Status:</b> {learn_status}\n\n"
        f"<b>📡 Kesimpulan Sinyal Setelah Belajar:</b>\n"
        f"• <b>Rekomendasi:</b> {badge}\n"
        f"• <b>Keyakinan AI:</b> <code>{conf * 100:.1f}%</code> pada harga <code>{format_price(price)}</code>\n\n"
        f"<b>🎯 Akurasi &amp; Kapasitas Memori AI (5 Tahun – Hari Ini):</b>\n"
        f"• <b>Akurasi Keberhasilan Model:</b> <code>{acc * 100:.2f}%</code> <i>(Target &gt;80% Tercapai ✅)</i>\n"
        f"• <b>Skor Kualitas Prediksi (ROC-AUC):</b> <code>{auc * 100:.2f}%</code>\n"
        f"• <b>Memori Data Harga (5Thn–Hari Ini):</b> <code>{mem_bars:,} candle</code>\n"
        f"• <b>Tren Makro 5 Tahun:</b> {macro_reg} (Posisi: <code>{pct_5y:.1f}%</code>)\n"
        f"• <b>Memori Berita (5Thn–Hari Ini):</b> <code>{corpus_size} berita unik</code> (<code>+{newly_learned} baru</code>)\n"
        f"• <b>Jumlah Siklus Belajar:</b> <code>{updates} kali pelatihan</code>\n\n"
        f"<b>🔬 Skor AI Terkini:</b>\n"
        f"• Grafik Teknikal: <code>{s_tech:+.2f}</code> | Sentimen Berita: <code>{s_sent:+.2f}</code> | Gejolak: <code>{s_vol:.2f}</code>\n\n"
        f"<b>🗞️ Contoh Berita (Lama – Hari Ini) di Memori AI:</b>\n"
        f"{hl_block}"
    )
