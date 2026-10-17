# Panduan Integrasi 4 EA MT5 ke NexusTrade AI

Repositori ini telah mengintegrasikan secara penuh 4 jenis Expert Advisor (EA) institusional dari paket `NX-EA-MT5`:

1. **NXBreak** (`Standalone/NXBreak.mq5`, `Experts/NXBreak.mq5`)
2. **NXParsial** (`Standalone/NXParsial.mq5`, `Experts/NXParsial.mq5`)
3. **NXReverse** (`Standalone/NXReverse.mq5`, `Experts/NXReverse.mq5`)
4. **NX-POC** (`Standalone/NX-POC.mq5`, `Experts/NX-POC.mq5`)

---

## 1. Struktur File di Proyek
```text
mt5_engine/NX-EA-MT5/
├── Standalone/
│   ├── NXBreak.mq5          # Single-file source NXBreak
│   ├── NXParsial.mq5        # Single-file source NXParsial (3-leg)
│   ├── NXReverse.mq5        # Single-file source NXReverse (1:5 RR & Recovery)
│   └── NX-POC.mq5           # Single-file source NX-POC (Volume Profile)
├── Experts/                 # Versi modular untuk MetaEditor
│   ├── NXBreak.mq5
│   ├── NXParsial.mq5
│   ├── NXReverse.mq5
│   └── NX-POC.mq5
├── Include/                 # Library modular MQL5
│   ├── NXInputs.mqh
│   ├── NXSignals.mqh
│   ├── NXTrade.mqh
│   └── NXEngine.mqh
├── SHA256SUMS.txt
├── STATIC-CHECKS.json
├── README-ID.md
├── FLOW-AND-REVIEW.md
└── INTEGRATION_GUIDE.md     # Panduan ini
```

---

## 2. Implementasi Python Engine (`ai_engine/auto_trader/nx_ea_engine.py`)

Logika matematika MQL5 telah direplikasi ke dalam modul Python murni:

### A. NXBreak (`EA_NX_BREAK`)
- **Prinsip:** Breakout candle berakurasi tinggi dengan filter bodi dan rentang volatilitas.
- **Filter Kualitas:**
  - Rentang candle sinyal: $0.4 \times ATR(14) \le \text{Range} \le 2.5 \times ATR(14)$.
  - Rasio bodi candle: $\frac{\text{Bodi}}{\text{Range}} \ge 55\%$.
  - Perhitungan $ATR(14)$ menggunakan *shift 2* agar candle pemicu tidak mendistorsi referensi volatilitasnya sendiri.
- **Kondisi Sinyal:**
  - **BUY:** Candle sinyal adalah *bullish* (`close > open`), harga pembukaan `open <= High[Candle 2]`, dan `close > High[Candle 2] + 0.05 * ATR`.
  - **SELL:** Candle sinyal adalah *bearish* (`close < open`), harga pembukaan `open >= Low[Candle 2]`, dan `close < Low[Candle 2] - 0.05 * ATR`.
- **Target & Proteksi:**
  - Stop Loss: 300 points (dibulatkan ke tick size).
  - Take Profit: 1:1 RR (300 points).

### B. NXParsial (`EA_NX_PARSIAL`)
- **Prinsip:** Menggunakan kriteria breakout dan kualitas yang identik dengan NXBreak, namun memecah eksekusi menjadi 3 leg independen dengan total budget risiko yang dibagi rata (`risk_budget / 3`).
- **Target Berjenjang (Multi-Target Basket):**
  - **Leg 1:** Stop Loss 300pt, Take Profit **1R** (300pt).
  - **Leg 2:** Stop Loss 300pt, Take Profit **3R** (900pt).
  - **Leg 3:** Stop Loss 300pt, Take Profit **6R** (1800pt).
- **Hasil Probabilistik:**
  - 3x SL: $-1.0R$
  - TP1 + 2x SL: $-0.333R$
  - TP1 + TP2 + 1x SL: $+1.0R$
  - TP1 + TP2 + TP3: $+3.333R$

### C. NXReverse (`EA_NX_REVERSE`)
- **Prinsip:** Mengikuti keselarasan tren multi-moving average, breakout SNR 20-bar, target tinggi 1:5 RR, dan sistem recovery switching jika terkena SL.
- **Konfluensi Tren:**
  - **BUY:** $EMA_{20} > EMA_{50} > EMA_{200}$ pada candle 1, kemiringan $EMA_{20}$ dan $EMA_{50}$ positif, kemiringan $EMA_{200} \ge 0$.
  - **SELL:** $EMA_{20} < EMA_{50} < EMA_{200}$ pada candle 1, kemiringan negatif.
- **Rolling Support/Resistance:**
  - Evaluasi 20 bar selesai (bar 2 hingga 21).
  - Breakout di atas Resistance $+ 0.05 \times ATR$ (BUY) atau di bawah Support $- 0.05 \times ATR$ (SELL).
- **Dynamic Swing Stop & Rasio Risk-Reward:**
  - Stop Loss: Titik terendah 10 bar terakhir $- 0.15 \times ATR$ (BUY) atau titik tertinggi $+ 0.15 \times ATR$ (SELL).
  - Jarak SL wajib berada di rentang $0.5 \times ATR$ hingga $3.0 \times ATR$.
  - Take Profit: **$5 \times \text{Jarak SL}$** ($1:5$ RR!).
- **State-Machine Recovery Switching:**
  - Jika trade normal ditutup karena kena SL, EA mengaktifkan jendela recovery selama 10 candle berikutnya untuk mencari setup berlawanan arah dengan $1:5$ RR.

### D. NX-POC (`EA_NX_POC`)
- **Prinsip:** Profil volume horizontal 100 bar dengan histogram 40-bin untuk mencari level harga paling banyak diperdagangkan (*Point of Control* / POC), lalu entry saat terjadi pemantulan (*rejection*).
- **Filter Histogram & ADX:**
  - Profil dihitung pada bar 2 hingga 101 (100 bar, tidak termasuk candle sinyal).
  - Histogram 40 bin harga; puncak volume bin tertinggi wajib mendominasi $\ge 1.2 \times$ rata-rata volume per bin (menghindari profil datar/bimodal).
  - $ADX(14) \le 25.0$ (memastikan pasar berada dalam fase konsolidasi/ranging).
- **Pinbar Rejection:**
  - Jarak harga ke POC wajib berada di rentang $0.5 \times ATR$ hingga $2.0 \times ATR$.
  - **BUY Rejection:** Harga berada di bawah POC, candle sinyal *bullish*, lower wick $\ge 35\%$ dari total rentang candle, dan `close > close[bar 2]`.
  - **SELL Rejection:** Harga berada di atas POC, candle sinyal *bearish*, upper wick $\ge 35\%$ dari total rentang candle, dan `close < close[bar 2]`.
- **Target:** Menuju POC ($1:1$ RR simetris).

---

## 3. Perintah Telegram Bot & Interaksi UI
Bot Telegram Aiogram v3 mendukung perintah langsung untuk ke-4 strategi ini:

| Perintah | Alias | Strategi & Parameter |
|---|---|---|
| `/ea_break <simbol> [lot]` | `/nxbreak`, `/ea-break` | Menjalankan NXBreak Breakout M1/M5 (SL 300pt, TP 1:1 RR) |
| `/ea_parsial <simbol> [lot]` | `/nxparsial`, `/ea-parsial` | Menjalankan NXParsial 3-Leg Basket (1R, 3R, 6R) |
| `/ea_reverse <simbol> [lot]` | `/nxreverse`, `/ea-reverse` | Menjalankan NXReverse EMA20/50/200 + 1:5 RR & Recovery |
| `/ea_poc <simbol> [lot]` | `/nxpoc`, `/ea-poc` | Menjalankan NX-POC Volume Profile Rejection (100-bar, 40-bin) |

Contoh pemakaian:
```text
/ea_break XAUUSD
/ea_parsial BTCUSD 0.03
/ea_reverse EURUSD 20m
/ea_poc XAUUSD 1h
```

---

## 4. Verifikasi Test Suite (`pytest`)
Semua logika matematika dan integrasi dieksekusi secara otomatis melalui:
```bash
.venv/bin/pytest tests/test_nx_ea_suite.py -v
```
Status: **6 Passed (100% Green)**.
