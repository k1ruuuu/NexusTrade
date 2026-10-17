//+------------------------------------------------------------------+
//|                                                  EA_Scalping.mq5 |
//|                                  NexusTrade AI Quantitative Team |
//|                                       https://nexustrade.ai      |
//+------------------------------------------------------------------+
#property copyright "NexusTrade AI Quantitative Team"
#property link      "https://nexustrade.ai"
#property version   "3.00"
#property description "EA Scalping MT5 (5-MA & MTF H1/M5): Action murni pada 1M/M1, analisa tren H1 & M5, 5 MA (Biru cross Hijau ke atas curam BUY, Biru cross Merah ke bawah curam SELL), SL/TP saling berdekatan, & auto re-entry seketika."
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Input Parameters
input group "=== Parameter Risiko & Lot ==="
input double   InpLotSize             = 0.01;      // Ukuran Lot Transaksi
input double   InpStopLossPips        = 12.0;      // Stop Loss (Pips - Berdekatan)
input double   InpTakeProfitPips      = 14.0;      // Take Profit (Pips - Berdekatan)
input ulong    InpMagicNumber         = 20261017;  // Magic Number EA Scalping
input ulong    InpSlippage            = 10;        // Max Slippage (Points)

input group "=== Timeframe Action & 5 Jenis MA ==="
input ENUM_TIMEFRAMES InpActionTimeframe     = PERIOD_M15;// Timeframe Action/Eksekusi (Default M15)
input int      InpMABlue              = 5;         // MA 1: Biru (EMA Cepat Signal)
input int      InpMAGreen             = 8;         // MA 2: Hijau (EMA Target Bullish Cross)
input int      InpMARed               = 13;        // MA 3: Merah (EMA Target Bearish Cross)
input int      InpMAYellow            = 34;        // MA 4: Kuning (Mid Baseline)
input int      InpMAPurple            = 55;        // MA 5: Ungu (Major Baseline)

input group "=== Analisa Tren Multi-Timeframe (H1 & M5 Utama) ==="
input int      InpH1FastEMA           = 20;        // H1 Fast EMA
input int      InpH1SlowEMA           = 50;        // H1 Slow EMA
input int      InpM5FastEMA           = 9;         // M5 Fast EMA
input int      InpM5SlowEMA           = 21;        // M5 Slow EMA

input group "=== Parameter Candlestick & Rejection ==="
input double   InpMinWickRatio        = 0.40;      // Rasio Minimum Sumbu Pinbar (40%)
input double   InpMinBodyRatio        = 0.50;      // Rasio Minimum Bodi Momentum (50%)

input group "=== Siklus Re-Entry Otomatis ==="
input bool     InpAutoReentryOnTP     = true;      // Auto Re-Entry Seketika Jika Kena TP
input bool     InpAutoReentryOnSL     = true;      // Auto Re-Entry Seketika Jika Kena SL

//--- Global Objects & Handles
CTrade         m_trade;
CSymbolInfo    m_symbol;
CPositionInfo  m_position;
CAccountInfo   m_account;

// Handles 5 MA M1
int            m_handle_blue          = INVALID_HANDLE;
int            m_handle_green         = INVALID_HANDLE;
int            m_handle_red           = INVALID_HANDLE;
int            m_handle_yellow        = INVALID_HANDLE;
int            m_handle_purple        = INVALID_HANDLE;

// Handles MTF Trend
int            m_handle_h1_fast       = INVALID_HANDLE;
int            m_handle_h1_slow       = INVALID_HANDLE;
int            m_handle_m5_fast       = INVALID_HANDLE;
int            m_handle_m5_slow       = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(!m_symbol.Name(_Symbol))
   {
      Print("Gagal menginisialisasi properti simbol: ", _Symbol);
      return INIT_FAILED;
   }
   m_symbol.Refresh();

   m_trade.SetExpertMagicNumber(InpMagicNumber);
   m_trade.SetDeviationInPoints(InpSlippage);
   m_trade.SetTypeFillingBySymbol(_Symbol);

   // Inisialisasi 5 MA pada M1
   m_handle_blue   = iMA(_Symbol, InpActionTimeframe, InpMABlue,   0, MODE_EMA, PRICE_CLOSE);
   m_handle_green  = iMA(_Symbol, InpActionTimeframe, InpMAGreen,  0, MODE_EMA, PRICE_CLOSE);
   m_handle_red    = iMA(_Symbol, InpActionTimeframe, InpMARed,    0, MODE_EMA, PRICE_CLOSE);
   m_handle_yellow = iMA(_Symbol, InpActionTimeframe, InpMAYellow, 0, MODE_EMA, PRICE_CLOSE);
   m_handle_purple = iMA(_Symbol, InpActionTimeframe, InpMAPurple, 0, MODE_EMA, PRICE_CLOSE);

   // Inisialisasi Tren H1 & M5
   m_handle_h1_fast = iMA(_Symbol, PERIOD_H1, InpH1FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   m_handle_h1_slow = iMA(_Symbol, PERIOD_H1, InpH1SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   m_handle_m5_fast = iMA(_Symbol, PERIOD_M5, InpM5FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   m_handle_m5_slow = iMA(_Symbol, PERIOD_M5, InpM5SlowEMA, 0, MODE_EMA, PRICE_CLOSE);

   PrintFormat("EA Scalping 5-MA Berhasil Dimuat pada %s (M1). SL=%.1f pips, TP=%.1f pips (Saling Berdekatan).",
               _Symbol, InpStopLossPips, InpTakeProfitPips);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(m_handle_blue);
   IndicatorRelease(m_handle_green);
   IndicatorRelease(m_handle_red);
   IndicatorRelease(m_handle_yellow);
   IndicatorRelease(m_handle_purple);

   IndicatorRelease(m_handle_h1_fast);
   IndicatorRelease(m_handle_h1_slow);
   IndicatorRelease(m_handle_m5_fast);
   IndicatorRelease(m_handle_m5_slow);
   Print("EA Scalping dinonaktifkan. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Menghitung Jarak Pip Size Sesuai Karakter Simbol                 |
//+------------------------------------------------------------------+
double GetPipSize()
{
   double point = m_symbol.Point();
   int digits = (int)m_symbol.Digits();
   if(digits == 3 || digits == 5)
      return point * 10.0;
   return point;
}

//+------------------------------------------------------------------+
//| Membaca Tren Multi-Timeframe (H1 & M5 Utama)                     |
//+------------------------------------------------------------------+
int ReadMTFTrend()
{
   double h1_f[1], h1_s[1];
   double m5_f[1], m5_s[1];

   if(CopyBuffer(m_handle_h1_fast, 0, 0, 1, h1_f) < 1) return 0;
   if(CopyBuffer(m_handle_h1_slow, 0, 0, 1, h1_s) < 1) return 0;
   if(CopyBuffer(m_handle_m5_fast, 0, 0, 1, m5_f) < 1) return 0;
   if(CopyBuffer(m_handle_m5_slow, 0, 0, 1, m5_s) < 1) return 0;

   int h1_bias = (h1_f[0] >= h1_s[0]) ? 1 : -1;
   int m5_bias = (m5_f[0] >= m5_s[0]) ? 1 : -1;

   if(h1_bias > 0 && m5_bias > 0) return 1;   // Strong Bullish
   if(h1_bias < 0 && m5_bias < 0) return -1;  // Strong Bearish
   return m5_bias;                             // Mengikuti momentum jangka pendek M5
}

//+------------------------------------------------------------------+
//| Membaca Karakter Candlestick M1                                  |
//+------------------------------------------------------------------+
int ReadCandlestick(string &pattern_desc)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, InpActionTimeframe, 1, 2, rates) < 2) return 0;

   double o1 = rates[0].open,  h1 = rates[0].high,  l1 = rates[0].low,  c1 = rates[0].close;
   double o2 = rates[1].open,  h2 = rates[1].high,  l2 = rates[1].low,  c2 = rates[1].close;

   double range = MathMax(h1 - l1, m_symbol.Point());
   double body  = MathAbs(c1 - o1);
   double lower_wick = MathMax(0.0, MathMin(o1, c1) - l1);
   double upper_wick = MathMax(0.0, h1 - MathMax(o1, c1));

   double lower_wick_ratio = lower_wick / range;
   double upper_wick_ratio = upper_wick / range;
   double body_ratio       = body / range;

   if(lower_wick_ratio >= InpMinWickRatio && c1 >= o1)
   {
      pattern_desc = StringFormat("PINBAR_REJECTION_BAWAH (%.0f%%)", lower_wick_ratio * 100);
      return 1;
   }
   if(upper_wick_ratio >= InpMinWickRatio && c1 <= o1)
   {
      pattern_desc = StringFormat("PINBAR_REJECTION_ATAS (%.0f%%)", upper_wick_ratio * 100);
      return -1;
   }
   if(c1 > o1 && c2 < o2 && c1 >= o2 && o1 <= c2)
   {
      pattern_desc = "BULLISH_ENGULFING";
      return 1;
   }
   if(c1 < o1 && c2 > o2 && c1 <= o2 && o1 >= c2)
   {
      pattern_desc = "BEARISH_ENGULFING";
      return -1;
   }
   if(c1 > o1 && body_ratio >= InpMinBodyRatio)
   {
      pattern_desc = StringFormat("MOMENTUM_SOLID_HIJAU (%.0f%%)", body_ratio * 100);
      return 1;
   }
   if(c1 < o1 && body_ratio >= InpMinBodyRatio)
   {
      pattern_desc = StringFormat("MOMENTUM_SOLID_MERAH (%.0f%%)", body_ratio * 100);
      return -1;
   }

   pattern_desc = "LILIN_NORMAL";
   return 0;
}

//+------------------------------------------------------------------+
//| Memeriksa Posisi Terbuka Milik EA Ini                            |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
   int total = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Symbol() == _Symbol && m_position.Magic() == InpMagicNumber)
            total++;
      }
   }
   return total;
}

//+------------------------------------------------------------------+
//| Eksekusi Scalping Otomatis (Entry dengan SL & TP Berdekatan)     |
//+------------------------------------------------------------------+
void ExecuteScalpingTrade(int side, const string reason)
{
   m_symbol.RefreshRates();
   double pip = GetPipSize();
   int digits = (int)m_symbol.Digits();

   double sl_distance = InpStopLossPips * pip;
   double tp_distance = InpTakeProfitPips * pip;

   if(side > 0) // BUY
   {
      double ask = m_symbol.Ask();
      double sl  = NormalizeDouble(ask - sl_distance, digits);
      double tp  = NormalizeDouble(ask + tp_distance, digits);

      if(m_trade.Buy(InpLotSize, _Symbol, ask, sl, tp, reason))
      {
         PrintFormat("🚀 [EA SCALPING BUY] Entry @ %.5f | SL: %.5f | TP: %.5f (SL/TP Berdekatan: %.1fp/%.1fp) | Alasan: %s",
                     ask, sl, tp, InpStopLossPips, InpTakeProfitPips, reason);
      }
      else
      {
         Print("Gagal kirim order BUY: ", m_trade.ResultRetcodeDescription());
      }
   }
   else if(side < 0) // SELL
   {
      double bid = m_symbol.Bid();
      double sl  = NormalizeDouble(bid + sl_distance, digits);
      double tp  = NormalizeDouble(bid - tp_distance, digits);

      if(m_trade.Sell(InpLotSize, _Symbol, bid, sl, tp, reason))
      {
         PrintFormat("🚀 [EA SCALPING SELL] Entry @ %.5f | SL: %.5f | TP: %.5f (SL/TP Berdekatan: %.1fp/%.1fp) | Alasan: %s",
                     bid, sl, tp, InpStopLossPips, InpTakeProfitPips, reason);
      }
      else
      {
         Print("Gagal kirim order SELL: ", m_trade.ResultRetcodeDescription());
      }
   }
}

//+------------------------------------------------------------------+
//| Event Handler Setiap Tick Market (OnTick)                        |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!m_symbol.RefreshRates()) return;

   // 1 Posisi aktif dalam satu waktu
   if(CountOpenPositions() > 0) return;

   // 1. Membaca Nilai 5 MA pada M1
   double b[2], g[2], r[2], y[1], p[1];
   if(CopyBuffer(m_handle_blue,   0, 0, 2, b) < 2) return;
   if(CopyBuffer(m_handle_green,  0, 0, 2, g) < 2) return;
   if(CopyBuffer(m_handle_red,    0, 0, 2, r) < 2) return;
   if(CopyBuffer(m_handle_yellow, 0, 0, 1, y) < 1) return;
   if(CopyBuffer(m_handle_purple, 0, 0, 1, p) < 1) return;

   double blue_slope = b[1] - b[0];
   double pip = GetPipSize();

   // 2. Analisa Tren Pasar Multi-Timeframe (H1 & M5)
   int mtf_trend = ReadMTFTrend();

   // 3. Membaca Karakter Candlestick M1
   string pattern_desc = "";
   int candle_bias = ReadCandlestick(pattern_desc);

   // 4. Deteksi Cross Curam:
   // BUY: MA Biru cross MA Hijau ke atas secara curam
   bool bull_cross_up = (b[0] <= g[0] && b[1] > g[1]) && (blue_slope > 0);
   bool bull_steep_exp = (b[1] > g[1]) && (blue_slope >= 0.10 * pip);

   // SELL: MA Biru cross MA Merah ke bawah secara curam
   bool bear_cross_down = (b[0] >= r[0] && b[1] < r[1]) && (blue_slope < 0);
   bool bear_steep_exp = (b[1] < r[1]) && (blue_slope <= -0.10 * pip);

   int entry_side = 0;
   string decision_reason = "";

   if((bull_cross_up || (bull_steep_exp && candle_bias >= 0)) && mtf_trend >= 0)
   {
      entry_side = 1;
      decision_reason = StringFormat("MA Biru Cross MA Hijau Naik Curam + Lilin %s + Tren MTF Bullish", pattern_desc);
   }
   else if((bear_cross_down || (bear_steep_exp && candle_bias <= 0)) && mtf_trend <= 0)
   {
      entry_side = -1;
      decision_reason = StringFormat("MA Biru Cross MA Merah Turun Curam + Lilin %s + Tren MTF Bearish", pattern_desc);
   }
   else if(candle_bias > 0 && b[1] >= g[1])
   {
      entry_side = 1;
      decision_reason = StringFormat("Rejection Lilin %s + MA Biru di Atas Hijau", pattern_desc);
   }
   else if(candle_bias < 0 && b[1] <= r[1])
   {
      entry_side = -1;
      decision_reason = StringFormat("Rejection Lilin %s + MA Biru di Bawah Merah", pattern_desc);
   }

   // 5. Eksekusi Entry dengan SL dan TP Berdekatan
   if(entry_side != 0)
   {
      ExecuteScalpingTrade(entry_side, decision_reason);
   }
}

//+------------------------------------------------------------------+
//| Event Handler Transaksi Trade (Auto Re-Entry Seketika TP / SL)   |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                         const MqlTradeRequest &request,
                         const MqlTradeResult &result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      ulong deal_ticket = trans.deal;
      if(HistoryDealSelect(deal_ticket))
      {
         long deal_entry = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
         long deal_magic = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
         string deal_sym = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
         long deal_reason = HistoryDealGetInteger(deal_ticket, DEAL_REASON);
         double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);

         if(deal_entry == DEAL_ENTRY_OUT && deal_magic == InpMagicNumber && deal_sym == _Symbol)
         {
            bool is_tp = (deal_reason == DEAL_REASON_TP || profit > 0);
            bool is_sl = (deal_reason == DEAL_REASON_SL || profit <= 0);

            PrintFormat("🔔 [EA SCALPING EXIT] Posisi Selesai (%s | PnL: $%.2f). Auto Re-Entry Seketika...",
                        (is_tp ? "TAKE PROFIT" : "STOP LOSS"), profit);

            if((is_tp && InpAutoReentryOnTP) || (is_sl && InpAutoReentryOnSL))
            {
               OnTick();
            }
         }
      }
   }
}
