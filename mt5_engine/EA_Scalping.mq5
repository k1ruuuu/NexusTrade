//+------------------------------------------------------------------+
//|                                                  EA_Scalping.mq5 |
//|                                  NexusTrade AI Quantitative Team |
//|                                       https://nexustrade.ai      |
//+------------------------------------------------------------------+
#property copyright "NexusTrade AI Quantitative Team"
#property link      "https://nexustrade.ai"
#property version   "2.00"
#property description "EA Scalping MT5: Membaca arah trend, membaca candlestick, memprediksi pergerakan market ke depan, entry SL & TP berdekatan, serta auto re-entry seketika pada TP maupun SL."
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

input group "=== Parameter Analisa Tren ==="
input int      InpFastEMA             = 9;         // Periode EMA Cepat
input int      InpMidEMA              = 21;        // Periode EMA Sedang
input int      InpSlowEMA             = 50;        // Periode EMA Lambat
input ENUM_TIMEFRAMES InpTimeframe    = PERIOD_M1; // Timeframe Analisa Scalping

input group "=== Parameter Candlestick & Rejection ==="
input double   InpMinWickRatio        = 0.40;      // Rasio Minimum Sumbu Pinbar (40%)
input double   InpMinBodyRatio        = 0.50;      // Rasio Minimum Bodi Momentum (50%)

input group "=== Siklus Re-Entry Otomatis ==="
input bool     InpAutoReentryOnTP     = true;      // Auto Re-Entry Seketika Jika Kena TP
input bool     InpAutoReentryOnSL     = true;      // Auto Re-Entry Seketika Jika Kena SL
input int      InpCooldownTicks       = 1;         // Jeda Tick Sebelum Re-Entry (1 Tick = Instan)

//--- Global Objects & Handles
CTrade         m_trade;
CSymbolInfo    m_symbol;
CPositionInfo  m_position;
CAccountInfo   m_account;

int            m_handle_fast_ema      = INVALID_HANDLE;
int            m_handle_mid_ema       = INVALID_HANDLE;
int            m_handle_slow_ema      = INVALID_HANDLE;

datetime       m_last_bar_time        = 0;
datetime       m_last_exit_time       = 0;
string         m_last_exit_reason     = "";

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

   // Inisialisasi Indikator Tren
   m_handle_fast_ema = iMA(_Symbol, InpTimeframe, InpFastEMA, 0, MODE_EMA, PRICE_CLOSE);
   m_handle_mid_ema  = iMA(_Symbol, InpTimeframe, InpMidEMA,  0, MODE_EMA, PRICE_CLOSE);
   m_handle_slow_ema = iMA(_Symbol, InpTimeframe, InpSlowEMA, 0, MODE_EMA, PRICE_CLOSE);

   if(m_handle_fast_ema == INVALID_HANDLE || m_handle_mid_ema == INVALID_HANDLE || m_handle_slow_ema == INVALID_HANDLE)
   {
      Print("Gagal membuat handle indikator EMA.");
      return INIT_FAILED;
   }

   PrintFormat("EA Scalping Berhasil Dimuat pada %s (%s). SL=%.1f pips, TP=%.1f pips (Saling Berdekatan).",
               _Symbol, EnumToString(InpTimeframe), InpStopLossPips, InpTakeProfitPips);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(m_handle_fast_ema);
   IndicatorRelease(m_handle_mid_ema);
   IndicatorRelease(m_handle_slow_ema);
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
//| Membaca Arah Tren (Trend Reading)                                |
//+------------------------------------------------------------------+
int ReadTrendDirection(double &e9_val, double &e21_val, double &e50_val)
{
   double fast[2], mid[2], slow[2];
   if(CopyBuffer(m_handle_fast_ema, 0, 0, 2, fast) < 2) return 0;
   if(CopyBuffer(m_handle_mid_ema,  0, 0, 2, mid)  < 2) return 0;
   if(CopyBuffer(m_handle_slow_ema, 0, 0, 2, slow) < 2) return 0;

   e9_val  = fast[1];
   e21_val = mid[1];
   e50_val = slow[1];

   bool bull = (fast[1] >= mid[1] && (fast[1] - fast[0]) >= 0);
   bool bear = (fast[1] <= mid[1] && (fast[1] - fast[0]) <= 0);

   if(bull) return 1;   // BULLISH
   if(bear) return -1;  // BEARISH
   return 0;            // NEUTRAL
}

//+------------------------------------------------------------------+
//| Membaca Anatomi Candlestick (Candle Reading)                     |
//+------------------------------------------------------------------+
int ReadCandlestick(string &pattern_desc)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, InpTimeframe, 1, 2, rates) < 2) return 0;

   double o1 = rates[0].open,  h1 = rates[0].high,  l1 = rates[0].low,  c1 = rates[0].close;
   double o2 = rates[1].open,  h2 = rates[1].high,  l2 = rates[1].low,  c2 = rates[1].close;

   double range = MathMax(h1 - l1, m_symbol.Point());
   double body  = MathAbs(c1 - o1);
   double lower_wick = MathMax(0.0, MathMin(o1, c1) - l1);
   double upper_wick = MathMax(0.0, h1 - MathMax(o1, c1));

   double lower_wick_ratio = lower_wick / range;
   double upper_wick_ratio = upper_wick / range;
   double body_ratio       = body / range;

   // 1. Bullish Pinbar Rejection Bawah (Lower Wick >= 40%)
   if(lower_wick_ratio >= InpMinWickRatio && c1 >= o1)
   {
      pattern_desc = StringFormat("BULLISH_PINBAR_REJECTION (Sumbu Bawah %.0f%%)", lower_wick_ratio * 100);
      return 1;
   }

   // 2. Bearish Pinbar Rejection Atas (Upper Wick >= 40%)
   if(upper_wick_ratio >= InpMinWickRatio && c1 <= o1)
   {
      pattern_desc = StringFormat("BEARISH_PINBAR_REJECTION (Sumbu Atas %.0f%%)", upper_wick_ratio * 100);
      return -1;
   }

   // 3. Bullish Engulfing
   if(c1 > o1 && c2 < o2 && c1 >= o2 && o1 <= c2)
   {
      pattern_desc = "BULLISH_ENGULFING";
      return 1;
   }

   // 4. Bearish Engulfing
   if(c1 < o1 && c2 > o2 && c1 <= o2 && o1 >= c2)
   {
      pattern_desc = "BEARISH_ENGULFING";
      return -1;
   }

   // 5. Bullish Momentum Body
   if(c1 > o1 && body_ratio >= InpMinBodyRatio)
   {
      pattern_desc = StringFormat("BULLISH_MOMENTUM_BODY (Bodi %.0f%%)", body_ratio * 100);
      return 1;
   }

   // 6. Bearish Momentum Body
   if(c1 < o1 && body_ratio >= InpMinBodyRatio)
   {
      pattern_desc = StringFormat("BEARISH_MOMENTUM_BODY (Bodi %.0f%%)", body_ratio * 100);
      return -1;
   }

   pattern_desc = "NORMAL_CANDLE";
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

   // Pastikan hanya 1 posisi scalping aktif dalam satu waktu
   if(CountOpenPositions() > 0) return;

   // 1. Membaca Arah Trend
   double e9, e21, e50;
   int trend = ReadTrendDirection(e9, e21, e50);

   // 2. Membaca Candlestick
   string pattern_desc = "";
   int candle_bias = ReadCandlestick(pattern_desc);

   // 3. Memprediksi Pergerakan Market Kedepannya & Menentukan Arah Entry
   int entry_side = 0;
   string decision_reason = "";

   if(trend > 0 && candle_bias >= 0)
   {
      entry_side = 1;
      decision_reason = StringFormat("Tren Bullish + Lilin %s ➔ Prediksi Lanjutan Naik", pattern_desc);
   }
   else if(trend < 0 && candle_bias <= 0)
   {
      entry_side = -1;
      decision_reason = StringFormat("Tren Bearish + Lilin %s ➔ Prediksi Lanjutan Turun", pattern_desc);
   }
   else if(candle_bias > 0)
   {
      entry_side = 1;
      decision_reason = StringFormat("Rejection Lilin %s ➔ Prediksi Reversal Naik", pattern_desc);
   }
   else if(candle_bias < 0)
   {
      entry_side = -1;
      decision_reason = StringFormat("Rejection Lilin %s ➔ Prediksi Reversal Turun", pattern_desc);
   }

   // 4. Eksekusi Entry dengan SL dan TP Berdekatan
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
   // Deteksi ketika deal penutupan posisi selesai
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

         // Jika posisi milik EA Scalping ditutup
         if(deal_entry == DEAL_ENTRY_OUT && deal_magic == InpMagicNumber && deal_sym == _Symbol)
         {
            bool is_tp = (deal_reason == DEAL_REASON_TP || profit > 0);
            bool is_sl = (deal_reason == DEAL_REASON_SL || profit <= 0);

            PrintFormat("🔔 [EA SCALPING EXIT] Posisi Selesai (%s | PnL: $%.2f). Menyiapkan Auto Re-Entry Seketika...",
                        (is_tp ? "TAKE PROFIT" : "STOP LOSS"), profit);

            if((is_tp && InpAutoReentryOnTP) || (is_sl && InpAutoReentryOnSL))
            {
               // Langsung panggil evaluasi tren & candlestick untuk entry seketika!
               OnTick();
            }
         }
      }
   }
}
