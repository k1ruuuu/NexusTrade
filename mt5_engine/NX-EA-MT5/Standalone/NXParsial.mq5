// Raw MQL5 source. Not compiled or executed.
#property strict
#property version "1.00"
#property description "Three hedging positions; total risk shared, targets 1R/3R/6R."

#define NX_PARSIAL
#define NX_NAME "NXParsial"
#define NX_DEFAULT_MAGIC 2601702

// Source module: NXEngine.mqh
#ifndef NX_ENGINE_MQH
#define NX_ENGINE_MQH

// Source module: NXInputs.mqh
#ifndef NX_INPUTS_MQH
#define NX_INPUTS_MQH

enum NX_VOLUME_MODE { NX_RISK_PERCENT=0, NX_FIXED_TOTAL_LOTS=1 };
enum NX_VOLUME_SOURCE { NX_TICK_VOLUME=0, NX_REAL_VOLUME=1 };

input group "General"
input ENUM_TIMEFRAMES InpTimeframe=PERIOD_CURRENT;
input ulong InpMagic=NX_DEFAULT_MAGIC;
input NX_VOLUME_MODE InpVolumeMode=NX_RISK_PERCENT;
input double InpRiskPercent=0.5;
input double InpFixedTotalLots=0.03;
input double InpCommissionPerLot=0.0;
input int InpMaxSpreadPoints=30;
input int InpDeviationPoints=10;
input int InpQuoteMaxAgeSeconds=10;
input int InpEntryWindowSeconds=30;
input int InpStartHour=0;
input int InpEndHour=24;
input bool InpAllowBuy=true;
input bool InpAllowSell=true;

input group "Signal quality (closed candles)"
input int InpATRPeriod=14;
input double InpMinBodyRatio=0.55;
input double InpMinSignalATR=0.4;
input double InpMaxSignalATR=2.5;
input double InpBreakBufferATR=0.05;
input double InpMaxEntryDriftATR=0.25;

#ifdef NX_BREAK
input group "NXBreak"
input int InpFixedSLPoints=300;
#endif
#ifdef NX_PARSIAL
input group "NXParsial (total risk shared by 3 legs)"
input int InpFixedSLPoints=300;
#endif
#ifdef NX_REVERSE
input group "NXReverse"
input int InpFastEMA=20;
input int InpMidEMA=50;
input int InpSlowEMA=200;
input int InpSNRBars=20;
input int InpSwingBars=10;
input double InpSwingBufferATR=0.15;
input double InpMinStopATR=0.5;
input double InpMaxStopATR=3.0;
input bool InpEnableRecovery=true;
input int InpRecoveryExpiryBars=10;
#endif
#ifdef NX_POC
input group "NX-POC (historical tick profile)"
input int InpProfileBars=100;
input int InpProfileBins=40;
input NX_VOLUME_SOURCE InpProfileVolume=NX_TICK_VOLUME;
input int InpProfileMaxTicks=500000;
input double InpPOCMinPeakRatio=1.2;
input double InpPOCMinDistanceATR=0.5;
input double InpPOCMaxDistanceATR=2.0;
input double InpRejectionWickRatio=0.35;
input int InpADXPeriod=14;
input double InpMaxADX=25.0;
#endif

#endif

ENUM_TIMEFRAMES nx_tf;
double nx_point=0,nx_tick=0;
int nx_digits=0;
datetime nx_observed_bar=0;

// Source module: NXSignals.mqh
#ifndef NX_SIGNALS_MQH
#define NX_SIGNALS_MQH

struct NXSignal
{
   int side;
   double atr;
   double stop;
   double target;
   double boundary;
   bool recovery;
};

int nx_atr=INVALID_HANDLE;
#ifdef NX_REVERSE
int nx_fast=INVALID_HANDLE,nx_mid=INVALID_HANDLE,nx_slow=INVALID_HANDLE;
#endif
#ifdef NX_POC
int nx_adx=INVALID_HANDLE;
#endif

bool NXBuffer(const int handle,const int shift,double &value)
{
   double values[];
   if(BarsCalculated(handle)<=shift || CopyBuffer(handle,0,shift,1,values)!=1)
      return false;
   value=values[0];
   return MathIsValidNumber(value) && value!=EMPTY_VALUE && value>0.0;
}

bool NXCreateIndicators()
{
   nx_atr=iATR(_Symbol,nx_tf,InpATRPeriod);
   if(nx_atr==INVALID_HANDLE) return false;
#ifdef NX_REVERSE
   nx_fast=iMA(_Symbol,nx_tf,InpFastEMA,0,MODE_EMA,PRICE_CLOSE);
   nx_mid=iMA(_Symbol,nx_tf,InpMidEMA,0,MODE_EMA,PRICE_CLOSE);
   nx_slow=iMA(_Symbol,nx_tf,InpSlowEMA,0,MODE_EMA,PRICE_CLOSE);
   if(nx_fast==INVALID_HANDLE || nx_mid==INVALID_HANDLE || nx_slow==INVALID_HANDLE)
      return false;
#endif
#ifdef NX_POC
   nx_adx=iADX(_Symbol,nx_tf,InpADXPeriod);
   if(nx_adx==INVALID_HANDLE) return false;
#endif
   return true;
}

void NXReleaseIndicators()
{
   if(nx_atr!=INVALID_HANDLE) IndicatorRelease(nx_atr);
#ifdef NX_REVERSE
   if(nx_fast!=INVALID_HANDLE) IndicatorRelease(nx_fast);
   if(nx_mid!=INVALID_HANDLE) IndicatorRelease(nx_mid);
   if(nx_slow!=INVALID_HANDLE) IndicatorRelease(nx_slow);
#endif
#ifdef NX_POC
   if(nx_adx!=INVALID_HANDLE) IndicatorRelease(nx_adx);
#endif
}

int NXRequiredBars()
{
   int count=InpATRPeriod+10;
#ifdef NX_REVERSE
   count=MathMax(count,MathMax(InpSlowEMA+10,MathMax(InpSNRBars,InpSwingBars)+3));
#endif
#ifdef NX_POC
   count=MathMax(count,MathMax(InpProfileBars+3,InpADXPeriod*3+10));
#endif
   return count;
}

bool NXLoadRates(MqlRates &rates[])
{
   int count=NXRequiredBars();
   ArraySetAsSeries(rates,true);
   if(!SeriesInfoInteger(_Symbol,nx_tf,SERIES_SYNCHRONIZED)) return false;
   if(CopyRates(_Symbol,nx_tf,0,count,rates)!=count) return false;
   for(int i=1;i<count;i++)
   {
      if(rates[i].time>=rates[i-1].time || rates[i].high<rates[i].low ||
         rates[i].close>rates[i].high || rates[i].close<rates[i].low ||
         rates[i].open>rates[i].high || rates[i].open<rates[i].low)
         return false;
   }
   return true;
}

bool NXQuality(const MqlRates &bar,const double atr)
{
   double range=bar.high-bar.low;
   if(atr<=0 || range<=0) return false;
   return range>=InpMinSignalATR*atr && range<=InpMaxSignalATR*atr &&
          MathAbs(bar.close-bar.open)/range>=InpMinBodyRatio;
}

int NXBreakDirection(const MqlRates &bar,const double upper,const double lower,
                     const double atr)
{
   double buffer=InpBreakBufferATR*atr;
   if(!NXQuality(bar,atr)) return 0;
   if(bar.close>upper+buffer && bar.open<=upper && bar.close>bar.open) return 1;
   if(bar.close<lower-buffer && bar.open>=lower && bar.close<bar.open) return -1;
   return 0;
}

#ifdef NX_POC
bool NXProfile(MqlRates &rates[],double &poc)
{
   // Exclude the signal candle to keep its confirmation target fixed.
   double low=rates[2].low, high=rates[2].high;
   long expected_ticks=0;
   double expected_volume=0;
   for(int i=2;i<InpProfileBars+2;i++)
   {
      low=MathMin(low,rates[i].low);
      high=MathMax(high,rates[i].high);
      if(rates[i].tick_volume<=0) return false;
      expected_ticks+=rates[i].tick_volume;
      expected_volume+=(double)rates[i].real_volume;
   }
   if(high-low<nx_tick*InpProfileBins || expected_ticks>InpProfileMaxTicks ||
      (InpProfileVolume==NX_REAL_VOLUME && expected_volume<=0)) return false;
   MqlTick ticks[];
   ulong from=(ulong)rates[InpProfileBars+1].time*1000;
   ulong until=(ulong)rates[1].time*1000-1;
   ResetLastError();
   int copied=CopyTicksRange(_Symbol,ticks,COPY_TICKS_ALL,from,until);
   int error=GetLastError();
   if(error!=0 || copied<=0 || copied!=expected_ticks || copied>InpProfileMaxTicks)
   {
      PrintFormat("POC skipped: tick coverage mismatch copied=%d expected=%I64d error=%d",
                  copied,expected_ticks,error);
      return false;
   }
   double width=(high-low)/InpProfileBins;
   double profile[];
   ArrayResize(profile,InpProfileBins);
   ArrayInitialize(profile,0.0);
   double total=0;
   bool last_chart=(SymbolInfoInteger(_Symbol,SYMBOL_CHART_MODE)==SYMBOL_CHART_MODE_LAST);
   for(int i=0;i<copied;i++)
   {
      double price=(last_chart ? ticks[i].last : ticks[i].bid);
      double volume=1.0;
      if(InpProfileVolume==NX_REAL_VOLUME)
      {
         // Quote ticks retain the previous trade volume; never count it twice.
         if((ticks[i].flags&(TICK_FLAG_BUY|TICK_FLAG_SELL))==0) continue;
         price=ticks[i].last;
         volume=(ticks[i].volume_real>0 ? ticks[i].volume_real : (double)ticks[i].volume);
      }
      if(price<=0 || volume<=0 || price<low-nx_tick*0.1 || price>high+nx_tick*0.1)
         return false;
      int bin=(int)MathFloor((price-low)/width);
      bin=MathMax(0,MathMin(InpProfileBins-1,bin));
      profile[bin]+=volume;
      total+=volume;
   }
   if(InpProfileVolume==NX_REAL_VOLUME &&
      MathAbs(total-expected_volume)>MathMax(1e-8,expected_volume*1e-6)) return false;
   int best=0;
   for(int j=1;j<InpProfileBins;j++)
      if(profile[j]>profile[best]) best=j;
   if(total<=0 || profile[best]<InpPOCMinPeakRatio*total/InpProfileBins) return false;
   for(int j=0;j<InpProfileBins;j++)
      if(j!=best && MathAbs(profile[j]-profile[best])<=profile[best]*1e-10)
         return false;
   poc=low+(best+0.5)*width;
   return true;
}
#endif

bool NXBuildSignal(MqlRates &rates[],NXSignal &signal)
{
   ZeroMemory(signal);
   // shift 1 is closed; ATR excludes the signal candle itself.
   if(!NXBuffer(nx_atr,2,signal.atr)) return false;
#ifdef NX_BREAK
   signal.side=NXBreakDirection(rates[1],rates[2].high,rates[2].low,signal.atr);
   signal.boundary=(signal.side>0 ? rates[2].high : rates[2].low);
#endif
#ifdef NX_PARSIAL
   signal.side=NXBreakDirection(rates[1],rates[2].high,rates[2].low,signal.atr);
   signal.boundary=(signal.side>0 ? rates[2].high : rates[2].low);
#endif
#ifdef NX_REVERSE
   double f1,f2,m1,m2,s1,s2;
   if(!NXBuffer(nx_fast,1,f1) || !NXBuffer(nx_fast,2,f2) ||
      !NXBuffer(nx_mid,1,m1) || !NXBuffer(nx_mid,2,m2) ||
      !NXBuffer(nx_slow,1,s1) || !NXBuffer(nx_slow,2,s2)) return false;
   double upper=rates[2].high,lower=rates[2].low;
   for(int i=3;i<InpSNRBars+2;i++)
   {
      upper=MathMax(upper,rates[i].high);
      lower=MathMin(lower,rates[i].low);
   }
   signal.side=NXBreakDirection(rates[1],upper,lower,signal.atr);
   if(signal.side>0 && !(f1>m1 && m1>s1 && f1>f2 && m1>m2 && s1>=s2))
      return false;
   if(signal.side<0 && !(f1<m1 && m1<s1 && f1<f2 && m1<m2 && s1<=s2))
      return false;
   signal.boundary=(signal.side>0 ? upper : lower);
   double swing=(signal.side>0 ? rates[1].low : rates[1].high);
   for(int i=2;i<=InpSwingBars;i++)
      swing=(signal.side>0 ? MathMin(swing,rates[i].low) : MathMax(swing,rates[i].high));
   signal.stop=swing-signal.side*InpSwingBufferATR*signal.atr;
#endif
#ifdef NX_POC
   double adx,poc;
   if(!NXBuffer(nx_adx,1,adx) || adx>InpMaxADX || !NXProfile(rates,poc))
      return false;
   double range=rates[1].high-rates[1].low;
   if(range<InpMinSignalATR*signal.atr || range>InpMaxSignalATR*signal.atr)
      return false;
   double lowerWick=MathMin(rates[1].open,rates[1].close)-rates[1].low;
   double upperWick=rates[1].high-MathMax(rates[1].open,rates[1].close);
   double distance=MathAbs(poc-rates[1].close);
   if(distance<InpPOCMinDistanceATR*signal.atr ||
      distance>InpPOCMaxDistanceATR*signal.atr) return false;
   if(rates[1].close<poc && rates[1].close>rates[1].open &&
      lowerWick/range>=InpRejectionWickRatio &&
      rates[1].close>rates[2].close) signal.side=1;
   if(rates[1].close>poc && rates[1].close<rates[1].open &&
      upperWick/range>=InpRejectionWickRatio &&
      rates[1].close<rates[2].close) signal.side=-1;
   signal.target=poc;
#endif
   return signal.side!=0;
}

#endif
// Source module: NXTrade.mqh
#ifndef NX_TRADE_MQH
#define NX_TRADE_MQH

#include <Trade/Trade.mqh>

CTrade nx_close;
string nx_key,nx_symbol_key,nx_mutex_file;
int nx_instance_file=INVALID_HANDLE;
datetime nx_last_cleanup=0;

uint NXHash(const string text)
{
   uint hash=2166136261;
   for(int i=0;i<StringLen(text);i++)
      hash=(hash^(uint)StringGetCharacter(text,i))*16777619;
   return hash;
}

double NXState(const string suffix)
{
   string key=nx_key+suffix;
   if(!GlobalVariableCheck(key)) return 0.0;
   return GlobalVariableGet(key);
}

bool NXStore(const string suffix,const double value)
{
   if(GlobalVariableSet(nx_key+suffix,value)==0)
   {
      Print("Persistent state write failed: ",suffix," error=",GetLastError());
      return false;
   }
   return true;
}

double NXLatch()
{
   if(!GlobalVariableCheck(nx_symbol_key+".halt")) return 0;
   return GlobalVariableGet(nx_symbol_key+".halt");
}

bool NXSetLatch(const double state)
{
   bool ok=(GlobalVariableSet(nx_symbol_key+".halt",state)!=0);
   GlobalVariablesFlush();
   if(!ok) Print("Cannot persist safety latch; remove EA and inspect account.");
   return ok;
}

bool NXInitState()
{
   if(AccountInfoInteger(ACCOUNT_LOGIN)<=0 || TimeCurrent()<=0) return false;
   string account=StringFormat("%I64d",AccountInfoInteger(ACCOUNT_LOGIN))+"|"+
                  AccountInfoString(ACCOUNT_SERVER);
   string scope=account+"|"+_Symbol;
   string prefix=(MQLInfoInteger(MQL_TESTER) ? "NXT" : "NXL");
   nx_symbol_key=prefix+"."+StringFormat("%u",NXHash(scope));
   nx_key=nx_symbol_key+"."+StringFormat("%I64u",InpMagic);
   nx_mutex_file=nx_symbol_key+".mutex";
   // Exclusive file handles prevent duplicate instances inside one terminal.
   nx_instance_file=FileOpen(nx_key+".instance",FILE_BIN|FILE_READ|FILE_WRITE);
   if(nx_instance_file==INVALID_HANDLE)
   {
      Print("Duplicate instance or unavailable state directory: ",nx_key);
      return false;
   }
   if(MQLInfoInteger(MQL_TESTER))
   {
      // Tester agents use their private sandbox; each run starts clean.
      string suffixes[]={".bar",".mode",".rd",".rt",".seenHi",".seenLo",".boot"};
      for(int i=0;i<ArraySize(suffixes);i++)
         if(!NXStore(suffixes[i],0)) return false;
      if(!NXSetLatch(0)) return false;
   }
   if(NXState(".boot")==0 && !NXStore(".boot",(double)TimeCurrent())) return false;
   GlobalVariablesFlush();
   nx_close.SetExpertMagicNumber(InpMagic);
   nx_close.SetDeviationInPoints(InpDeviationPoints);
   nx_close.SetAsyncMode(false);
   nx_close.SetTypeFillingBySymbol(_Symbol);
   Print("State prefix=",nx_key," symbol safety key=",nx_symbol_key,".halt");
   if(NXLatch()!=0) Print("Safety latch active. Inspect positions, orders and history before manual reset.");
   return true;
}

bool NXAnyExposure()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
      if(PositionGetTicket(i)>0 && PositionGetString(POSITION_SYMBOL)==_Symbol)
         return true;
   for(int i=OrdersTotal()-1;i>=0;i--)
      if(OrderGetTicket(i)>0 && OrderGetString(ORDER_SYMBOL)==_Symbol)
         return true;
   return false;
}

bool NXBatchExposureMatches(const int expected)
{
   int count=0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      if(PositionGetTicket(i)==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC)!=InpMagic) return false;
      count++;
   }
   for(int i=OrdersTotal()-1;i>=0;i--)
      if(OrderGetTicket(i)>0 && OrderGetString(ORDER_SYMBOL)==_Symbol) return false;
   return count==expected;
}

bool NXOurPosition(const int index,ulong &ticket)
{
   ticket=PositionGetTicket(index);
   return ticket>0 && PositionGetString(POSITION_SYMBOL)==_Symbol &&
          (ulong)PositionGetInteger(POSITION_MAGIC)==InpMagic;
}

bool NXTradingAllowed()
{
   return TerminalInfoInteger(TERMINAL_CONNECTED) &&
          TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) &&
          MQLInfoInteger(MQL_TRADE_ALLOWED) &&
          AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) &&
          AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
}

bool NXQuote(MqlTick &quote)
{
   if(!SymbolInfoTick(_Symbol,quote) || quote.ask<=0 || quote.bid<=0 ||
      quote.ask<quote.bid || quote.time<=0) return false;
   long age=(long)TimeTradeServer()-(long)quote.time;
   if(age<0 || age>InpQuoteMaxAgeSeconds) return false;
   return (quote.ask-quote.bid)/nx_point<=InpMaxSpreadPoints+1e-8;
}

bool NXSession()
{
   MqlDateTime now;
   TimeToStruct(TimeCurrent(),now);
   if(InpStartHour<InpEndHour)
      return now.hour>=InpStartHour && now.hour<InpEndHour;
   return now.hour>=InpStartHour || now.hour<InpEndHour;
}

bool NXEntryWindow()
{
   datetime current=iTime(_Symbol,nx_tf,0);
   return current==nx_observed_bar && TimeCurrent()>=current &&
          TimeCurrent()-current<=InpEntryWindowSeconds;
}

double NXRoundPrice(const double price)
{
   return NormalizeDouble(MathRound(price/nx_tick)*nx_tick,nx_digits);
}

double NXFloorVolume(const double volume)
{
   double step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(step<=0 || volume<=0) return 0;
   return NormalizeDouble(MathFloor((volume+step*1e-9)/step)*step,8);
}

bool NXSideAllowed(const int side)
{
   if((side>0 && !InpAllowBuy) || (side<0 && !InpAllowSell)) return false;
   ENUM_SYMBOL_TRADE_MODE mode=(ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_MODE);
   if(mode==SYMBOL_TRADE_MODE_DISABLED || mode==SYMBOL_TRADE_MODE_CLOSEONLY) return false;
   if(side>0 && mode==SYMBOL_TRADE_MODE_SHORTONLY) return false;
   if(side<0 && mode==SYMBOL_TRADE_MODE_LONGONLY) return false;
   long order_mode=SymbolInfoInteger(_Symbol,SYMBOL_ORDER_MODE);
   return (order_mode&SYMBOL_ORDER_MARKET)!=0 &&
          (order_mode&SYMBOL_ORDER_SL)!=0 && (order_mode&SYMBOL_ORDER_TP)!=0;
}

bool NXFillMode(ENUM_ORDER_TYPE_FILLING &mode)
{
   ENUM_SYMBOL_TRADE_EXECUTION execution=(ENUM_SYMBOL_TRADE_EXECUTION)
      SymbolInfoInteger(_Symbol,SYMBOL_TRADE_EXEMODE);
   long flags=SymbolInfoInteger(_Symbol,SYMBOL_FILLING_MODE);
   if(execution==SYMBOL_TRADE_EXECUTION_INSTANT ||
      execution==SYMBOL_TRADE_EXECUTION_REQUEST || (flags&SYMBOL_FILLING_FOK)!=0)
   {
      mode=ORDER_FILLING_FOK;
      return true;
   }
#ifdef NX_PARSIAL
   // Three equal layers require full fills per request.
   return false;
#else
   if((flags&SYMBOL_FILLING_IOC)!=0)
   {
      mode=ORDER_FILLING_IOC;
      return true;
   }
   return false;
#endif
}

bool NXStopsValid(const int side,const double entry,const double sl,
                  const double tp,const MqlTick &quote)
{
   if(sl<=0 || tp<=0 || !MathIsValidNumber(sl) || !MathIsValidNumber(tp)) return false;
   double minimum=MathMax(nx_tick,SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*nx_point);
   double closePrice=(side>0 ? quote.bid : quote.ask);
   return side*(entry-sl)>0 && side*(tp-entry)>0 &&
          side*(closePrice-sl)>=minimum-1e-9*nx_tick &&
          side*(tp-closePrice)>=minimum-1e-9*nx_tick;
}

bool NXPlan(const NXSignal &signal,const MqlTick &quote,double &entry,
            double &sl,double &targets[])
{
   entry=(signal.side>0 ? quote.ask : quote.bid);
   double distance=0;
#ifdef NX_BREAK
   distance=MathCeil(InpFixedSLPoints*nx_point/nx_tick)*nx_tick;
#endif
#ifdef NX_PARSIAL
   distance=MathCeil(InpFixedSLPoints*nx_point/nx_tick)*nx_tick;
#endif
#ifdef NX_REVERSE
   sl=NormalizeDouble((signal.side>0 ? MathFloor(signal.stop/nx_tick) :
                        MathCeil(signal.stop/nx_tick))*nx_tick,nx_digits);
   distance=signal.side*(entry-sl);
   if(distance<InpMinStopATR*signal.atr || distance>InpMaxStopATR*signal.atr)
      return false;
#endif
#ifdef NX_POC
   double target=NXRoundPrice(signal.target);
   distance=signal.side*(target-entry);
   if(distance<InpPOCMinDistanceATR*signal.atr ||
      distance>InpPOCMaxDistanceATR*signal.atr) return false;
#endif
   if(distance<nx_tick) return false;
#ifndef NX_REVERSE
   sl=NXRoundPrice(entry-signal.side*distance);
#endif
   distance=signal.side*(entry-sl);
   int legs=1;
#ifdef NX_PARSIAL
   legs=3;
#endif
   ArrayResize(targets,legs);
   for(int i=0;i<legs;i++)
   {
      double reward=1.0;
#ifdef NX_PARSIAL
      reward=(i==0 ? 1.0 : (i==1 ? 3.0 : 6.0));
#endif
#ifdef NX_REVERSE
      reward=5.0;
#endif
      targets[i]=NXRoundPrice(entry+signal.side*reward*distance);
      if(!NXStopsValid(signal.side,entry,sl,targets[i],quote)) return false;
   }
   return true;
}

bool NXUnitRisk(const int side,const double entry,const double sl,double &risk)
{
   ENUM_ORDER_TYPE type=(side>0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   double reference=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN),profit;
   if(reference<=0 || !OrderCalcProfit(type,_Symbol,reference,entry,sl,profit) || profit>=0)
      return false;
   risk=(-profit/reference)+InpCommissionPerLot;
   return MathIsValidNumber(risk) && risk>0;
}

bool NXVolume(const int side,const double entry,const double sl,const int legs,
              double &volume,double &budget)
{
   budget=AccountInfoDouble(ACCOUNT_EQUITY)*InpRiskPercent/100.0;
   if(budget<=0) return false;
   double unit;
   // Deviation reserves entry slippage; stop gaps remain unbounded.
   double adverse=entry+side*InpDeviationPoints*nx_point;
   if(!NXUnitRisk(side,adverse,sl,unit)) return false;
   double total=(InpVolumeMode==NX_RISK_PERCENT ? budget/unit : InpFixedTotalLots);
   volume=NXFloorVolume(MathMin(total/legs,SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX)));
   if(volume<SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN)) return false;
   if(volume*legs*unit>budget+1e-8) return false;
   double limit=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_LIMIT);
   if(limit>0 && volume*legs>limit+1e-8) return false;
   return true;
}

void NXMakeRequest(const int side,const double entry,const double sl,const double tp,
                   const double volume,const ENUM_ORDER_TYPE_FILLING filling,
                   const string comment,MqlTradeRequest &request)
{
   ZeroMemory(request);
   request.action=TRADE_ACTION_DEAL;
   request.magic=InpMagic;
   request.symbol=_Symbol;
   request.volume=volume;
   request.type=(side>0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   request.price=entry;
   request.sl=sl;
   request.tp=tp;
   request.deviation=InpDeviationPoints;
   request.type_filling=filling;
   request.type_time=ORDER_TIME_GTC;
   request.comment=comment;
}

bool NXCheck(MqlTradeRequest &request)
{
   MqlTradeCheckResult check={};
   // Preflight success may be zero; it is distinct from a filled order.
   if(!OrderCheck(request,check) || (check.retcode!=0 && check.retcode!=TRADE_RETCODE_DONE))
   {
      PrintFormat("OrderCheck rejected: %u %s error=%d",check.retcode,check.comment,GetLastError());
      return false;
   }
   return true;
}

void NXCleanup()
{
   if(NXLatch()!=2 || !GlobalVariableCheck(nx_symbol_key+".owner") ||
      GlobalVariableGet(nx_symbol_key+".owner")!=(double)InpMagic ||
      !NXTradingAllowed() || TimeCurrent()-nx_last_cleanup<5) return;
   nx_last_cleanup=TimeCurrent();
   bool outstanding=false;
   for(int i=OrdersTotal()-1;i>=0;i--)
   {
      ulong ticket=OrderGetTicket(i);
      if(ticket==0 || OrderGetString(ORDER_SYMBOL)!=_Symbol ||
         (ulong)OrderGetInteger(ORDER_MAGIC)!=InpMagic) continue;
      outstanding=true;
      bool ok=nx_close.OrderDelete(ticket);
      PrintFormat("Rollback delete %I64u ok=%d ret=%u",ticket,ok,nx_close.ResultRetcode());
      uint code=nx_close.ResultRetcode();
      if(code!=TRADE_RETCODE_DONE && !NXDefiniteRejection(code))
      {
         NXSetLatch(1);
         return;
      }
   }
   // Close oldest first, including accounts enforcing FIFO.
   int maximum_closes=PositionsTotal();
   for(int attempt=0;attempt<maximum_closes;attempt++)
   {
      ulong oldest=0;
      long earliest=LONG_MAX;
      for(int i=PositionsTotal()-1;i>=0;i--)
      {
         ulong ticket;
         if(!NXOurPosition(i,ticket)) continue;
         long opened=PositionGetInteger(POSITION_TIME_MSC);
         if(opened<earliest || (opened==earliest && (oldest==0 || ticket<oldest)))
         {
            earliest=opened;
            oldest=ticket;
         }
      }
      if(oldest==0) break;
      outstanding=true;
      bool ok=nx_close.PositionClose(oldest,(ulong)InpDeviationPoints);
      PrintFormat("Rollback close %I64u ok=%d ret=%u",oldest,ok,nx_close.ResultRetcode());
      uint code=nx_close.ResultRetcode();
      if(!ok || code!=TRADE_RETCODE_DONE || PositionSelectByTicket(oldest))
      {
         if(code!=TRADE_RETCODE_DONE_PARTIAL && !NXDefiniteRejection(code))
            NXSetLatch(1);
         break;
      }
   }
   // Keep the halt after rollback; operator reconciliation is required.
   if(!outstanding) NXSetLatch(1);
}

bool NXDefiniteRejection(const uint code)
{
   switch(code)
   {
      case TRADE_RETCODE_REQUOTE:
      case TRADE_RETCODE_REJECT:
      case TRADE_RETCODE_INVALID:
      case TRADE_RETCODE_INVALID_VOLUME:
      case TRADE_RETCODE_INVALID_PRICE:
      case TRADE_RETCODE_INVALID_STOPS:
      case TRADE_RETCODE_TRADE_DISABLED:
      case TRADE_RETCODE_MARKET_CLOSED:
      case TRADE_RETCODE_NO_MONEY:
      case TRADE_RETCODE_PRICE_CHANGED:
      case TRADE_RETCODE_PRICE_OFF:
      case TRADE_RETCODE_INVALID_EXPIRATION:
      case TRADE_RETCODE_TOO_MANY_REQUESTS:
      case TRADE_RETCODE_SERVER_DISABLES_AT:
      case TRADE_RETCODE_CLIENT_DISABLES_AT:
      case TRADE_RETCODE_INVALID_FILL:
      case TRADE_RETCODE_LIMIT_ORDERS:
      case TRADE_RETCODE_LIMIT_VOLUME:
      case TRADE_RETCODE_INVALID_ORDER:
         return true;
   }
   return false;
}

bool NXExecute(const NXSignal &signal,const MqlRates &bar)
{
   MqlTick quote;
   if(!NXTradingAllowed() || !NXSession() || !NXEntryWindow() || !NXQuote(quote) ||
      !NXSideAllowed(signal.side)) return false;
   double entry,sl,targets[],volume,budget;
   ENUM_ORDER_TYPE_FILLING filling;
   if(!NXFillMode(filling) || !NXPlan(signal,quote,entry,sl,targets) ||
      !NXVolume(signal.side,entry,sl,ArraySize(targets),volume,budget))
   {
      Print("Signal skipped: filling policy, stops, volume or risk constraints.");
      return false;
   }
   double margin=0;
   if(!OrderCalcMargin((signal.side>0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL),
                       _Symbol,volume*ArraySize(targets),entry,margin) ||
      margin>AccountInfoDouble(ACCOUNT_MARGIN_FREE)) return false;
   for(int i=0;i<ArraySize(targets);i++)
   {
      MqlTradeRequest preflight;
      NXMakeRequest(signal.side,entry,sl,targets[i],volume,filling,NX_NAME,preflight);
      if(!NXCheck(preflight)) return false;
   }
   // Journal before sending: a crash cannot silently permit a new cycle.
   if(GlobalVariableSet(nx_symbol_key+".owner",(double)InpMagic)==0) return false;
   if(!NXSetLatch(1)) return false;
#ifdef NX_REVERSE
   if(!NXStore(".mode",signal.recovery ? 2 : 0)) return false;
   if(!NXStore(".rd",0)) return false;
   GlobalVariablesFlush();
#endif
   double usedRisk=0;
   for(int i=0;i<ArraySize(targets);i++)
   {
      MqlTick fresh={};
      double unit=0;
      bool valid=NXBatchExposureMatches(i) && NXTradingAllowed() &&
                 NXSession() && NXEntryWindow() && NXQuote(fresh);
      double freshEntry=(signal.side>0 ? fresh.ask : fresh.bid);
      valid=valid && MathAbs(freshEntry-bar.close)<=InpMaxEntryDriftATR*signal.atr &&
            NXStopsValid(signal.side,freshEntry,sl,targets[i],fresh) &&
            NXUnitRisk(signal.side,freshEntry+signal.side*InpDeviationPoints*nx_point,sl,unit);
#ifndef NX_POC
      double closePrice=(signal.side>0 ? fresh.bid : fresh.ask);
      valid=valid && signal.side*(closePrice-signal.boundary)>InpBreakBufferATR*signal.atr;
#endif
      if(!valid || usedRisk+unit*volume*(ArraySize(targets)-i)>budget+1e-8)
      {
         if(i>0) NXSetLatch(2); else NXSetLatch(0);
         return false;
      }
      MqlTradeRequest request;
      MqlTradeResult result={};
      string comment=NX_NAME+(signal.recovery ? ":R" : ":N")+StringFormat(":%d",i+1);
      NXMakeRequest(signal.side,freshEntry,sl,targets[i],volume,filling,comment,request);
      if(!NXCheck(request))
      {
         if(i>0) NXSetLatch(2); else NXSetLatch(0);
         return false;
      }
      bool sent=OrderSend(request,result);
      PrintFormat("%s sent=%d ret=%u deal=%I64u order=%I64u volume=%.8f price=%.*f %s",
                  comment,sent,result.retcode,result.deal,result.order,result.volume,
                  nx_digits,result.price,result.comment);
      // Any ambiguous result remains halted, without automatic resend.
      if(!sent || result.retcode!=TRADE_RETCODE_DONE || result.deal==0)
      {
         if(NXDefiniteRejection(result.retcode))
         {
            NXSetLatch(i>0 ? 2 : 0);
            return false;
         }
         if(result.retcode==TRADE_RETCODE_DONE_PARTIAL) NXSetLatch(2);
         Print("Execution not confirmed in full. Safety halt retained; reconcile server history.");
         return false;
      }
      double filledUnit;
      if(MathAbs(result.volume-volume)>SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP)*0.1 ||
         !NXUnitRisk(signal.side,result.price,sl,filledUnit))
      {
         NXSetLatch(2);
         return false;
      }
      usedRisk+=filledUnit*result.volume;
      if(usedRisk>budget+1e-8)
      {
         NXSetLatch(2);
         return false;
      }
      if(!HistoryDealSelect(result.deal)) return false;
      long identifier=HistoryDealGetInteger(result.deal,DEAL_POSITION_ID);
      bool found=false;
      for(int p=PositionsTotal()-1;p>=0;p--)
      {
         ulong ticket;
         if(!NXOurPosition(p,ticket) || PositionGetInteger(POSITION_IDENTIFIER)!=identifier)
            continue;
         found=true;
         if(MathAbs(PositionGetDouble(POSITION_SL)-sl)>nx_tick*0.1 ||
            MathAbs(PositionGetDouble(POSITION_TP)-targets[i])>nx_tick*0.1)
         {
            NXSetLatch(2);
            return false;
         }
      }
      if(!found)
      {
         // A fast exit or delayed position update requires reconciliation.
         if(ArraySize(targets)>1) NXSetLatch(2);
         return false;
      }
   }
   return NXSetLatch(0);
}

#endif

bool NXInputsValid()
{
   if(InpMagic==0 || InpMagic>2147483647 || InpRiskPercent<=0 || InpRiskPercent>5 ||
      InpFixedTotalLots<=0 || InpCommissionPerLot<0 || InpMaxSpreadPoints<=0 ||
      InpDeviationPoints<0 || InpQuoteMaxAgeSeconds<1 || InpEntryWindowSeconds<1 ||
      InpEntryWindowSeconds>PeriodSeconds(nx_tf) ||
      InpStartHour<0 || InpStartHour>23 || InpEndHour<0 || InpEndHour>24 ||
      InpStartHour==InpEndHour || InpATRPeriod<2 ||
      InpMinBodyRatio<=0 || InpMinBodyRatio>1 || InpMinSignalATR<=0 ||
      InpMaxSignalATR<InpMinSignalATR || InpBreakBufferATR<0 ||
      InpMaxEntryDriftATR<=0 || (!InpAllowBuy && !InpAllowSell)) return false;
   if(InpVolumeMode!=NX_RISK_PERCENT && InpVolumeMode!=NX_FIXED_TOTAL_LOTS) return false;
#ifdef NX_BREAK
   if(InpFixedSLPoints<1) return false;
#endif
#ifdef NX_PARSIAL
   if(InpFixedSLPoints<1 || AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      return false;
#endif
#ifdef NX_REVERSE
   if(InpFastEMA<2 || InpMidEMA<=InpFastEMA || InpSlowEMA<=InpMidEMA ||
      InpSlowEMA>5000 || InpSNRBars<2 || InpSNRBars>5000 ||
      InpSwingBars<2 || InpSwingBars>5000 || InpSwingBufferATR<0 ||
      InpMinStopATR<=0 || InpMaxStopATR<InpMinStopATR ||
      InpRecoveryExpiryBars<1 || InpRecoveryExpiryBars>1000) return false;
#endif
#ifdef NX_POC
   if(InpProfileBars<10 || InpProfileBars>5000 || InpProfileBins<5 ||
      InpProfileBins>500 || InpPOCMinDistanceATR<=0 ||
      InpProfileMaxTicks<1000 || InpProfileMaxTicks>2000000 || InpPOCMinPeakRatio<=1 ||
      InpPOCMaxDistanceATR<InpPOCMinDistanceATR ||
      InpRejectionWickRatio<=0 || InpRejectionWickRatio>=1 ||
      InpADXPeriod<2 || InpADXPeriod>1000 || InpMaxADX<=0 || InpMaxADX>100)
      return false;
   if(InpProfileVolume!=NX_TICK_VOLUME && InpProfileVolume!=NX_REAL_VOLUME) return false;
#endif
   return InpATRPeriod<=1000 && MathIsValidNumber(InpRiskPercent) &&
          MathIsValidNumber(InpFixedTotalLots) && MathIsValidNumber(InpCommissionPerLot);
}

#ifdef NX_REVERSE
bool NXRefreshRecovery()
{
   if(!HistorySelect((datetime)NXState(".boot"),TimeCurrent())) return false;
   ulong seen=((ulong)(uint)NXState(".seenHi")<<32) | (ulong)(uint)NXState(".seenLo");
   ulong identifier=0;
   for(int i=HistoryDealsTotal()-1;i>=0;i--)
   {
      ulong deal=HistoryDealGetTicket(i);
      if(deal==0 || HistoryDealGetString(deal,DEAL_SYMBOL)!=_Symbol ||
         (ulong)HistoryDealGetInteger(deal,DEAL_MAGIC)!=InpMagic) continue;
      ENUM_DEAL_ENTRY entry=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal,DEAL_ENTRY);
      if(entry==DEAL_ENTRY_IN)
      {
         identifier=(ulong)HistoryDealGetInteger(deal,DEAL_POSITION_ID);
         break;
      }
   }
   if(identifier==0) return true;
   if(!HistorySelectByPosition(identifier)) return false;
   ulong latest=0;
   double remaining=0;
   for(int i=0;i<HistoryDealsTotal();i++)
   {
      ulong deal=HistoryDealGetTicket(i);
      ENUM_DEAL_ENTRY entry=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal,DEAL_ENTRY);
      if(entry==DEAL_ENTRY_IN) remaining+=HistoryDealGetDouble(deal,DEAL_VOLUME);
      else if(entry==DEAL_ENTRY_OUT || entry==DEAL_ENTRY_OUT_BY)
      {
         remaining-=HistoryDealGetDouble(deal,DEAL_VOLUME);
         latest=deal;
      }
      else return false;
   }
   // Flat positions and settled history must agree before another entry.
   if(latest==0 || MathAbs(remaining)>SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP)*0.1)
      return false;
   if(latest==seen) return true;
   long reason=HistoryDealGetInteger(latest,DEAL_REASON);
   long type=HistoryDealGetInteger(latest,DEAL_TYPE);
   datetime when=(datetime)HistoryDealGetInteger(latest,DEAL_TIME);
   // Mark before arming: interruption may lose a recovery, never duplicate it.
   if(!NXStore(".seenHi",(double)(uint)(latest>>32)) ||
      !NXStore(".seenLo",(double)(uint)(latest&0xFFFFFFFF))) return false;
   GlobalVariablesFlush();
   int mode=(int)NXState(".mode");
   int direction=0;
   if(InpEnableRecovery && reason==DEAL_REASON_SL && mode==0)
      direction=(type==DEAL_TYPE_BUY ? 1 : (type==DEAL_TYPE_SELL ? -1 : 0));
   if(!NXStore(".rd",direction) || !NXStore(".rt",(double)when) ||
      !NXStore(".mode",direction!=0 ? 1 : 0)) return false;
   GlobalVariablesFlush();
   if(direction!=0) Print("Recovery armed; opposite confirmation required, direction=",direction);
   return true;
}

bool NXApplyRecovery(NXSignal &signal,const MqlRates &bar)
{
   int required=(int)NXState(".rd");
   if(required==0) return true;
   datetime stopped=(datetime)NXState(".rt");
   int shift=iBarShift(_Symbol,nx_tf,stopped,false);
   if(!InpEnableRecovery || shift<0 || shift>InpRecoveryExpiryBars)
   {
      if(!NXStore(".rd",0) || !NXStore(".mode",0)) return false;
      GlobalVariablesFlush();
      return true;
   }
   // The confirmation candle must start after the stop event.
   if(bar.time<=stopped || signal.side!=required) return false;
   signal.recovery=true;
   return true;
}
#endif

int OnInit()
{
   nx_tf=(InpTimeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)_Period : InpTimeframe);
   nx_point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   nx_tick=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   nx_digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   if(!NXInputsValid() || nx_point<=0 || nx_tick<=0 ||
      SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN)<=0 ||
      SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP)<=0)
   {
      Print("Invalid inputs, symbol properties, or account type (NXParsial requires hedging).");
      return INIT_PARAMETERS_INCORRECT;
   }
   // Brokers deleting stops at rollover cannot preserve this fixed-stop contract.
   if(SymbolInfoInteger(_Symbol,SYMBOL_ORDER_GTC_MODE)==SYMBOL_ORDERS_DAILY)
   {
      Print("Symbol deletes protective stops at rollover; unsupported.");
      return INIT_FAILED;
   }
   if(!NXInitState() || !NXCreateIndicators()) return INIT_FAILED;
   nx_observed_bar=iTime(_Symbol,nx_tf,0);
   Print(NX_NAME," initialized. Closed-bar signals; first entry only on a subsequent bar.");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   NXReleaseIndicators();
   if(nx_instance_file!=INVALID_HANDLE) FileClose(nx_instance_file);
   GlobalVariablesFlush();
}

void NXProcessBar(const datetime currentBar)
{
   if(NXLatch()!=0 || NXAnyExposure() || !NXTradingAllowed()) return;
   if(TimeCurrent()<currentBar || TimeCurrent()-currentBar>InpEntryWindowSeconds) return;
   if(NXState(".bar")>=(double)currentBar) return;
   if(!NXStore(".bar",(double)currentBar)) return;
   GlobalVariablesFlush();
#ifdef NX_REVERSE
   if(!NXRefreshRecovery()) return;
#endif
   MqlRates rates[];
   if(!NXLoadRates(rates) || rates[0].time!=currentBar) return;
   NXSignal signal;
   if(!NXBuildSignal(rates,signal)) return;
#ifdef NX_REVERSE
   if(!NXApplyRecovery(signal,rates[1])) return;
#endif
   MqlTick quote;
   if(!NXQuote(quote)) return;
   double entry=(signal.side>0 ? quote.ask : quote.bid);
   if(MathAbs(entry-rates[1].close)>InpMaxEntryDriftATR*signal.atr) return;
#ifndef NX_POC
   double closePrice=(signal.side>0 ? quote.bid : quote.ask);
   if(signal.side*(closePrice-signal.boundary)<=InpBreakBufferATR*signal.atr) return;
#endif
   NXExecute(signal,rates[1]);
}

void OnTick()
{
   int mutex=FileOpen(nx_mutex_file,FILE_BIN|FILE_READ|FILE_WRITE);
   if(mutex==INVALID_HANDLE) return;
   NXCleanup();
   datetime current=iTime(_Symbol,nx_tf,0);
   if(current>0 && current!=nx_observed_bar)
   {
      if(nx_observed_bar==0)
         nx_observed_bar=current;
      else
      {
         nx_observed_bar=current;
         NXProcessBar(current);
      }
   }
   FileClose(mutex);
}

#endif
