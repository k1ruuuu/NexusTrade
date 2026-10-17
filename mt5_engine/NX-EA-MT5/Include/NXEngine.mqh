#ifndef NX_ENGINE_MQH
#define NX_ENGINE_MQH

#include "NXInputs.mqh"

ENUM_TIMEFRAMES nx_tf;
double nx_point=0,nx_tick=0;
int nx_digits=0;
datetime nx_observed_bar=0;

#include "NXSignals.mqh"
#include "NXTrade.mqh"

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
