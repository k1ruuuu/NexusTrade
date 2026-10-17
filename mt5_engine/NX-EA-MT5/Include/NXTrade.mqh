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
