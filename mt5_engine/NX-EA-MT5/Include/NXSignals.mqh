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
