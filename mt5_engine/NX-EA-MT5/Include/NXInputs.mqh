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
