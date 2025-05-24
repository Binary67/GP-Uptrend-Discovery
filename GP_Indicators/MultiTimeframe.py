import pandas as pd
import pandas_ta as ta


def DailyRSI(Price, Period):
    """Calculate RSI on daily timeframe"""
    try:
        Period = int(abs(Period)) + 1
        Period = min(max(Period, 2), 30)
        if not hasattr(Price, '__len__'):
            return 50.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        DailyPrice = PriceSeries.resample('D').last()
        DailyRSI = ta.rsi(DailyPrice, length=Period)
        
        # Reindex to hourly frequency
        return DailyRSI.reindex(PriceSeries.index, method='ffill')
    except:
        return pd.Series([50.0] * len(Price)) if hasattr(Price, '__len__') else 50.0


def WeeklyRSI(Price, Period):
    """Calculate RSI on weekly timeframe"""
    try:
        Period = int(abs(Period)) + 1
        Period = min(max(Period, 2), 10)
        if not hasattr(Price, '__len__'):
            return 50.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        WeeklyPrice = PriceSeries.resample('W').last()
        WeeklyRSI = ta.rsi(WeeklyPrice, length=Period)
        
        return WeeklyRSI.reindex(PriceSeries.index, method='ffill')
    except:
        return pd.Series([50.0] * len(Price)) if hasattr(Price, '__len__') else 50.0


def DailyMACD(Price, FastPeriod, SlowPeriod):
    """Calculate MACD on daily timeframe"""
    try:
        FastPeriod = int(abs(FastPeriod)) + 1
        SlowPeriod = int(abs(SlowPeriod)) + 1
        FastPeriod = min(max(FastPeriod, 2), 20)
        SlowPeriod = min(max(SlowPeriod, FastPeriod + 1), 50)
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        DailyPrice = PriceSeries.resample('D').last()
        DailyMACDResult = ta.macd(DailyPrice, fast=FastPeriod, slow=SlowPeriod)
        DailyMACDLine = DailyMACDResult.iloc[:, 0]
        
        return DailyMACDLine.reindex(PriceSeries.index, method='ffill')
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def WeeklyMACD(Price, FastPeriod, SlowPeriod):
    """Calculate MACD on weekly timeframe"""
    try:
        FastPeriod = int(abs(FastPeriod)) + 1
        SlowPeriod = int(abs(SlowPeriod)) + 1
        FastPeriod = min(max(FastPeriod, 2), 10)
        SlowPeriod = min(max(SlowPeriod, FastPeriod + 1), 20)
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        WeeklyPrice = PriceSeries.resample('W').last()
        WeeklyMACDResult = ta.macd(WeeklyPrice, fast=FastPeriod, slow=SlowPeriod)
        WeeklyMACDLine = WeeklyMACDResult.iloc[:, 0]
        
        return WeeklyMACDLine.reindex(PriceSeries.index, method='ffill')
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def GetMultiTimeframeIndicators():
    """Return list of multi-timeframe indicator primitives for registration"""
    return [
        ("DailyRSI", DailyRSI, 2),
        ("WeeklyRSI", WeeklyRSI, 2),
        ("DailyMACD", DailyMACD, 3),
        ("WeeklyMACD", WeeklyMACD, 3),
    ]