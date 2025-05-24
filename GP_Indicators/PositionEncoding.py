import pandas as pd


def PriceToDaily(Price, Type):
    """Calculate price ratio to daily high/low/close"""
    try:
        Type = int(abs(Type)) % 3  # 0=high, 1=low, 2=close
        if not hasattr(Price, '__len__'):
            return 1.0
        
        # Resample to daily
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        Daily = PriceSeries.resample('D').agg(['high', 'low', 'close'])
        
        # Forward fill to align with hourly data
        if Type == 0:
            DailyValue = Daily['high'].reindex(PriceSeries.index, method='ffill')
        elif Type == 1:
            DailyValue = Daily['low'].reindex(PriceSeries.index, method='ffill')
        else:
            DailyValue = Daily['close'].reindex(PriceSeries.index, method='ffill')
        
        # Calculate ratio
        return PriceSeries / DailyValue
    except:
        return pd.Series([1.0] * len(Price)) if hasattr(Price, '__len__') else 1.0


def PriceToWeekly(Price, Type):
    """Calculate price ratio to weekly high/low/close"""
    try:
        Type = int(abs(Type)) % 3
        if not hasattr(Price, '__len__'):
            return 1.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        Weekly = PriceSeries.resample('W').agg(['high', 'low', 'close'])
        
        if Type == 0:
            WeeklyValue = Weekly['high'].reindex(PriceSeries.index, method='ffill')
        elif Type == 1:
            WeeklyValue = Weekly['low'].reindex(PriceSeries.index, method='ffill')
        else:
            WeeklyValue = Weekly['close'].reindex(PriceSeries.index, method='ffill')
        
        return PriceSeries / WeeklyValue
    except:
        return pd.Series([1.0] * len(Price)) if hasattr(Price, '__len__') else 1.0


def DistanceFromDailyMA(Price, Period):
    """Calculate distance from daily moving average as percentage"""
    try:
        Period = int(abs(Period)) + 1
        Period = min(max(Period, 5), 50)
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        DailyPrice = PriceSeries.resample('D').last()
        DailyMA = DailyPrice.rolling(window=Period).mean()
        DailyMAHourly = DailyMA.reindex(PriceSeries.index, method='ffill')
        
        return (PriceSeries - DailyMAHourly) / DailyMAHourly
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def DistanceFromWeeklyMA(Price, Period):
    """Calculate distance from weekly moving average as percentage"""
    try:
        Period = int(abs(Period)) + 1
        Period = min(max(Period, 2), 20)
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        WeeklyPrice = PriceSeries.resample('W').last()
        WeeklyMA = WeeklyPrice.rolling(window=Period).mean()
        WeeklyMAHourly = WeeklyMA.reindex(PriceSeries.index, method='ffill')
        
        return (PriceSeries - WeeklyMAHourly) / WeeklyMAHourly
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def PositionInDailyRange(Price):
    """Calculate position within daily range (0-1)"""
    try:
        if not hasattr(Price, '__len__'):
            return 0.5
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        
        # Calculate daily high/low from the price series itself
        DailyHigh = PriceSeries.resample('D').max().reindex(PriceSeries.index, method='ffill')
        DailyLow = PriceSeries.resample('D').min().reindex(PriceSeries.index, method='ffill')
        DailyRange = DailyHigh - DailyLow
        
        # Avoid division by zero
        DailyRange = DailyRange.replace(0, 1)
        
        Position = (PriceSeries - DailyLow) / DailyRange
        return Position.clip(0, 1)
    except:
        return pd.Series([0.5] * len(Price)) if hasattr(Price, '__len__') else 0.5


def PositionInWeeklyRange(Price):
    """Calculate position within weekly range (0-1)"""
    try:
        if not hasattr(Price, '__len__'):
            return 0.5
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        
        # Calculate weekly high/low from the price series itself
        WeeklyHigh = PriceSeries.resample('W').max().reindex(PriceSeries.index, method='ffill')
        WeeklyLow = PriceSeries.resample('W').min().reindex(PriceSeries.index, method='ffill')
        WeeklyRange = WeeklyHigh - WeeklyLow
        
        # Avoid division by zero
        WeeklyRange = WeeklyRange.replace(0, 1)
        
        Position = (PriceSeries - WeeklyLow) / WeeklyRange
        return Position.clip(0, 1)
    except:
        return pd.Series([0.5] * len(Price)) if hasattr(Price, '__len__') else 0.5


def GetPositionEncodingPrimitives():
    """Return list of position encoding primitives for registration"""
    return [
        ("PriceToDaily", PriceToDaily, 2),
        ("PriceToWeekly", PriceToWeekly, 2),
        ("DistanceFromDailyMA", DistanceFromDailyMA, 2),
        ("DistanceFromWeeklyMA", DistanceFromWeeklyMA, 2),
        ("PositionInDailyRange", PositionInDailyRange, 1),
        ("PositionInWeeklyRange", PositionInWeeklyRange, 1),
    ]