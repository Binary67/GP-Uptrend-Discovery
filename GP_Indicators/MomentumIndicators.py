import pandas as pd


def HourlyMomentum(Price, Hours):
    """Calculate momentum over specified hours"""
    try:
        Hours = int(abs(Hours)) + 1
        Hours = min(max(Hours, 1), 24)
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        return (PriceSeries - PriceSeries.shift(Hours)) / PriceSeries.shift(Hours)
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def DailyMomentum(Price, Days):
    """Calculate momentum over specified days"""
    try:
        Days = int(abs(Days)) + 1
        Days = min(max(Days, 1), 20)
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        HoursToShift = Days * 24
        return (PriceSeries - PriceSeries.shift(HoursToShift)) / PriceSeries.shift(HoursToShift)
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def WeeklyMomentum(Price, Weeks):
    """Calculate momentum over specified weeks"""
    try:
        Weeks = int(abs(Weeks)) + 1
        Weeks = min(max(Weeks, 1), 4)
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        HoursToShift = Weeks * 24 * 7
        return (PriceSeries - PriceSeries.shift(HoursToShift)) / PriceSeries.shift(HoursToShift)
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def MomentumAlignment(Price):
    """Calculate momentum alignment score across timeframes"""
    try:
        if not hasattr(Price, '__len__'):
            return 0.0
        
        PriceSeries = pd.Series(Price) if not isinstance(Price, pd.Series) else Price
        
        # Calculate momentum at different timeframes
        # Hourly: 6 hours (quarter day)
        HourlyMom = (PriceSeries - PriceSeries.shift(6)) / PriceSeries.shift(6)
        
        # Daily: 5 days (trading week)
        DailyMom = (PriceSeries - PriceSeries.shift(24 * 5)) / PriceSeries.shift(24 * 5)
        
        # Weekly: 2 weeks
        WeeklyMom = (PriceSeries - PriceSeries.shift(24 * 14)) / PriceSeries.shift(24 * 14)
        
        # Fill NaN values with 0
        HourlyMom = HourlyMom.fillna(0)
        DailyMom = DailyMom.fillna(0)
        WeeklyMom = WeeklyMom.fillna(0)
        
        # Count positive momentums
        PositiveCount = (HourlyMom > 0).astype(int) + (DailyMom > 0).astype(int) + (WeeklyMom > 0).astype(int)
        
        # Alignment score: -1 (all negative) to +1 (all positive)
        AlignmentScore = (PositiveCount - 1.5) / 1.5
        
        # Weight by magnitude if all aligned
        FullyAligned = (PositiveCount == 3) | (PositiveCount == 0)
        MagnitudeBonus = (abs(HourlyMom) + abs(DailyMom) + abs(WeeklyMom)) / 3
        
        Result = AlignmentScore.copy()
        Result[FullyAligned] = Result[FullyAligned] * (1 + MagnitudeBonus[FullyAligned])
        
        return Result
    except:
        return pd.Series([0.0] * len(Price)) if hasattr(Price, '__len__') else 0.0


def GetMomentumIndicators():
    """Return list of momentum indicator primitives for registration"""
    return [
        ("HourlyMomentum", HourlyMomentum, 2),
        ("DailyMomentum", DailyMomentum, 2),
        ("WeeklyMomentum", WeeklyMomentum, 2),
        ("MomentumAlignment", MomentumAlignment, 1),
    ]