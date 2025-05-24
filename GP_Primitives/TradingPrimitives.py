import pandas as pd


def Lag(Data, Periods):
    """Get lagged values of data"""
    try:
        Periods = int(abs(Periods)) + 1
        Periods = min(max(Periods, 1), 50)
        if hasattr(Data, 'shift'):
            return Data.shift(Periods)
        elif hasattr(Data, '__len__'):
            return pd.Series(Data).shift(Periods)
        return Data
    except:
        return Data


def Change(Data, Periods):
    """Calculate change over periods"""
    try:
        Periods = int(abs(Periods)) + 1
        Periods = min(max(Periods, 1), 50)
        if hasattr(Data, 'diff'):
            return Data.diff(Periods)
        elif hasattr(Data, '__len__'):
            return pd.Series(Data).diff(Periods)
        return 0
    except:
        return 0


def CrossOver(Series1, Series2):
    """Detect when Series1 crosses above Series2"""
    try:
        if hasattr(Series1, '__len__') and hasattr(Series2, '__len__'):
            S1 = pd.Series(Series1) if not hasattr(Series1, 'shift') else Series1
            S2 = pd.Series(Series2) if not hasattr(Series2, 'shift') else Series2
            Cross = (S1.shift(1) <= S2.shift(1)) & (S1 > S2)
            return Cross.astype(float) * 2 - 1
        return -1
    except:
        return -1


def CrossUnder(Series1, Series2):
    """Detect when Series1 crosses below Series2"""
    try:
        if hasattr(Series1, '__len__') and hasattr(Series2, '__len__'):
            S1 = pd.Series(Series1) if not hasattr(Series1, 'shift') else Series1
            S2 = pd.Series(Series2) if not hasattr(Series2, 'shift') else Series2
            Cross = (S1.shift(1) >= S2.shift(1)) & (S1 < S2)
            return Cross.astype(float) * 2 - 1
        return -1
    except:
        return -1


def GetTradingPrimitives():
    """Return list of trading-specific primitives for registration"""
    return [
        ("Lag", Lag, 2),
        ("Change", Change, 2),
        ("CrossOver", CrossOver, 2),
        ("CrossUnder", CrossUnder, 2),
    ]