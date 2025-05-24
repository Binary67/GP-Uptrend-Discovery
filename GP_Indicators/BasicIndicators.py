import pandas_ta as ta


def EMA(Data, Period):
    """Exponential Moving Average wrapper"""
    Period = int(abs(Period)) + 1  # Ensure positive integer
    Period = min(max(Period, 2), 200)  # Clamp between 2 and 200
    if isinstance(Data, (int, float)):
        return Data
    return ta.ema(Data, length=Period)


def SMA(Data, Period):
    """Simple Moving Average wrapper"""
    Period = int(abs(Period)) + 1
    Period = min(max(Period, 2), 200)
    if isinstance(Data, (int, float)):
        return Data
    return ta.sma(Data, length=Period)


def RSI(Data, Period):
    """Relative Strength Index wrapper"""
    Period = int(abs(Period)) + 1
    Period = min(max(Period, 2), 100)
    if isinstance(Data, (int, float)):
        return 50.0  # Neutral RSI
    return ta.rsi(Data, length=Period)


def MACD(Data, FastPeriod, SlowPeriod):
    """MACD wrapper - returns MACD line"""
    FastPeriod = int(abs(FastPeriod)) + 1
    SlowPeriod = int(abs(SlowPeriod)) + 1
    FastPeriod = min(max(FastPeriod, 2), 50)
    SlowPeriod = min(max(SlowPeriod, FastPeriod + 1), 100)
    if isinstance(Data, (int, float)):
        return 0
    MACDResult = ta.macd(Data, fast=FastPeriod, slow=SlowPeriod)
    return MACDResult.iloc[:, 0]  # Return MACD line


def BollingerBands(Data, Period, StdDev):
    """Bollinger Bands wrapper - returns band width"""
    Period = int(abs(Period)) + 1
    Period = min(max(Period, 2), 100)
    StdDev = abs(StdDev) + 0.1
    StdDev = min(max(StdDev, 0.5), 4.0)
    if isinstance(Data, (int, float)):
        return 0
    BBands = ta.bbands(Data, length=Period, std=StdDev)
    # Return bandwidth (upper - lower)
    return BBands.iloc[:, 2] - BBands.iloc[:, 0]


def ATR(High, Low, Close):
    """Average True Range wrapper"""
    if isinstance(High, (int, float)) or isinstance(Low, (int, float)) or isinstance(Close, (int, float)):
        return 0
    return ta.atr(High, Low, Close, length=14)


def Momentum(Data, Period):
    """Momentum wrapper - Rate of Change"""
    Period = int(abs(Period)) + 1
    Period = min(max(Period, 1), 100)
    if isinstance(Data, (int, float)):
        return 0
    return ta.roc(Data, length=Period)


def GetBasicIndicators():
    """Return list of basic indicator primitives for registration"""
    return [
        ("EMA", EMA, 2),
        ("SMA", SMA, 2),
        ("RSI", RSI, 2),
        ("MACD", MACD, 3),
        ("BollingerBands", BollingerBands, 3),
        ("ATR", ATR, 3),
        ("Momentum", Momentum, 2),
    ]