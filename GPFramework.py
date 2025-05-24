import random
import numpy as np
from deap import base, creator, tools, gp
import operator
import math
import pandas_ta as ta
import pandas as pd


def DesignIndividualRepresentation():
    """Design the individual representation (tree structure) for trading signals"""
    
    PSet = gp.PrimitiveSet("MAIN", 5)
    
    PSet.addPrimitive(operator.add, 2)
    PSet.addPrimitive(operator.sub, 2)
    PSet.addPrimitive(operator.mul, 2)
    PSet.addPrimitive(ProtectedDiv, 2)
    PSet.addPrimitive(operator.neg, 1)
    PSet.addPrimitive(math.sin, 1)
    PSet.addPrimitive(math.cos, 1)
    PSet.addPrimitive(operator.abs, 1)
    PSet.addPrimitive(Max, 2)
    PSet.addPrimitive(Min, 2)
    PSet.addPrimitive(Sqrt, 1)
    PSet.addPrimitive(Log, 1)
    
    # Technical Analysis Indicators
    PSet.addPrimitive(EMA, 2)  # EMA(data, period)
    PSet.addPrimitive(SMA, 2)  # SMA(data, period)
    PSet.addPrimitive(RSI, 2)  # RSI(data, period)
    PSet.addPrimitive(MACD, 3)  # MACD(data, fast, slow)
    PSet.addPrimitive(BollingerBands, 3)  # BB(data, period, std)
    PSet.addPrimitive(ATR, 3)  # ATR(high, low, close)
    PSet.addPrimitive(Momentum, 2)  # Momentum(data, period)
    
    # Relative Position Encoding
    PSet.addPrimitive(PriceToDaily, 2)  # PriceToDaily(price, type) where type: 0=high, 1=low, 2=close
    PSet.addPrimitive(PriceToWeekly, 2)  # PriceToWeekly(price, type)
    PSet.addPrimitive(DistanceFromDailyMA, 2)  # DistanceFromDailyMA(price, period)
    PSet.addPrimitive(DistanceFromWeeklyMA, 2)  # DistanceFromWeeklyMA(price, period)
    PSet.addPrimitive(PositionInDailyRange, 1)  # PositionInDailyRange(price) - uses global high/low
    PSet.addPrimitive(PositionInWeeklyRange, 1)  # PositionInWeeklyRange(price) - uses global high/low
    
    # Timeframe Momentum Cascade
    PSet.addPrimitive(HourlyMomentum, 2)  # HourlyMomentum(price, hours)
    PSet.addPrimitive(DailyMomentum, 2)  # DailyMomentum(price, days)
    PSet.addPrimitive(WeeklyMomentum, 2)  # WeeklyMomentum(price, weeks)
    PSet.addPrimitive(MomentumAlignment, 1)  # MomentumAlignment(price) - calculates all internally
    
    # Higher Timeframe Indicators
    PSet.addPrimitive(DailyRSI, 2)  # DailyRSI(price, period)
    PSet.addPrimitive(WeeklyRSI, 2)  # WeeklyRSI(price, period)
    PSet.addPrimitive(DailyMACD, 3)  # DailyMACD(price, fast, slow)
    PSet.addPrimitive(WeeklyMACD, 3)  # WeeklyMACD(price, fast, slow)
    
    # Comparison Operators
    PSet.addPrimitive(GreaterThan, 2)
    PSet.addPrimitive(LessThan, 2)
    PSet.addPrimitive(GreaterEqual, 2)
    PSet.addPrimitive(LessEqual, 2)
    PSet.addPrimitive(Equal, 2)
    
    # Logical Operators
    PSet.addPrimitive(And, 2)
    PSet.addPrimitive(Or, 2)
    PSet.addPrimitive(Not, 1)
    
    # Conditional Operator
    PSet.addPrimitive(IfThenElse, 3)
    
    # Trading-Specific Operators
    PSet.addPrimitive(Lag, 2)  # Lag(data, periods)
    PSet.addPrimitive(Change, 2)  # Change(data, periods)
    PSet.addPrimitive(CrossOver, 2)  # CrossOver(series1, series2)
    PSet.addPrimitive(CrossUnder, 2)  # CrossUnder(series1, series2)
    
    PSet.addEphemeralConstant("Rand101", lambda: random.uniform(-1, 1))
    PSet.addEphemeralConstant("RandPeriod", lambda: random.randint(5, 50))
    
    PSet.renameArguments(ARG0='Open')
    PSet.renameArguments(ARG1='High')
    PSet.renameArguments(ARG2='Low')
    PSet.renameArguments(ARG3='Close')
    PSet.renameArguments(ARG4='Volume')
    
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    return PSet


def GeneratePopulation(PSet, PopulationSize=300, MinDepth = 2, MaxDepth = 6):
    """Generate initial population of trading signal individuals"""
    
    Toolbox = base.Toolbox()
    
    Toolbox.register("Expr", gp.genHalfAndHalf, pset=PSet, min_=MinDepth, max_=MaxDepth)
    Toolbox.register("Individual", tools.initIterate, creator.Individual, Toolbox.Expr)
    Toolbox.register("Population", tools.initRepeat, list, Toolbox.Individual)
    
    # Register genetic operators
    RegisterGeneticOperators(Toolbox, PSet)
    
    Population = Toolbox.Population(n=PopulationSize)
    
    return Population, Toolbox


def RegisterGeneticOperators(Toolbox, PSet):
    """Register crossover and mutation operators with the toolbox"""
    
    # Crossover - One-point crossover (most common for GP trees)
    Toolbox.register("mate", gp.cxOnePoint)
    
    # Multiple mutation operators
    Toolbox.register("mutUniform", gp.mutUniform, expr=Toolbox.Expr, pset=PSet)
    Toolbox.register("mutNodeReplace", gp.mutNodeReplacement, pset=PSet)
    Toolbox.register("mutEphemeral", gp.mutEphemeral, mode="all")
    
    # Combined mutation strategy
    def MutateIndividual(Individual):
        """Apply different mutation strategies with varying probabilities"""
        Rand = random.random()
        if Rand < 0.5:
            # 50% chance: Uniform mutation (replace subtrees)
            return Toolbox.mutUniform(Individual)
        elif Rand < 0.8:
            # 30% chance: Node replacement (replace single nodes)
            return Toolbox.mutNodeReplace(Individual)
        else:
            # 20% chance: Ephemeral constant mutation
            return Toolbox.mutEphemeral(Individual)
    
    Toolbox.register("mutate", MutateIndividual)
    
    # Apply bloat control to prevent trees from growing too large
    Toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    Toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


def ProtectedDiv(Left, Right):
    """Protected division to avoid division by zero"""
    try:
        return Left / Right
    except ZeroDivisionError:
        return 1


def Max(Left, Right):
    """Maximum of two values"""
    return max(Left, Right)


def Min(Left, Right):
    """Minimum of two values"""
    return min(Left, Right)


def Sqrt(Value):
    """Protected square root"""
    return math.sqrt(abs(Value))


def Log(Value):
    """Protected logarithm"""
    try:
        return math.log(abs(Value)) if Value != 0 else 0
    except:
        return 0


# Technical Analysis Wrapper Functions
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


# Comparison Operators
def GreaterThan(Left, Right):
    """Return 1 if Left > Right, else -1"""
    try:
        # Handle series comparisons
        if hasattr(Left, '__len__') and hasattr(Right, '__len__'):
            return (Left > Right).astype(float) * 2 - 1
        return 1 if Left > Right else -1
    except:
        return -1


def LessThan(Left, Right):
    """Return 1 if Left < Right, else -1"""
    try:
        if hasattr(Left, '__len__') and hasattr(Right, '__len__'):
            return (Left < Right).astype(float) * 2 - 1
        return 1 if Left < Right else -1
    except:
        return -1


def GreaterEqual(Left, Right):
    """Return 1 if Left >= Right, else -1"""
    try:
        if hasattr(Left, '__len__') and hasattr(Right, '__len__'):
            return (Left >= Right).astype(float) * 2 - 1
        return 1 if Left >= Right else -1
    except:
        return -1


def LessEqual(Left, Right):
    """Return 1 if Left <= Right, else -1"""
    try:
        if hasattr(Left, '__len__') and hasattr(Right, '__len__'):
            return (Left <= Right).astype(float) * 2 - 1
        return 1 if Left <= Right else -1
    except:
        return -1


def Equal(Left, Right):
    """Return 1 if Left == Right (with tolerance), else -1"""
    try:
        Tolerance = 0.0001
        if hasattr(Left, '__len__') and hasattr(Right, '__len__'):
            return (abs(Left - Right) < Tolerance).astype(float) * 2 - 1
        return 1 if abs(Left - Right) < Tolerance else -1
    except:
        return -1


# Logical Operators
def And(Left, Right):
    """Logical AND - both must be positive"""
    try:
        if hasattr(Left, '__len__') and hasattr(Right, '__len__'):
            return ((Left > 0) & (Right > 0)).astype(float) * 2 - 1
        return 1 if Left > 0 and Right > 0 else -1
    except:
        return -1


def Or(Left, Right):
    """Logical OR - at least one must be positive"""
    try:
        if hasattr(Left, '__len__') and hasattr(Right, '__len__'):
            return ((Left > 0) | (Right > 0)).astype(float) * 2 - 1
        return 1 if Left > 0 or Right > 0 else -1
    except:
        return -1


def Not(Value):
    """Logical NOT - negate the sign"""
    try:
        if hasattr(Value, '__len__'):
            return -Value
        return -Value
    except:
        return 1


# Conditional Operator
def IfThenElse(Condition, TrueValue, FalseValue):
    """If Condition > 0 return TrueValue, else FalseValue"""
    try:
        if hasattr(Condition, '__len__'):
            # Handle series data
            import pandas as pd
            import numpy as np
            Result = pd.Series(index=Condition.index if hasattr(Condition, 'index') else range(len(Condition)))
            Mask = Condition > 0
            Result[Mask] = TrueValue[Mask] if hasattr(TrueValue, '__len__') else TrueValue
            Result[~Mask] = FalseValue[~Mask] if hasattr(FalseValue, '__len__') else FalseValue
            return Result
        return TrueValue if Condition > 0 else FalseValue
    except:
        return FalseValue


# Trading-Specific Operators
def Lag(Data, Periods):
    """Get lagged values of data"""
    try:
        Periods = int(abs(Periods)) + 1
        Periods = min(max(Periods, 1), 50)
        if hasattr(Data, 'shift'):
            return Data.shift(Periods)
        elif hasattr(Data, '__len__'):
            import pandas as pd
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
            import pandas as pd
            return pd.Series(Data).diff(Periods)
        return 0
    except:
        return 0


def CrossOver(Series1, Series2):
    """Detect when Series1 crosses above Series2"""
    try:
        if hasattr(Series1, '__len__') and hasattr(Series2, '__len__'):
            import pandas as pd
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
            import pandas as pd
            S1 = pd.Series(Series1) if not hasattr(Series1, 'shift') else Series1
            S2 = pd.Series(Series2) if not hasattr(Series2, 'shift') else Series2
            Cross = (S1.shift(1) >= S2.shift(1)) & (S1 < S2)
            return Cross.astype(float) * 2 - 1
        return -1
    except:
        return -1


# Multi-Timeframe Functions - Method 2: Relative Position Encoding
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


# Multi-Timeframe Functions - Method 3: Timeframe Momentum Cascade
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


# Higher Timeframe Indicators
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


def EvaluateFitness(Individual, Data, PSet):
    """
    Evaluate fitness using Precision-Recall Balance with Early Detection Bonus
    
    Fitness = (2 × Precision × Recall) / (Precision + Recall) + EarlyBonus
    
    Parameters:
    - Individual: GP tree individual to evaluate
    - Data: Market data DataFrame with OHLCV and labels
    - PSet: Primitive set for compilation
    
    Returns:
    - Tuple containing fitness score (higher is better)
    """
    
    # Compile the individual into a callable function
    CompiledFunc = gp.compile(expr=Individual, pset=PSet)
    
    # Minimum data points needed for indicators
    MinDataPoints = 200
    
    try:
        # Get full data as pandas Series (not numpy arrays)
        Open = Data['Open']
        High = Data['High']
        Low = Data['Low']
        Close = Data['Close']
        Volume = Data['Volume']
        
        # Evaluate the GP individual on entire dataset at once
        SignalValues = CompiledFunc(Open, High, Low, Close, Volume)
        
        # Handle different output types
        if isinstance(SignalValues, (int, float)):
            # If scalar, create series with same value
            SignalValues = pd.Series([SignalValues] * len(Data), index=Data.index)
        elif hasattr(SignalValues, '__len__') and not isinstance(SignalValues, pd.Series):
            # Convert to pandas Series if needed
            SignalValues = pd.Series(SignalValues, index=Data.index)
        
        # Ensure we have valid signal values
        if len(SignalValues) != len(Data):
            # If lengths don't match, likely due to indicator lag, pad with zeros
            if len(SignalValues) < len(Data):
                # Pad beginning with zeros
                Padding = pd.Series([0] * (len(Data) - len(SignalValues)))
                SignalValues = pd.concat([Padding, SignalValues.reset_index(drop=True)], ignore_index=True)
                SignalValues.index = Data.index
            else:
                # Trim to match data length
                SignalValues = SignalValues.iloc[:len(Data)]
        
        # Generate binary predictions (vectorized)
        Predictions = (SignalValues > 0).astype(int)
        
        # Get actual labels
        Labels = Data['Label']
        
        # Skip the first MinDataPoints for evaluation (need sufficient history for indicators)
        # Also skip last 10 points to avoid looking ahead
        ValidIndices = Data.index[MinDataPoints:len(Data) - 10]
        
        # Apply slicing to get valid predictions and labels
        ValidPredictions = Predictions.loc[ValidIndices]
        ValidLabels = Labels.loc[ValidIndices]
        
        # Calculate confusion matrix components using vectorized operations
        TruePositives = ((ValidPredictions == 1) & (ValidLabels == 1)).sum()
        FalsePositives = ((ValidPredictions == 1) & (ValidLabels == 0)).sum()
        FalseNegatives = ((ValidPredictions == 0) & (ValidLabels == 1)).sum()
        
        # Calculate early detection bonus for true positives
        TruePositiveMask = (ValidPredictions == 1) & (ValidLabels == 1)
        if TruePositiveMask.any():
            # Get days early for true positive predictions
            DaysEarlyValues = Data.loc[ValidIndices, 'DaysUntilCrossover'][TruePositiveMask]
            ValidDaysEarly = DaysEarlyValues[DaysEarlyValues > 0]
            TotalDaysEarly = ValidDaysEarly.sum()
            NumValidPredictions = len(ValidDaysEarly)
        else:
            TotalDaysEarly = 0
            NumValidPredictions = 0
        
    except Exception as e:
        # If vectorized evaluation fails, return zero fitness
        # Uncomment for debugging: print(f"EvaluateFitness error: {e}")
        return (0.0,)
    
    # Calculate metrics
    Precision = TruePositives / (TruePositives + FalsePositives) if (TruePositives + FalsePositives) > 0 else 0
    Recall = TruePositives / (TruePositives + FalseNegatives) if (TruePositives + FalseNegatives) > 0 else 0
    
    # F1 Score (Precision-Recall Balance)
    F1Score = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    
    # Early detection bonus
    AvgDaysEarly = (TotalDaysEarly / NumValidPredictions) if NumValidPredictions > 0 else 0
    EarlyBonus = AvgDaysEarly / 10  # Normalize to 0-1 range for 0-10 days
    
    # Final fitness score
    Fitness = F1Score + (0.2 * EarlyBonus)  # Weight early detection at 30% of F1 score
    
    return (Fitness,)  # Return as tuple for DEAP compatibility


def RegisterFitnessEvaluation(Toolbox, Data, PSet):
    """Register the fitness evaluation function with the toolbox"""
    Toolbox.register("evaluate", EvaluateFitness, Data=Data, PSet=PSet)
    
    # Register selection operator
    Toolbox.register("select", tools.selTournament, tournsize=3)