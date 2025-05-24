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
    Fitness = F1Score + (0.3 * EarlyBonus)  # Weight early detection at 30% of F1 score
    
    return (Fitness,)  # Return as tuple for DEAP compatibility


def RegisterFitnessEvaluation(Toolbox, Data, PSet):
    """Register the fitness evaluation function with the toolbox"""
    Toolbox.register("evaluate", EvaluateFitness, Data=Data, PSet=PSet)
    
    # Register selection operator
    Toolbox.register("select", tools.selTournament, tournsize=3)