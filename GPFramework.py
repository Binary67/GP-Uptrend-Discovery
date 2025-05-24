import random
import numpy as np
from deap import base, creator, tools, gp
import operator
import math
import pandas_ta as ta


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
    
    # Initialize prediction tracking
    TruePositives = 0
    FalsePositives = 0
    FalseNegatives = 0
    TotalDaysEarly = 0
    ValidPredictions = 0
    
    # Get required data columns
    Open = Data['Open'].values
    High = Data['High'].values
    Low = Data['Low'].values
    Close = Data['Close'].values
    Volume = Data['Volume'].values
    Labels = Data['Label'].values  # 1 for uptrend within 3-10 days, 0 otherwise
    
    # Evaluate the individual on each data point
    MinDataPoints = 200  # Minimum data points needed for indicators
    
    for i in range(MinDataPoints, len(Data) - 10):
        try:
            # Get data up to current point
            OpenSlice = Data['Open'].iloc[:i+1]
            HighSlice = Data['High'].iloc[:i+1]
            LowSlice = Data['Low'].iloc[:i+1]
            CloseSlice = Data['Close'].iloc[:i+1]
            VolumeSlice = Data['Volume'].iloc[:i+1]
            
            # Evaluate the GP individual
            SignalValue = CompiledFunc(OpenSlice, HighSlice, LowSlice, CloseSlice, VolumeSlice)
            
            # Handle array outputs from technical indicators
            if hasattr(SignalValue, '__len__') and not isinstance(SignalValue, str):
                SignalValue = SignalValue.iloc[-1] if hasattr(SignalValue, 'iloc') else SignalValue[-1]
            
            # Generate binary prediction (threshold at 0)
            Prediction = 1 if SignalValue > 0 else 0
            ActualLabel = Labels[i]
            
            # Update confusion matrix
            if Prediction == 1 and ActualLabel == 1:
                TruePositives += 1
                # Use actual days until crossover from DataLabel
                DaysEarly = Data['DaysUntilCrossover'].iloc[i]
                if DaysEarly > 0:  # Valid crossover prediction
                    TotalDaysEarly += DaysEarly
                    ValidPredictions += 1
            elif Prediction == 1 and ActualLabel == 0:
                FalsePositives += 1
            elif Prediction == 0 and ActualLabel == 1:
                FalseNegatives += 1
                
        except Exception as e:
            # Handle any errors in evaluation (e.g., invalid operations)
            continue
    
    # Calculate metrics
    Precision = TruePositives / (TruePositives + FalsePositives) if (TruePositives + FalsePositives) > 0 else 0
    Recall = TruePositives / (TruePositives + FalseNegatives) if (TruePositives + FalseNegatives) > 0 else 0
    
    # F1 Score (Precision-Recall Balance)
    F1Score = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    
    # Early detection bonus
    AvgDaysEarly = (TotalDaysEarly / ValidPredictions) if ValidPredictions > 0 else 0
    EarlyBonus = AvgDaysEarly / 10  # Normalize to 0-1 range for 0-10 days
    
    # Final fitness score
    Fitness = F1Score + (0.3 * EarlyBonus)  # Weight early detection at 30% of F1 score
    
    return (Fitness,)  # Return as tuple for DEAP compatibility


def RegisterFitnessEvaluation(Toolbox, Data, PSet):
    """Register the fitness evaluation function with the toolbox"""
    Toolbox.register("evaluate", EvaluateFitness, Data=Data, PSet=PSet)
    
    # Register selection operator
    Toolbox.register("select", tools.selTournament, tournsize=3)