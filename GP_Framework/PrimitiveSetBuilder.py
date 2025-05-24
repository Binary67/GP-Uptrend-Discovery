import random
from deap import gp
from GP_Primitives.MathPrimitives import GetMathPrimitives
from GP_Primitives.ComparisonPrimitives import GetComparisonPrimitives
from GP_Primitives.TradingPrimitives import GetTradingPrimitives
from GP_Indicators.BasicIndicators import GetBasicIndicators
from GP_Indicators.MultiTimeframe import GetMultiTimeframeIndicators
from GP_Indicators.PositionEncoding import GetPositionEncodingPrimitives
from GP_Indicators.MomentumIndicators import GetMomentumIndicators


def BuildPrimitiveSet():
    """Build and return the complete primitive set for GP"""
    
    # Create primitive set with 5 inputs (OHLCV)
    PSet = gp.PrimitiveSet("MAIN", 5)
    
    # Register all primitives from different modules
    RegisterPrimitives(PSet)
    
    # Add ephemeral constants
    PSet.addEphemeralConstant("Rand101", lambda: random.uniform(-1, 1))
    PSet.addEphemeralConstant("RandPeriod", lambda: random.randint(5, 50))
    
    # Rename arguments for clarity
    PSet.renameArguments(ARG0='Open')
    PSet.renameArguments(ARG1='High')
    PSet.renameArguments(ARG2='Low')
    PSet.renameArguments(ARG3='Close')
    PSet.renameArguments(ARG4='Volume')
    
    return PSet


def RegisterPrimitives(PSet):
    """Register all primitives from various modules"""
    
    # Get all primitive lists
    PrimitiveLists = [
        GetMathPrimitives(),
        GetComparisonPrimitives(),
        GetTradingPrimitives(),
        GetBasicIndicators(),
        GetMultiTimeframeIndicators(),
        GetPositionEncodingPrimitives(),
        GetMomentumIndicators(),
    ]
    
    # Register each primitive
    for PrimitiveList in PrimitiveLists:
        for Name, Function, Arity in PrimitiveList:
            PSet.addPrimitive(Function, Arity, name=Name)


def GetAllPrimitiveFunctions():
    """Get a dictionary of all primitive functions for evaluation"""
    
    Functions = {}
    
    # Import all necessary functions
    from GP_Primitives.MathPrimitives import ProtectedDiv, Max, Min, Sqrt, Log
    from GP_Primitives.ComparisonPrimitives import (
        GreaterThan, LessThan, GreaterEqual, LessEqual, Equal,
        And, Or, Not, IfThenElse
    )
    from GP_Primitives.TradingPrimitives import Lag, Change, CrossOver, CrossUnder
    from GP_Indicators.BasicIndicators import EMA, SMA, RSI, MACD, BollingerBands, ATR, Momentum
    from GP_Indicators.MultiTimeframe import DailyRSI, WeeklyRSI, DailyMACD, WeeklyMACD
    from GP_Indicators.PositionEncoding import (
        PriceToDaily, PriceToWeekly, DistanceFromDailyMA, DistanceFromWeeklyMA,
        PositionInDailyRange, PositionInWeeklyRange
    )
    from GP_Indicators.MomentumIndicators import (
        HourlyMomentum, DailyMomentum, WeeklyMomentum, MomentumAlignment
    )
    
    # Add all functions to dictionary
    LocalVars = locals()
    AllPrimitives = [
        GetMathPrimitives(),
        GetComparisonPrimitives(),
        GetTradingPrimitives(),
        GetBasicIndicators(),
        GetMultiTimeframeIndicators(),
        GetPositionEncodingPrimitives(),
        GetMomentumIndicators(),
    ]
    
    for PrimitiveList in AllPrimitives:
        for Name, Function, _ in PrimitiveList:
            Functions[Name] = Function
    
    return Functions