import random
import numpy as np
from deap import base, creator, tools, gp
import operator
import math


def DesignIndividualRepresentation():
    """Design the individual representation (tree structure) for trading signals"""
    
    PSet = gp.PrimitiveSet("MAIN", 10)
    
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
    
    PSet.addEphemeralConstant("Rand101", lambda: random.uniform(-1, 1))
    
    PSet.renameArguments(ARG0='Open')
    PSet.renameArguments(ARG1='High')
    PSet.renameArguments(ARG2='Low')
    PSet.renameArguments(ARG3='Close')
    PSet.renameArguments(ARG4='Volume')
    PSet.renameArguments(ARG5='RSI')
    PSet.renameArguments(ARG6='MACD')
    PSet.renameArguments(ARG7='BB_Upper')
    PSet.renameArguments(ARG8='BB_Lower')
    PSet.renameArguments(ARG9='ATR')
    
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    return PSet


def GeneratePopulation(PSet, PopulationSize=300, MinDepth=2, MaxDepth=6):
    """Generate initial population of trading signal individuals"""
    
    Toolbox = base.Toolbox()
    
    Toolbox.register("Expr", gp.genHalfAndHalf, pset=PSet, min_=MinDepth, max_=MaxDepth)
    Toolbox.register("Individual", tools.initIterate, creator.Individual, Toolbox.Expr)
    Toolbox.register("Population", tools.initRepeat, list, Toolbox.Individual)
    
    Population = Toolbox.Population(n=PopulationSize)
    
    return Population, Toolbox


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