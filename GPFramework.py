"""
GP Framework - Main orchestrator for the Genetic Programming system
This module provides the high-level interface to the modular GP components
"""

from GP_Framework.PrimitiveSetBuilder import BuildPrimitiveSet
from GP_Framework.PopulationManager import GeneratePopulation, InitializeCreators
from GP_Framework.GeneticOperators import RegisterGeneticOperators
from GP_Framework.FitnessEvaluator import RegisterFitnessEvaluation


def DesignIndividualRepresentation():
    """Design the individual representation (tree structure) for trading signals"""
    
    # Initialize DEAP creators
    InitializeCreators()
    
    # Build the primitive set with all operators and indicators
    PSet = BuildPrimitiveSet()
    
    return PSet


def CreateGPFramework(Data, PopulationSize=300, MinDepth=2, MaxDepth=6):
    """
    Create complete GP framework with all components configured
    
    Parameters:
    - Data: Market data DataFrame with OHLCV and labels
    - PopulationSize: Number of individuals in population
    - MinDepth: Minimum tree depth
    - MaxDepth: Maximum tree depth
    
    Returns:
    - PSet: Primitive set
    - Population: Initial population
    - Toolbox: Configured DEAP toolbox
    """
    
    # Design individual representation
    PSet = DesignIndividualRepresentation()
    
    # Generate population
    Population, Toolbox = GeneratePopulation(
        PSet, 
        PopulationSize=PopulationSize,
        MinDepth=MinDepth,
        MaxDepth=MaxDepth
    )
    
    # Register genetic operators
    RegisterGeneticOperators(Toolbox, PSet)
    
    # Register fitness evaluation
    RegisterFitnessEvaluation(Toolbox, Data, PSet)
    
    return PSet, Population, Toolbox


# For backward compatibility - maintain original function names
def GeneratePopulation(PSet, PopulationSize=300, MinDepth=2, MaxDepth=6):
    """Generate initial population of trading signal individuals"""
    from GP_Framework.PopulationManager import GeneratePopulation as GenPop
    return GenPop(PSet, PopulationSize, MinDepth, MaxDepth)


def RegisterGeneticOperators(Toolbox, PSet):
    """Register crossover and mutation operators with the toolbox"""
    from GP_Framework.GeneticOperators import RegisterGeneticOperators as RegOps
    RegOps(Toolbox, PSet)


def RegisterFitnessEvaluation(Toolbox, Data, PSet):
    """Register the fitness evaluation function with the toolbox"""
    from GP_Framework.FitnessEvaluator import RegisterFitnessEvaluation as RegFit
    RegFit(Toolbox, Data, PSet)