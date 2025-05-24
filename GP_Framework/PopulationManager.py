from deap import base, creator, tools, gp


def InitializeCreators():
    """Initialize DEAP creators for fitness and individuals"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def CreateToolbox(PSet, MinDepth=2, MaxDepth=6):
    """Create and configure the DEAP toolbox"""
    
    Toolbox = base.Toolbox()
    
    # Register tree generation
    Toolbox.register("Expr", gp.genHalfAndHalf, pset=PSet, min_=MinDepth, max_=MaxDepth)
    Toolbox.register("Individual", tools.initIterate, creator.Individual, Toolbox.Expr)
    Toolbox.register("Population", tools.initRepeat, list, Toolbox.Individual)
    
    return Toolbox


def GeneratePopulation(PSet, PopulationSize=300, MinDepth=2, MaxDepth=6):
    """Generate initial population of trading signal individuals"""
    
    # Initialize creators
    InitializeCreators()
    
    # Create toolbox
    Toolbox = CreateToolbox(PSet, MinDepth, MaxDepth)
    
    # Generate population
    Population = Toolbox.Population(n=PopulationSize)
    
    return Population, Toolbox