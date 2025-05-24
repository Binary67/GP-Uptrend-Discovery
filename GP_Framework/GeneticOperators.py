import random
import operator
from deap import gp, tools


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
    
    # Register selection operator
    Toolbox.register("select", tools.selTournament, tournsize=3)