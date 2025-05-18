import random
import pandas as pd
from sklearn.metrics import f1_score
from deap import base, creator, tools, gp

from PrimitiveSetFactory import PrimitiveSetFactory


class GPEngine:
    def __init__(self, Data, ColumnNames, PopulationSize=100, Generations=10,
                 EliteSize=5, CrossoverProb=0.5, MutationProb=0.2,
                 LookAheadPenalty=True, PenaltyWeight=0.05):
        self.Data = Data
        self.ColumnNames = ColumnNames
        self.PopulationSize = PopulationSize
        self.Generations = Generations
        self.EliteSize = EliteSize
        self.CrossoverProb = CrossoverProb
        self.MutationProb = MutationProb
        self.LookAheadPenalty = LookAheadPenalty
        self.PenaltyWeight = PenaltyWeight
        self.Factory = PrimitiveSetFactory()
        self.Pset = self.Factory.CreatePrimitiveSet(ColumnNames)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        self.Toolbox = base.Toolbox()
        self.Toolbox.register("expr", gp.genHalfAndHalf, pset=self.Pset, min_=1, max_=2)
        self.Toolbox.register("individual", tools.initIterate, creator.Individual, self.Toolbox.expr)
        self.Toolbox.register("population", tools.initRepeat, list, self.Toolbox.individual)
        self.Toolbox.register("compile", gp.compile, pset=self.Pset)
        self.Toolbox.register("evaluate", self.EvaluateIndividual)
        self.Toolbox.register("select", tools.selTournament, tournsize=3)
        self.Toolbox.register("mate", gp.cxOnePoint)
        self.Toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.Toolbox.register("mutate", gp.mutUniform, expr=self.Toolbox.expr_mut, pset=self.Pset)

    def EvaluateIndividual(self, Individual):
        Function = self.Toolbox.compile(expr=Individual)
        Arguments = [self.Data[Name] for Name in self.ColumnNames]
        Output = Function(*Arguments)
        Predictions = pd.Series(Output).astype(bool)
        Labels = self.Data["IsUpcomingUptrend"].astype(bool)
        Score = f1_score(Labels, Predictions)
        if self.LookAheadPenalty:
            EarlyCount = 0
            Limit = len(Predictions) - 10
            for Index in range(Limit):
                if Predictions.iloc[Index] and not Labels.iloc[Index]:
                    if Labels.iloc[Index + 1 : Index + 11].any():
                        EarlyCount += 1
            Score -= self.PenaltyWeight * EarlyCount / len(Predictions)
            Score = max(0.0, Score)
        return Score,

    def Evolve(self):
        Population = self.Toolbox.population(n=self.PopulationSize)
        FitnessValues = list(map(self.Toolbox.evaluate, Population))
        for Individual, Value in zip(Population, FitnessValues):
            Individual.fitness.values = Value
        for GenerationIndex in range(self.Generations):
            Elite = tools.selBest(Population, self.EliteSize)
            Offspring = self.Toolbox.select(Population, len(Population) - self.EliteSize)
            Offspring = list(map(self.Toolbox.clone, Offspring))
            for Child1, Child2 in zip(Offspring[::2], Offspring[1::2]):
                if random.random() < self.CrossoverProb:
                    self.Toolbox.mate(Child1, Child2)
                    del Child1.fitness.values
                    del Child2.fitness.values
            for Individual in Offspring:
                if random.random() < self.MutationProb:
                    self.Toolbox.mutate(Individual)
                    del Individual.fitness.values
            Population = Elite + Offspring
            InvalidIndividuals = [Ind for Ind in Population if not Ind.fitness.valid]
            FitnessValues = list(map(self.Toolbox.evaluate, InvalidIndividuals))
            for Ind, Value in zip(InvalidIndividuals, FitnessValues):
                Ind.fitness.values = Value
        Best = tools.selBest(Population, 1)[0]
        return Best

