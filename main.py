from DataCleaning import DataCleaner
from DataDownloader import YFinanceDownloader
from DataLabel import DataLabel
from GPFramework import DesignIndividualRepresentation, GeneratePopulation, RegisterFitnessEvaluation, EvaluateFitness
from deap import algorithms, tools, gp
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Download and prepare data
    print("1. Downloading market data...")
    DataDownloaderObj = YFinanceDownloader('AAPL', '2020-01-01', '2024-12-31', '1d')
    TradingData = DataDownloaderObj.DownloadData()
    print(f"   Downloaded {len(TradingData)} days of data")

    print("\n2. Cleaning data...")
    DataCleanerObj = DataCleaner(TradingData)
    
    CleanedData = (DataCleanerObj
                   .ValidateOHLC()
                   .GetCleanedData())
    print(f"   Cleaned data shape: {CleanedData.shape}")
    
    print("\n3. Labeling data for uptrend prediction...")
    DataLabelObj = DataLabel(CleanedData)
    LabeledData = DataLabelObj.LabelUptrend()
    
    print("\n   Label Distribution:")
    print(f"   Positive labels (crossover in 3-10 days): {LabeledData['Label'].sum()}")
    print(f"   Negative labels: {(LabeledData['Label'] == 0).sum()}")
    print(f"   Class balance: {LabeledData['Label'].mean():.3%} positive")
    
    # Split data into train/test sets
    SplitPoint = int(len(LabeledData) * 0.8)
    TrainData = LabeledData.iloc[:SplitPoint].copy()
    TestData = LabeledData.iloc[SplitPoint:].copy()
    print(f"\n4. Data split:")
    print(f"   Training: {len(TrainData)} samples")
    print(f"   Testing: {len(TestData)} samples")
    
    # Initialize GP framework
    print("\n5. Initializing Genetic Programming framework...")
    PSet = DesignIndividualRepresentation()
    
    # Generate initial population
    PopulationSize = 500
    Population, Toolbox = GeneratePopulation(PSet, PopulationSize=PopulationSize)
    
    # Register fitness evaluation with training data
    RegisterFitnessEvaluation(Toolbox, TrainData, PSet)
    
    # Statistics setup
    Stats = tools.Statistics(lambda ind: ind.fitness.values)
    Stats.register("avg", np.mean)
    Stats.register("std", np.std)
    Stats.register("min", np.min)
    Stats.register("max", np.max)
    
    # Evolution parameters
    NGEN = 25  # Number of generations
    CXPB = 0.7  # Crossover probability
    MUTPB = 0.2  # Mutation probability
    
    print(f"\n6. Starting evolution:")
    print(f"   Population Size: {PopulationSize}")
    print(f"   Generations: {NGEN}")
    print(f"   Crossover Probability: {CXPB}")
    print(f"   Mutation Probability: {MUTPB}\n")
    
    # Hall of Fame to track best individuals
    HallOfFame = tools.HallOfFame(10)
    
    # # Run the evolution
    Population, Logbook = algorithms.eaSimple(
        Population, 
        Toolbox, 
        cxpb=CXPB, 
        mutpb=MUTPB, 
        ngen=NGEN, 
        stats=Stats, 
        halloffame=HallOfFame,
        verbose=True
    )
    
    # Analyze results
    print("\n\n=== Evolution Results ===")
    print("\nTop 5 Best Individuals:")
    for i, BestInd in enumerate(HallOfFame[:5]):
        print(f"\n{i+1}. Training Fitness: {BestInd.fitness.values[0]:.4f}")
        print(f"   Expression: {str(BestInd)}")
        
        # Evaluate on test set
        TestFitness = EvaluateFitness(BestInd, TestData, PSet)
        print(f"   Test Fitness: {TestFitness[0]:.4f}")
    
    # Detailed analysis of best individual
    print("\n\n=== Best Individual Analysis ===")
    BestIndividual = HallOfFame[0]
    
    # Compile and test the best individual
    CompiledBest = gp.compile(expr=BestIndividual, pset=PSet)
    
    # Generate predictions on test set
    Predictions = []
    for i in range(200, len(TestData)):
        try:
            Signal = CompiledBest(
                TestData['Open'].iloc[:i+1],
                TestData['High'].iloc[:i+1],
                TestData['Low'].iloc[:i+1],
                TestData['Close'].iloc[:i+1],
                TestData['Volume'].iloc[:i+1]
            )
            if hasattr(Signal, '__len__') and not isinstance(Signal, str):
                Signal = Signal.iloc[-1] if hasattr(Signal, 'iloc') else Signal[-1]
            Predictions.append(1 if Signal > 0 else 0)
        except:
            Predictions.append(0)
    
    # Calculate final metrics
    ActualLabels = TestData['Label'].iloc[200:].values
    Predictions = np.array(Predictions[:len(ActualLabels)])
    
    TruePositives = np.sum((Predictions == 1) & (ActualLabels == 1))
    FalsePositives = np.sum((Predictions == 1) & (ActualLabels == 0))
    FalseNegatives = np.sum((Predictions == 0) & (ActualLabels == 1))
    TrueNegatives = np.sum((Predictions == 0) & (ActualLabels == 0))
    
    Precision = TruePositives / (TruePositives + FalsePositives) if (TruePositives + FalsePositives) > 0 else 0
    Recall = TruePositives / (TruePositives + FalseNegatives) if (TruePositives + FalseNegatives) > 0 else 0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
    
    print(f"\nTest Set Performance:")
    print(f"Precision: {Precision:.3f}")
    print(f"Recall: {Recall:.3f}")
    print(f"F1 Score: {F1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"True Positives:  {TruePositives}")
    print(f"False Positives: {FalsePositives}")
    print(f"False Negatives: {FalseNegatives}")
    print(f"True Negatives:  {TrueNegatives}")
    
    # Save best individual
    print(f"\n\nBest Strategy Found:")
    print(f"{str(BestIndividual)}")