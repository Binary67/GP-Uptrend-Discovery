import pandas as pd
from deap import gp


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