import pandas as pd


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


def GetComparisonPrimitives():
    """Return list of comparison and logical primitives for registration"""
    return [
        # Comparison operators
        ("GreaterThan", GreaterThan, 2),
        ("LessThan", LessThan, 2),
        ("GreaterEqual", GreaterEqual, 2),
        ("LessEqual", LessEqual, 2),
        ("Equal", Equal, 2),
        
        # Logical operators
        ("And", And, 2),
        ("Or", Or, 2),
        ("Not", Not, 1),
        
        # Conditional
        ("IfThenElse", IfThenElse, 3),
    ]