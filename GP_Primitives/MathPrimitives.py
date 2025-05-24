import math
import operator


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


def GetMathPrimitives():
    """Return list of math primitives for registration"""
    return [
        # Basic arithmetic
        ("add", operator.add, 2),
        ("sub", operator.sub, 2),
        ("mul", operator.mul, 2),
        ("ProtectedDiv", ProtectedDiv, 2),
        ("neg", operator.neg, 1),
        
        # Trigonometric
        ("sin", math.sin, 1),
        ("cos", math.cos, 1),
        
        # Other math functions
        ("abs", operator.abs, 1),
        ("Max", Max, 2),
        ("Min", Min, 2),
        ("Sqrt", Sqrt, 1),
        ("Log", Log, 1),
    ]