import operator
import pandas as pd
from deap import gp


class PrimitiveSetFactory:
    def __init__(self):
        self.Indicators = {}
        self.DefaultArithmeticOperators = [
            (operator.add, 2),
            (operator.sub, 2),
            (operator.mul, 2),
            (self.SafeDivide, 2),
        ]
        self.DefaultComparisonOperators = [
            (operator.lt, 2),
            (operator.le, 2),
            (operator.gt, 2),
            (operator.ge, 2),
            (operator.eq, 2),
            (operator.ne, 2),
        ]
        self.AddIndicator("RSI", self.RSI, 1)
        self.AddIndicator("ATR", self.ATR, 3)
        self.AddIndicator("BollingerUpper", self.BollingerUpper, 1)
        self.AddIndicator("BollingerLower", self.BollingerLower, 1)

    @staticmethod
    def SafeDivide(X, Y):
        return X / Y if Y != 0 else 0

    @staticmethod
    def RSI(Close, Period=14):
        Delta = Close.diff()
        Gain = Delta.clip(lower=0)
        Loss = -Delta.clip(upper=0)
        AvgGain = Gain.rolling(window=Period).mean()
        AvgLoss = Loss.rolling(window=Period).mean()
        Rs = AvgGain / AvgLoss
        ReturnValue = 100 - (100 / (1 + Rs))
        return ReturnValue

    @staticmethod
    def ATR(High, Low, Close, Period=14):
        Hl = High - Low
        Hc = (High - Close.shift()).abs()
        Lc = (Low - Close.shift()).abs()
        Tr = pd.concat([Hl, Hc, Lc], axis=1).max(axis=1)
        ReturnValue = Tr.rolling(window=Period).mean()
        return ReturnValue

    @staticmethod
    def BollingerUpper(Close, Period=20, StdDev=2):
        Sma = Close.rolling(window=Period).mean()
        Std = Close.rolling(window=Period).std()
        ReturnValue = Sma + StdDev * Std
        return ReturnValue

    @staticmethod
    def BollingerLower(Close, Period=20, StdDev=2):
        Sma = Close.rolling(window=Period).mean()
        Std = Close.rolling(window=Period).std()
        ReturnValue = Sma - StdDev * Std
        return ReturnValue

    def AddIndicator(self, Name, Function, Arity):
        self.Indicators[Name] = (Function, Arity)

    def CreatePrimitiveSet(self, ColumnNames, ExtraIndicators=None):
        Pset = gp.PrimitiveSet("MAIN", len(ColumnNames))
        for Index, Name in enumerate(ColumnNames):
            Pset.renameArguments(**{f"ARG{Index}": Name})
        for Function, Arity in self.DefaultArithmeticOperators:
            Pset.addPrimitive(Function, Arity)
        for Function, Arity in self.DefaultComparisonOperators:
            Pset.addPrimitive(Function, Arity)
        IndicatorDict = dict(self.Indicators)
        if ExtraIndicators:
            IndicatorDict.update(ExtraIndicators)
        for Name, (Function, Arity) in IndicatorDict.items():
            Pset.addPrimitive(Function, Arity, name=Name)
        return Pset
