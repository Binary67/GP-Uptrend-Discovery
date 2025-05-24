import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame.copy()
        
    def RemoveDuplicates(self):
        """Remove duplicate rows based on index (datetime)"""
        self.DataFrame = self.DataFrame[~self.DataFrame.index.duplicated(keep='first')]
        return self
        
    def HandleMissingValues(self, Method='forward'):
        """Handle missing values using forward fill, backward fill, or interpolation"""
        if Method == 'forward':
            self.DataFrame = self.DataFrame.fillna(method='ffill')
        elif Method == 'backward':
            self.DataFrame = self.DataFrame.fillna(method='bfill')
        elif Method == 'interpolate':
            self.DataFrame = self.DataFrame.interpolate(method='time')
        elif Method == 'drop':
            self.DataFrame = self.DataFrame.dropna()
        return self
        
    def RemoveOutliers(self, Columns=None, Method='iqr', Threshold=3):
        """Remove outliers using IQR or Z-score method"""
        if Columns is None:
            Columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
        for Column in Columns:
            if Column not in self.DataFrame.columns:
                continue
                
            if Method == 'iqr':
                Q1 = self.DataFrame[Column].quantile(0.25)
                Q3 = self.DataFrame[Column].quantile(0.75)
                IQR = Q3 - Q1
                LowerBound = Q1 - Threshold * IQR
                UpperBound = Q3 + Threshold * IQR
                self.DataFrame = self.DataFrame[
                    (self.DataFrame[Column] >= LowerBound) & 
                    (self.DataFrame[Column] <= UpperBound)
                ]
            elif Method == 'zscore':
                Mean = self.DataFrame[Column].mean()
                Std = self.DataFrame[Column].std()
                self.DataFrame = self.DataFrame[
                    (np.abs(self.DataFrame[Column] - Mean) <= Threshold * Std)
                ]
        return self
        
    def ValidateOHLC(self):
        """Validate OHLC data integrity"""
        # Ensure High >= Low
        InvalidRows = self.DataFrame['High'] < self.DataFrame['Low']
        if InvalidRows.any():
            self.DataFrame = self.DataFrame[~InvalidRows]
            
        # Ensure High >= Open and High >= Close
        InvalidRows = (self.DataFrame['High'] < self.DataFrame['Open']) | \
                     (self.DataFrame['High'] < self.DataFrame['Close'])
        if InvalidRows.any():
            self.DataFrame = self.DataFrame[~InvalidRows]
            
        # Ensure Low <= Open and Low <= Close
        InvalidRows = (self.DataFrame['Low'] > self.DataFrame['Open']) | \
                     (self.DataFrame['Low'] > self.DataFrame['Close'])
        if InvalidRows.any():
            self.DataFrame = self.DataFrame[~InvalidRows]
            
        return self
        
    def RemoveZeroVolume(self):
        """Remove rows with zero or negative volume"""
        self.DataFrame = self.DataFrame[self.DataFrame['Volume'] > 0]
        return self
        
    def ResampleData(self, Frequency):
        """Resample data to a different frequency"""
        Aggregation = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        self.DataFrame = self.DataFrame.resample(Frequency).agg(Aggregation)
        self.DataFrame = self.DataFrame.dropna()
        return self
        
    def GetCleanedData(self):
        """Return the cleaned DataFrame"""
        return self.DataFrame