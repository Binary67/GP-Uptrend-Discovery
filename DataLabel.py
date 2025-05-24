import pandas as pd
import numpy as np
np.NaN = np.nan
import pandas_ta as ta


class DataLabel:
    def __init__(self, DataFrame):
        """
        Initialize DataLabel with a DataFrame containing OHLCV data.
        
        Args:
            DataFrame (pd.DataFrame): DataFrame with OHLCV data
        """
        self.DataFrame = DataFrame
    
    def LabelUptrend(self):
        """
        Label the data with uptrend signals based on EMA 12 > EMA 50.
        
        Returns:
            pd.DataFrame: DataFrame with added 'Uptrend' column
        """
        # Calculate EMA 12 and EMA 50
        self.DataFrame['EMA12'] = ta.ema(self.DataFrame['Close'], length=12)
        self.DataFrame['EMA50'] = ta.ema(self.DataFrame['Close'], length=50)
        
        # Label uptrend where EMA 12 > EMA 50
        self.DataFrame['Uptrend'] = (self.DataFrame['EMA12'] > self.DataFrame['EMA50']).astype(int)
        
        return self.DataFrame