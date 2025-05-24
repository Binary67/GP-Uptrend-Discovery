from DataCleaning import DataCleaner
from DataDownloader import YFinanceDownloader
from DataLabel import DataLabel
import pandas as pd
import numpy as np

if __name__ == "__main__":
    DataDownloaderObj = YFinanceDownloader('AAPL', '2020-01-01', '2024-12-31', '1d')
    TradingData = DataDownloaderObj.DownloadData()

    DataCleanerObj = DataCleaner(TradingData)
    
    CleanedData = (DataCleanerObj
                   .ValidateOHLC()
                   .GetCleanedData())
    
    DataLabelObj = DataLabel(CleanedData)
    LabeledData = DataLabelObj.LabelUptrend()
    
    print(LabeledData)