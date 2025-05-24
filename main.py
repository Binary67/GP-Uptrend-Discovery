from DataCleaning import DataCleaner
from DataDownloader import YFinanceDownloader
from DataLabel import DataLabel
from GPFramework import DesignIndividualRepresentation, GeneratePopulation
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
    
    print("Labeled Data Shape:", LabeledData.shape)
    print("\nFirst 5 rows of labeled data:")
    print(LabeledData.head())
    
    PSet = DesignIndividualRepresentation()

    
    PopulationSize = 100
    Population, Toolbox = GeneratePopulation(PSet, PopulationSize=PopulationSize)
    print(f"\nGenerated population of {len(Population)} individuals")

    for EachPopulation in Population[:10]:
        print(EachPopulation)