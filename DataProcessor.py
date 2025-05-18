import pandas as pd
from pandas import DataFrame
from pandas import read_csv, read_parquet

from DataDownloader import YFinanceDownloader

class DataProcessor:
    def __init__(self, FilePath=None, Ticker=None, StartDate=None, EndDate=None, Interval='1d'):
        self.FilePath = FilePath
        self.Ticker = Ticker
        self.StartDate = StartDate
        self.EndDate = EndDate
        self.Interval = Interval

    def LoadDataFromFile(self, Path):
        if Path.endswith('.csv'):
            Data = read_csv(Path)
        else:
            Data = read_parquet(Path)
        if 'Datetime' in Data.columns:
            Data['Datetime'] = pd.to_datetime(Data['Datetime'])
            Data = Data.set_index('Datetime')
        elif 'Date' in Data.columns:
            Data['Date'] = pd.to_datetime(Data['Date'])
            Data = Data.set_index('Date')
        Data = Data.sort_index()
        return Data

    def ForwardFillMissingCandles(self, Data):
        if not Data.index.is_monotonic_increasing:
            Data = Data.sort_index()
        Frequency = pd.infer_freq(Data.index)
        if Frequency is None:
            Frequency = self.Interval
        CompleteIndex = pd.date_range(start=Data.index[0], end=Data.index[-1], freq=Frequency)
        Data = Data.reindex(CompleteIndex)
        Data = Data.ffill()
        return Data

    def AddEmaColumns(self, Data):
        Data['EMA12'] = Data['Close'].ewm(span=12, adjust=False).mean()
        Data['EMA50'] = Data['Close'].ewm(span=50, adjust=False).mean()
        return Data

    def CreateLabels(self, Data: DataFrame) -> DataFrame:
        """Create the IsUpcomingUptrend column using a look-ahead."""
        Data['IsUpcomingUptrend'] = Data['EMA12'].shift(-1) > Data['EMA50'].shift(-1)
        Data = Data.iloc[:-1].copy()
        Data['IsUpcomingUptrend'] = Data['IsUpcomingUptrend'].astype(bool)
        return Data

    def GetProcessedData(self):
        if self.FilePath:
            Data = self.LoadDataFromFile(self.FilePath)
        else:
            Downloader = YFinanceDownloader(self.Ticker, self.StartDate, self.EndDate, self.Interval)
            Data = Downloader.DownloadData()
        Data = self.ForwardFillMissingCandles(Data)
        Data = self.AddEmaColumns(Data)
        Data = self.CreateLabels(Data)
        Data.columns.name = None
        return Data
