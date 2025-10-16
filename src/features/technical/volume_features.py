# src/features/technical/volume_features.py
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VolumeFeatures:
    """
    Computes volume-based indicators: MA volume, OBV, volume spikes.
    """

    def __init__(self, df: pd.DataFrame, equity: str):
        self.df = df.copy()
        self.equity = equity
        

    def ma_volume(self, period: int = 14) -> pd.Series:
        return self.df["Volume"].rolling(period).mean()

    def obv(self) -> pd.Series:
        obv = pd.Series(index=self.df.index, dtype=float)
        obv.iloc[0] = 0
        for i in range(1, len(self.df)):
            if self.df["Close"].iloc[i] > self.df["Close"].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.df["Volume"].iloc[i]
            elif self.df["Close"].iloc[i] < self.df["Close"].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.df["Volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    def compute_all(self) -> pd.DataFrame:
        
        df_vol = pd.DataFrame(index=self.df.index)
        df_vol["MA_Vol_14"] = self.ma_volume(14)
        df_vol["OBV"] = self.obv()
        
        return df_vol
