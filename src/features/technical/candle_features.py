# src/features/technical/candle_features.py
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CandleFeatures:
    """
    Computes candlestick patterns: Doji, Hammer, Engulfing, etc.
    """

    def __init__(self, df: pd.DataFrame, equity: str):
        self.df = df.copy()
        self.equity = equity
        

    def is_doji(self, threshold: float = 0.1) -> pd.Series:
        body = abs(self.df["Close"] - self.df["Open"])
        return (body / self.df["Close"]) < threshold

    def is_hammer(self) -> pd.Series:
        body = abs(self.df["Close"] - self.df["Open"])
        lower_shadow = self.df["Open"].where(self.df["Close"] > self.df["Open"], self.df["Close"]) - self.df["Low"]
        upper_shadow = self.df["High"] - self.df["Close"].where(self.df["Close"] > self.df["Open"], self.df["Open"])
        return (lower_shadow >= 2 * body) & (upper_shadow <= body)

    def compute_all(self) -> pd.DataFrame:
        
        df_candle = pd.DataFrame(index=self.df.index)
        df_candle["Doji"] = self.is_doji()
        df_candle["Hammer"] = self.is_hammer()

        return df_candle
