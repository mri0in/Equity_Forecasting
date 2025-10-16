# src/features/technical/trend_features.py
import pandas as pd
from typing import Literal
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TrendFeatures:
    """
    Computes trend indicators: SMA, EMA, MACD, Bollinger Bands, etc.
    """

    def __init__(self, df: pd.DataFrame, equity: str):
        """
        Args:
            df (pd.DataFrame): OHLCV DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'.
            equity (str): Equity ticker.
        """
        self.df = df.copy()
        self.equity = equity
        

    def sma(self, period: int = 14) -> pd.Series:
        return self.df["Close"].rolling(period).mean()

    def ema(self, period: int = 14) -> pd.Series:
        return self.df["Close"].ewm(span=period, adjust=False).mean()

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        fast_ema = self.df["Close"].ewm(span=fast, adjust=False).mean()
        slow_ema = self.df["Close"].ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Hist": hist})

    def bollinger_bands(self, period: int = 20, std: float = 2) -> pd.DataFrame:
        sma = self.df["Close"].rolling(period).mean()
        rstd = self.df["Close"].rolling(period).std()
        upper = sma + std * rstd
        lower = sma - std * rstd
        return pd.DataFrame({"BB_upper": upper, "BB_lower": lower})

    def compute_all(self) -> pd.DataFrame:
        """
        Computes all trend features and returns DataFrame.
        Caches the result.
        """
        
        df_trend = pd.DataFrame(index=self.df.index)
        df_trend["SMA_14"] = self.sma(14)
        df_trend["EMA_14"] = self.ema(14)
        df_trend = pd.concat([df_trend, self.macd(), self.bollinger_bands()], axis=1)

        return df_trend
