# src/features/technical/momentum_features.py
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MomentumFeatures:
    """
    Computes momentum indicators: RSI, ROC, Stochastic Oscillator.
    """

    def __init__(self, df: pd.DataFrame, equity: str):
        self.df = df.copy()
        self.equity = equity
        

    def rsi(self, period: int = 14) -> pd.Series:
        delta = self.df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def roc(self, period: int = 12) -> pd.Series:
        return ((self.df["Close"] - self.df["Close"].shift(period)) / self.df["Close"].shift(period)) * 100

    def compute_all(self) -> pd.DataFrame:
        
        df_mom = pd.DataFrame(index=self.df.index)
        df_mom["RSI_14"] = self.rsi(14)
        df_mom["ROC_12"] = self.roc(12)

        return df_mom
