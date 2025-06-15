import pandas as pd

class TrendFeatureBuilder:
    def __init__(self, sma_window: int = 10, ema_span: int = 10):
        """
        Initializes trend feature parameters.
        
        Args:
            sma_window (int): Window size for Simple Moving Average (default 10)
            ema_span (int): Span for Exponential Moving Average (default 10)
        """
        self.sma_window = sma_window
        self.ema_span = ema_span

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds trend-based technical indicators:
        - sma: Simple Moving Average over sma_window periods
        - ema: Exponential Moving Average over ema_span periods
        
        Args:
            df (pd.DataFrame): Stock price dataframe
        
        Returns:
            pd.DataFrame: DataFrame with added trend features
        """
        df[f"sma_{self.sma_window}"] = df["Close"].rolling(window=self.sma_window).mean()
        df[f"ema_{self.ema_span}"] = df["Close"].ewm(span=self.ema_span, adjust=False).mean()
        return df
