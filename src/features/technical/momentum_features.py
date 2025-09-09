import pandas as pd

class MomentumFeatureBuilder:
    def __init__(self, roc_periods: int = 5):
        """
        Initializes momentum feature parameters.
        
        Args:
            roc_periods (int): Number of periods for Rate of Change (default 5)
        """
        self.roc_periods = roc_periods

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds momentum-based indicators:
        - roc: Rate of Change over roc_periods
        
        Args:
            df (pd.DataFrame): Stock price dataframe
        
        Returns:
            pd.DataFrame: DataFrame with added momentum features
        """
        df[f"roc_{self.roc_periods}"] = df["Close"].pct_change(periods=self.roc_periods)
        return df
