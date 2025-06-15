import pandas as pd

class VolumeFeatureBuilder:
    def __init__(self, rolling_window: int = 5):
        """
        Initializes volume feature parameters.
        
        Args:
            rolling_window (int): Window size for volume rolling average (default 5)
        """
        self.rolling_window = rolling_window

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds volume-based features:
        - volume_rolling_mean: Moving average of volume over rolling_window periods
        
        Args:
            df (pd.DataFrame): Stock price dataframe
        
        Returns:
            pd.DataFrame: DataFrame with added volume features
        """
        df[f"volume_rolling_mean_{self.rolling_window}"] = df["Volume"].rolling(window=self.rolling_window).mean()
        return df
