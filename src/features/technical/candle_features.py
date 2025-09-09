import pandas as pd

class CandleFeatureBuilder:
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds candlestick-related features to the dataframe:
        - open_close_diff: Difference between closing and opening price
        - high_low_range: Daily price range
        - candle_body: Absolute size of candle body
        - upper_shadow: Wick above the body
        - lower_shadow: Wick below the body
        """
        df["open_close_diff"] = df["Close"] - df["Open"]
        df["high_low_range"] = df["High"] - df["Low"]
        df["candle_body"] = (df["Close"] - df["Open"]).abs()
        df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        return df
