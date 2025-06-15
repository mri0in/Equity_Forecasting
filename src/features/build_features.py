import pandas as pd
from features.candle_features import CandleFeatureBuilder
from features.trend_features import TrendFeatureBuilder
from features.momentum_features import MomentumFeatureBuilder
from features.volume_features import VolumeFeatureBuilder
from utils.logger import get_logger

# Initialize module-level logger
logger = get_logger(__name__)

class FeatureBuilder:
    """
    Orchestrates the application of multiple technical feature engineering steps.
    """

    def __init__(self, sma_window: int = 10, ema_span: int = 10, roc_periods: int = 5, volume_rolling_window: int = 5):
        """
        Initializes all feature builders such as Candles, Momentum, Trends, Volume into one and with their corresponding configuration parameters.
        
        Args:
            sma_window (int): Window size for Simple Moving Average
            ema_span (int): Span size for Exponential Moving Average
            roc_periods (int): Number of periods for Rate of Change
            volume_rolling_window (int): Rolling window for volume features
        """
        
        self.candle = CandleFeatureBuilder()
        self.trend = TrendFeatureBuilder(sma_window, ema_span)
        self.momentum = MomentumFeatureBuilder(roc_periods)
        self.volume = VolumeFeatureBuilder(volume_rolling_window)

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all feature engineering steps sequentially.

        Args:
            df (pd.DataFrame): Input stock price data

        Returns:
            pd.DataFrame: Transformed DataFrame with all features added
        """
        try:
            logger.info("Starting candle features...")
            df = self.candle.add_features(df)
            logger.info("Candle features added.")

            logger.info("Starting trend features...")
            df = self.trend.add_features(df)
            logger.info("Trend features added.")

            logger.info("Starting momentum features...")
            df = self.momentum.add_features(df)
            logger.info("Momentum features added.")

            logger.info("Starting volume features...")
            df = self.volume.add_features(df)
            logger.info("Volume features added.")

            logger.info("All features built successfully.")

        except Exception as e:
            logger.error(f"Feature building failed: {e}")
            raise

        return df
