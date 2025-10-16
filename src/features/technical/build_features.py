# src/features/technical/build_features.py
import pandas as pd
from src.features.technical.candle_features import CandleFeatures
from src.features.technical.trend_features import TrendFeatures
from src.features.technical.momentum_features import MomentumFeatures
from src.features.technical.volume_features import VolumeFeatures
from src.utils.cache_manager import CacheManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureBuilder:
    """
    Orchestrates the application of multiple technical feature engineering steps
    (Candlestick, Trend, Momentum, Volume) and caches results for efficiency.
    """

    def __init__(self, equity: str):
        """
        Initializes FeatureBuilder with OHLCV DataFrame and equity ticker.

        Args:
            equity (str): Equity ticker (used for cache keys)
        """
        self.equity = equity
        self.cache_manager = CacheManager.get_instance()
    
    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sequentially applies all technical feature engineering steps.
        Uses caching to avoid recomputation.

        Returns:
            pd.DataFrame: DataFrame with all features appended
        """
        cache_key = f"features_{self.ticker}"
        refresh_cache = False  # Set to True to force recomputation
        ttl_seconds = 30 * 24 * 60 * 60  # 1 month

        
        # Check cache first
        if not refresh_cache and self.cache_manager.exists(cache_key, module="features"):
            cached_result = self.cache_manager.load(cache_key, module="features")
            if cached_result is not None:
                return cached_result

        # Initialize feature submodules with current df
        candle = CandleFeatures(df, self.equity)
        trend = TrendFeatures(df, self.equity)
        momentum = MomentumFeatures(df, self.equity)
        volume = VolumeFeatures(df, self.equity)

        try:
            logger.info(f"Building candle features for {self.equity}...")
            df_candle = candle.compute_all()
            logger.info("Candle features built.")

            logger.info(f"Building trend features for {self.equity}...")
            df_trend = trend.compute_all()
            logger.info("Trend features built.")

            logger.info(f"Building momentum features for {self.equity}...")
            df_momentum = momentum.compute_all()
            logger.info("Momentum features built.")

            logger.info(f"Building volume features for {self.equity}...")
            df_volume = volume.compute_all()
            logger.info("Volume features built.")

            # Merge all features
            df_all = pd.concat([df_candle, df_trend, df_momentum, df_volume], axis=1)

            # Save final merged features to cache with expiry time(ttl) of 1 month
            self.cache_manager.save(cache_key, df_all, module="features", ttl=ttl_seconds)
            logger.info(f"All features built and cached successfully for {self.equity}.")

        except Exception as e:
            logger.error(f"Feature building failed for {self.equity}: {e}")
            raise

        return df_all
