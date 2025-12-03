# src/pipeline/B_feature_pipeline.py
"""
Feature Building Pipeline
-------------------------
Loads preprocessed equity data from cache, generates technical indicators
using FeatureBuilder, and caches final feature-enriched dataset for modeling.
"""

import pandas as pd
from typing import Optional
from src.features.technical.build_features import FeatureBuilder
from src.utils.cache_manager import CacheManager
from src.utils.logger import get_logger


class FeaturePipeline:
    """
    Handles feature engineering from preprocessed equity data
    """

    def __init__(self, equity: Optional[str] = None):
        self.equity = equity
        self.cache_manager = CacheManager.get_instance()
        self.logger = get_logger(self.__class__.__name__)
        self.feature_builder = FeatureBuilder(equity=equity)

    # -----------------------------------------------------------------
    # Pipeline Execution
    # -----------------------------------------------------------------
    def run(self, refresh_cache: bool = False) -> pd.DataFrame:
        """
        Executes feature building:
        - Load preprocessed data from cache
        - Build all technical indicators
        - Cache final feature dataset
        """
        try:
            self.logger.info(f"Starting feature building for {self.equity}...")

            preprocessed_cache_key = f"preprocessed_{self.equity}"
            df_preprocessed = self.cache_manager.load(preprocessed_cache_key, module="features")
            self.logger.info(f"Loaded preprocessed data from cache for {self.equity}")

            # Step 1: Build features
            df_features = self.feature_builder.build_all(df_preprocessed)

            # Step 2: Cache final features
            feature_cache_key = f"features_{self.equity}"
            self.cache_manager.save(df_features, feature_cache_key, module="features")
            self.logger.info(f"Feature generation completed and cached for {self.equity}")

            return df_features

        except FileNotFoundError:
            self.logger.error(f"Preprocessed data not found for {self.equity}. Run preprocessing first.")
            raise
        except Exception as e:
            self.logger.error(f"Feature pipeline failed for {self.equity}: {e}", exc_info=True)
            raise
