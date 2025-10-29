"""
preprocess_data.py
==================
Orchestrates data preprocessing and feature building steps with caching.

Pipeline Overview:
------------------
1. Load raw stock data from API cache
2. Clean raw stock data (remove NaNs, retain essential columns)
3. Generate technical indicators using FeatureBuilder
4. Cache and return final feature-enriched dataset for downstream modeling
"""

import pandas as pd
from typing import List, Optional
from datetime import timedelta
from src.features.technical.build_features import FeatureBuilder
from src.utils.cache_manager import CacheManager
from src.utils.logger import get_logger


class DataPreprocessor:
    """
    Handles preprocessing and feature preparation for equity data.
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        equity: Optional[str] = None
    ):
        """
        Initialize the DataPreprocessor.

        Args:
            required_columns (Optional[List[str]]): List of columns to retain from raw data.
            equity (Optional[str]): Stock symbol (for logging and cache keys).
        """
        self.required_columns = required_columns
        self.equity = equity
        self.cache_manager = CacheManager.get_instance()
        self.logger = get_logger(self.__class__.__name__)
        self.feature_builder = FeatureBuilder(equity=equity)  # df passed in run_pipeline

    # -----------------------------------------------------------------
    # Data Cleaning
    # -----------------------------------------------------------------
    def drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows containing any missing values."""
        initial_shape = df.shape
        df_clean = df.dropna()
        self.logger.info(f"[{self.equity}] Dropped missing values: {initial_shape} → {df_clean.shape}")
        return df_clean

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retain only required columns."""
        if self.required_columns:
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"[{self.equity}] Missing expected columns: {missing_cols}")

            available_cols = [col for col in self.required_columns if col in df.columns]
            df = df[available_cols]
            self.logger.info(f"[{self.equity}] Selected columns: {df.columns.tolist()}")
        return df

    # -----------------------------------------------------------------
    # Full Preprocessing + Feature Engineering Pipeline
    # -----------------------------------------------------------------
    def run_pipeline(self, df: pd.DataFrame, refresh_cache: bool = False) -> pd.DataFrame:
        """
        Executes the full pipeline:
        1. Load raw data from data_processor cache
        2. Clean raw data
        3. Build all technical features
        4. Cache final features in 'features' module

        Args:
            df (pd.DataFrame): Raw equity data as fallback if cache is empty.
            refresh_cache (bool): If True, ignore cache and rebuild features.

        Returns:
            pd.DataFrame: Feature-enriched DataFrame ready for model training
        """
        try:
            self.logger.info(f"Starting preprocessing pipeline for {self.equity}...")

            # Load raw data from data_processor cache
            raw_cache_key = f"data_processor_{self.equity}"
            raw_df = None
            if not refresh_cache:
                try:
                    raw_df = self.cache_manager.load(raw_cache_key, module="data_processor")
                    self.logger.info(f"Loaded raw data from cache for {self.equity}")
                except FileNotFoundError:
                    self.logger.info(f"No cached raw data for {self.equity}, using passed df")

            df_to_process = raw_df if raw_df is not None else df

            # Step 1: Clean raw data
            df_clean = self.drop_missing(df_to_process)
            df_clean = self.select_columns(df_clean)

            # Step 2: Build technical features and then it caches internally
            df_features = self.feature_builder.build_all(df_clean)

            self.logger.info(f"Preprocessing + feature generation completed for {self.equity}")
            return df_features

        except Exception as e:
            self.logger.error(f"Pipeline failed for {self.equity}: {e}", exc_info=True)
            raise
