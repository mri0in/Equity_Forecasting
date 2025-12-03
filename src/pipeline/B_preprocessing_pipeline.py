# src/pipeline/B_preprocessing_pipeline.py
"""
Preprocessing Pipeline
----------------------
Loads raw equity data from cache, cleans it (drop missing rows, select columns),
and caches cleaned data for downstream feature building.
"""

import pandas as pd
from typing import List, Optional
from src.utils.cache_manager import CacheManager
from src.utils.logger import get_logger


class PreprocessingPipeline:
    """
    Handles preprocessing of raw equity data (cleaning & column selection)
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        equity: Optional[str] = None
    ):
        self.required_columns = required_columns
        self.equity = equity
        self.cache_manager = CacheManager.get_instance()
        self.logger = get_logger(self.__class__.__name__)

    # -----------------------------------------------------------------
    # Data Cleaning Utilities
    # -----------------------------------------------------------------
    def drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_shape = df.shape
        df_clean = df.dropna()
        self.logger.info(f"[{self.equity}] Dropped missing values: {initial_shape} → {df_clean.shape}")
        return df_clean

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.required_columns:
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"[{self.equity}] Missing expected columns: {missing_cols}")
            available_cols = [col for col in self.required_columns if col in df.columns]
            df = df[available_cols]
            self.logger.info(f"[{self.equity}] Selected columns: {df.columns.tolist()}")
        return df

    # -----------------------------------------------------------------
    # Pipeline Execution
    # -----------------------------------------------------------------
    def run(self, df: pd.DataFrame, refresh_cache: bool = False) -> pd.DataFrame:
        """
        Executes preprocessing steps:
        - Load raw data from cache
        - Clean data
        - Cache cleaned data for feature building
        """
        try:
            self.logger.info(f"Starting preprocessing for {self.equity}...")

            # Load raw data from cache
            raw_cache_key = f"data_processor_{self.equity}"
            raw_df = None
            if not refresh_cache:
                try:
                    raw_df = self.cache_manager.load(raw_cache_key, module="data_processor")
                    self.logger.info(f"Loaded raw data from cache for {self.equity}")
                except FileNotFoundError:
                    self.logger.info(f"No cached raw data for {self.equity}, using passed df")

            df_to_process = raw_df if raw_df is not None else df

            # Step 1: Clean
            df_clean = self.drop_missing(df_to_process)
            df_clean = self.select_columns(df_clean)

            # Step 2: Cache cleaned data
            clean_cache_key = f"preprocessed_{self.equity}"
            self.cache_manager.save(df_clean, clean_cache_key, module="features")
            self.logger.info(f"Preprocessing completed and cached for {self.equity}")

            return df_clean

        except Exception as e:
            self.logger.error(f"Preprocessing failed for {self.equity}: {e}", exc_info=True)
            raise
