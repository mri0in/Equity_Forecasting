import pandas as pd
from typing import List, Optional
from utils.logger import get_logger


class DataPreprocessor:
    """
    Class responsible for preprocessing equity data.
    Applies standard data cleaning techniques such as:
    - Removing rows with missing values
    - Selecting only relevant columns
    """

    def __init__(self, required_columns: Optional[List[str]] = None):
        """
        Initializes the preprocessor.

        Args:
            required_columns (Optional[List[str]]): List of column names to keep from the dataset.
            If None, all columns will be retained.
        """
        self.required_columns = required_columns
        self.logger = get_logger(self.__class__.__name__)  # Logger specific to this class

    def drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows that contain any missing (NaN) values.

        Args:
            df (pd.DataFrame): Input DataFrame with raw data.

        Returns:
            pd.DataFrame: DataFrame with rows containing NaNs removed.
        """
        initial_shape = df.shape  # Save original shape for logging
        df_clean = df.dropna()  # Drop rows with any NaN values
        self.logger.info(f"Dropped missing values: {initial_shape} -> {df_clean.shape}")
        return df_clean

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retains only the columns specified during class initialization.

        Args:
            df (pd.DataFrame): Cleaned DataFrame (no missing values).

        Returns:
            pd.DataFrame: DataFrame with only selected columns retained.
        """
        if self.required_columns:
            # Identify any columns that are missing from the DataFrame
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing expected columns: {missing_cols}")

            # Retain only available required columns
            df = df[[col for col in self.required_columns if col in df.columns]]
            self.logger.info(f"Selected columns: {df.columns.tolist()}")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full preprocessing pipeline.

        Steps:
        1. Remove rows with missing values
        2. Retain only specified columns

        Args:
            df (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: Cleaned and formatted DataFrame.
        """
        self.logger.info("Starting data preprocessing pipeline.")
        df = self.drop_missing(df)       # Step 1: Drop rows with NaNs
        df = self.select_columns(df)     # Step 2: Select relevant columns
        self.logger.info("Completed data preprocessing.")
        return df
