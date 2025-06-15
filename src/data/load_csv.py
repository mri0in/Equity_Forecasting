import os
import pandas as pd
from typing import Optional
from utils.logger import get_logger


class CSVLoader:
    """
    A class to load CSV files into pandas DataFrames with robust error handling and logging.

    Attributes:
        file_path (str): Path to the CSV file.
        data (Optional[pd.DataFrame]): Loaded DataFrame, defaults to None.
        logger: Logger instance for logging events and errors.
    """

    def __init__(self, file_path: str):
        """
        Initializes the CSVLoader with a given file path.

        Args:
            file_path (str): Path to the CSV file.

        Raises:
            TypeError: If the input path is not a string.
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string.")
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        self.logger = get_logger(__name__)

    def load_csv(self) -> pd.DataFrame:
        """
        Loads the CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame. Empty DataFrame on failure.

        Raises:
            FileNotFoundError: If the file path does not exist.
            pd.errors.ParserError: If the CSV is malformed.
            Exception: For any other unforeseen errors.
        """
        if not os.path.exists(self.file_path):
            self.logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            self.data = pd.read_csv(self.file_path)
            self.logger.info(f"Loaded data from {self.file_path}, shape: {self.data.shape}")
        except pd.errors.ParserError as e:
            self.logger.error(f"Malformed CSV: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading CSV: {e}")
            raise

        return self.data
