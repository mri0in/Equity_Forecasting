import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EquityHistory:
    """
    Manages local equity price history stored as CSV files.

    Responsibilities:
    - Load cached equity CSVs from datalake
    - Refresh data via yfinance if stale or missing
    - Always return a valid pandas DataFrame
    """

    def __init__(self, raw_data_dir: str = "datalake/data/raw") -> None:
        self.raw_data_dir = Path(raw_data_dir)

        if not self.raw_data_dir.exists():
            logger.warning(
                "Equity history directory does not exist: %s",
                self.raw_data_dir,
            )
        elif not self.raw_data_dir.is_dir():
            raise ValueError(
                f"Expected directory but got file: {self.raw_data_dir}"
            )

    def get_equity_data(self, equity: str) -> pd.DataFrame:
        """
        Fetch equity history from local cache or Yahoo Finance.

        Rules:
        - CSV name must be {EQUITY}.csv
        - Cache is valid for 24 hours
        - Always returns a DataFrame or raises
        """
        if not equity or not isinstance(equity, str):
            raise ValueError("Equity symbol must be a non-empty string")

        equity = equity.upper().strip()
        csv_path = self.raw_data_dir / f"{equity}.csv"

        logger.info("Fetching equity history for %s", equity)

        # ---------------------------
        # Load from cache if fresh
        # ---------------------------
        if csv_path.exists():
            if not csv_path.is_file():
                raise ValueError(f"Expected CSV file but got directory: {csv_path}")

            file_age = datetime.now() - datetime.fromtimestamp(csv_path.stat().st_mtime)
            if file_age < timedelta(days=1):
                logger.info("Using cached equity data for %s", equity)
                df = pd.read_csv(csv_path)
                return self._validate_dataframe(df, equity)

        # ---------------------------
        # Download fresh data
        # ---------------------------
        logger.info("Downloading fresh data for %s from Yahoo Finance", equity)

        try:
            df = yf.download(
                equity,
                period="1y",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception as e:
            logger.exception("Yahoo Finance download failed for %s", equity)
            raise RuntimeError(f"Failed to download data for {equity}") from e

        if df.empty:
            raise ValueError(f"No data returned from Yahoo Finance for {equity}")

        # Persist
        try:
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path)
        except Exception as e:
            logger.warning("Failed to write equity CSV: %s (%s)", csv_path, e)

        return self._validate_dataframe(df, equity)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, equity: str) -> pd.DataFrame:
        """
        Ensure dataframe is usable by downstream models.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Equity data must be a pandas DataFrame")

        if df.empty:
            raise ValueError(f"Equity dataframe is empty for {equity}")

        # If datetime index, keep it but ensure columns are numeric
        df = df.copy()

        # Coerce all columns to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            raise ValueError(
                f"No usable numeric columns after coercion for equity {equity}"
            )

        # Drop rows where all numeric values are NaN
        numeric_df = numeric_df.dropna(how="all")

        if numeric_df.empty:
            raise ValueError(
                f"Numeric columns exist but contain only NaNs for {equity}"
            )

        return df