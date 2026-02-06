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
        - Cache is valid if it already contains the latest available market data
        - Always returns a DataFrame or raises
        """
        if not equity or not isinstance(equity, str):
            raise ValueError("Equity symbol must be a non-empty string")

        equity = equity.upper().strip()
        csv_path = self.raw_data_dir / f"{equity}.csv"

        logger.info("Fetching equity history for %s", equity)

        # ---------------------------
        # Use cache if it already has the latest data
        # ---------------------------
        if csv_path.exists():
            if not csv_path.is_file():
                raise ValueError(f"Expected CSV file but got directory: {csv_path}")

            try:
                # Simple read: just load the CSV with Date as index
                local_df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            except Exception as e:
                logger.warning("Failed to read local CSV %s: %s", csv_path, e)
                local_df = None

            if local_df is not None and not local_df.empty:
                # determine local last date
                try:
                    if hasattr(local_df.index, "max"):
                        local_last = pd.to_datetime(local_df.index.max())
                        local_last_date = local_last.date()
                    elif "Date" in local_df.columns:
                        local_last_date = pd.to_datetime(local_df["Date"]).max().date()
                    else:
                        local_last_date = None
                except Exception:
                    local_last_date = None

                # try to get the latest available date from Yahoo (1-day query)
                online_last_date = None
                try:
                    recent = yf.download(
                        equity,
                        period="1d",
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                    )
                    if not recent.empty:
                        online_last_date = pd.to_datetime(recent.index.max()).date()
                except Exception:
                    logger.debug("Failed to fetch 1-day data to validate cache for %s", equity)

                # If we couldn't determine online date, prefer local cache
                if local_last_date is not None and (online_last_date is None or local_last_date >= online_last_date):
                    logger.info("Using cached equity data for %s (last local date: %s, online last date: %s)", equity, local_last_date, online_last_date)
                    return self._validate_dataframe(local_df, equity)

                logger.info("Local cache is stale or missing latest date for %s (local: %s, online: %s)", equity, local_last_date, online_last_date)

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

        # Normalize columns: flatten MultiIndex if present, keep only OHLCV columns
        if isinstance(df.columns, pd.MultiIndex):
            # If MultiIndex, flatten to single level (use first level)
            df.columns = df.columns.get_level_values(0)
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Keep only OHLCV columns (some may be "adj close" instead of "close")
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        adj_cols = ['open', 'high', 'low', 'adj close', 'volume']
        
        # Use whichever set matches the available columns
        available = df.columns.tolist()
        keep_cols = [c for c in ohlcv_cols if c in available]
        if not keep_cols:
            keep_cols = [c for c in adj_cols if c in available]
        
        if keep_cols:
            df = df[keep_cols]
        
        # Persist
        try:
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index_label='Date')
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