"""
Ingestion Pipeline Module
-------------------------
Downloads raw OHLCV data for multiple tickers from Yahoo Finance
and saves them to datalake/data/raw/ as individual CSV files.

Includes:
- Retry logic
- Rate-limiting (sleep between requests)
- Logging
- OOP structure
"""

import os
import time
import logging
from typing import List, Optional

import yfinance as yf

RAW_DATA_DIR = "datalake/data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IngestionPipeline:
    """
    Pipeline class that handles downloading raw equity data using Yahoo Finance.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str = "2013-01-01",
        end_date: Optional[str] = None,
        sleep_time: float = 2.0,
        max_retries: int = 3,
    ) -> None:
        """
        :param tickers: List of ticker symbols (NSE/BSE allowed)
        :param start_date: Start date for download
        :param end_date: End date, optional
        :param sleep_time: Sleep between requests to avoid rate limits
        :param max_retries: Retry attempts per ticker
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.sleep_time = sleep_time
        self.max_retries = max_retries

    def _download_ticker(self, ticker: str) -> None:
        """
        Download a single ticker with retry logic.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[{ticker}] Download attempt {attempt}/{self.max_retries}...")
                df = yf.download(ticker, start=self.start_date, end=self.end_date)

                if df.empty:
                    raise ValueError(f"No data returned for ticker {ticker}")

                # Save to raw data directory
                save_path = os.path.join(RAW_DATA_DIR, f"{ticker.replace('.', '_')}.csv")
                df.to_csv(save_path)

                logger.info(f"[{ticker}] Saved raw data → {save_path}")
                return

            except Exception as e:
                logger.error(f"[{ticker}] Error on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    logger.error(f"[{ticker}] FAILED after {self.max_retries} attempts.")
                else:
                    time.sleep(2)

        # Sleep between tickers
        time.sleep(self.sleep_time)

    def run(self) -> None:
        """
        Run ingestion for all tickers.
        """
        logger.info(f"Starting ingestion for tickers: {self.tickers}")
        for ticker in self.tickers:
            self._download_ticker(ticker)
        logger.info("Ingestion completed for all tickers.")
