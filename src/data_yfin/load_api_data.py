# src/data/load_api_data.py
import pandas as pd
import yfinance as yf
from typing import Optional
from src.utils.logger import get_logger
from src.utils.cache_manager import CacheManager

class APILoader:
    """
    Class to fetch historical equity data from Yahoo Finance and convert it into a pandas DataFrame.
    Integrates caching to avoid redundant API calls and speed up repeated requests.

    Attributes:
        ticker (str): Equity ticker symbol (e.g., 'AAPL', 'INFY.NS').
        period (str): Historical data period (default '1y').
        interval (str): Data interval (default '1d').
        logger (logging.Logger): Logger instance for this class.
        cache_manager (CacheManager): Global cache manager instance.
        data (pd.DataFrame or None): DataFrame holding fetched historical data.
    """

    def __init__(self, ticker: str, period: str = "1y", interval: str = "1d"):
        """
        Initialize APILoader with ticker and Yahoo Finance parameters.

        Args:
            ticker (str): Equity ticker symbol.
            period (str, optional): Historical period (default '1y').
            interval (str, optional): Data interval (default '1d').
        """
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Ticker must be a non-empty string.")

        self.ticker = ticker.upper().strip()
        self.period = period
        self.interval = interval
        self.logger = get_logger(self.__class__.__name__)
        self.cache_manager = CacheManager.get_instance()
        self.data: Optional[pd.DataFrame] = None

        self.logger.info(f"APILoader(load_api_data) initialized for {self.ticker} | period={self.period} | interval={self.interval}")

    def fetch(self) -> pd.DataFrame:
        """
        Fetch historical equity data from Yahoo Finance or retrieve from cache if available.

        Returns:
            pd.DataFrame: Historical OHLCV data for the equity.
        """
        cache_key = f"yahoo:{self.ticker}:{self.period}:{self.interval}"

        # 1) Attempt to retrieve from cache
        cached_data = self.cache_manager.load(cache_key, module="data_processor")
        if cached_data is not None:
            self.logger.info(f"Cache hit for {self.ticker} | returning cached data")
            self.data = cached_data
            return self.data

        # 2) Fetch from Yahoo Finance
        try:
            self.logger.info(f"Fetching Yahoo Finance data for {self.ticker}")
            ticker_obj = yf.Ticker(self.ticker)
            hist = ticker_obj.history(period=self.period, interval=self.interval)

            if hist.empty:
                self.logger.warning(f"No historical data found for {self.ticker}")
                self.data = pd.DataFrame()
            else:
                self.data = hist
                # 3) Store in cache for future retrieval
                self.cache_manager.save(cache_key, self.data, module="data_processor")
                self.logger.info(f"Data fetched and cached for {self.ticker} | shape={self.data.shape}")

        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance data for {self.ticker}: {e}")
            self.data = pd.DataFrame()

        return self.data
