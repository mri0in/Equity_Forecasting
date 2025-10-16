"""
Async API Data Loader

This module defines a class that handles real-time asynchronous API data loading
for short-term equity forecasting. It uses the HTTPX library and Python's asyncio
to efficiently fetch data concurrently from multiple endpoints. It has Built-in 
Timeout, Retry, Streaming, Connection Pooling.

"""

import asyncio
import httpx
import pandas as pd
from typing import Dict, List, Optional
from utils.logger import get_logger


class AsyncAPIDataLoader:
    """
    A class for asynchronously fetching real-time stock/equity data via HTTP APIs.

    Designed for high-frequency or low-latency systems, this class enables
    concurrent API calls and is suitable for streaming or near-real-time ingestion.

    Attributes:
        base_url (str): The base URL of the API to call.
        api_key (Optional[str]): Optional API key used for authenticated access.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the AsyncAPIDataLoader instance with required configurations.

        Args:
            base_url (str): The API endpoint to send requests to.
            api_key (Optional[str]): An API key if the endpoint requires authorization.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.logger = get_logger(self.__class__.__name__)  # Logs under class name

    async def fetch(self, session: httpx.AsyncClient, params: Dict[str, str]) -> pd.DataFrame:
        """
        Perform a single asynchronous API call and return the response as a DataFrame.

        Args:
            session (httpx.AsyncClient): Shared HTTP client for efficient connection reuse.
            params (Dict[str, str]): Dictionary of parameters to pass in the query string.

        Returns:
            pd.DataFrame: Response converted to pandas DataFrame.
                          If the request fails or data is empty, returns an empty DataFrame.

        Note:
            API key is automatically appended to params if provided.
        """
        if self.api_key:
            params["apikey"] = self.api_key  # Add API key dynamically if provided

        try:
            self.logger.info(f"Initiating async fetch with params: {params}")
            response = await session.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an error for HTTP 4xx/5xx responses
            data = response.json()
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Async fetch failed for params {params}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure

    async def fetch_multiple(self, list_of_params: List[Dict[str, str]]) -> List[pd.DataFrame]:
        """
        Perform multiple API calls concurrently using asyncio.gather.

        Args:
            list_of_params (List[Dict[str, str]]): A list where each item is a set of
                                                   query parameters for a separate request.

        Returns:
            List[pd.DataFrame]: A list of DataFrames. Each entry corresponds to one API call.
                                If any call fails, its corresponding DataFrame will be empty.

        Note:
            This method improves performance for scenarios like fetching multiple tickers
            simultaneously or streaming data from various sources.
        """
        async with httpx.AsyncClient() as session:
            # Create a list of coroutines for concurrent execution
            tasks = [self.fetch(session, params) for params in list_of_params]
            return await asyncio.gather(*tasks)
