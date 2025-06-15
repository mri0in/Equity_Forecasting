import requests
import pandas as pd
from utils.logger import get_logger

class APILoader:
    """
    Class to fetch data from an API endpoint and convert it into a pandas DataFrame.

    Attributes:
        api_url (str): URL of the API endpoint.
        headers (dict): Optional HTTP headers for the API request.
        params (dict): Optional query parameters for the API request.
        logger (logging.Logger): Logger instance for this class.
        data (pd.DataFrame or None): DataFrame holding the fetched data.
    """

    def __init__(self, api_url: str, headers: dict = None, params: dict = None):
        """
        Initialize the APILoader with API endpoint details.

        Args:
            api_url (str): The URL of the API to fetch data from.
            headers (dict, optional): HTTP headers to send with the request. Defaults to None.
            params (dict, optional): Query parameters for the request. Defaults to None.
        """
        self.api_url = api_url
        self.headers = headers if headers else {}
        self.params = params if params else {}
        self.logger = get_logger(self.__class__.__name__)
        self.data = None

    def fetch(self) -> pd.DataFrame:
        """
        Make a GET request to the API and convert the JSON response to a pandas DataFrame.

        Returns:
            pd.DataFrame: Data fetched from the API, or empty DataFrame on failure.
        """
        try:
            self.logger.info(f"Sending GET request to API: {self.api_url} with params: {self.params}")
            response = requests.get(self.api_url, headers=self.headers, params=self.params)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            json_data = response.json()
            # Convert JSON data to DataFrame — adjust this based on API response structure
            self.data = pd.json_normalize(json_data)
            self.logger.info(f"Successfully fetched data, shape: {self.data.shape}")
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err}")
            self.data = pd.DataFrame()
        except requests.exceptions.RequestException as req_err:
            self.logger.error(f"Request exception: {req_err}")
            self.data = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.data = pd.DataFrame()
        return self.data
