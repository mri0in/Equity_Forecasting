import pytest
from unittest.mock import patch
from src.data.load_api_data import APILoader
from tests.mocks.load_api_mocks import (
    mock_successful_response,
    mock_http_error_response,
    mock_request_exception,
    mock_malformed_json_response
)

# === Test Case: TC20250613_API_001 ===
# Description : Test successful API response is converted into a valid DataFrame.
# Component   : src/data/load_csv.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-13 21:18
@patch("src.data.load_api_data.requests.get", side_effect=mock_successful_response)
def test_api_fetch_success(mock_get):
    
    loader = APILoader(api_url="https://fakeapi.com/data")
    df = loader.fetch()

    assert not df.empty
    mock_get.assert_called_once()

# === Test Case: TC20250613_API_002 ===
# Description : Test HTTP 4xx/5xx errors are caught and handled with empty DataFrame.
# Component   : src/data/load_api_data.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-13 21:20
@patch("src.data.load_api_data.requests.get", side_effect=mock_http_error_response)
def test_api_http_error(mock_get):
    
    loader = APILoader(api_url="https://fakeapi.com/data")
    df = loader.fetch()

    assert df.empty
    mock_get.assert_called_once()

# === Test Case: TC20250613_API_003 ===
# Description : Test network-related exceptions like timeout are handled.
# Component   : src/data/load_api_data.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-13 21:21
@patch("src.data.load_api_data.requests.get", side_effect=mock_request_exception)
def test_api_request_exception(mock_get):
    
    loader = APILoader(api_url="https://fakeapi.com/data")
    df = loader.fetch()

    assert df.empty
    mock_get.assert_called_once()

# === Test Case: TC20250613_API_004 ===
# Description : Test malformed JSON raises an exception and returns empty DataFrame.
# Component   : src/data/load_api_data.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-13 21:22
@patch("src.data.load_api_data.requests.get", side_effect=mock_malformed_json_response)
def test_api_malformed_json(mock_get):
   
    loader = APILoader(api_url="https://fakeapi.com/data")
    df = loader.fetch()

    assert df.empty
    mock_get.assert_called_once()
