import pytest
import pandas as pd
import httpx
from unittest.mock import patch, MagicMock,AsyncMock

from src.data.async_api_data import AsyncAPIDataLoader
from tests.mocks.async_api_mocks import (
    get_successful_async_response,
    get_http_error_response,
    get_malformed_json_response,
    get_failed_async_response

)
# === Test Case: TC20250614_async_001 ===
# Description : Test that async API succesfully fetch and returns correct DataFrame of the passed script.
# Component   : src/data/async_api_data.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-14 21:31
@pytest.mark.asyncio
@patch("src.data.async_api_data.httpx.AsyncClient.get")
async def test_successful_async_fetch(mock_get):
    # async mock for session was created and .get was assigned to return a mocked response
    mock_session = AsyncMock()
    mock_session.get.return_value = get_successful_async_response()

    loader = AsyncAPIDataLoader("https://fakeapi.com")
    result = await loader.fetch(session=mock_session, params={"symbol": "AAPL"})

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "symbol" in result.columns
    assert result.iloc[0]["price"] == 150

    mock_session.get.assert_awaited_once_with("https://fakeapi.com", params={"symbol": "AAPL"})

# === Test Case: TC20250614_async_002 ===
# Description : Test that in case of HTTP error (e.g. 4xx/5xx) during an api call, an empty DataFrame is returned. 
# Component   : src/data/async_api_data.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-14 21:34
@pytest.mark.asyncio
@patch("src.data.async_api_data.httpx.AsyncClient.get")
async def test_async_http_error_handling(mock_get):
    
    mock_get.return_value = get_http_error_response()

    loader = AsyncAPIDataLoader("https://fakeapi.com")
    result = await loader.fetch(httpx.AsyncClient(), {"symbol": "AAPL"})

    assert isinstance(result, pd.DataFrame)
    assert result.empty

# === Test Case: TC20250614_async_003 ===
# Description : Test in case of invalid or malformed JSON an empty DF is returned.
# Component   : src/data/async_api_data.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-14 21:43
@pytest.mark.asyncio
@patch("src.data.async_api_data.httpx.AsyncClient.get")
async def test_async_malformed_json(mock_get):
    
    mock_get.return_value = get_malformed_json_response()

    loader = AsyncAPIDataLoader("https://fakeapi.com")
    result = await loader.fetch(httpx.AsyncClient(), {"symbol": "AAPL"})

    assert isinstance(result, pd.DataFrame)
    assert result.empty

# === Test Case: TC20250614_async_004 ===
# Description : Test that when async API fetches multiple list concurrent a list of result is returned.
# Component   : src/data/async_api_data.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-14 21:58
@pytest.mark.asyncio
@patch("src.data.async_api_data.httpx.AsyncClient", autospec=True)
async def test_concurrent_async_fetch(mock_async_client_class):
    # Create mock session
    mock_session = AsyncMock()
    # When async with is used, __aenter__ returns this mock session
    mock_async_client_class.return_value.__aenter__.return_value = mock_session

    # Set .get to always return a successful async response
    mock_session.get.return_value = get_successful_async_response()

    # Instantiate and call
    loader = AsyncAPIDataLoader("https://fakeapi.com")
    param_list = [{"symbol": "TSLA"}, {"symbol": "TSLA"}]
    results = await loader.fetch_multiple(param_list)

    # Assertions
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(df, pd.DataFrame) for df in results)
    assert all(not df.empty for df in results)
    assert all("symbol" in df.columns for df in results)

    # Ensure both API calls were made
    assert mock_session.get.await_count == 2

# === Test Case: TC20250615_async_005 ===
# Description : Test when multiple async api calls returns a partial failure result one of the DataFrame should be empty.
# Component   : src/data/async_api_data.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-15 21:09
@pytest.mark.asyncio
@patch("src.data.async_api_data.httpx.AsyncClient")
async def test_concurrent_async_fetch_partial_failure(mock_async_client_class):
    
    mock_session = AsyncMock()
    mock_async_client_class.return_value.__aenter__.return_value = mock_session

    # First call succeeds, second call fails
    mock_session.get.side_effect = [
        get_successful_async_response(),
        get_failed_async_response()
    ]

    loader = AsyncAPIDataLoader("https://fakeapi.com")
    param_list = [{"symbol": "AAPL"}, {"symbol": "GOOG"}]
    results = await loader.fetch_multiple(param_list)

    # First DF should be valid and not empty
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], pd.DataFrame)
    assert not results[0].empty
    assert "symbol" in results[0].columns

    # Second one should be an empty DF due to failure
    assert isinstance(results[1], pd.DataFrame)
    assert results[1].empty

    assert mock_session.get.await_count == 2    
