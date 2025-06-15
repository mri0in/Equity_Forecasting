from unittest.mock import AsyncMock, MagicMock
import httpx


def get_successful_async_response():
    """
    Returns an AsyncMock that simulates a successful async API response.
    """

    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json = MagicMock(return_value=[{"symbol": "TSLA", "price": 150}])
    return mock_response
    
def get_http_error_response():
    """
    Returns an AsyncMock that simulates an HTTP error during the API call.
    """
    mock_response = AsyncMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="404 Not Found",
        request=None,
        response=None
    )
    return mock_response


def get_malformed_json_response():
    """
    Returns an AsyncMock that simulates a malformed JSON response.
    """

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = "malformed"  
    return mock_resp

def get_failed_async_response():
    mock_response = AsyncMock()
    mock_response.raise_for_status.side_effect = Exception("API failure")
    return mock_response