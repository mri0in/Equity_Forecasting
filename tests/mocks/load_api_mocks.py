import requests
from unittest.mock import Mock
import json


def mock_successful_response(*args, **kwargs):
    """
    Mocked successful API response with valid JSON data.
    """
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"date": "2023-01-01", "open": 100, "close": 110},
        {"date": "2023-01-02", "open": 110, "close": 115},
    ]
    mock_resp.raise_for_status = Mock()
    return mock_resp


def mock_http_error_response(*args, **kwargs):
    """
    Mocked API response that raises HTTPError (4xx or 5xx).
    """
    mock_resp = Mock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
    return mock_resp


def mock_request_exception(*args, **kwargs):
    """
    Mocked scenario where the API call raises a RequestException (e.g., timeout).
    """
    raise requests.exceptions.RequestException("Timeout occurred")


def mock_malformed_json_response(*args, **kwargs):
    """
    Mocked response with invalid JSON content.
    """
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = Mock()
    mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    return mock_resp
