"""Tests for input sanitization module."""

import pytest
import pandas as pd
import numpy as np
from src.validation.sanitizer import InputSanitizer, BatchSanitizer


class TestInputSanitizer:
    """Test InputSanitizer class."""
    
    def test_sanitize_ticker_valid(self):
        """Test sanitizing valid tickers."""
        assert InputSanitizer.sanitize_ticker("AAPL") == "AAPL"
        assert InputSanitizer.sanitize_ticker("aapl") == "AAPL"
        assert InputSanitizer.sanitize_ticker("  AAPL  ") == "AAPL"
        assert InputSanitizer.sanitize_ticker("BRK.B") == "BRK.B"
        assert InputSanitizer.sanitize_ticker("SPY-OLD") == "SPY-OLD"
    
    def test_sanitize_ticker_invalid(self):
        """Test sanitizing invalid tickers."""
        assert InputSanitizer.sanitize_ticker("") is None
        assert InputSanitizer.sanitize_ticker(None) is None
        assert InputSanitizer.sanitize_ticker("THISLONGTICKERISWAYTOOLONG") is None
        assert InputSanitizer.sanitize_ticker("AAPL@#$") is None
        assert InputSanitizer.sanitize_ticker(12345) is None
    
    def test_sanitize_horizon_valid(self):
        """Test sanitizing valid horizons."""
        assert InputSanitizer.sanitize_horizon(1) == 1
        assert InputSanitizer.sanitize_horizon(30) == 30
        assert InputSanitizer.sanitize_horizon(365) == 365
        assert InputSanitizer.sanitize_horizon(30.7) == 30
        assert InputSanitizer.sanitize_horizon("60") == 60
    
    def test_sanitize_horizon_invalid(self):
        """Test sanitizing invalid horizons."""
        assert InputSanitizer.sanitize_horizon(0) is None
        assert InputSanitizer.sanitize_horizon(-10) is None
        assert InputSanitizer.sanitize_horizon(400) is None
        assert InputSanitizer.sanitize_horizon("not_a_number") is None
        assert InputSanitizer.sanitize_horizon(None) is None
    
    def test_sanitize_dataframe_valid(self):
        """Test sanitizing valid DataFrames."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]})
        result = InputSanitizer.sanitize_dataframe(df)
        assert result is not None
        assert len(result) == 3
    
    def test_sanitize_dataframe_remove_inf(self):
        """Test removing infinite values."""
        df = pd.DataFrame({"A": [1, np.inf, 3], "B": [4.0, -np.inf, 6.0]})
        result = InputSanitizer.sanitize_dataframe(df, remove_inf=True)
        assert result is not None
        assert np.isnan(result["A"].iloc[1])
        assert np.isnan(result["B"].iloc[1])
    
    def test_sanitize_dataframe_remove_nan(self):
        """Test removing NaN values."""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4.0, 5.0, 6.0]})
        result = InputSanitizer.sanitize_dataframe(df, remove_nan=True)
        assert result is not None
        assert len(result) == 2
    
    def test_sanitize_dataframe_invalid(self):
        """Test with invalid input."""
        assert InputSanitizer.sanitize_dataframe(None) is None
        assert InputSanitizer.sanitize_dataframe([1, 2, 3]) is None
        assert InputSanitizer.sanitize_dataframe(pd.DataFrame()) is None
    
    def test_sanitize_list_valid(self):
        """Test sanitizing valid lists."""
        result = InputSanitizer.sanitize_list([1, 2, 3])
        assert result == [1, 2, 3]
        
        result = InputSanitizer.sanitize_list(["a", "b"])
        assert result == ["a", "b"]
    
    def test_sanitize_list_filter_type(self):
        """Test filtering by type."""
        result = InputSanitizer.sanitize_list([1, "a", 2, "b", 3], allowed_type=int)
        assert result == [1, 2, 3]
        
        result = InputSanitizer.sanitize_list([1, "a", 2, "b"], allowed_type=str)
        assert result == ["a", "b"]
    
    def test_sanitize_list_empty(self):
        """Test with empty list."""
        assert InputSanitizer.sanitize_list([], allow_empty=True) == []
        assert InputSanitizer.sanitize_list([], allow_empty=False) is None
    
    def test_sanitize_list_invalid(self):
        """Test with invalid input."""
        assert InputSanitizer.sanitize_list(None) is None
        assert InputSanitizer.sanitize_list("not a list") is None
    
    def test_sanitize_numeric_range_valid(self):
        """Test sanitizing numeric values in range."""
        assert InputSanitizer.sanitize_numeric_range(50, min_val=0, max_val=100) == 50
        assert InputSanitizer.sanitize_numeric_range(0, min_val=0) == 0
        assert InputSanitizer.sanitize_numeric_range(100.5, max_val=200) == 100.5
    
    def test_sanitize_numeric_range_clamp(self):
        """Test clamping out-of-range values."""
        assert InputSanitizer.sanitize_numeric_range(-10, min_val=0, strict=False) == 0
        assert InputSanitizer.sanitize_numeric_range(200, max_val=100, strict=False) == 100
    
    def test_sanitize_numeric_range_strict(self):
        """Test strict range checking."""
        assert InputSanitizer.sanitize_numeric_range(-10, min_val=0, strict=True) is None
        assert InputSanitizer.sanitize_numeric_range(200, max_val=100, strict=True) is None
    
    def test_sanitize_numeric_range_invalid(self):
        """Test with invalid values."""
        assert InputSanitizer.sanitize_numeric_range(np.nan) is None
        assert InputSanitizer.sanitize_numeric_range(np.inf) is None
        assert InputSanitizer.sanitize_numeric_range("not a number") is None
        assert InputSanitizer.sanitize_numeric_range(None) is None
    
    def test_sanitize_dict_valid(self):
        """Test sanitizing valid dicts."""
        d = {"a": 1, "b": 2}
        result = InputSanitizer.sanitize_dict(d)
        assert result == d
    
    def test_sanitize_dict_required_keys(self):
        """Test checking required keys."""
        d = {"a": 1, "b": 2}
        assert InputSanitizer.sanitize_dict(d, required_keys=["a"]) == d
        assert InputSanitizer.sanitize_dict(d, required_keys=["a", "c"]) is None
    
    def test_sanitize_dict_allowed_keys(self):
        """Test filtering to allowed keys."""
        d = {"a": 1, "b": 2, "c": 3}
        result = InputSanitizer.sanitize_dict(d, allowed_keys=["a", "c"])
        assert result == {"a": 1, "c": 3}
    
    def test_sanitize_dict_invalid(self):
        """Test with invalid input."""
        assert InputSanitizer.sanitize_dict(None) is None
        assert InputSanitizer.sanitize_dict([1, 2, 3]) is None
    
    def test_sanitize_string_valid(self):
        """Test sanitizing valid strings."""
        assert InputSanitizer.sanitize_string("hello") == "hello"
        assert InputSanitizer.sanitize_string("  hello  ") == "hello"
    
    def test_sanitize_string_length(self):
        """Test string length validation."""
        assert InputSanitizer.sanitize_string("hello", max_length=10) == "hello"
        assert InputSanitizer.sanitize_string("hello", max_length=3) is None
    
    def test_sanitize_string_invalid(self):
        """Test with invalid input."""
        assert InputSanitizer.sanitize_string(None) is None
        assert InputSanitizer.sanitize_string(123) is None
        assert InputSanitizer.sanitize_string("  ") is None


class TestBatchSanitizer:
    """Test BatchSanitizer class."""
    
    def test_batch_sanitize_tickers(self):
        """Test batch ticker sanitization."""
        sanitizer = BatchSanitizer()
        tickers = ["AAPL", "invalid@#$", "MSFT", "toolongtickernamexyz"]
        results = sanitizer.sanitize_tickers(tickers)
        
        assert results[tickers[0]] is True  # AAPL
        assert results[tickers[1]] is False  # invalid chars
        assert results[tickers[2]] is True  # MSFT
        assert results[tickers[3]] is False  # too long
    
    def test_get_stats(self):
        """Test statistics calculation."""
        sanitizer = BatchSanitizer()
        tickers = ["AAPL", "invalid@", "MSFT"]
        sanitizer.sanitize_tickers(tickers)
        stats = sanitizer.get_stats()
        
        assert "tickers" in stats
        assert stats["tickers"]["total"] == 3
        assert stats["tickers"]["passed"] == 2
        assert stats["tickers"]["failed"] == 1


class TestSanitizationScenarios:
    """Test real-world sanitization scenarios."""
    
    def test_corrupted_csv_data(self):
        """Test sanitizing corrupted CSV data."""
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02", "invalid"],
            "Price": [100, np.inf, -50],
            "Volume": [1000, np.nan, 500]
        })
        
        result = InputSanitizer.sanitize_dataframe(
            df,
            remove_inf=True,
            remove_nan=True
        )
        
        assert result is not None
        # After removing inf and NaN, only the first and third rows remain
        # (first has no NaN after inf->NaN conversion started, third has no NaN)
        assert len(result) >= 1
    
    def test_batch_ticker_list_with_duplicates(self):
        """Test sanitizing ticker list with duplicates."""
        sanitizer = BatchSanitizer()
        tickers = ["AAPL", "aapl", "MSFT", "msft"]
        results = sanitizer.sanitize_tickers(tickers)
        
        # First AAPL and aapl should both pass (after sanitization)
        assert results["AAPL"] is True
        assert results["aapl"] is True
        assert results["MSFT"] is True
        assert results["msft"] is True
    
    def test_api_parameter_validation(self):
        """Test validating API parameters."""
        # Validate ticker
        ticker = InputSanitizer.sanitize_ticker("INVALID@#$")
        assert ticker is None
        
        # Validate horizon
        horizon = InputSanitizer.sanitize_horizon(400)
        assert horizon is None
        
        # Validate dict with required fields
        params = {"ticker": "AAPL", "horizon": 30}
        result = InputSanitizer.sanitize_dict(params, required_keys=["ticker", "horizon"])
        assert result is not None
    
    def test_missing_data_handling(self):
        """Test handling DataFrames with missing data."""
        df = pd.DataFrame({
            "A": [1, 2, None, 4],
            "B": [None, 5, 6, 7],
            "C": [8, 9, 10, 11]
        })
        
        # Remove NaN rows - only rows with no NaN values survive
        result = InputSanitizer.sanitize_dataframe(df, remove_nan=True)
        assert len(result) == 2  # Rows at indices 1 and 3 have no NaN
        
        # Just replace inf without removing NaN
        result2 = InputSanitizer.sanitize_dataframe(df, remove_inf=True, remove_nan=False)
        assert len(result2) == 4
    
    def test_extreme_values(self):
        """Test handling extreme values."""
        values = [
            (np.inf, False),  # Should fail
            (-np.inf, False),  # Should fail
            (1e308, True),    # Large but valid
            (-1e308, True),   # Large negative but valid
            (np.nan, False),  # Should fail
        ]
        
        for value, should_pass in values:
            result = InputSanitizer.sanitize_numeric_range(value)
            if should_pass:
                assert result is not None
            else:
                assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
